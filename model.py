from __future__ import print_function, division
import argparse
import numpy as np
from pathlib import Path
import random
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import core.datasets as datasets
from core.utils.utils import transformation_from_parameters, disp_to_depth, filter_base_params, filter_specific_params


# --- Assume necessary imports are in place ---
# MemFlow imports
from core.Memflow import build_network
# Dispnet Imports
from core.depth_nets.core.mocha_stereo import Mocha
from loss import disp_pyramid_loss, sequence_loss

# PoseNet Imports
from core.posenet import PoseNet
from tool import BackprojectDepth, Logger, Project3D

try:
    from torch.cuda.amp.grad_scaler import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    print("dummy GradScaler for PyTorch < 1.6")
    class GradScaler:
        def __init__(self, enabled=True):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(object):
    def __init__(self, args):
        self.args = args

    def stage_1_train(self):
        # dataset_loader
        train_loader = datasets.fetch_dataloader(self.args)
        # train_loader = datasets.fetch_clean_dataloader(self.args)

        # param setting
        # MemFlow
        model_flow = nn.DataParallel(build_network(self.args), device_ids=self.args.gpus)
        print("Parameter Count: %d" % count_parameters(model_flow))

        if self.args.restore_flow_ckpt is not None:
            model_flow.load_state_dict(torch.load(self.args.restore_flow_ckpt), strict=False)

        model_flow.cuda()
        model_flow.train()

        # aanet
        model_depth = Mocha(self.args)
        print("Parameter Count: %d" % count_parameters(model_depth))

        if self.args.restore_disp_ckpt is not None:
            model_depth.load_state_dict(torch.load(self.args.restore_disp_ckpt), strict=False)

        model_depth = nn.DataParallel(model_depth, device_ids=self.args.gpus)

        model_depth.cuda()
        model_depth.train()

        # if self.args.stage != 'chairs':
        #     model_flow.module.freeze_bn()

        # training process
        flow_optimizer = optim.AdamW(model_flow.parameters(), lr=self.args.lr, weight_decay=self.args.wdecay, eps=self.args.epsilon)
        flow_scheduler = optim.lr_scheduler.OneCycleLR(flow_optimizer, self.args.lr, self.args.num_steps+100,
            pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')


        # disp optimizer
        specific_params = list(filter(filter_specific_params,
                                  model_depth.named_parameters()))
        base_params = list(filter(filter_base_params,
                                model_depth.named_parameters()))
        specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
        base_params = [kv[1] for kv in base_params]

        specific_lr = self.args.lr * 0.1
        milestones = [400, 600, 800, 900]
        params_group = [
            {'params': base_params, 'lr': self.args.lr},
            {'params': specific_params, 'lr': specific_lr},
        ]
        depth_optimizer = optim.Adam(params_group, weight_decay=self.args.wdecay*10)
        depth_scheduler = optim.lr_scheduler.MultiStepLR(depth_optimizer, milestones=milestones, gamma=0.5, last_epoch=-1)

        # depth_scheduler = optim.lr_scheduler.OneCycleLR(depth_optimizer, self.args.lr, self.args.num_steps+100,
        #     pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
        # optimizer, scheduler = fetch_optimizer(self.args, model_flow)

        total_steps = 0
        flow_scaler = GradScaler(enabled=self.args.mixed_precision)
        depth_scaler = GradScaler(enabled=self.args.mixed_precision)
        logger = Logger('checkpoints/', model_flow, flow_scheduler)

        VAL_FREQ = 5000
        VAL_SUMMARY_FREQ = 500

        add_noise = False
        should_keep_training = True

        adaptive_weight = 0.0

        while should_keep_training:
            for i_batch, data_blob in enumerate(train_loader):
                flow_optimizer.zero_grad()
                depth_optimizer.zero_grad()
                image1_left = data_blob['left_1'].cuda()
                image2_left = data_blob['left_2'].cuda()
                image1_right = data_blob['right_1'].cuda()
                image2_right = data_blob['right_2'].cuda()
                flow = data_blob['flow'].cuda()
                valid = data_blob['valid'].cuda()
                disp = data_blob['disp'].cuda()
                # image1, image2, flow, valid, disp  = [x.cuda() for x in data_blob]

                disp_mask = (disp > 0) & (disp < self.args.max_disp)

                if self.args.load_pseudo_gt:
                    pseudo_gt_disp = data_blob['pseudo_disp'].cuda()
                    pseudo_mask = (pseudo_gt_disp > 0) & (pseudo_gt_disp < self.args.max_disp) & (~disp_mask)  # inverse mask

                if not disp_mask.any():
                    continue

                if add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1_left = (image1_left + stdv * torch.randn(*image1_left.shape).cuda()).clamp(0.0, 255.0)
                    image2_left = (image2_left + stdv * torch.randn(*image2_left.shape).cuda()).clamp(0.0, 255.0)
                    image1_right = (image1_right + stdv * torch.randn(*image1_right.shape).cuda()).clamp(0.0, 255.0)
                    image2_right = (image2_right + stdv * torch.randn(*image2_right.shape).cuda()).clamp(0.0, 255.0)

                flow_predictions,_ = model_flow(image1_left, image2_left, iters=self.args.iters)
                _,depth_predictions = model_depth(image1_left, image1_right)  # list of H/12, H/6, H/3, H/2, H

                flow_loss, flow_metrics = sequence_loss(flow_predictions, flow, valid, self.args.gamma)
                if self.args.load_pseudo_gt:
                    disp_loss, disp_metrics = disp_pyramid_loss(depth_predictions, disp, disp_mask, pseudo_gt_disp, pseudo_mask, self.args.load_pseudo_gt)
                else:
                    disp_loss, disp_metrics = disp_pyramid_loss(depth_predictions, disp, disp_mask, disp, disp_mask, self.args.load_pseudo_gt)

                # flow network update
                flow_scaler.scale(flow_loss).backward()
                flow_scaler.unscale_(flow_optimizer)
                torch.nn.utils.clip_grad_norm_(model_flow.parameters(), self.args.clip)

                flow_scaler.step(flow_optimizer)
                flow_scheduler.step()
                flow_scaler.update()

                # disp network update
                depth_scaler.scale(adaptive_weight * disp_loss).backward()
                depth_scaler.unscale_(depth_optimizer)
                # torch.nn.utils.clip_grad_norm_(model_flow.parameters(), self.args.clip)

                depth_scaler.step(depth_optimizer)
                depth_scheduler.step()
                depth_scaler.update()


                total_steps += 1

                # print info
                dict_metric = dict(flow_metrics, **disp_metrics)
                logger.push(dict_metric)
                # logger.push(disp_metrics)

                # test
                # disp = depth_predictions[-1][-1].detach().cpu().numpy()  # [H, W]
                # skimage.io.imsave('pred_disp/' + str(total_steps) + '.png', (disp * 256.).astype(np.uint16))

                if total_steps % VAL_SUMMARY_FREQ == 0:
                    img_summary = dict()
                    img_summary['left_1'] = image1_left
                    img_summary['right_1'] = image1_right
                    img_summary['gt_flow'] = flow
                    img_summary['pred_flow'] = flow_predictions[-1]
                    img_summary['gt_disp'] = disp
                    img_summary['pred_disp'] = depth_predictions[-1]
                    logger.save_image('train', img_summary)


                if total_steps % VAL_FREQ == 0:
                    FLOW_PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, 'raft')
                    torch.save(model_flow.state_dict(), FLOW_PATH)

                    DISP_PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, 'aanet')
                    torch.save(model_depth.state_dict(), DISP_PATH)


                    # results = {}
                    # for val_dataset in self.args.validation:
                    #     if val_dataset == 'chairs':
                    #         results.update(evaluate.validate_chairs(model_flow.module))
                    #     elif val_dataset == 'sintel':
                    #         results.update(evaluate.validate_sintel(model_flow.module))
                    #     elif val_dataset == 'kitti':
                    #         results.update(evaluate.validate_kitti(model_flow.module))

                    # logger.write_dict(results)

                    # model_flow.train()
                    # if self.args.stage != 'chairs':
                    #     model_flow.module.freeze_bn()

                if total_steps > self.args.num_steps:
                    should_keep_training = False
                    break

        logger.close()
        FLOW_PATH = 'checkpoints/%s.pth' % 'memflow'
        DISP_PATH = 'checkpoints/%s.pth' % 'mocha'
        torch.save(model_flow.state_dict(), FLOW_PATH)
        torch.save(model_depth.state_dict(), DISP_PATH)

        return FLOW_PATH, DISP_PATH



    # def generate_depth_foggy_images(self):
    #     if not os.path.isdir('generate_images'):
    #         os.mkdir('generate_images')

    #     if not os.path.isdir('pred_depth'):
    #         os.mkdir('pred_depth')

    #     if not os.path.isdir('pred_disp'):
    #         os.mkdir('pred_disp')

    #     fx = 721.53
    #     baseline = 53.72 # cm

    #     k = 0.88  #atmospheric
    #     beta = 0.06   #attenuation factor

    #     IMAGENET_MEAN = [0.485, 0.456, 0.406]
    #     IMAGENET_STD = [0.229, 0.224, 0.225]
    #     test_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    #     # all_samples = sorted(glob(self.args.data_dir + '/training/image_2/*.png'))
    #     all_samples = sorted(glob(self.args.data_dir + 'KITTI/training/image_2/*.png'))
    #     num_samples = len(all_samples)
    #     print('=> %d samples found in the data dir' % num_samples)

    #     # param setting
    #     # aanet
    #     model_disp = AANet(self.args)
    #     print("Parameter Count: %d" % count_parameters(model_disp))

    #     model_disp.load_state_dict(torch.load(self.args.restore_disp_ckpt), strict=False)

    #     model_disp = nn.DataParallel(model_disp, device_ids=self.args.gpus)

    #     model_disp.cuda()
    #     model_disp.eval()

    #     for i, sample_name in enumerate(all_samples):
    #         if i % 100 == 0:
    #             print('=> Inferencing %d/%d' % (i, num_samples))

    #         left_name = sample_name
    #         right_name = left_name.replace('image_2', 'image_3')

    #         left = np.array(Image.open(left_name).convert('RGB')).astype(np.float32)
    #         right = np.array(Image.open(right_name).convert('RGB')).astype(np.float32)

    #         temp_left = cv2.imread(left_name)

    #         sample = {'left': left,
    #               'right': right}

    #         sample = test_transform(sample)  # to tensor and normalize

    #         left = sample['left'].cuda()  # [3, H, W]
    #         left = left.unsqueeze(0)  # [1, 3, H, W]
    #         right = sample['right'].cuda()
    #         right = right.unsqueeze(0)

    #         ori_height, ori_width = left.size()[2:]

    #         # Automatic
    #         factor = 48
    #         img_height = math.ceil(ori_height / factor) * factor
    #         img_width = math.ceil(ori_width / factor) * factor

    #         if ori_height < img_height or ori_width < img_width:
    #             top_pad = img_height - ori_height
    #             right_pad = img_width - ori_width

    #             # Pad size: (left_pad, right_pad, top_pad, bottom_pad)
    #             left = F.pad(left, (0, right_pad, top_pad, 0))
    #             right = F.pad(right, (0, right_pad, top_pad, 0))

    #         with torch.no_grad():
    #             pred_disp = model_disp(left, right)[-1]

    #         if pred_disp.size(-1) < left.size(-1):
    #             pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
    #             pred_disp = F.interpolate(pred_disp, (left.size(-2), left.size(-1)),
    #                                     mode='bilinear') * (left.size(-1) / pred_disp.size(-1))
    #             pred_disp = pred_disp.squeeze(1)  # [B, H, W]

    #         # Crop
    #         if ori_height < img_height or ori_width < img_width:
    #             if right_pad != 0:
    #                 pred_disp = pred_disp[:, top_pad:, :-right_pad]
    #             else:
    #                 pred_disp = pred_disp[:, top_pad:]

    #         disp = pred_disp[0].detach().cpu().numpy()  # [H, W]
    #         saved_disp_name = 'pred_disp/' + os.path.basename(left_name)
    #         disp = (disp * 256.).astype(np.uint16)
    #         skimage.io.imsave(saved_disp_name, disp)

    #         saved_depth_name = 'pred_depth/' + os.path.basename(left_name)
    #         depth = 1/disp * fx * baseline
    #         im_color=cv2.applyColorMap(cv2.convertScaleAbs(depth,alpha=15),cv2.COLORMAP_BONE)
    #         im=Image.fromarray(im_color)
    #         im.save(saved_depth_name)

    #         saved_foggy_name = 'generate_images/' + os.path.basename(left_name)
    #         fog = gemerate_haze(temp_left, depth, k, beta)
    #         cv2.waitKey(3)
    #         cv2.imwrite(saved_foggy_name, fog)
