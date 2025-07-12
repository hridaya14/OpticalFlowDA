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

# --- Assume necessary imports are in place ---
# MemFlow imports
from core.Networks import build_network
import core.datasets_video as datasets
from core.loss import sequence_loss as memflow_sequence_loss
from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger
from core.utils.logger import Logger

# Dispnet Imports
from ucda.models import AANet
from ucda.losses import disp_pyramid_loss

# PoseNet Imports
from ucda.models import PoseNet, BackprojectDepth, Project3D
from ucda.losses import motion_consis_loss, transformation_from_parameters

try:
    from torch.cuda.amp import GradScaler
except ImportError:  # Dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self, enabled=False): self.enabled = enabled
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): optimizer.step()
        def update(self): pass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(object):
    def __init__(self, args):
        self.args = args

    def stage_1_train(gpu, cfg):
        """Stage 1: Trains MemFlow and AANet jointly."""
        rank = cfg.node_rank * cfg.gpus + gpu
        torch.cuda.set_device(rank)
        if cfg.DDP:
            dist.init_process_group(
                backend='nccl', init_method='env://', world_size=cfg.world_size, rank=rank)

        # 1. --- Initialize Models ---
        model_flow = build_network(cfg)  # MemFlow
        model_depth = AANet(cfg.aanet_args)  # AANet

        if cfg.DDP:
            model_flow = nn.SyncBatchNorm.convert_sync_batchnorm(
                model_flow).cuda()
            model_flow = nn.parallel.DistributedDataParallel(
                model_flow, device_ids=[rank])
        else:
            model_flow = nn.DataParallel(model_flow, device_ids=[gpu]).cuda()

        model_depth = nn.DataParallel(model_depth, device_ids=[gpu]).cuda()

        if rank == 0:
            loguru_logger.info("MemFlow Parameter Count: %d" %
                               count_parameters(model_flow))
            loguru_logger.info("AANet Parameter Count: %d" %
                               count_parameters(model_depth))

        # 2. --- Load Checkpoints (if any) ---
        if cfg.restore_ckpt:
            print(f"[Loading MemFlow ckpt from {cfg.restore_ckpt}]")
            # Loading logic here...
        if cfg.restore_disp_ckpt:
            print(f"[Loading AANet ckpt from {cfg.restore_disp_ckpt}]")
            model_depth.load_state_dict(torch.load(
                cfg.restore_disp_ckpt, map_location='cpu'), strict=False)

        model_flow.train()
        model_depth.train()

        # 3. --- DataLoaders and Optimizers ---
        train_loader = datasets.fetch_dataloader(cfg, DDP=cfg.DDP, rank=rank)

        # MemFlow Optimizer
        flow_optimizer, flow_scheduler = fetch_optimizer(
            model_flow, cfg.trainer)

        # AANet Optimizer
        # ... (AANet optimizer setup from your UCDA code)
        depth_optimizer = optim.Adam(model_depth.parameters(
        ), lr=cfg.trainer.lr, weight_decay=cfg.trainer.wdecay*10)
        depth_scheduler = optim.lr_scheduler.MultiStepLR(
            depth_optimizer, milestones=[400, 600, 800, 900], gamma=0.5)

        # 4. --- Training Loop ---
        total_steps = 0
        flow_scaler = GradScaler(enabled=cfg.mixed_precision)
        depth_scaler = GradScaler(enabled=cfg.mixed_precision)
        logger = Logger(model_flow, flow_scheduler, cfg)

        should_keep_training = True
        while should_keep_training:
            for i_batch, data_blob in enumerate(train_loader):
                flow_optimizer.zero_grad()
                depth_optimizer.zero_grad()

                # Unpack data
                images = data_blob['images'].cuda()
                image1_left = data_blob['left_1'].cuda()
                image1_right = data_blob['right_1'].cuda()
                flows, valids = data_blob['flow'].cuda(
                ), data_blob['valid'].cuda()
                disp, disp_mask = data_blob['disp'].cuda(
                ), data_blob['disp_mask'].cuda()

                # --- Forward Passes ---
                images_norm = 2 * (images / 255.0) - 1.0
                with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                    flow_predictions = model_flow(
                        images_norm, iters=cfg.trainer.iters)
                    depth_predictions = model_depth(image1_left, image1_right)

                # --- Loss Calculation ---
                flow_loss, flow_metrics = memflow_sequence_loss(
                    flow_predictions, flows, valids, cfg.trainer.gamma)
                disp_loss, disp_metrics = disp_pyramid_loss(
                    depth_predictions, disp, disp_mask)

                # --- Backward & Optimize Flow ---
                flow_scaler.scale(flow_loss).backward(retain_graph=True)
                flow_scaler.unscale_(flow_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model_flow.parameters(), cfg.trainer.clip)
                flow_scaler.step(flow_optimizer)
                flow_scheduler.step()
                flow_scaler.update()

                # --- Backward & Optimize Depth ---
                depth_scaler.scale(disp_loss).backward()
                depth_scaler.unscale_(depth_optimizer)
                depth_scaler.step(depth_optimizer)
                depth_scheduler.step()
                depth_scaler.update()

                total_steps += 1
                if rank == 0:
                    logger.push({**flow_metrics, **disp_metrics})

                if total_steps >= cfg.trainer.num_steps:
                    should_keep_training = False
                    break
            if not should_keep_training:
                break

        # --- Save final models ---
        if rank == 0:
            torch.save(model_flow.state_dict(), f'{
                       cfg.log_dir}/{cfg.name}_memflow_stage1.pth')
            torch.save(model_depth.state_dict(), f'{
                       cfg.log_dir}/{cfg.name}_aanet_stage1.pth')

    def stage_2_train(gpu, cfg):
        """Stage 2: Adds geometric consistency loss using a frozen PoseNet."""
        rank = cfg.node_rank * cfg.gpus + gpu
        torch.cuda.set_device(rank)
        if cfg.DDP:
            dist.init_process_group(
                backend='nccl', init_method='env://', world_size=cfg.world_size, rank=rank)

        # 1. --- Initialize All Models ---
        model_flow = build_network(cfg)  # MemFlow
        model_depth = AANet(cfg.aanet_args)  # AANet
        model_pose = PoseNet(cfg.posenet_args)  # PoseNet

        # Load Stage 1 models
        print("[Loading Stage 1 MemFlow model...]")
        model_flow.load_state_dict(torch.load(
            cfg.stage1_memflow_ckpt, map_location='cpu'))
        print("[Loading Stage 1 AANet model...]")
        model_depth.load_state_dict(torch.load(
            cfg.stage1_aanet_ckpt, map_location='cpu'))

        # Load PRETRAINED PoseNet and freeze it
        print("[Loading PRETRAINED PoseNet model...]")
        model_pose.load_state_dict(torch.load(
            cfg.restore_pose_ckpt, map_location='cpu'))
        for param in model_pose.parameters():
            param.requires_grad = False

        # Setup DDP/DataParallel and move to GPU
        # ... (model setup as in Stage 1)
        model_flow.cuda().train()
        model_depth.cuda().train()  # AANet can also be fine-tuned
        model_pose.cuda().eval()   # PoseNet is for inference only

        # 2. --- Initialize Projection helpers ---
        h, w = cfg.image_size
        backproject_depth = BackprojectDepth(cfg.batch_size, h, w).cuda()
        project_3d = Project3D(cfg.batch_size, h, w).cuda()

        # 3. --- Optimizers (only for trainable models) ---
        flow_optimizer, flow_scheduler = fetch_optimizer(
            model_flow, cfg.trainer)
        depth_optimizer = optim.Adam(model_depth.parameters(
        ), lr=cfg.trainer.lr * 0.1)  # Lower LR for fine-tuning

        # 4. --- Training Loop ---
        # ... (setup logger, scalers, etc. as in Stage 1)
        train_loader = datasets.fetch_dataloader(cfg, DDP=cfg.DDP, rank=rank)

        while should_keep_training:
            for i_batch, data_blob in enumerate(train_loader):
                # ... (zero grads, unpack data as in Stage 1)

                # --- Forward Passes ---
                with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                    flow_predictions = model_flow(
                        images_norm, iters=cfg.trainer.iters)
                    depth_predictions = model_depth(image1_left, image1_right)

                # --- Calculate Rigid Flow (from PoseNet) ---
                with torch.no_grad():
                    pred_poses = model_pose(
                        image1_left, image2_left)  # Simplified
                    pred_disp_final = depth_predictions[-1]
                    # Convert disparity to depth
                    _, depth = disp_to_depth(pred_disp_final, 0.1, 100)
                    cam_points, _ = backproject_depth(depth, inv_K)
                    rigid_flow = project_3d(cam_points, K, pred_poses)

                # --- Loss Calculation ---
                flow_loss, flow_metrics = memflow_sequence_loss(
                    flow_predictions, flows, valids, cfg.trainer.gamma)
                disp_loss, disp_metrics = disp_pyramid_loss(
                    depth_predictions, disp, disp_mask)

                # Add Geometric Loss
                geo_loss, motion_metrics = motion_consis_loss(
                    flow_predictions, rigid_flow, valids)
                total_flow_loss = flow_loss + cfg.geo_weight * geo_loss

                # --- Backward & Optimize ---
                # ... (Separate backward steps for total_flow_loss and disp_loss)

        # --- Save final models ---
        if rank == 0:
            torch.save(model_flow.state_dict(), f'{
                       cfg.log_dir}/{cfg.name}_memflow_stage2.pth')
            torch.save(model_depth.state_dict(), f'{
                       cfg.log_dir}/{cfg.name}_aanet_stage2.pth')

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
