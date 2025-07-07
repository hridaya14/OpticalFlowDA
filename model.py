import os
import sys
import ptlflow


sys.path.append('core')
import datasets
import argparse
import os
import os.path as osp
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import skimage
from core.datasets import KittiCleanFoggyDataset

from torch.utils.data import DataLoader
# from aanet import AANet

from glob import glob
# from tool import gemerate_haze

from PIL import Image
from utils import transforms

from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils import flow_utils


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Model(object):
    def __init__(self,args):
        self.args = args

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

    def stage_1_train(self):
        # Loading dataset
        train_dataset = KittiCleanFoggyDataset()
        train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)


        #loading models and weights
        clean_flow_model = ptlflow.get_model("flowformer", ckpt_path="kitti")
        clean_flow_model.to(device)
        clean_flow_model.eval()

        for batch in train_loader:

            clean_images = batch['clean_images'].to(device)
            foggy_images = batch['foggy_images']

            with torch.no_grad():
                clean_predictions = clean_flow_model({'images': clean_images})
                final_clean_flow = clean_predictions['flows'][-1]












