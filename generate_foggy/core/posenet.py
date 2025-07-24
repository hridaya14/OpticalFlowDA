import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.pose_nets.pose_cnn import PoseCNN
from core.pose_nets.pose_decoder import PoseDecoder
from core.pose_nets.resnet_encoder import ResnetEncoder


# optical flow network
def PoseNet(args):
    posenet = {}
    if args.pose_model_type == 'seperate_pose':
        posenet['pose_encoder'] = ResnetEncoder(50, True, 2)
        posenet['pose_decoder'] = PoseDecoder(posenet['pose_encoder'].num_ch_enc, 1, 2)

        return posenet
    elif args.pose_model_type == 'pose_cnn':
        posenet['pose_cnn'] = PoseCNN(2)

    return posenet


