from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/SceneFlow', type=str, help='Training dataset')

    parser.add_argument('--stage', help="determines which dataset to use for training")

    parser.add_argument('--restore_flow_ckpt', help="restore checkpoint")
    parser.add_argument('--restore_disp_ckpt', help="restore checkpoint")
    parser.add_argument('--restore_pose_encoder_ckpt', help="restore checkpoint")
    parser.add_argument('--restore_pose_decoder_ckpt', help="restore checkpoint")

    # syn foggy
    parser.add_argument('--restore_flow_synimg_ckpt', help="restore checkpoint")
    # real foggy
    parser.add_argument('--restore_flow_realimg_ckpt', help="restore checkpoint")

    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--val_batch_size', default=64, type=int, help='Batch size for validation')

    # camera size
    parser.add_argument('--camera_size', type=int, nargs='+', default=[375, 1242])

    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # RAFT
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')

    parser.add_argument('--use_context_attention', action='store_true', help='Use spatial context for attention')
    parser.add_argument('--use_non_local', type=bool, default=False)


    # AANet
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')

    parser.add_argument('--feature_similarity', default='correlation', type=str,
                        help='Similarity measure for matching cost')
    parser.add_argument('--num_downsample', default=2, type=int, help='Number of downsample layer for feature extraction')
    parser.add_argument('--num_scales', default=3, type=int, help='Number of stages when using parallel aggregation')

    parser.add_argument('--load_pseudo_gt', action='store_true', help='Load pseudo gt for supervision')

    #Mocha
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

    # PoseNet
    parser.add_argument('--num_layers', default=18, type=int, choices=[18, 34, 50, 101, 152], help='number of resnet layers')
    parser.add_argument('--pose_model_type', default='seperate_pose', type=str,
                        help='Similarity measure for matching cost')

    # disp-to-depth
    parser.add_argument('--min_depth', default=0.1, type=float, help='minimum depth')
    parser.add_argument('--max_depth', default=100.0, type=float, help='maximum depth')

    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)

    torch.backends.cudnn.benchmark = True

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    model = Model(args)
    print("=> Start training\n\n")

    model.stage_1_train()

    print("=> End training\n\n")
