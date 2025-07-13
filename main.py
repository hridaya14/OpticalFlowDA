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
# from model import Model

# =======================================================
#               START DIAGNOSTIC BLOCK
# =======================================================
print("--- GPU DIAGNOSTICS ---")
print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is NOT available.")

print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
print("-----------------------")
# =======================================================
#                END DIAGNOSTIC BLOCK
# =======================================================


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

    #Memflow
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--use_context_attention', action='store_true', help='Use spatial context for attention')
    parser.add_argument('--use_non_local', type=bool, default=False)

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
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    parser.add_argument('--load_pseudo_gt', action='store_true', help='Load pseudo gt for supervision')

    # PoseNet
    parser.add_argument('--num_layers', default=18, type=int, choices=[18, 34, 50, 101, 152], help='number of resnet layers')
    parser.add_argument('--pose_model_type', default='seperate_pose', type=str,
                        help='Similarity measure for matching cost')


    args = parser.parse_args()

    # if args.stage == 'kitti':
    #     from core.Memflow.configs.kitti_memflownet import get_cfg
    # elif args.stage == 'sintel':
    #     from core.Memflow.configs.sintel_memflownet import get_cfg
    # # Add other stages as needed
    # # elif args.stage == 'sintel':
    # #     from configs.sintel_stereo_flow_pose import get_cfg
    # else:
    #     raise ValueError(f"Unknown stage: {args.stage}")

    # # --- Merging args into cfg ---
    # cfg = get_cfg()
    # cfg.update(vars(args))

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)

    torch.backends.cudnn.benchmark = True

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    # model = Model(cfg)
    print("=> Start training\n\n")

    # model.stage_1_train()

    print("=> End training\n\n")
