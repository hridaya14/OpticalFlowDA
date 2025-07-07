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

    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
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

    parser.add_argument('--load_pseudo_gt', action='store_true', help='Load pseudo gt for supervision')


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
