


import argparse

import numpy as np
from core.posenet import PoseNet
import torch


def load_posenet(args):
    model_pose = {}
    model_pose['encoder'] = PoseNet(args)['pose_encoder']
    model_pose['decoder'] = PoseNet(args)['pose_decoder']
    model_pose['decoder'].load_state_dict(torch.load(args.restore_pose_decoder_ckpt), strict=False)
    print("Parameter Count: %d" % count_parameters(model_pose['encoder']))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    parser = argparse.ArgumentParser()
    # PoseNet

    parser.add_argument('--restore_pose_encoder_ckpt', help="restore checkpoint")
    parser.add_argument('--restore_pose_decoder_ckpt', help="restore checkpoint")

    parser.add_argument('--num_layers', default=18, type=int, choices=[18, 34, 50, 101, 152], help='number of resnet layers')
    parser.add_argument('--pose_model_type', default='seperate_pose', type=str,
                        help='Similarity measure for matching cost')

    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)

    load_posenet(args)


