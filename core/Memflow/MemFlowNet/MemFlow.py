import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .update import GMAUpdateBlock
from ..encoders import twins_svt_large
from .cnn import BasicEncoder
from .corr import CorrBlock
from ...utils.utils import coords_grid
from .sk import SKUpdateBlock6_Deep_nopoolres_AllDecoder
from .sk2 import SKUpdateBlock6_Deep_nopoolres_AllDecoder2_Mem_skflow
from .memory_util import *

# Flash attention removed

class MemFlowNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.hidden_dim = 128
        self.context_dim = 128

        cfg.corr_radius = 4
        cfg.corr_levels = 4

        # feature network, context network, and update block
        if cfg.cnet == 'twins':
            print("[Using twins as context encoder]")
            self.cnet = twins_svt_large(pretrained=self.cfg.pretrain)
            self.proj = nn.Conv2d(256, 256, 1)
        elif cfg.cnet == 'basicencoder':
            print("[Using basicencoder as context encoder]")
            self.cnet = BasicEncoder(output_dim=256, norm_fn='batch')

        if cfg.fnet == 'twins':
            print("[Using twins as feature encoder]")
            self.fnet = twins_svt_large(pretrained=self.cfg.pretrain)
            self.channel_convertor = nn.Conv2d(256, 256, 1, padding=0, bias=False)
        elif cfg.fnet == 'basicencoder':
            print("[Using basicencoder as feature encoder]")
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance')

        if self.cfg.gma == "GMA":
            print("[Using GMA]")
            self.update_block = GMAUpdateBlock(self.cfg, hidden_dim=128)
        elif self.cfg.gma == 'GMA-SK':
            print("[Using GMA-SK]")
            self.cfg.cost_heads_num = 1
            self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder(args=self.cfg, hidden_dim=128)
        elif self.cfg.gma == 'GMA-SK2':
            print("[Using GMA-SK2]")
            self.cfg.cost_heads_num = 1
            self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder2_Mem_skflow(args=self.cfg, hidden_dim=128)

        print("[Using corr_fn {}]".format(self.cfg.corr_fn))

        self.att = nn.MultiheadAttention(embed_dim=self.context_dim, num_heads=1, batch_first=True)
        self.train_avg_length = cfg.train_avg_length

    def encode_features(self, frame, flow_init=None):
        if len(frame.shape) == 5:
            need_reshape = True
            b, t = frame.shape[:2]
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            need_reshape = False
        else:
            raise NotImplementedError

        fmaps = self.fnet(frame).float()
        if self.cfg.fnet == 'twins':
            fmaps = self.channel_convertor(fmaps)
        if need_reshape:
            fmaps = fmaps.view(b, t, *fmaps.shape[-3:])
            frame = frame.view(b, t, *frame.shape[-3:])
            coords0, coords1 = self.initialize_flow(frame[:, 0, ...])
        else:
            coords0, coords1 = self.initialize_flow(frame)
        if flow_init is not None:
            coords1 = coords1 + flow_init

        return coords0, coords1, fmaps

    def encode_context(self, frame):
        if len(frame.shape) == 5:
            need_reshape = True
            b, t = frame.shape[:2]
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            need_reshape = False
        else:
            raise NotImplementedError

        cnet = self.cnet(frame)
        if self.cfg.cnet == 'twins':
            cnet = self.proj(cnet)

        net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        query = key = inp

        if need_reshape:
            query = query.view(b, t, *query.shape[-3:]).transpose(1, 2).contiguous()
            key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()
            net = net.view(b, t, *net.shape[-3:])
            inp = inp.view(b, t, *inp.shape[-3:])

        return query, key, net, inp

    def predict_flow(self, net, inp, coords0, coords1, fmaps, query, ref_keys, ref_values, test_mode=False):
        corr_fn = CorrBlock(fmaps[:, 0, ...], fmaps[:, 1, ...],
                            num_levels=self.cfg.corr_levels, radius=self.cfg.corr_radius)
        flow_predictions = []

        # Flatten query/key/value from [B, C, H, W] to [B, L, C]
        query = query.flatten(start_dim=2).permute(0, 2, 1)        # [B, L, C]
        ref_keys = ref_keys.flatten(start_dim=2).permute(0, 2, 1)  # [B, L, C]

        for _ in range(self.cfg.decoder_depth):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # [B, num_corr_levels, H, W]
            flow = coords1 - coords0

            # Get motion features and value for this frame
            motion_features, current_value = self.update_block.get_motion_and_value(flow, corr)
            current_value = current_value.unsqueeze(2)  # [B, C, 1, H, W]

            # Update value memory
            value = current_value if ref_values is None else torch.cat([ref_values, current_value], dim=2)  # [B, C, N, H, W]
            value = value.flatten(start_dim=2).permute(0, 2, 1)  # [B, L, C]

            # Attention: query [B, L, C], key [B, L, C], value [B, L, C]
            hidden_states, _ = self.att(query, ref_keys, value)  # [B, L, C]
            hidden_states = hidden_states.permute(0, 2, 1).reshape(motion_features.shape)

            # Fusion and update
            motion_features_global = motion_features + self.update_block.aggregator.gamma * hidden_states
            net, up_mask, delta_flow = self.update_block(net, inp, motion_features, motion_features_global)

            coords1 = coords1 + delta_flow
            flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up, current_value
        else:
            return flow_predictions, current_value


    def initialize_flow(self, img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, images, b, cfg):
        query, key, net, inp = self.encode_context(images[:, :-1, ...])
        coords0, coords1, fmaps = self.encode_features(images)
        values = None
        video_flow_predictions = []
        for ti in range(0, cfg.input_frames - 1):
            if ti < cfg.num_ref_frames:
                ref_values = values
                ref_keys = key[:, :, :ti + 1]
            else:
                indices = [torch.randperm(ti)[:cfg.num_ref_frames - 1] for _ in range(b)]
                ref_values = torch.stack([
                    values[bi, :, indices[bi]] for bi in range(b)
                ], 0)
                ref_keys = torch.stack([
                    key[bi, :, indices[bi]] for bi in range(b)
                ], 0)
                ref_keys = torch.cat([ref_keys, key[:, :, ti].unsqueeze(2)], dim=2)

            flow_pr, current_value = self.predict_flow(net[:, ti], inp[:, ti], coords0, coords1,
                                                       fmaps[:, ti:ti + 2], query[:, :, ti], ref_keys,
                                                       ref_values)
            values = current_value if values is None else torch.cat([values, current_value], dim=2)
            video_flow_predictions.append(torch.stack(flow_pr, dim=0))

        video_flow_predictions = torch.stack(video_flow_predictions, dim=2)
        return video_flow_predictions
