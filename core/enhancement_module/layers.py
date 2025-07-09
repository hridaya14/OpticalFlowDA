import torch
import torch.nn as nn


class CNNStem(nn.Module):
    '''
        Extract early texture edges and reduce noise before attention layers.
    '''
    def __init__(self, in_channels=6, out_channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class FogAttentionBlock(nn.Module):
    '''
        Use both spatial and channel-wise attention to model haze/fog patterns.
    '''
    def __init__(self, dim, heads=4, window_size=8):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Tanh()
        )

    def forward(self, x):
        res = x
        x = self.depthwise(x)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # B, HW, C
        x_attn, _ = self.attn(x_flat, x_flat, x_flat)
        x_attn = x_attn.transpose(1, 2).view(B, C, H, W)

        # Channel Attention
        w = self.channel_fc(x_attn)
        x_attn = x_attn * w

        return res + x_attn


class FogSuppression(nn.Module):
    '''
    Predict the residual haze or transmission to subtract it from input.
    '''
    def __init__(self, channels):
        super().__init__()
        self.fog_predictor = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Tanh()
        )

    def forward(self, x):
        fog_residual = self.fog_predictor(x)
        return x - fog_residual


class Reconstruction(nn.Module):
    '''
    Reconstruct high-res output and refine enhanced images.
    '''
    def __init__(self, in_channels=64, out_channels=6):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.final = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x):
        x = self.up(x)
        return self.final(x)
