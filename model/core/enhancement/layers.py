import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.modules.mha import MHA

class CNNStem(nn.Module):
    '''
        Extracts early texture and edge information and reduces spatial size for efficient processing.
    '''
    def __init__(self, in_channels=6, out_channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class FogAttentionBlock(nn.Module):
    '''
    FlashAttention-enhanced fog attention block.
    - Uses FlashAttention for better performance and memory usage.
    - Keeps same API and structure as FogAttentionBlock to be drop-in replaceable.
    '''
    def __init__(self, dim, heads=4, with_positional_bias=True):
        super().__init__()

        self.conv_local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        # FlashAttention v2 module from MemFlow paper
        self.attn = MHA(
            embed_dim=dim,
            num_heads=heads,
            dropout=0.0,
            causal=False  # No temporal causality needed
        )

        self.norm = nn.LayerNorm(dim)
        self.with_positional_bias = with_positional_bias
        if with_positional_bias:
            self.rel_bias = nn.Parameter(torch.zeros(1, dim))  # Simple scalar bias per channel

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.gate_conv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        res = x
        x = self.conv_local(x)
        B, C, H, W = x.shape

        x_flat = x.flatten(2).transpose(1, 2).contiguous()  # [B, HW, C]
        x_norm = self.norm(x_flat)

        if self.with_positional_bias:
            x_norm = x_norm + self.rel_bias

        # FlashAttention expects (B, T, C) input
        x_attn = self.attn(x_norm)  # [B, HW, C]
        x_attn = x_attn.transpose(1, 2).view(B, C, H, W)

        weights = self.channel_attn(x_attn)
        x_attn = x_attn * weights

        fused = res + x_attn
        gate = torch.sigmoid(self.gate_conv(fused))
        return fused * gate


class IlluminationEstimator(nn.Module):
    '''
        Estimates a pixel-wise illumination map using shallow convs
    '''
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),  # Output 1-channel light map
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)


class ReflectanceEstimator(nn.Module):
    '''
        Predicts the fog-free structure of the scene (scene radiance).
    '''
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)


class AdaptiveColorCorrector(nn.Module):
    '''
        Learns a residual RGB correction map to undo color shifts caused by fog.
    '''

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 3, 1)  # RGB residual
        )

    def forward(self, x):
        return self.conv(x)

class UR2PInspiredDehazer(nn.Module):
    '''
        Fuses reflectance, illumination, and color correction into a clean feature representation.
    '''
    def __init__(self, channels):
        super().__init__()
        self.illum = IlluminationEstimator(channels)
        self.reflect = ReflectanceEstimator(channels)
        self.color = AdaptiveColorCorrector(channels)
        self.merge = nn.Conv2d(channels + 1 + 3, channels, 1)  # Combine R, I, and C

    def forward(self, feat, original):
        I = self.illum(feat)
        R = self.reflect(feat)
        C = self.color(feat)
        original = F.interpolate(original, size=R.shape[2:], mode='bilinear')
        merged_input = torch.cat([R, I.expand_as(R), C], dim=1)
        fused = self.merge(merged_input)
        return fused + original

class Reconstruction(nn.Module):
    '''
        Upsamples and refines feature maps back into image space with 6 channels (2 frames × RGB).
    '''
    def __init__(self, in_channels=64, out_channels=6):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.final = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.up(x)
        return self.final(x)
