import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 8, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = F.relu(self.pointwise(self.depthwise(x)))
        b, c, _, _ = out.shape
        scale = self.fc(self.pool(out).view(b, c)).view(b, c, 1, 1)
        return out * scale

class MiniEnhanceBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, in_channels)
        self.attn = DepthwiseSEBlock(in_channels)
        self.res = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out = self.attn(self.norm(x))
        return out + self.res(out)

class FogEnhancer(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=32, num_blocks=3):
        super().__init__()
        self.initial = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[MiniEnhanceBlock(hidden_dim) for _ in range(num_blocks)])
        self.final = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # process each frame independently
        feat = self.initial(x)
        feat = self.blocks(feat)
        out = self.final(feat)
        out = out + x  # residual connection
        return out.view(B, T, C, H, W)
