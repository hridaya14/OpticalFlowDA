import torch.nn as nn
from layers import CNNStem, FogAttentionBlock, FogSuppression, Reconstruction


class FogFormerEnhancer(nn.Module):
    def __init__(self, in_channels=6, hidden_dim=64, num_blocks=3):
        super().__init__()
        self.stem = CNNStem(in_channels, hidden_dim)
        self.blocks = nn.Sequential(*[FogAttentionBlock(hidden_dim) for _ in range(num_blocks)])
        self.suppress = FogSuppression(hidden_dim)
        self.reconstruct = Reconstruction(hidden_dim, in_channels)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.suppress(x)
        return self.reconstruct(x)
