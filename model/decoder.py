import torch
from torch import nn

from .fusion import Fusion
from .reass import Reassemble


class ConvDecoder(nn.Module):
    def __init__(self, patch_rc, **kwargs):
        super().__init__()
        stage_dim=[96, 192, 384, 768]

        self.reass = nn.ModuleList([
            Reassemble(idx, dim, patch_rc, **kwargs)
            for idx, dim in enumerate(stage_dim)
        ])
        self.fusions = nn.ModuleList([Fusion(**kwargs) for _ in stage_dim])

    def forward(self, feats):
        feat_maps = [rsmbl(feat) for rsmbl, feat in zip(self.reass, feats)]

        feat = None
        for fusion, next_feat in zip(self.fusions, reversed(feat_maps)):
            feat = fusion(next_feat, feat)

        return feat


class DepthPredHead(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()

        self.pred_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(feat_dim // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.pred_head(x).squeeze(1)
        x = x.clamp(0.0, 10.0)

        return x
