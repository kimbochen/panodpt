import torch
from torch import nn


class Fusion(nn.Module):
    def __init__(self, feat_dim=256, use_bn=False, **kwargs):
        super().__init__()

        self.res_conv1 = ResidualConvUnit(feat_dim, use_bn)
        self.res_conv2 = ResidualConvUnit(feat_dim, use_bn)
        self.resample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.project = nn.Conv2d(feat_dim, feat_dim, kernel_size=1, stride=1)

    def forward(self, x, prev_x=None):
        x = self.res_conv1(x)

        if prev_x is not None:
            x = x + prev_x

        x = self.res_conv2(x)  # [B, feat_dim, h, w]
        x = self.resample(x)   # [B, feat_dim, 2*h, 2*w]
        x = self.project(x)    # [B, feat_dim, 2*h, 2*w]

        return x


class ResidualConvUnit(nn.Module):
    def __init__(self, feat_dim=256, use_bn=False):
        super().__init__()
        conv_args = dict(kernel_size=3, stride=1, padding=1, bias=not use_bn)

        self.activation = nn.ReLU()
        self.conv1 = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, **conv_args),
            nn.BatchNorm2d(feat_dim) if use_bn else nn.Identity(),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, **conv_args),
            nn.BatchNorm2d(feat_dim) if use_bn else nn.Identity()
        )

    def forward(self, x):
        skip_conn = x

        x = self.activation(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x = x + skip_conn

        return x
