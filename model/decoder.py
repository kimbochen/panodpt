import torch
from torch import nn


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
    def __init__(self, dmax, out_size, feat_dim=256):
        super().__init__()

        self.pred_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(feat_dim // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        self.max_val = dmax
        self.scale = nn.Upsample(out_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.pred_head(x).squeeze(1)
        x = x.clamp(0.0, self.max_val)
        x = self.scale(x.unsqueeze(1)).squeeze(1)

        return x


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


class Reassemble(nn.Module):
    def __init__(self, stage, dim, patch_rc, embd_dim=768, feat_dim=256):
        super().__init__()

        self.readout_proj = ReadoutProject(embd_dim)
        self.concat = nn.Unflatten(2, patch_rc)
        self.project = nn.Conv2d(embd_dim, dim, kernel_size=1)

        if stage == 0:
            resample = nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=4)
        elif stage == 1:
            resample = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        elif stage == 2:
            resample = nn.Identity()
        elif stage == 3:
            resample = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError(f'Invalid reassemble stage {stage}.')

        self.resample = nn.Sequential(
            resample,
            nn.Conv2d(dim, feat_dim, kernel_size=3, padding=1)
        )

        self.dim = dim
        self.feat_dim = feat_dim

    def forward(self, x):
        x = self.readout_proj(x)
        x = self.concat(x.transpose(1, 2))
        x = self.project(x)
        x = self.resample(x)

        return x

class ReadoutProject(nn.Module):
    def __init__(self, embd_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * embd_dim, embd_dim),
            nn.GELU()
        )

    def forward(self, x):
        readout, x = torch.split(x, [1, x.size(1)-1], dim=1)
        readout = readout.expand_as(x)
        x = torch.cat([x, readout], dim=-1)
        x = self.mlp(x)

        return x
