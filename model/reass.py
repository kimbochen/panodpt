import torch
from torch import nn


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
