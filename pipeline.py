import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from numpy import pi as PI

from datasets import MP3dPSP
from model.patch_ops import TangentPatch, Scatter2D, polar_coord_grid


DEVICE = 'cuda:0'


def pipeline():
    torch.manual_seed(3985)

    ds = MP3dPSP('val')
    dl = DataLoader(ds, batch_size=4)

    grid = polar_coord_grid(fov=5*PI/180.0, patch_dim=16, npatch=864)
    x_grid, y_grid = map(lambda g: g.to(DEVICE), grid)
    tan_patch = TangentPatch(x_grid, y_grid)
    scatter2d = Scatter2D(x_grid, y_grid, 400, 1024)

    data = next(iter(dl))
    xb, gt = map(lambda b: b.to(DEVICE), data)
    gt = gt.unsqueeze(1)

    with torch.no_grad():
        xb_patch, gt_patch = map(tan_patch, (xb, gt))

    return xb_patch.shape


if __name__ == '__main__':
    pipeline()
