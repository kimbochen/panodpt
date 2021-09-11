import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_mean
from numpy import pi as PI


class ToTangentPatch(nn.Module):
    def __init__(self, fov, patch_dim, npatch, img_h, img_w):
        super().__init__()
        x_grid, y_grid = polar_coord_grid(fov, patch_dim, npatch)
        self.tan_patch = TangentPatch(x_grid, y_grid)
        self.scatter2d = Scatter2D(x_grid, y_grid, img_h, img_w)

    def forward(self, xb, gt):
        with torch.no_grad():
            xb_patch = self.tan_patch(xb)
            gt_patch = self.tan_patch(gt.unsqueeze(1))
            gt = self.scatter2d(gt_patch)

        return xb_patch, gt


class TangentPatch(nn.Module):
    def __init__(self, theta, phi):
        super().__init__()

        # Scale grids to [-1, 1]
        x_coord = (theta / PI).flatten(0, 1)
        y_coord = (-phi / PI * 2.0).flatten(0, 1)

        # Create flattened grid and unflatten op
        self.register_buffer("grid", torch.stack([x_coord, y_coord], dim=-1))
        self.unflatten = nn.Unflatten(-2, theta.shape[:2])

    def forward(self, x):
        batch_grid = self.grid.expand(x.size(0), *self.grid.size())
        tan_patch = F.grid_sample(x, batch_grid, align_corners=True)
        tan_patch = self.unflatten(tan_patch)  # (N, C, P, D, D)
        return tan_patch


class Scatter2D(nn.Module):
    def __init__(self, theta, phi, y_max, x_max):
        super().__init__()

        # theta: [-pi, pi] -> [0, x_max], phi: [pi/2, -pi/2] -> [0, y_max]
        self.y_max, self.x_max = y_max, x_max
        x_coord = (theta / PI + 1.0) * (self.x_max - 1) / 2.0
        y_coord = (-phi / PI + 0.5) * (self.y_max - 1)

        # Convert data type and dimension
        x_coord = x_coord.round().long().flatten(0, 1)
        y_coord = y_coord.round().long().flatten(0, 1)

        # Create image coordinates
        y_coord = scatter_mean(y_coord, x_coord, dim_size=x_max)
        self.register_buffer('x_coord', x_coord.unsqueeze(0))
        self.register_buffer('y_coord', y_coord.unsqueeze(0))

    def forward(self, x):
        x = x.flatten(-3, -2)
        x = scatter_mean(x, self.x_coord.expand_as(x), dim_size=self.x_max)
        x = scatter_mean(x, self.y_coord.expand_as(x), dim_size=self.y_max, dim=-2)
        return x.squeeze(1)


def polar_coord_grid(fov, patch_dim, npatch):
    assert fov < PI and fov > 0.0, f'Invalid FOV: got {fov} rad.'

    # Create sphereical grid in Cartesian coordinates
    amp = torch.tan(torch.as_tensor(fov / 2.0, dtype=torch.float32))
    tan_ax = torch.linspace(-amp, amp, steps=patch_dim)
    tan_y, tan_x = torch.meshgrid(-tan_ax, tan_ax)

    # Convert grid to polar coordinates
    theta = torch.atan2(tan_x, torch.as_tensor(1.0))
    phi = torch.atan2(tan_y, torch.sqrt(tan_x ** 2.0 + 1.0))

    # Create rotated grids
    u = 2.0 * PI * torch.rand(npatch, 1, 1)
    v = torch.asin(2.0 * torch.rand(npatch, 1, 1) - 1.0)
    theta, phi = theta + u, phi + v

    # Convert to reference angles
    theta[theta > PI] -= 2.0 * PI
    theta[theta < -PI] += 2.0 * PI
    phi[phi > PI / 2.0] -= PI
    phi[phi < -PI / 2.0] += PI

    return theta, phi
