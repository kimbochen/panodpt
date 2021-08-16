import torch
import numpy as np
import torch.nn.functional as F


def psp_xyz(h_fov, v_fov, u, v, out_hw, in_rot):
    def get_axis(fov, nstep):
        amp = torch.tan(fov / 2.0)
        return torch.linspace(-amp, amp, steps=nstep)
    out_h, out_w = out_hw
    axis_x, axis_y = get_axis(h_fov, out_w), get_axis(-v_fov, out_h)

    grid_y, grid_x = torch.meshgrid(axis_y, axis_x)
    grid_z = torch.ones(out_h, out_w, dtype=torch.float32)
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)

    cos_v, sin_v = torch.cos(v), torch.sin(v)
    Rx = torch.tensor([[1, 0, 0], [0, cos_v, -sin_v], [0, sin_v, cos_v]])

    cos_u, sin_u = torch.cos(u), torch.sin(u)
    Ry = torch.tensor([[cos_u, 0, sin_u], [0, 1, 0], [-sin_u, 0, cos_u]])

    ang = torch.tensor([0.0, 0.0, 1.0]) @ Rx @ Ry
    ux, uy, uz = torch.split(ang, 1)
    cos_i, sin_i = torch.cos(in_rot), torch.sin(in_rot)
    cross = torch.tensor([
        [cos_i,       sin_i * -uz, sin_i * uy],
        [sin_i * uz,  cos_i,       sin_i * -ux],
        [sin_i * -uy, sin_i * ux,  cos_i]
    ])
    Ri = cross + (1.0 - cos_i) * torch.outer(ang, ang)

    xyz = grid @ Rx @ Ry @ Ri

    return xyz

def xyz2uv(xyz):
    x, y, z = torch.split(xyz, 1, dim=-1)
    u = torch.atan2(x, z)
    c = torch.sqrt(x ** 2 + z ** 2)
    v = torch.atan2(y, c)
    uv = torch.cat([u, v], dim=-1)
    return uv

def uv2coor(uv, h, w):
    u, v = torch.split(uv, 1, dim=-1)
    x = (u / (2 * np.pi) + 0.5) * w - 0.5
    y = (-v / np.pi + 0.5) * h - 0.5
    return torch.cat([x, y], dim=-1)

def pad_image(img):
    w = img.size(2)
    pad_u = torch.roll(img[:, 0, :].unsqueeze(1), w // 2, 2)
    pad_d = torch.roll(img[:, -1, :].unsqueeze(1), w // 2, 2)
    img = torch.cat([img, pad_d, pad_u], dim=1)
    return img

def e2p(img, fov_deg, u_deg, v_deg, out_hw, in_rot_deg=0, mode='bilinear'):
    to_radian = lambda deg: torch.as_tensor(deg * np.pi / 180.0)
    h_fov, v_fov = map(to_radian, fov_deg)
    u, v, in_rot = map(to_radian, (-u_deg, v_deg, in_rot_deg))
    h, w = img.shape[1:]

    pad_img = pad_image(img)
    xyz = psp_xyz(h_fov, v_fov, u, v, out_hw, in_rot)
    uv = xyz2uv(xyz)
    xy = uv2coor(uv, h, w)
    grid = (xy / torch.tensor([[[w, h]]]) - 0.5) * 2.0

    pad_img = pad_img.unsqueeze(0)
    grid = grid.unsqueeze(0)
    psp_img = F.grid_sample(
        pad_img, grid, mode,
        padding_mode='reflection', align_corners=True
    )

    return psp_img.squeeze(0)
