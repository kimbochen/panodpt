from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .base import MP3dDepthDataset

SPLIT = {
    'train': 'scenes_train.txt',
    'val': 'mp3d_scenes_test.txt',
    'test': 'scenes_val.txt'
}
ROOT = Path('/mnt/home_6T/sunset/matterport3d')


class MP3D(MP3dDepthDataset):
    def __init__(self, mode, dmax, crop_h):
        rand = (mode == 'train')
        super().__init__(
            root=ROOT/'mp3d_rgbd',
            scene_txt=ROOT/SPLIT[mode],
            dmin=0.01, dmax=dmax,
            rand_rotate=rand, rand_flip=rand, rand_gamma=rand,
            rand_pitch=0, rand_roll=0, fix_pitch=0, fix_roll=0
        )
        self.transform = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.CenterCrop([crop_h, 1024])
        ])

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        rgb = self.transform(sample['x'])
        depth = self.transform(sample['depth']).squeeze(0)
        return rgb, depth
