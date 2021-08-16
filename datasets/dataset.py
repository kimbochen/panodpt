from pathlib import Path

import torch
import numpy as np
import torchvision.transforms as T

from .base_dataset import MP3dDepthDataset
from .eqr2psp import e2p

SPLIT = {
    'train': 'scenes_train.txt',
    'val': 'mp3d_scenes_test.txt',
    'test': 'scenes_val.txt'
}
ROOT = Path('/mnt/home_6T/sunset/matterport3d')


class MP3dDataset(MP3dDepthDataset):
    def __init__(self, mode, resize_h):
        assert mode in ['train', 'val', 'test'], f'Invalid mode {mode}.'

        rgbd_path = ROOT / 'mp3d_rgbd'
        scene_txt = ROOT / SPLIT[mode]

        tsfm = (mode == 'train')
        rand_kwargs = dict(
            dmin=0.01, dmax=10,
            rand_rotate=tsfm, rand_flip=tsfm, rand_gamma=tsfm,
            rand_pitch=0, rand_roll=0, fix_pitch=0, fix_roll=0
        )
        super().__init__(rgbd_path, scene_txt, **rand_kwargs)

        self.data_tsfm = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.CenterCrop([384, 1024])
        ])
        self.resize = T.Resize(resize_h)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        rgb = self.resize(self.data_tsfm(sample['x']))
        depth = self.data_tsfm(sample['depth']).squeeze(0)
        return rgb, depth

    def __len__(self):
        return super().__len__()


class MP3dPSP(MP3dDepthDataset):
    def __init__(self, mode):
        rfg = (mode == 'train')
        super().__init__(
            root=ROOT / 'mp3d_rgbd', scene_txt=ROOT / SPLIT[mode],
            dmin=0.01, dmax=10,
            rand_rotate=rfg, rand_flip=rfg, rand_gamma=rfg,
            rand_pitch=0, rand_roll=0, fix_pitch=0, fix_roll=0
        )

        self.transform = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.CenterCrop([400, 1024])
        ])

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        rgb = self.transform(sample['x'])
        depth = self.transform(sample['depth']).squeeze(0)
        return rgb, depth
