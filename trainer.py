from argparse import ArgumentParser
from functools import partial

import pytorch_lightning as pl
import torch.optim as opt
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader
from numpy import pi as PI

from data.dataset import MP3D
from data.patch_op import ToTangentPatch
from model import ConvDecoder, DepthPredHead, ViTBackbone
from util import Delta1


NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

FOV = 5.0 * PI / 180.0
PATCH_DIM = 16
IMG_H, IMG_W = 384, 1024

PATCH_RC = (18, 48)
NPATCH = 864
DMAX = 10.0

LR = 1e-4
MAX_EPOCHS = 120
GAMMA = 0.9


class PanoDPT(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.norm = T.Normalize(mean=NORM_MEAN, std=NORM_STD)
        self.tan_patch = ToTangentPatch(FOV, PATCH_DIM, NPATCH, IMG_H, IMG_W)
        self.model = nn.Sequential(
            ViTBackbone(npatch=NPATCH, patch_dim=PATCH_DIM),
            ConvDecoder(PATCH_RC),
            DepthPredHead(dmax=DMAX, out_size=[IMG_H, IMG_W])
        )

        self.loss_fn = nn.L1Loss()
        self.lr = LR
        self.poly_coeff = lambda epoch: (1.0 - epoch / MAX_EPOCHS) ** GAMMA

        self.log = partial(self.log, on_step=False, on_epoch=True) 
        self.train_metric = Delta1()
        self.val_metric = Delta1()

    def step(self, xb, gt):
        xb = self.norm(xb)
        xb_patch, gt = self.tan_patch(xb, gt)

        pred = self.model(xb_patch)
        valid_depth = (gt > 0.0)

        return pred[valid_depth], gt[valid_depth]

    def training_step(self, batch, idx):
        pred, gt = self.step(*batch)

        loss = self.loss_fn(pred, gt)
        self.log('MAE/Train', loss)

        delta1 = self.train_metric(pred, gt)
        self.log('Delta1/Train', delta1)

        return loss

    def validation_step(self, batch, idx):
        pred, gt = self.step(*batch)

        loss = self.loss_fn(pred, gt)
        self.log('MAE/Val', loss)

        delta1 = self.val_metric(pred, gt)
        self.log('Delta1/Val', delta1)

        return loss

    def configure_optimizers(self):
        optimizer = opt.Adam(self.parameters(), lr=self.lr)
        scheduler = opt.lr_scheduler.LambdaLR(optimizer, self.poly_coeff)
        return [optimizer], [scheduler]


def main():
    pl.seed_everything(3985, workers=True)

    parser = ArgumentParser('PanoDPT trainer script')
    parser.add_argument('--gpu', '-g', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--testrun', '-t', action='store_true')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    args.precision = 16
    args.deterministic = True
    args.max_epochs = MAX_EPOCHS
    args.log_every_n_steps = MAX_EPOCHS
    args.gpus = args.gpu
    args.fast_dev_run = args.testrun
    args.accelerator = 'ddp' if len(args.gpus) > 1 else None

    if not args.fast_dev_run:
        args.logger = pl.loggers.TensorBoardLogger('.', 'new_logs')
        args.logger.log_hyperparams({
            'fov': FOV / PI * 180.0, 'patch_dim': PATCH_DIM, 'npatch': NPATCH,
            'lr': LR, 'epochs': MAX_EPOCHS, 'gamma': GAMMA
        })

    trainer = pl.Trainer.from_argparse_args(args)

    model = PanoDPT()

    dl = partial(DataLoader, batch_size=4, num_workers=4, pin_memory=True)
    ds = partial(MP3D, dmax=DMAX, crop_h=IMG_H)
    train_dl, val_dl = dl(ds('train')), dl(ds('val'))

    trainer.fit(model, train_dl, val_dl)

if __name__ == '__main__':
    main()
