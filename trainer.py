from argparse import ArgumentParser

import torch
import torch.optim as opt
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Metric
from pytorch_lightning import LightningModule, Trainer, seed_everything
from numpy import pi as PI

from datasets import MP3dDataset
from model import ConvDecoder, DepthPredHead, ViTBackbone
from model.patch_ops import TangentPatch, Scatter2D, polar_coord_grid


class PanoDPT(LightningModule):
    def __init__(self, patch_rc, max_epochs, lr, gamma, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['gpu', 'testrun', 'bsnw'])

        npatch = patch_rc[0]*patch_rc[1]
        self.norm = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        x_grid, y_grid = polar_coord_grid(fov=5.0*PI/180.0, patch_dim=16, npatch=npatch)
        self.tan_patch = TangentPatch(x_grid, y_grid)
        self.scatter2d = Scatter2D(x_grid, y_grid, 16*patch_rc[0], 16*patch_rc[1])

        self.backbone = ViTBackbone(npatch=npatch, dropout=False)
        self.decoder = ConvDecoder(patch_rc)
        self.pred_head = DepthPredHead()

        self.loss_fn = nn.L1Loss()
        self.lr = lr
        self.poly_lr = lambda epoch: (1 - epoch / max_epochs) ** gamma

        self.train_metric = Delta1()
        self.val_metric = Delta1()

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        x = self.pred_head(x)
        return x

    def step(self, xb, gt):
        with torch.no_grad():
            xb_patch = self.tan_patch(self.norm(xb))
            gt_patch = self.tan_patch(gt.unsqueeze(1))
            gt = self.scatter2d(gt_patch)

        pred = self.forward(xb_patch)
        valid_depth = (gt > 0.0)

        return pred[valid_depth], gt[valid_depth]

    def training_step(self, batch, batch_idx):
        pred, gt = self.step(*batch)

        loss = self.loss_fn(pred, gt)
        delta1 = self.train_metric(pred, gt)

        self.log('MAE/Train', loss, on_step=False, on_epoch=True)
        self.log('Delta1/Train', delta1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        pred, gt = self.step(*batch)

        loss = self.loss_fn(pred, gt)
        delta1 = self.val_metric(pred, gt)

        self.log('MAE/Val', loss, on_step=False, on_epoch=True)
        self.log('Delta1/Val', delta1, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = opt.Adam(self.parameters(), lr=self.lr)
        scheduler = opt.lr_scheduler.LambdaLR(optimizer, self.poly_lr)
        return [optimizer], [scheduler]


class Delta1(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('correct', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('n_batch', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, pred, gt):
        pred, gt = pred.detach(), gt.detach()
        self.correct += (torch.max(pred / gt, gt / pred) < 1.25).float().mean()
        self.n_batch += 1

    def compute(self):
        return self.correct / self.n_batch


if __name__ == '__main__':
    parser = ArgumentParser('DPT trainer.')

    parser.add_argument('--gpu', '-g', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--bsnw', nargs='+', type=int, default=[4, 4])
    parser.add_argument('--patch_rc', nargs='+', type=int, default=[18, 48])
    parser.add_argument('--max_epochs', type=int, default=90)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--testrun', '-t', action='store_true')

    args = parser.parse_args()

    seed_everything(3985)

    model = PanoDPT(**vars(args))

    def get_dataloader(mode):
        patch_rc = [16 * d for d in args.patch_rc]
        ds = MP3dDataset(mode, patch_rc)
        bs, nw = args.bsnw
        dl = DataLoader(ds, batch_size=bs, num_workers=nw, pin_memory=True)
        return dl
    train_dl, val_dl = get_dataloader('train'), get_dataloader('val')

    trainer = Trainer(
        gpus=args.gpu, accelerator='ddp', precision=16, deterministic=True,
        max_epochs=args.max_epochs, fast_dev_run=args.testrun
    )

    trainer.fit(model, train_dl, val_dl)
