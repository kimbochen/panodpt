import timm
import torch
import torchvision.transforms as T
from torch import nn

PATCH_D = 24  # ViT config


class ViTBackbone(nn.Module):
    def __init__(self, npatch, dropout):
        super().__init__()

        self.model = timm.create_model(
            'vit_base_patch16_384', num_classes=0, pretrained=True
        )
        self.patch_embed = PatchEmbed(patch_dim=16)
        self.upsample = nn.Upsample(npatch, mode='nearest')
        self.dropout = dropout
        self.stages = [2, 5, 8, 11]

    def forward(self, x):
        # Patch projection
        x = self.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Resize positional embedding
        token, grid = self.model.pos_embed.split([1, 576], 1)
        grid = self.upsample(grid.transpose(1, 2)).transpose(1, 2)
        pos_embed = torch.cat([token, grid], dim=1)

        # ViT forward
        x = x + pos_embed
        if self.dropout:
            x = self.model.pos_drop(x)

        features = []
        for stage, block in enumerate(self.model.blocks):
            x = block(x)
            if stage in self.stages:
                features.append(x)

        return features


class PatchEmbed(nn.Module):
    def __init__(self, patch_dim):
        super().__init__()
        self.linear = nn.Linear(3*patch_dim*patch_dim, 768)

    def forward(self, x):
        x = x.transpose(1, 2)  # N x P x C x D x D
        x = x.flatten(2)       # N x P x CDD
        x = self.linear(x)     # N x P x 768
        return x
