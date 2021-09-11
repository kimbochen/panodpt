import timm
import torch
import torchvision.transforms as T
from torch import nn

PATCH_D = 24
EMBED_D = 768


class ViTBackbone(nn.Module):
    def __init__(self, npatch, patch_dim, dropout=False):
        super().__init__()

        self.model = timm.create_model(
            'vit_base_patch16_384', num_classes=0, pretrained=True
        )
        self.patch_embed = PatchEmbed(patch_dim=patch_dim)
        self.upsample = nn.Upsample(npatch, mode='nearest')
        self.dropout = dropout
        self.stages = [2, 5, 8, 11]

    def forward(self, x):
        '''x: N x C x H x W'''

        # Patch projection
        x = self.patch_embed(x)
        x = self.upsample(x.transpose(1, 2)).transpose(1, 2)
        cls_token = self.model.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Resize positional embedding
        token, grid = self.model.pos_embed.split([1, PATCH_D ** 2], 1)
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
        self.project = nn.Linear(3*(patch_dim**2), EMBED_D)

    def forward(self, x):
        '''x: N x C x P x D x D'''

        x = x.transpose(1, 2).flatten(2)  # N x P x CDD
        x = self.project(x)               # N x P x EMBED_D

        return x


class ViTConv(nn.Module):
    def __init__(self, patch_rc, dropout=False):
        super().__init__()

        self.model = timm.create_model(
            'vit_base_patch16_384', num_classes=0, pretrained=True
        )
        self.upsample = nn.Upsample(patch_rc, mode='bilinear', align_corners=True)
        self.dropout = dropout
        self.stages = [2, 5, 8, 11]

    def forward(self, x):
        '''x: N x C x H x W'''

        # Patch projection
        x = self.model.patch_embed.proj(x)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.model.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Resize positional embedding
        token, grid = self.model.pos_embed.split([1, PATCH_D ** 2], 1)
        grid = grid.transpose(1, 2).reshape(1, -1, PATCH_D, PATCH_D)
        grid = self.upsample(grid).flatten(2).transpose(1, 2)
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
