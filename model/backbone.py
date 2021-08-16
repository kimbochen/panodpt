import timm
import torch
import torchvision.transforms as T
from torch import nn

PATCH_D = 24  # ViT config


class ViTBackbone(nn.Module):
    def __init__(self, patch_rc, dropout):
        super().__init__()

        self.model = timm.create_model(
            'vit_base_patch16_384', num_classes=0, pretrained=True
        )
        self.norm = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.upsample = nn.Upsample(patch_rc, mode='bilinear', align_corners=True)
        self.dropout = dropout
        self.stages = [2, 5, 8, 11]

    def forward(self, x):
        # Normalize image
        x = self.norm(x)

        # Patch projection
        x = self.model.patch_embed.proj(x)
        x = x.flatten(2).transpose(1, 2)

        cls_token = self.model.cls_token
        cls_token = cls_token.expand(x.size(0), -1, -1)

        x = torch.cat([cls_token, x], dim=1)

        # Resize positional embedding
        token, grid = torch.split(self.model.pos_embed, [1, PATCH_D ** 2], 1)

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
