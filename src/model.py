"""ViT backbone + direct mesh/joint regression for 3D hand pose estimation."""
import torch
import torch.nn as nn
import timm

from src.mano_layer import MANOHead


class HandPoseModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "vit_huge_patch14_dinov2.lvd142m",
        img_size: int = 224,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, img_size=img_size, num_classes=0
        )
        feat_dim = self.backbone.num_features

        self.mano = MANOHead(feat_dim=feat_dim)

        self.head_camera = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        for m in self.head_camera.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        vertices, joints_3d = self.mano(features)
        camera = self.head_camera(features)

        return {
            "joints_3d": joints_3d,
            "vertices": vertices,
            "camera": camera,
        }
