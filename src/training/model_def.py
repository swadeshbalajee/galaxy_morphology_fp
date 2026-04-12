from __future__ import annotations

import torch.nn as nn
from torchvision import models

SUPPORTED_BACKBONES = {
    'resnet18': models.resnet18,
}


def build_model(num_classes: int, backbone_name: str = 'resnet18'):
    if backbone_name not in SUPPORTED_BACKBONES:
        raise ValueError(f'Unsupported backbone: {backbone_name}')

    model = SUPPORTED_BACKBONES[backbone_name](weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
