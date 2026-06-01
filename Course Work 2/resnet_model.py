import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class WoodResNetDetector(nn.Module):
    """
    Transfer Learning model based on ResNet18 pretrained on ImageNet.

       Architecture:
        ResNet18 backbone (pretrained, 18 conv layers, 512 output features)
            ↓
        Dropout(0.4)       — regularization, prevents overfitting
            ↓
        Linear(512 → 1)    — our custom binary output layer
            ↓
        Sigmoid            — converts output to probability [0, 1]
    """

    def __init__(self, pretrained=True):
        super(WoodResNetDetector, self).__init__()

        # Load ResNet18 — with ImageNet pretrained weights if pretrained=True
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Remove the original final layer (it outputs 1000 classes for ImageNet)
        # and replace it with our own binary classifier head
        num_features = backbone.fc.in_features 

        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

        self.resnet = backbone

    def forward(self, x):
        return self.resnet(x)
