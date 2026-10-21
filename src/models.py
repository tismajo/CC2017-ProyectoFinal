"""Modelos: CNN desde cero y modelos preentrenados de torchvision adaptados a 43 clases."""
from torch import nn
from torchvision import models

from .data import NUM_CLASSES


def conv_block(cin, cout, use_bn=True, p_drop=0.1):
    """[Conv-BN-ReLU] x2 -> MaxPool -> Dropout2d."""
    layers = []
    for c_in in (cin, cout):
        layers += [nn.Conv2d(c_in, cout, 3, padding=1, bias=not use_bn)]
        if use_bn:
            layers += [nn.BatchNorm2d(cout)]
        layers += [nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(2), nn.Dropout2d(p_drop)]
    return nn.Sequential(*layers)


class TrafficCNN(nn.Module):
    """CNN de 3 bloques convolucionales (32/64/128 filtros) con pooling global."""

    def __init__(self, num_classes=NUM_CLASSES, channels=(32, 64, 128), use_bn=True, p_drop_conv=0.1, p_drop_fc=0.5):
        super().__init__()
        blocks, cin = [], 3
        for c in channels:
            blocks.append(conv_block(cin, c, use_bn, p_drop_conv))
            cin = c
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(cin, 256), nn.ReLU(inplace=True), nn.Dropout(p_drop_fc), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    if freeze_backbone:
        set_trainable(model.features, False)
    return model


def build_resnet18(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if freeze_backbone:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("fc.")
    return model


def set_trainable(module, trainable=True):
    for p in module.parameters():
        p.requires_grad = trainable


def unfreeze_last_blocks(mobilenet, n_blocks=3):
    """Descongela los últimos n bloques de mobilenet.features para fine-tuning."""
    for block in mobilenet.features[-n_blocks:]:
        set_trainable(block, True)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
