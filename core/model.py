import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES


def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, NUM_CLASSES),
    )

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("layer4.1") or name.startswith("fc")

    return model