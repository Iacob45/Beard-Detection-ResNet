import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES


def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=NUM_CLASSES,
    )

    return model