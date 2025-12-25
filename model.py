import torch.nn as nn
from torchvision.models import resnet18

def get_model(num_classes=2, pretrained=True):
    if pretrained:
        model = resnet18(weights="IMAGENET1K_V1")
    else:
        model = resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
