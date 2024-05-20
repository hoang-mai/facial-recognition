import torch
import torchvision.models as models
import torch.nn as nn

model = models.resnet50(pretrained=True)

model.fc = nn.Linear(2048, 7)
