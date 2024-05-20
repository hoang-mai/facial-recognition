
import torchvision.models as models
import torch.nn as nn
# Tải mô hình VGG19 đã được pretrained
model = models.vgg19(pretrained=True)


num_classes = 7

model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

