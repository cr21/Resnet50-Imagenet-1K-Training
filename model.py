import torch.nn as nn
import torch.nn as nn
import torchvision.models as models

class ResNet50Wrapper(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50Wrapper, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)