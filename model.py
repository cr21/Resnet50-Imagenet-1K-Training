import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Wrapper(nn.Module):
    def __init__(self, num_classes=1000, use_checkpoint=False):
        super(ResNet50Wrapper, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Optionally enable gradient checkpointing at initialization
        if use_checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, x):
        return self.model(x)

    def use_gradient_checkpointing(self):
        """Enable gradient checkpointing if not already enabled."""
        self.model.gradient_checkpointing_enable()
