import torch.nn as nn

from torchvision import models

from torchvision.models import ResNet18_Weights

from model_bases.base import EnhancedModel


class ResNetModel(EnhancedModel):
    """
    Simple feedforward neural network.
    """

    def __init__(self, num_classes, freeze=False, **kwargs):
        """
        Initialize the resnet model.
        :param num_classes: The number of classes.
        :param freeze: Whether to freeze the model except the last layer.
        :param kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.weight = ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=self.weight)
        self.preprocess = self.weight.transforms()
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass.
        :param x: The input.
        :return: The output.
        """
        x = self.preprocess(x)
        return self.model(x)

