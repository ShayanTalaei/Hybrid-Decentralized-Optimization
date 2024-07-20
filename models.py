import numpy as np
import torch
from torch import nn
import torchvision.models as models
from torch.nn import Transformer
from torchvision.models import ResNet101_Weights, ResNet50_Weights, ResNet18_Weights

from TransformerModels.base import GPTBaseClassification
from datasets.datasets import get_model_shape


def get_temp_state_dict(input_shape, n_class, conv_number=2, hidden=128, num_layer=2, model_name=None,
                        freeze_model=False, out_channels=8, device='cpu', config=None):
    hidden_layers = [hidden] * num_layer
    hidden_layers.append(n_class)
    if model_name == 'resnet':
        model = ResNetModel(n_class, freeze=freeze_model, device=device)
    elif model_name == 'transformer':
        vars(config)['device'] = device
        vars(config)['n_class'] = n_class
        vars(config)['sequence_length'] = input_shape[0]

        model = GPTBaseClassification(config)
    else:
        model = CustomNN(input_shape, hidden_layers, conv_number=conv_number, out_channels=out_channels, device=device)
    state_dict = model.state_dict()
    return state_dict


def get_model(dataset_name, conv_number=2, hidden=128, num_layer=2, out_channels=8, config=None, **kwargs):
    kwargs['dataset_name'] = dataset_name
    input_shape, n_class = get_model_shape(dataset_name)
    hidden_layers = [hidden] * num_layer
    hidden_layers.append(n_class)
    if kwargs['model_name'] == 'resnet':
        model = ResNetModel(n_class, freeze=kwargs['freeze_model'], **kwargs)
    elif kwargs['model_name'] == 'transformer':
        vars(config)['device'] = kwargs['device']
        vars(config)['n_class'] = n_class
        vars(config)['sequence_length'] = input_shape[0]

        model = GPTBaseClassification(config)
    else:
        model = CustomNN(input_shape, hidden_layers, conv_number=conv_number, out_channels=out_channels, **kwargs)

    return model


class EnhancedModel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.acc_def = "class number"
        device_name = kwargs.get("device", 'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device_name)
        self.to(self.device)

    def get_sd(self):
        return self.state_dict()

    def evaluate(self, dataloader, criterion):
        model = self
        model_initial_training = model.training
        model.eval()

        total_loss = 0
        count = 0
        corrects = 0
        data_count = 0
        result = {}

        with torch.no_grad():
            for data in dataloader:
                xb, yb = data
                xb, yb = xb.to(self.device), yb.to(self.device)
                count += 1
                outputs = model(xb)
                loss = criterion(outputs, yb)
                data_count += xb.shape[0]
                total_loss += loss
                if self.acc_def == "class number":
                    preds = torch.argmax(outputs, dim=1)
                    corrects += torch.sum((preds == yb).float())
                else:
                    corrects += torch.sum((torch.abs(outputs - yb) < 0.01).float())

        result["Loss"] = float(total_loss / count)
        result["Accuracy"] = float(corrects / data_count)
        model.train(model_initial_training)
        return result


class CustomNN(EnhancedModel):
    """
    Simple feedforward neural network.
    """

    def __init__(self, input_shape, hidden, activation='relu', sigmoid_output=True, conv_number=1, out_channels=8,
                 **kwargs):
        """
        Initialize the neural network.
        :param input_shape: The shape of the input.
        :param hidden: A list of hidden layer sizes.
        :param activation: The activation function to use.
        :param sigmoid_output: Whether to use a sigmoid activation on the output.
        :param conv_number: The number of convolutional layers to use.
        """
        super().__init__(**kwargs)
        self.seq = nn.Sequential()
        if conv_number > 0:
            self.seq.append(nn.Conv2d(in_channels=input_shape[0], out_channels=out_channels, kernel_size=3, stride=1,
                                      padding=1,
                                      bias=True))
            self.seq.append(nn.BatchNorm2d(out_channels))
            self.seq.append(nn.ReLU())
            for i in range(conv_number - 1):
                self.seq.append(nn.Conv2d(in_channels=out_channels * (2 ** i), out_channels=out_channels * (2 ** (i + 1)), kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bias=True))
                self.seq.append(nn.BatchNorm2d(out_channels * (2 ** (i + 1))))
                self.seq.append(nn.ReLU())
                self.seq.append(nn.MaxPool2d(kernel_size=2))
            input_size = out_channels * (2 ** (conv_number - 1)) * ((input_shape[1] // (2 ** (conv_number - 1))) *
                                                          (input_shape[2] // (2 ** (conv_number - 1))))
        else:
            input_size = np.prod(input_shape)
        self.seq.append(nn.Flatten())
        self.seq.append(nn.Linear(input_size, hidden[0]))
        self.seq.append(nn.BatchNorm1d(hidden[0]))
        for i in range(len(hidden) - 1):
            if activation == 'relu':
                self.seq.append(nn.ReLU())
            else:
                self.seq.append(nn.Tanh())
            self.seq.append(nn.Linear(hidden[i], hidden[i + 1]))
            self.seq.append(nn.BatchNorm1d(hidden[i + 1]))
        if sigmoid_output:
            self.seq.append(nn.Sigmoid())
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass.
        :param x: The input.
        :return: The output.
        """
        return self.seq(x)

    def _init_weights(self, module):
        module.weight.data.uniform_(-100, 100)
        module.bias.data.zero_()


class ResNetModel(EnhancedModel):
    """
    Simple feedforward neural network.
    """

    def __init__(self, num_classes, freeze=False, **kwargs):
        """
        Initialize the resnet model.
        :param num_classes: The number of classes.
        :param grad_mode: The grad mode.
        :param criterion: The criterion.
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
