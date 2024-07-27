import numpy as np
import torch
from torch import nn
import torchvision.models as models
from torch.nn import Transformer
from torchvision.models import ResNet101_Weights, ResNet50_Weights, ResNet18_Weights

from model_bases.base import CustomNN
from datasets.datasets import get_model_shape
from model_bases.resnet_base import resnet20_cifar
from model_bases.transformer_base import TransformerModel


def get_temp_state_dict(input_shape, n_class, conv_number=2, hidden=128, num_layer=2, model_name=None,
                        freeze_model=False, out_channels=8, device='cpu', config=None):
    hidden_layers = [hidden] * num_layer
    hidden_layers.append(n_class)
    if model_name == 'resnet':
        # model = ResNetCifar(n_class, freeze=freeze_model, device=device)
        model = resnet20_cifar()
    elif model_name == 'transformer':
        vars(config)['device'] = device
        vars(config)['n_class'] = n_class
        vars(config)['sequence_length'] = input_shape[0]

        model = TransformerModel(config)
    elif model_name == 'vtransformer':
        vars(config)['device'] = device
        vars(config)['n_class'] = n_class
        vars(config)['sequence_length'] = torch.prod(torch.tensor(input_shape)).item()
        vars(config)['vocab_size'] = 256

        model = TransformerModel(config, is_v=True)
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
        # model = ResNetCifar(n_class, freeze=kwargs['freeze_model'], **kwargs)
        model = resnet20_cifar()
    elif kwargs['model_name'] == 'transformer':
        vars(config)['device'] = kwargs['device']
        vars(config)['n_class'] = n_class
        vars(config)['sequence_length'] = input_shape[0]

        model = TransformerModel(config)
    elif kwargs['model_name'] == 'vtransformer':
        vars(config)['device'] = kwargs['device']
        vars(config)['n_class'] = n_class
        vars(config)['sequence_length'] = torch.prod(torch.tensor(input_shape)).item()
        vars(config)['vocab_size'] = 256

        model = TransformerModel(config, is_v=True)
    else:
        model = CustomNN(input_shape, hidden_layers, conv_number=conv_number, out_channels=out_channels, **kwargs)

    return model

