from torch.utils.data import Dataset
from torchvision.datasets import FashionMNIST
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import v2
import torch

EXTRACTED_DATASETS = ['birds', 'flowers', 'pets', 'food101']
DATASETS = EXTRACTED_DATASETS + ['fmnist', 'year_pred', 'mnist', 'cifar10', 'cifar100']
NUM_CLASSES = {'birds': 500,
               'flowers': 102,
               'cifar10': 10,
               'pets': 37,
               'food101': 101,
               'fmnist': 10,
               'year_pred': 1,
               'mnist': 10,
               'cifar100': 100,
               }
INPUT_DIM = {'mnist': [1, 28, 28],
             'fmnist': [1, 28, 28],
             'cifar10': [3, 32, 32],
             'cifar100': [3, 32, 32],
             'flowers': [3, 224, 224],
             'birds': [3, 224, 224],
             'pets': [3, 224, 224],
             'food101': [3, 224, 224],
             'year_pred': [90],
             }


class CustomDataset(Dataset):
    def __init__(self, x, labels):
        self.x = x
        self.labels = labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx]


def get_model_shape(dataset_name):
    assert dataset_name in DATASETS
    return INPUT_DIM[dataset_name], NUM_CLASSES[dataset_name]


def get_dataset(dataset_name, path=None, **kwargs):
    if dataset_name == "mnist":
        transform_mnist = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.1307,), (0.3081,))
        ])
        dataset = MNIST(path + "data", train=True, download=True, transform=transform_mnist)
        test_dataset = MNIST(path + "data", train=False, download=True, transform=transform_mnist)
    elif dataset_name == "fmnist":
        transform_fmnist = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.1307,), (0.3081,))
        ])
        dataset = FashionMNIST(path + "data", train=True, download=True, transform=transform_fmnist)
        test_dataset = FashionMNIST(path + "data", train=False, download=True, transform=transform_fmnist)
    elif dataset_name == "cifar10":
        transform_cifar = v2.Compose([
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToTensor(),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = CIFAR10(path + "data", train=True, download=True, transform=transform_cifar)
        test_dataset = CIFAR10(path + "data", train=False, download=True, transform=transform_cifar)
    elif dataset_name == "cifar100":
        transform_cifar = v2.Compose([
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToTensor(),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = CIFAR100(path + "data", train=True, download=True, transform=transform_cifar)
        test_dataset = CIFAR100(path + "data", train=False, download=True, transform=transform_cifar)
    elif dataset_name in EXTRACTED_DATASETS:
        dataset, test_dataset = get_extracted_dataset(dataset_name, path, **kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented yet.")

    input_shape = INPUT_DIM[dataset_name]
    n_class = NUM_CLASSES[dataset_name]

    return dataset, test_dataset, input_shape, n_class


def get_extracted_dataset(dataset_name, path, finetuning="fully_finetuned", augmentation="unaugmented"):
    train_X, train_labels = torch.load(path + f'data/{dataset_name}_{finetuning}_{augmentation}/' + 'train_data')
    test_X, test_labels = torch.load(path + f'data/{dataset_name}_{finetuning}_{augmentation}/' + 'test_data')
    train_set = CustomDataset(train_X, train_labels)
    val_set = CustomDataset(test_X, test_labels)
    return train_set, val_set
