import torch
from torch.utils.data import Dataset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import random
import pickle

import pdb

EXTRACTED_DATASETS = ['birds', 'flowers', 'cifar10', 'pets', 'food101']
DATASETS = EXTRACTED_DATASETS + ['mnist', 'year_pred']

ROOT = "../Extracted Datasets/" #../

def get_batchsize(dataset_name):
    if dataset_name in EXTRACTED_DATASETS:
        return 128
    elif dataset_name == "year_pred":
        return 128
    elif dataset_name == "fmnist":
        return 128
        
class CustomDataset(Dataset):
    def __init__(self, X, labels):
        self.X = X
        self.labels = labels
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]

def get_dataset(dataset_name, **kwargs):
    if dataset_name in EXTRACTED_DATASETS:
        return get_extracted_dataset(dataset_name, **kwargs)
    elif dataset_name == "year_pred":
        return get_year_pred_dataset()
    elif dataset_name == "fmnist":
        return get_fashion_mnist_dataset()
    
def get_extracted_dataset(dataset_name, finetuning="fully_finetuned", augmentation="unaugmented"):
    ROOT = "Extracted Datasets/" #../
    train_X, train_labels = torch.load(ROOT+ f'{dataset_name}_{finetuning}_{augmentation}/'+'train_data')
    test_X, test_labels   = torch.load(ROOT+ f'{dataset_name}_{finetuning}_{augmentation}/'+'test_data')
    train_set = CustomDataset(train_X, train_labels)
    val_set  = CustomDataset(test_X, test_labels)
    return train_set, val_set

def cast(lst, dtype=torch.float32):
    return list(map(lambda x: torch.tensor(x).to(dtype), lst))

def get_year_pred_dataset():
#     with open("../../data/YearPred/YearPredictionMSD.pickle", "rb") as f:
#         X, label = pickle.load(f)
# #     with open("../../data/YearPred/Year_Pred.pickle", "rb") as f:
# #         X, label = pickle.load(f)
# #         X = list(X)
# #     pdb.set_trace()
#     X = list(map(lambda x: torch.tensor(x).to(torch.float32), X))
#     label = list(map(lambda x: torch.tensor([x]).to(torch.float32), label))
    X, label = torch.load("Extracted Datasets/year_prediction/data")
    joint = list(zip(X, label))
    random.shuffle(joint)
    X, label = zip(*joint)    

    train_size = 2**17 #462000 #2**14 
    X_train, label_train = X[:train_size], label[:train_size]
    X_val, label_val = X[-10000:], label[-10000:]

    train_set = CustomDataset(X_train, label_train)
    val_set   = CustomDataset(X_val, label_val)
    return train_set, val_set

def get_fashion_mnist_dataset():
    ROOT = "../../data/"
    train_ds = FashionMNIST(ROOT + "Fashion MNIST/train_data", download=True, train=True,  transform=ToTensor())
    val_ds   = FashionMNIST(ROOT + "Fashion MNIST/test_data" , download=True, train=False, transform=ToTensor()) 
    return train_ds, val_ds