import torch
from torch import Tensor, nn
from torch.nn import Linear
from copy import copy, deepcopy

import pdb

EXTRACTED_DATASETS = ['birds', 'flowers', 'cifar10', 'pets', 'food101']
NUM_CLASSES = {'birds':500, 'flowers':102, 'cifar10':10, 'pets':37, 'food101': 101}
DATASETS = EXTRACTED_DATASETS + ['fmnist', 'year_pred']


def get_in_out_shape(dataset_name):
    if dataset_name in EXTRACTED_DATASETS:
        return (2048, NUM_CLASSES[dataset_name])
    elif dataset_name == "year_pred":
        return (90, 1)
    else:
        print("Not implemented yet.\n in out shape")

def get_criterion(dataset_name, reduction='mean'):
    if dataset_name in EXTRACTED_DATASETS:
        return torch.nn.CrossEntropyLoss(reduction=reduction)
    elif dataset_name == "year_pred":
        return torch.nn.MSELoss(reduction=reduction) 
    elif dataset_name == "fmnist":
        return torch.nn.CrossEntropyLoss(reduction=reduction)
                
class EnhancedModel(nn.Module):
    
    def __init__(self, grad_mode, criterion, **kwargs):
        super().__init__()
        assert grad_mode in ['first order', 
                             'zeroth order delta', 
                             'zeroth order forward-mode AD',
                             'zeroth order bidirectional delta', 
                             'zeroth order unbiased delta',
                             'temp']
        self.grad_mode = grad_mode
        self.criterion = criterion
        self.past_move = None
        self.acc_def = "class number"
        if 'zeroth order' in grad_mode:
            self.random_vecs = kwargs.get('random vecs', 100)
        device_name = kwargs.get("device name", "cuda:0")
        self.device = torch.device(device_name)
        self.to(self.device)
        
    @staticmethod
    def average_models(model_1, model_2):
        sd_1 = model_1.state_dict()
        sd_2 = model_2.state_dict()
        
        averaged_sd = deepcopy(sd_1)
        for key in sd_1:
            averaged_sd[key] = (sd_1[key] + sd_2[key]) / 2
        return averaged_sd
    
    def get_sd(self):
        return self.state_dict()
    
    def compute_grad(self, Xb, yb, lr=None):
        assert self.grad_mode in ['first order', 'zeroth order forward-mode AD']
        
        self.zero_grad()
        outputs = self(Xb)
        loss = self.criterion(outputs, yb)
#         pdb.set_trace()
        loss.backward()
        actual_grad = [param.grad for param in self.parameters()]
        grad = torch.cat([gr.flatten() for gr in actual_grad], 0)
        if self.grad_mode == 'zeroth order forward-mode AD':
            y_vectors   = torch.randn(self.random_vecs, *grad.shape, device=self.device) #, device=self.device
            efficiency = torch.matmul(y_vectors, grad)
            grad = torch.mean(efficiency.unsqueeze(1) * y_vectors, dim=0)
        return grad, loss
        
    def move(self, vector, lr):
        if self.past_move == None:
            self.past_move = vector
        next_move = 0.9*self.past_move + 0.1*vector
        params = torch.nn.utils.parameters_to_vector(self.parameters())
        torch.nn.utils.vector_to_parameters(params -  (lr * next_move), self.parameters())
        self.past_move = next_move
        
    def evaluate(self, dataloader):
        model = self
        model_initial_training = model.training
        model.eval()

        total_loss = 0
        count = 0
        corrects = 0
        data_count = 0
        criterion = self.criterion
        result = {}

        with torch.no_grad():
            for data in dataloader:
                Xb, yb = data
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                count += 1
                outputs = model(Xb)
                loss = criterion(outputs, yb)
                data_count += Xb.shape[0]
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
    
def one_three_multiplication(one, three): #One.shape= (b), three.shape= (b, m, n). Returns the product of each number in one to each layer of three
        shape = three.shape
        temp = torch.diag(one) @ three.reshape(shape[0], -1)
        return temp.reshape(shape)
    
class LinearEnhancedModel(EnhancedModel):
    
    def __init__(self, in_out_dim, grad_mode, criterion, **kwargs):
        EnhancedModel.__init__(self, grad_mode, criterion, **kwargs)
        self.linear = Linear(*in_out_dim)
        if grad_mode != 'temp':
            self.dim = sum([p.numel() for p in self.parameters()])
            self.dataset_name = kwargs.get('dataset_name', None)
            self.criterion_non_reduce = get_criterion(dataset_name = self.dataset_name,
                                               reduction='none')
        self.to(self.device)
    
    def forward(self, Xb):
        return self.linear(Xb)
      
    def perturb(self, search_radiuses):
        perturbations, new_params = [], []
        vecs_count = self.random_vecs
        for i, param in enumerate(self.parameters()):
            stacked_parameter = (torch.t(param)).repeat((vecs_count, ) + 
                                                        (1, )* (len(param.size())))
            
            random_perturbation = torch.Tensor([]) 
            random_perturbation = torch.randn_like((torch.t(param)).repeat(
                (self.random_vecs, ) + (1, )* (len(param.size()))))
            
            perturbation_amount = random_perturbation
            perturbations.append(perturbation_amount)
            perturbation_amount = (perturbation_amount.reshape(vecs_count, -1)*search_radiuses[:, None]).reshape(
                                                                                        perturbation_amount.shape)

            new_params.append(torch.cat((stacked_parameter + perturbation_amount,
                                           stacked_parameter - perturbation_amount), dim = 0))
        return perturbations, new_params    
        
    def compute_grad_zeroth_order_delta(self, Xb, yb, v):
        ## Perform local search
        search_radiuses = torch.ones(self.random_vecs, device=self.device)*v #
        perturbations, new_params = self.perturb(search_radiuses=search_radiuses)
        grad, loss = self.compute_average_grad(Xb, yb, perturbations, new_params, search_radiuses)
        return grad, loss    
        
    def compute_grad(self, Xb, yb, **kwargs):
        if self.grad_mode in 'zeroth order delta':
            lr = kwargs.get('lr', None)
            v = lr / (self.dim + 6)
            grad, loss = self.compute_grad_zeroth_order_delta(Xb, yb, v)
        elif self.grad_mode in ['first order', 'zeroth order forward-mode AD']:
            grad, loss = super().compute_grad(Xb, yb)
    
        return grad, loss
    
    def compute_average_grad(self, Xb, yb, perturbations, new_params, search_radiuses):
        vecs_count = perturbations[0].shape[0]
        with torch.no_grad():
            X = self.batch_forward(Xb, new_params, vecs_count)
            
            losses = self.vector_batch_loss(X, yb)
            ratios = torch.div((losses[0:vecs_count] - losses[vecs_count:2*vecs_count]), 2*search_radiuses)
            avg_loss = torch.mean(losses)
            grads = []
            for u, param in zip(perturbations, self.parameters()):
                u = one_three_multiplication(ratios, u)
                mean_u = torch.mean(u, 0)
                grad = torch.t(mean_u)
                grads.append(grad.flatten())
            return torch.cat(grads), avg_loss
        
    def batch_forward(self, Xb, new_params, vecs_count):
        batch_size = len(Xb)
#         if self.dataset_name == "mnsit":
#             X = Xb.view(Xb.size(0), -1)
#             X = X.repeat(vecs_count*2, 1, 1)
#             X = torch.bmm(X, new_params[0])
#             X += new_params[1].unsqueeze(1).repeat(1, batch_size, 1)
#             X = F.relu(X)
#             X = torch.bmm(X, new_params[2])
#             X += new_params[3].unsqueeze(1).repeat(1, batch_size, 1)
#         elif self.dataset_name == "year_pred": ## Todo change this!
#             X = Xb.view(Xb.size(0), -1)
        X = Xb.repeat(vecs_count*2, 1, 1)
        X = torch.bmm(X, new_params[0])
        X += new_params[1].unsqueeze(1).repeat(1, batch_size, 1)
        return X   
    
    def vector_batch_loss(self, X, y):
        X_shape = X.shape
        independent_vecs = X_shape[0]
        batch_size = X_shape[1]
        if(self.dataset_name == "year_pred"):
            reshape = (-1, )
            y = y.flatten()
        else:
            reshape = (-1, X_shape[2])
        X_2d = torch.reshape(X, reshape)
        y_repeated = y.repeat(independent_vecs)
        losses = self.criterion_non_reduce(X_2d, y_repeated)
        losses = losses.reshape(independent_vecs, batch_size)
        losses = torch.mean(losses, 1)
        return losses

def get_temp_state_dict(dataset_name):
    assert dataset_name in DATASETS
    
    if dataset_name in EXTRACTED_DATASETS + ['year_pred']:
        in_out_dim = get_in_out_shape(dataset_name)
        model = LinearEnhancedModel(in_out_dim=in_out_dim, grad_mode='temp', criterion=None)
    elif dataset_name == "fmnist":
        model = FashionMnistNet(grad_mode='temp', criterion=None)
    state_dict = model.state_dict()
    return state_dict

class FashionMnistNet(EnhancedModel):
    
    def __init__(self, grad_mode, criterion, **kwargs):
        super(FashionMnistNet, self).__init__(grad_mode, criterion, **kwargs)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        self.to(self.device)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out

def get_model(dataset_name, grad_mode, **kwargs):
    assert dataset_name in DATASETS
    kwargs['dataset_name'] = dataset_name
    criterion = get_criterion(dataset_name)
    if dataset_name in EXTRACTED_DATASETS:
        in_out_dim = get_in_out_shape(dataset_name)
        model = LinearEnhancedModel(in_out_dim=in_out_dim, grad_mode=grad_mode, criterion=criterion, **kwargs)
    elif dataset_name == "year_pred":
        in_out_dim = get_in_out_shape(dataset_name)
        model = LinearEnhancedModel(in_out_dim=in_out_dim, grad_mode=grad_mode, criterion=criterion, **kwargs)
        model.acc_def = "in neighbourhood"
    elif dataset_name == "fmnist":
        model = FashionMnistNet(grad_mode=grad_mode, criterion=criterion, **kwargs)
    return model

def get_group_models(dataset_name, group, initial_state_dict, **kwargs):
    count, grad_mode = group['count'], group['grad_mode']
    models = []
    for i in range(count):
        if 'zeroth order' in grad_mode:
            if 'random vecs' in group:
                kwargs['random vecs'] = group['random vecs']
        model = get_model(dataset_name, grad_mode, **kwargs)
        
        model.load_state_dict(initial_state_dict)
        models.append(model)
    return models