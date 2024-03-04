import numpy as np
import torch
from torch import nn
from torch.nn import Linear
from copy import deepcopy

from datasets import get_model_shape


def get_temp_state_dict(dataset_name, input_shape, n_class, conv_number=2, hidden=128, num_layer=2):
    hidden_layers = [hidden] * num_layer
    hidden_layers.append(n_class)
    model = CustomNN(input_shape, hidden_layers, grad_mode='temp', criterion=None, conv_number=conv_number)
    # if dataset_name in ['birds', 'flowers', 'pets', 'food101', 'year_pred', 'mnist', 'cifar10', 'cifar100']:
    #     model = LinearEnhancedModel(input_shape, n_class, grad_mode='temp', criterion=None)
    # elif dataset_name == "fmnist":
    #     model = FashionMnistNet(grad_mode='temp', criterion=None)
    # else:
    #     raise ValueError(f"Dataset {dataset_name} not implemented yet.")
    state_dict = model.state_dict()
    return state_dict


def get_model(dataset_name, grad_mode, conv_number=2, hidden=128, num_layer=2, **kwargs):
    kwargs['dataset_name'] = dataset_name
    criterion = get_criterion(dataset_name)
    input_shape, n_class = get_model_shape(dataset_name)
    hidden_layers = [hidden] * num_layer
    hidden_layers.append(n_class)
    model = CustomNN(input_shape, hidden_layers, grad_mode=grad_mode, criterion=criterion, conv_number=conv_number)

    # if dataset_name in ['birds', 'flowers', 'pets', 'food101', 'mnist', 'cifar10', 'cifar100']:
    #     model = LinearEnhancedModel(input_shape, n_class, grad_mode=grad_mode, criterion=criterion, **kwargs)
    # elif dataset_name == "year_pred":
    #     model = LinearEnhancedModel(input_shape, n_class, grad_mode=grad_mode, criterion=criterion, **kwargs)
    #     model.acc_def = "in neighbourhood"
    # elif dataset_name == "fmnist":
    #     model = FashionMnistNet(grad_mode=grad_mode, criterion=criterion, **kwargs)
    # else:
    #     raise ValueError(f"Dataset {dataset_name} not implemented yet.")
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


def communicate(node_1, node_2):
    model_1, model_2 = node_1['model'], node_2['model']
    avg_state_dict = EnhancedModel.average_models(model_1, model_2)
    model_1.load_state_dict(avg_state_dict)
    model_2.load_state_dict(avg_state_dict)


def take_step(node, lr):
    steps = 1
    taken_steps = 0
    model, iterator = node['model'], node['iterator']
    total_loss = 0
    while taken_steps < steps:
        try:
            Xb, yb = next(iterator)
            Xb, yb = Xb.to(model.device), yb.to(model.device)
            grad, loss = model.compute_grad(Xb, yb, lr=lr)
            model.move(grad, lr)
            taken_steps += 1
            total_loss += loss
        except StopIteration:
            iterator = iter(node['data loader'])
            node['iterator'] = iterator
    node['steps'] += steps
    return total_loss / steps


def one_three_multiplication(one,
                             three):
    # One.shape= (b), three.shape= (b, m, n). Returns the product of each number in one to each layer of three
    shape = three.shape
    temp = torch.diag(one) @ three.reshape(shape[0], -1)
    return temp.reshape(shape)


def get_criterion(dataset_name, reduction='mean'):
    if dataset_name == "year_pred":
        return torch.nn.MSELoss(reduction=reduction)
    else:
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
        device_name = kwargs.get("device name", 'cuda:0' if torch.cuda.is_available() else 'cpu')
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

    def compute_grad(self, xb, yb, lr=None):
        assert self.grad_mode in ['first order', 'zeroth order forward-mode AD']

        self.zero_grad()
        outputs = self(xb)
        loss = self.criterion(outputs, yb)
        #         pdb.set_trace()
        loss.backward()
        actual_grad = [param.grad for param in self.parameters()]
        grad = torch.cat([gr.flatten() for gr in actual_grad], 0)
        if self.grad_mode == 'zeroth order forward-mode AD':
            y_vectors = torch.randn(self.random_vecs, *grad.shape, device=self.device)  # , device=self.device
            efficiency = torch.matmul(y_vectors, grad)
            grad = torch.mean(efficiency.unsqueeze(1) * y_vectors, dim=0)
        return grad, loss

    def move(self, vector, lr):
        if self.past_move is None:
            self.past_move = vector
        next_move = 0.9 * self.past_move + 0.1 * vector
        params = torch.nn.utils.parameters_to_vector(self.parameters())
        torch.nn.utils.vector_to_parameters(params - (lr * next_move), self.parameters())
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


# class LinearEnhancedModel(EnhancedModel):
#
#     def __init__(self, input_shape, output_shape, grad_mode, criterion, **kwargs):
#         EnhancedModel.__init__(self, grad_mode, criterion, **kwargs)
#         inp = 1
#         for i in input_shape:
#             inp *= i
#         self.flatten = nn.Flatten()
#         self.linear = Linear(inp, output_shape, bias=True)
#         if grad_mode != 'temp':
#             self.dim = sum([p.numel() for p in self.parameters()])
#             self.dataset_name = kwargs.get('dataset_name', None)
#             self.criterion_non_reduce = get_criterion(dataset_name=self.dataset_name,
#                                                       reduction='none')
#         self.to(self.device)
#
#     def forward(self, xb):
#         xb = self.flatten(xb)
#         return self.linear(xb)
#
#     def perturb(self, search_radiuses):
#         perturbations, new_params = [], []
#         vecs_count = self.random_vecs
#         for i, param in enumerate(self.parameters()):
#             stacked_parameter = (torch.t(param)).repeat((vecs_count,) +
#                                                         (1,) * (len(param.size())))
#
#             random_perturbation = torch.randn_like((torch.t(param)).repeat(
#                 (self.random_vecs,) + (1,) * (len(param.size()))))
#
#             perturbation_amount = random_perturbation
#             perturbations.append(perturbation_amount)
#             perturbation_amount = (perturbation_amount.reshape(vecs_count, -1) * search_radiuses[:, None]).reshape(
#                 perturbation_amount.shape)
#
#             new_params.append(torch.cat((stacked_parameter + perturbation_amount,
#                                          stacked_parameter - perturbation_amount), dim=0))
#         return perturbations, new_params
#
#     def compute_grad_zeroth_order_delta(self, xb, yb, v):
#         # Perform local search
#         search_radiuses = torch.ones(self.random_vecs, device=self.device) * v  #
#         perturbations, new_params = self.perturb(search_radiuses=search_radiuses)
#         grad, loss = self.compute_average_grad(xb, yb, perturbations, new_params, search_radiuses)
#         return grad, loss
#
#     def compute_grad(self, xb, yb, **kwargs):
#         if self.grad_mode in 'zeroth order delta':
#             lr = kwargs.get('lr', None)
#             v = lr / (self.dim + 6)
#             grad, loss = self.compute_grad_zeroth_order_delta(xb, yb, v)
#         elif self.grad_mode in ['first order', 'zeroth order forward-mode AD']:
#             grad, loss = super().compute_grad(xb, yb)
#         else:
#             raise ValueError("Unknown grad mode")
#
#         return grad, loss
#
#     def batch_forward(self, xb, new_params, vecs_count):
#         batch_size = len(xb)
#         #         if self.dataset_name == "mnsit":
#         #             x = xb.view(xb.size(0), -1)
#         #             x = x.repeat(vecs_count*2, 1, 1)
#         #             x = torch.bmm(x, new_params[0])
#         #             x += new_params[1].unsqueeze(1).repeat(1, batch_size, 1)
#         #             x = F.relu(x)
#         #             x = torch.bmm(x, new_params[2])
#         #             x += new_params[3].unsqueeze(1).repeat(1, batch_size, 1)
#         #         elif self.dataset_name == "year_pred": ## Todo change this!
#         #             x = xb.view(xb.size(0), -1)
#         x = xb.repeat(vecs_count * 2, 1, 1)
#         x = torch.bmm(x, new_params[0])
#         x += new_params[1].unsqueeze(1).repeat(1, batch_size, 1)
#         return x
#
#     def compute_average_grad(self, xb, yb, perturbations, new_params, search_radiuses):
#         vecs_count = perturbations[0].shape[0]
#         with torch.no_grad():
#             x = self.batch_forward(xb, new_params, vecs_count)
#
#             losses = self.vector_batch_loss(x, yb)
#             ratios = torch.div((losses[0:vecs_count] - losses[vecs_count:2 * vecs_count]), 2 * search_radiuses)
#             avg_loss = torch.mean(losses)
#             grads = []
#             for u, param in zip(perturbations, self.parameters()):
#                 u = one_three_multiplication(ratios, u)
#                 mean_u = torch.mean(u, 0)
#                 grad = torch.t(mean_u)
#                 grads.append(grad.flatten())
#             return torch.cat(grads), avg_loss
#
#     def vector_batch_loss(self, x, y):
#         x_shape = x.shape
#         independent_vecs = x_shape[0]
#         batch_size = x_shape[1]
#         if self.dataset_name == "year_pred":
#             reshape = (-1,)
#             y = y.flatten()
#         else:
#             reshape = (-1, x_shape[2])
#         x_2d = torch.reshape(x, reshape)
#         y_repeated = y.repeat(independent_vecs)
#         losses = self.criterion_non_reduce(x_2d, y_repeated)
#         losses = losses.reshape(independent_vecs, batch_size)
#         losses = torch.mean(losses, 1)
#         return losses
#
#
# class FashionMnistNet(EnhancedModel):
#
#     def __init__(self, grad_mode, criterion, **kwargs):
#         super(FashionMnistNet, self).__init__(grad_mode, criterion, **kwargs)
#
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#
#         self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
#         self.drop = nn.Dropout2d(0.25)
#         self.fc2 = nn.Linear(in_features=600, out_features=120)
#         self.fc3 = nn.Linear(in_features=120, out_features=10)
#         self.to(self.device)
#
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc1(out)
#         out = self.drop(out)
#         out = self.fc2(out)
#         out = self.fc3(out)
#
#         return out


class CustomNN(EnhancedModel):
    """
    Simple feedforward neural network.
    """

    def __init__(self, input_shape, hidden, grad_mode, criterion, activation='relu', sigmoid_output=True, conv_number=1,
                 **kwargs):
        """
        Initialize the neural network.
        :param input_shape: The shape of the input.
        :param hidden: A list of hidden layer sizes.
        :param activation: The activation function to use.
        :param sigmoid_output: Whether to use a sigmoid activation on the output.
        :param conv_number: The number of convolutional layers to use.
        """
        super().__init__(grad_mode, criterion, **kwargs)
        self.seq = nn.Sequential()
        if conv_number > 0:
            self.seq.append(nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1,
                                      padding=1,
                                      bias=True))
            self.seq.append(nn.BatchNorm2d(32))
            self.seq.append(nn.ReLU())
            for i in range(conv_number - 1):
                self.seq.append(nn.Conv2d(in_channels=32 * (2 ** i), out_channels=32 * (2 ** (i + 1)), kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bias=True))
                self.seq.append(nn.BatchNorm2d(32 * (2 ** (i + 1))))
                self.seq.append(nn.ReLU())
                self.seq.append(nn.MaxPool2d(kernel_size=2))
            input_size = 32 * (2 ** (conv_number - 1)) * ((input_shape[1] // (2 ** (conv_number - 1))) *
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
