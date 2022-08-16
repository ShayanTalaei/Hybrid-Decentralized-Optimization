import torch
from torch import Tensor, nn
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from models import *
from datasets import get_batchsize

class HybridSGDTrainer:
    
    def __init__(self, population_args, dataset_name, train_set, test_set, 
                 initial_state_dict):
        self.dataset_name = dataset_name
        self.population_args = population_args
        self.population_count, self.nodes = 0, []
        assert self.population_count % 2 == 0
        self.setup_models(initial_state_dict)
        
        self.test_loader = None
        self.setup_dataloaders(train_set, test_set)
        
        self.total_steps = 0
        self.training_loss = None
        self.history = []
        
    def setup_models(self, initial_state_dict):
        for group in self.population_args:
            models = get_group_models(self.dataset_name, group, initial_state_dict)
            self.population_count += group['count']
            self.nodes += [{'model': model, 'steps': 0} for model in models]
            
    def setup_dataloaders(self, train_set, test_set):
        zeroth_orders, first_orders = [], []
        for node in self.nodes:
            if node['model'].grad_mode == 'first order':
                first_orders.append(node)
            else:
                zeroth_orders.append(node)
        
        batch_size = get_batchsize(self.dataset_name)
        N_0 = len(zeroth_orders)
        for i, node in enumerate(zeroth_orders):
            sampler = DistributedSampler(train_set, N_0, i, shuffle = True)
            dataloader_i = DataLoader(train_set, batch_size=batch_size, num_workers=4, sampler=sampler)
            node['data loader'] = dataloader_i
            node['iterator'] = iter(dataloader_i)
        N_1 = len(first_orders)
        for i, node in enumerate(first_orders):
            sampler = DistributedSampler(train_set, N_1, i, shuffle = True)
            dataloader_i = DataLoader(train_set, batch_size=batch_size, num_workers=4, sampler=sampler)
            node['data loader'] = dataloader_i
            node['iterator'] = iter(dataloader_i)
            
        self.test_loader = DataLoader(test_set, batch_size=4*batch_size, num_workers=4)
            
    def train(self, lr_schedule):
        if self.population_count == 1:
            return self.train_solo(lr_schedule)
        
        population_factor = self.population_count // 2
        for steps, lr, log_period in lr_schedule:
            steps_to_go = steps * population_factor
            for taken_steps in range(steps_to_go):
                if self.total_steps % (log_period*population_factor) == 0:
                    self.evaluate()
                
                node_1, node_2 = np.random.choice(self.nodes, size = 2, replace = False)
                loss_1 = self.take_step(node_1, lr)
                loss_2 = self.take_step(node_2, lr)
                loss = (loss_1 + loss_2)/2
                if loss > 10**4: # Diverged!
                    return self.history
                self.communicate(node_1, node_2)
                self.training_loss = self.training_loss*0.95 + loss*0.05 if self.training_loss!=None else loss
                self.total_steps += 1
        return self.history
            
    def train_solo(self, lr_schedule):
        assert self.population_count == 1
        node = self.nodes[0]
        for steps, lr, log_period in lr_schedule:
            for step in range(steps):
                if self.total_steps % log_period == 0:
                    self.evaluate()

                loss = self.take_step(node, lr)
                if loss > 10**4: # Diverged!
                        return self.history
                self.training_loss = self.training_loss*0.95 + loss*0.05 if self.training_loss!=None else loss
                self.total_steps += 1
        return self.history
            
    def take_step(self, node, lr):
        steps = 1 
        taken_steps = 0
        model, iterator = node['model'], node['iterator']
        total_loss = 0
        while(taken_steps < steps):
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
    
    def communicate(self, node_1, node_2):
        model_1, model_2 = node_1['model'], node_2['model']
        avg_state_dict = EnhancedModel.average_models(model_1, model_2)
        model_1.load_state_dict(avg_state_dict)
        model_2.load_state_dict(avg_state_dict)
        
    def evaluate(self, idx='mid'):
        if idx == 'mid':
            idx_steps = sorted([(i, node['steps']) for i, node in enumerate(self.nodes)],
                               key=lambda x: x[1])
            idx = idx_steps[int(self.population_count//2)][0]
        node = self.nodes[idx]
        model, steps = node['model'], node['steps']
        result = model.evaluate(self.test_loader)
        validation_loss = result['Loss']
        validation_accuracy = result['Accuracy']
        training_loss = self.training_loss.data if self.training_loss else 0
        print("Steps: {:5.0f}, Training loss: {:.4f}, Validation loss: {:.4f}, Validation accuracy: {:.2f}".format(steps, training_loss, validation_loss, validation_accuracy))
        result_dict = {'Steps': int(steps), 'Training loss': float(training_loss), 
                       'Validation loss': float(validation_loss), 'Validation accuracy': float(validation_accuracy)}
        self.history.append(result_dict)
        