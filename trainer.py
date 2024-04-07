import numpy as np
import torch

import optimizers
from models import get_model
from mpi4py import MPI

from datasets import get_criterion


class HybridSGDTrainer:

    def __init__(self, rank, size, comm, fn, grad_mode, dataset_name, train_loader, test_loader, initial_state_dict,
                 lr, conv_number=2, hidden=128, num_layer=2, model_name=None, freeze_model=False, random_vecs=200,
                 momentum=0.0):
        self.dataset_name = dataset_name
        self.rank = rank
        self.size = size
        self.comm = comm
        self.fn = fn
        self.lr = lr
        self.model = get_model(dataset_name, conv_number=conv_number, hidden=hidden, num_layer=num_layer,
                               model_name=model_name, freeze_model=freeze_model, random_vecs=random_vecs)

        self.model.load_state_dict(initial_state_dict)
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.train_iterator = iter(train_loader)
        assert grad_mode in ['first_order', 'zeroth_order_forward-mode_AD', 'zeroth_order_simple']
        self.grad_mode = grad_mode
        self.criterion = get_criterion(dataset_name)
        grad_mode_to_opt = {'first_order': 'SGD', 'zeroth_order_forward-mode_AD': 'ZAD', 'zeroth_order_simple': 'ZAD'}
        opt_args = {'lr': lr, 'momentum': momentum}
        if self.grad_mode.startswith('zeroth_order'):
            opt_args['random_vec'] = random_vecs
            opt_args['names'] = list(n for n, _ in self.model.named_parameters())
            opt_args['grad_mode'] = grad_mode
        self.optimizer = getattr(optimizers, grad_mode_to_opt[grad_mode])(self.model.parameters(), **opt_args)

        self.training_loss = None
        self.history = []
        self.steps = 0

        total_elements = 0
        for param in list(self.model.parameters()):
            total_elements += param.data.nelement()
        buffer_size = 0
        for buf in list(self.model.buffers()):
            buffer_size += buf.data.nelement()
        model_size = total_elements + buffer_size

        self.model_copy = torch.empty(model_size, dtype=torch.float64, device='cpu')
        self.partner_model = torch.empty(model_size, dtype=torch.float64, device='cpu')
        self.partner_buf = MPI.memory.fromaddress(self.partner_model.data_ptr(),
                                                  self.partner_model.nelement() * self.partner_model.element_size())
        buf = MPI.memory.fromaddress(self.model_copy.data_ptr(),
                                     self.model_copy.nelement() * self.model_copy.element_size())
        self.win = MPI.Win.Create(buf, comm=self.comm)

    def take_step(self):
        steps = 1
        taken_steps = 0
        total_loss = 0
        while taken_steps < steps:
            try:
                Xb, yb = next(self.train_iterator)
                Xb, yb = Xb.to(self.model.device), yb.to(self.model.device)
                loss = self.optimizer.optimize(self.model, Xb, yb, self.criterion)
                taken_steps += 1
                total_loss += loss
            except StopIteration:
                iterator = iter(self.train_loader)
        return total_loss / steps

    def train(self, steps, log_period):
        self.model.train()
        if self.size == 1:
            return self.train_solo(steps, log_period)

        for taken_steps in range(steps):
            if self.steps % log_period == 0:
                print(f"Rank {self.rank} steps: {self.steps}")
                self.comm.Barrier()
                if self.rank == 0:
                    self.evaluate()
                self.comm.Barrier()

            print(f"Rank {self.rank} steps: {self.steps} before take step")

            loss = self.take_step()
            print(f"Rank {self.rank} steps: {self.steps} after take step")
            if loss > 10 ** 4:  # Diverged!
                return self.history

            print(f"Rank {self.rank} steps: {self.steps} before lock")

            self.win.Lock(self.rank, lock_type=MPI.LOCK_EXCLUSIVE)
            print(f"Rank {self.rank} steps: {self.steps} after lock")

            self.model_to_copy(self.model_copy)
            print(f"Rank {self.rank} steps: {self.steps} after model to copy")

            self.win.Unlock(self.rank)
            print(f"Rank {self.rank} steps: {self.steps} after unlock")

            partner_rank = np.random.randint(self.size)
            while partner_rank == self.rank:
                partner_rank = np.random.randint(self.size)

            print(f"Rank {self.rank} steps: {self.steps} before lock partner")

            self.win.Lock(partner_rank, lock_type=MPI.LOCK_SHARED)
            print(f"Rank {self.rank} steps: {self.steps} after lock partner")

            self.win.Get((self.partner_buf, MPI.FLOAT), target_rank=partner_rank)

            print(f"Rank {self.rank} steps: {self.steps} after get")

            self.win.Unlock(partner_rank)
            print(f"Rank {self.rank} steps: {self.steps} after unlock partner")

            self.partner_model[:] = (self.partner_model + self.model_copy) / 2

            self.copy_to_model(self.partner_model)
            print(f"Rank {self.rank} steps: {self.steps} after copy to model")

            self.training_loss = self.training_loss * 0.95 + loss * 0.05 if self.training_loss is not None else loss
            self.steps += 1
        return self.history

    def train_solo(self, steps, log_period):
        assert self.size == 1
        for step in range(steps):
            if self.steps % log_period == 0:
                self.evaluate()

            loss = self.take_step()
            if loss > 10 ** 4:  # Diverged!
                return self.history
            self.training_loss = self.training_loss * 0.95 + loss * 0.05 if self.training_loss is not None else loss
            self.steps += 1
        return self.history

    def evaluate(self):
        self.model.eval()
        result = self.model.evaluate(self.test_loader, self.criterion)
        validation_loss = result['Loss']
        validation_accuracy = result['Accuracy']
        training_loss = self.training_loss if self.training_loss else 0
        print(
            "Steps: {:5.0f}, Training loss: {:.4f}, Validation loss: {:.4f}, Validation accuracy: {:.2f}"
            .format(self.steps,
                    training_loss,
                    validation_loss,
                    validation_accuracy)
        )
        result_dict = {'Steps': int(self.steps), 'Training loss': float(training_loss),
                       'Validation loss': float(validation_loss), 'Validation accuracy': float(validation_accuracy)}
        self.history.append(result_dict)
        self.model.train()

    def copy_to_model(self, model_copy_tensor):
        counter = 0
        for param in self.model.parameters():
            t = param.data
            t.view(-1)[:] = model_copy_tensor[counter: counter + t.nelement()].to(t.device)
            counter += t.nelement()

    def model_to_copy(self, model_copy_tensor):
        counter = 0
        for param in self.model.parameters():
            t = param.data
            model_copy_tensor[counter: counter + t.nelement()] = t.view(-1).to(model_copy_tensor.device)
            counter += t.nelement()

