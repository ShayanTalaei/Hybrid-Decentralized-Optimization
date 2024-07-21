import numpy as np
import torch
import wandb

import optimizers
from models import get_model
from mpi4py import MPI
import pytorch_warmup as warmup

from datasets.datasets import get_criterion


class HybridSGDTrainer:

    def __init__(self, rank, size, comm, fn, grad_mode, dataset_name, train_loader, test_loader, initial_state_dict,
                 lr, conv_number=2, hidden=128, num_layer=2, model_name=None, freeze_model=False, random_vecs=200,
                 momentum=0.0, scheduler=False, scheduler_warmup_steps=0, warmup_steps=0, total_step_number=200,
                 log_period=10, v_step=10.0, out_channels=8, is_cuda_aware=False, device='cpu', config=None):
        self.dataset_name = dataset_name
        self.rank = rank
        self.size = size
        self.comm = comm
        self.fn = fn
        self.lr = lr
        self.total_step_number = total_step_number
        self.log_period = log_period
        self.is_cuda_aware = is_cuda_aware
        self.warmup_steps = warmup_steps
        self.model = get_model(dataset_name, conv_number=conv_number, hidden=hidden, num_layer=num_layer,
                               model_name=model_name, freeze_model=freeze_model, random_vecs=random_vecs,
                               out_channels=out_channels, device=device, config=config)

        self.model.load_state_dict(initial_state_dict)
        self.test_loader = test_loader
        self.train_loader = train_loader
        assert grad_mode in ['first_order', 'zeroth_order_forward-mode_AD', 'zeroth_order_rge', 'zeroth_order_cge', 'zeroth_order_forward-mode_AD_sim']
        self.grad_mode = grad_mode
        self.criterion = get_criterion(dataset_name)
        grad_mode_to_opt = {'first_order': 'SGD', 'zeroth_order_forward-mode_AD': 'ZAD', 'zeroth_order_rge': 'ZAD', 'zeroth_order_cge': 'ZAD', 'zeroth_order_forward-mode_AD_sim': 'ZAD'}
        opt_args = {'lr': lr, 'momentum': momentum, 'weight_decay': config.weight_decay}
        if self.grad_mode.startswith('zeroth_order'):
            opt_args['random_vec'] = random_vecs
            opt_args['names'] = list(n for n, _ in self.model.named_parameters())
            opt_args['grad_mode'] = grad_mode
            opt_args['v_step'] = v_step
            opt_args['device'] = device
        if model_name == 'transformer' or model_name == 'vtransformer':
            params, names = self.model.get_parameter_group_specs()
            if self.grad_mode.startswith('zeroth_order'):
                opt_args['names'] = names[0] + names[1]
        else:
            params = self.model.parameters()
        self.optimizer = getattr(optimizers, grad_mode_to_opt[grad_mode])(params, **opt_args)
        self.scheduler = None
        if scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                        T_max=self.total_step_number)
        self.scheduler_warmup_steps = scheduler_warmup_steps
        self.warmup_scheduler = None
        if self.scheduler_warmup_steps > 0:
            self.warmup_scheduler = warmup.LinearWarmup(self.optimizer, warmup_period=scheduler_warmup_steps)
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

        if self.size > 1:
            if self.is_cuda_aware:
                self.model_copy = torch.zeros(model_size, dtype=torch.float64, device=self.model.device)
                self.partner_model = torch.zeros(model_size, dtype=torch.float64, device=self.model.device)
            else:
                self.model_copy = torch.zeros(model_size, dtype=torch.float64, device="cpu")
                self.partner_model = torch.zeros(model_size, dtype=torch.float64, device="cpu")
            self.partner_buf = MPI.memory.fromaddress(self.partner_model.data_ptr(),
                                                      self.partner_model.nelement() * self.partner_model.element_size())
            # print('Rank:', self.rank, 'Model size:', model_size, self.model_copy.size(), self.model_copy.nelement() * self.model_copy.element_size())
            self.buf = MPI.memory.fromaddress(self.model_copy.data_ptr(),
                                              self.model_copy.nelement() * self.model_copy.element_size())
            self.comm.Barrier()
            # self.win = MPI.Win.Create(buf, comm=self.comm)
            self.win = MPI.Win.Allocate(self.model_copy.nelement() * self.model_copy.element_size(), comm=self.comm)
            self.comm.Barrier()

    def take_step(self, data, target):
        steps = 1
        taken_steps = 0
        total_loss = 0
        while taken_steps < steps:
            # print("Rank: ", self.rank, 'optimize')
            data, target = data.to(self.model.device), target.to(self.model.device)
            # while True:
            #     try:
            loss = self.optimizer.optimize(self.model, data, target, self.criterion)
                #     break
                # except Exception as e:
                #     print("Rank: ", self.rank, e)
                #     print("Rank: ", self.rank, "Retrying...")
            taken_steps += 1
            total_loss += loss
        if self.warmup_scheduler is not None:
            with self.warmup_scheduler.dampening():
                if self.scheduler is not None and self.warmup_scheduler.last_step + 1 >= self.scheduler_warmup_steps:
                    self.scheduler.step()
        elif self.scheduler is not None:
            self.scheduler.step()
        return total_loss / steps

    def train(self):
        self.model.train()
        if self.size == 1:
            return self.train_solo()

        for taken_steps in range((self.total_step_number + self.warmup_steps) // len(self.train_loader) + 1):
            # step_loss = 0
            for (data, target) in self.train_loader:

                if self.steps % self.log_period == 0:
                    print(f"Rank {self.rank} steps: {self.steps} evaluate")
                    self.comm.Barrier()
                    if self.rank == 0:
                        self.evaluate()
                    self.comm.Barrier()
                # print(f"Rank {self.rank} steps: {self.steps} before take step")

                loss = self.take_step(data, target)
                # step_loss += loss
                # print(f"Rank {self.rank} steps: {self.steps} after take step")
                if loss is None or loss > 10 ** 4:  # Diverged!
                    # step_loss /= len(self.train_loader)
                    # self.training_loss = self.training_loss * 0.95 + step_loss * 0.05 if self.training_loss is not None else step_loss
                    self.training_loss = self.training_loss * 0.95 + loss * 0.05 if self.training_loss is not None else loss
                    # self.training_loss = loss
                    return self.history
                if self.steps < self.warmup_steps:
                    self.training_loss = self.training_loss * 0.95 + loss * 0.05 if self.training_loss is not None else loss
                    # self.training_loss = loss
                    self.steps += 1
                    continue

                # print(f"Rank {self.rank} steps: {self.steps} before lock")

                self.win.Lock(self.rank, lock_type=MPI.LOCK_EXCLUSIVE)
                # print(f"Rank {self.rank} steps: {self.steps} after lock")

                self.model_to_copy(self.model_copy)
                self.win.Put(self.buf, target_rank=self.rank)
                # print(f"Rank {self.rank} steps: {self.steps} after model to copy")

                self.win.Unlock(self.rank)
                # print(f"Rank {self.rank} steps: {self.steps} after unlock")

                pairs = np.empty(self.size, dtype=np.int32)
                if self.rank == 0:
                    per = np.random.permutation(self.size)
                    for i in range(0, self.size-1, 2):
                        pairs[per[i]] = per[i + 1]
                        pairs[per[i + 1]] = per[i]

                    if self.size % 2 == 1:
                        pairs[per[-1]] = -1
                    self.comm.Bcast(pairs, root=0)
                    self.comm.Barrier()
                else:
                    self.comm.Barrier()
                    self.comm.Bcast(pairs, root=0)
                # partner_rank = np.random.randint(self.size)
                # while partner_rank == self.rank:
                #     partner_rank = np.random.randint(self.size)
                partner_rank = pairs[self.rank]
                # print(f"Rank {self.rank} steps: {self.steps} before lock partner")
                # print(f"Rank {self.rank} steps: {self.steps} partner rank: {partner_rank}")
                if partner_rank != -1:
                    self.win.Lock(partner_rank, lock_type=MPI.LOCK_SHARED)
                    # print(f"Rank {self.rank} steps: {self.steps} after lock partner")

                    self.win.Get((self.partner_buf, MPI.FLOAT), target_rank=partner_rank)

                    # print(f"Rank {self.rank} steps: {self.steps} after get")

                    self.win.Unlock(partner_rank)
                    # print(f"Rank {self.rank} steps: {self.steps} after unlock partner")

                    self.partner_model[:] = (self.partner_model + self.model_copy) / 2 if any(self.partner_model) else self.model_copy

                    self.copy_to_model(self.partner_model)
                self.training_loss = self.training_loss * 0.95 + loss * 0.05 if self.training_loss is not None else loss
                # self.training_loss = loss
                self.steps += 1
                if self.steps == self.total_step_number + self.warmup_steps:
                    return self.history
                # print(f"Rank {self.rank} steps: {self.steps} after copy to model")
            # step_loss /= len(self.train_loader)
            # self.training_loss = self.training_loss * 0.95 + step_loss * 0.05 if self.training_loss is not None else step_loss
            # self.steps += 1
        return self.history

    def train_solo(self):
        assert self.size == 1
        for taken_steps in range((self.total_step_number + self.warmup_steps) // len(self.train_loader) + 1):
            # step_loss = 0
            for (data, target) in self.train_loader:
                if self.steps % self.log_period == 0:
                    self.evaluate()
                    if self.dataset_name == 'bracket' and self.history[-1]['eval/accuracy'] < 0.6 and self.steps > 200:
                        return self.history
                loss = self.take_step(data, target)
                # step_loss += loss
                # self.training_loss = self.training_loss * 0.95 + loss * 0.05 if self.training_loss is not None else loss
                self.training_loss = loss
                if loss is None or loss is torch.nan or loss > 10 ** 4:  # Diverged!
                    # step_loss /= len(self.train_loader)
                    # self.training_loss = self.training_loss * 0.95 + step_loss * 0.05 if self.training_loss is not None else step_loss
                    return self.history
                self.steps += 1
                if self.steps == self.total_step_number + self.warmup_steps:
                    return self.history
            # self.training_loss = self.training_loss * 0.95 + step_loss * 0.05 if self.training_loss is not None else step_loss
            # self.steps += 1
        return self.history

    def evaluate(self):
        self.model.eval()
        result = self.model.evaluate(self.test_loader, self.criterion)
        validation_loss = result['Loss']
        validation_accuracy = result['Accuracy']
        training_loss = self.training_loss if self.training_loss else 0
        print(f"Rank {self.rank}: " +
            "Steps: {:5.0f}, Training loss: {:.4f}, Validation loss: {:.4f}, Validation accuracy: {:.2f}"
            .format(self.steps,
                    training_loss,
                    validation_loss,
                    validation_accuracy)
        )
        result_dict = {'step': int(self.steps), 'train/loss': float(training_loss),
                       'eval/loss': float(validation_loss), 'eval/accuracy': float(validation_accuracy)}
        if self.rank == 0:
            wandb.log(result_dict)
        self.history.append(result_dict)
        self.model.train()

    def copy_to_model(self, model_copy_tensor):
        counter = 0
        if not self.is_cuda_aware:
            for param in self.model.parameters():
                t = param.data
                t.view(-1)[:] = model_copy_tensor[counter: counter + t.nelement()].to(t.device)
                counter += t.nelement()
        else:
            for param in self.model.parameters():
                t = param.data
                t.view(-1)[:] = model_copy_tensor[counter: counter + t.nelement()]
                counter += t.nelement()

    def model_to_copy(self, model_copy_tensor):
        counter = 0
        if not self.is_cuda_aware:
            for param in self.model.parameters():
                t = param.data
                model_copy_tensor[counter: counter + t.nelement()] = t.view(-1).to(model_copy_tensor.device)
                counter += t.nelement()
        else:
            for param in self.model.parameters():
                t = param.data
                model_copy_tensor[counter: counter + t.nelement()] = t.view(-1)
                counter += t.nelement()

