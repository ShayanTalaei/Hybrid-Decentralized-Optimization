import collections

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
                 momentum0=0.0, scheduler=False, scheduler_warmup_steps=0, warmup_steps=0, total_step_number=200,
                 log_period=10, v_step=10.0, out_channels=8, is_cuda_aware=False, device='cpu', config=None,
                 momentum1=0.0, concurrency=1, exchange_period=0, verbose=True, clear_cache=False):
        self.verbose = verbose
        self.dataset_name = dataset_name
        self.rank = rank
        self.size = size
        self.comm = comm
        self.clear_cache = clear_cache
        self.exchange_period = exchange_period
        self.fn = fn
        self.lr = lr
        self.concurrency = concurrency
        self.total_step_number = total_step_number
        self.log_period = log_period
        self.is_cuda_aware = is_cuda_aware
        self.warmup_steps = warmup_steps
        for turn in range(self.size // self.concurrency + 1):
            if self.rank // self.concurrency == turn:
                # Initialize the model
                self.model = get_model(dataset_name, conv_number=conv_number, hidden=hidden, num_layer=num_layer,
                                       model_name=model_name, freeze_model=freeze_model, random_vecs=random_vecs,
                                       out_channels=out_channels, device=device, config=config)
                # Load the synchronized initial state dict
                initial_state_dict = collections.OrderedDict(
                    {key: value.to(device) for key, value in initial_state_dict.items()}
                )
                self.model.load_state_dict(initial_state_dict)
                if self.concurrency < self.size:
                    torch.cuda.empty_cache()
            if self.concurrency < self.size:
                self.comm.Barrier()

        self.test_loader = test_loader
        self.train_loader = train_loader
        assert grad_mode in ['first_order', 'zeroth_order_forward-mode_AD', 'zeroth_order_rge', 'zeroth_order_cge',
                             'zeroth_order_forward-mode_AD_sim']
        # Initialize the optimizer
        self.grad_mode = grad_mode
        self.criterion = get_criterion(dataset_name)
        grad_mode_to_opt = {'first_order': 'SGD', 'zeroth_order_forward-mode_AD': 'ZAD', 'zeroth_order_rge': 'ZAD',
                            'zeroth_order_cge': 'ZAD', 'zeroth_order_forward-mode_AD_sim': 'ZAD'}
        opt_args = {'lr': lr, 'momentum': momentum1, 'weight_decay': config.weight_decay}
        if self.grad_mode.startswith('zeroth_order'):
            opt_args['random_vec'] = random_vecs
            opt_args['names'] = list(n for n, _ in self.model.named_parameters())
            opt_args['grad_mode'] = grad_mode
            opt_args['v_step'] = v_step
            opt_args['device'] = device
            opt_args['momentum'] = momentum0
        if model_name == 'transformer' or model_name == 'vtransformer':
            params, names = self.model.get_parameter_group_specs()
            if self.grad_mode.startswith('zeroth_order'):
                opt_args['names'] = names[0] + names[1]
        elif model_name == 'resnet':
            params = [v for _, v in self.model.named_parameters()]
            params = [{'params': params}]
        else:
            params = self.model.parameters()
        self.optimizer = getattr(optimizers, grad_mode_to_opt[grad_mode])(params, **opt_args)
        # Initialize the learning rate scheduler
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

    def take_step(self, data, target):
        steps = 1
        taken_steps = 0
        total_loss = 0

        # Train the model on the batched data
        while taken_steps < steps:
            data, target = data.to(self.model.device), target.to(self.model.device)
            loss = self.optimizer.optimize(self.model, data, target, self.criterion)
            taken_steps += 1
            total_loss += loss

        # Step the learning rate scheduler
        # In the first steps, the warmup scheduler is used.
        # After the warmup steps, the cosine annealing scheduler is used.
        if self.warmup_scheduler is not None:
            with self.warmup_scheduler.dampening():
                if self.scheduler is not None and self.warmup_scheduler.last_step + 1 >= self.scheduler_warmup_steps:
                    self.scheduler.step()
        elif self.scheduler is not None:
            self.scheduler.step()

        return total_loss / steps

    def train(self):
        self.model.train()

        # If the model is trained on a single worker, use the solo training method to avoid communication checks
        if self.size == 1:
            return self.train_solo()

        # Train the model with the hybrid SGD method
        for taken_steps in range((self.total_step_number + self.warmup_steps) // len(self.train_loader) + 1):
            for (data, target) in self.train_loader:

                # Evaluate the model every log_period steps
                if self.steps % self.log_period == 0:
                    self.evaluate()
                    if self.clear_cache:
                        torch.cuda.empty_cache()

                loss = None
                if len(data) < 2:
                    # coopy data to increase the size of the batch
                    data = torch.cat([data, data])
                    target = torch.cat([target, target])

                # Train the model on the batched data
                for turn in range(self.size // self.concurrency + 1):
                    if self.rank // self.concurrency == turn:
                        loss = self.take_step(data, target)
                        if self.concurrency < self.size:
                            torch.cuda.empty_cache()
                    if self.concurrency < self.size:
                        self.comm.Barrier()

                # Check if the loss is nan or diverged
                if loss is None or loss > 10 ** 4:
                    self.training_loss = loss
                    return self.history

                # We do not perform communication in the warmup steps.
                # You can set the exchange_period to alternate between communicating and not communicating
                # after each exchange_period steps.
                if self.steps < self.warmup_steps or (
                        self.exchange_period != 0 and self.steps // self.exchange_period % 2 == 0):
                    self.training_loss = loss
                    self.steps += 1
                    continue

                # Make pairs of workers to exchange the model parameters
                pairs = np.empty(self.size, dtype=np.int32)
                if self.rank == 0:
                    per = np.random.permutation(self.size)
                    for i in range(0, self.size - 1, 2):
                        pairs[per[i]] = per[i + 1]
                        pairs[per[i + 1]] = per[i]

                    if self.size % 2 == 1:
                        pairs[per[-1]] = -1
                self.comm.Bcast(pairs, root=0)

                # Exchange the model parameters with the paired worker and average the parameters
                partner_rank = pairs[self.rank]
                if partner_rank != -1:
                    model_copy = self.get_flat_model()
                    # We set the model_copy to the CPU to avoid memory issues while sending the model_copy.
                    # Note that you can set the model_copy to the GPU if you have enough memory and MPI is CUDA aware.
                    model_copy = model_copy.to('cpu')
                    data_received = self.comm.sendrecv(sendobj=model_copy, dest=partner_rank, source=partner_rank,
                                                       sendtag=0, recvtag=0)
                    new_model_param = (data_received + model_copy) / 2 if any(data_received) else model_copy
                    new_model_param = new_model_param.to(self.model.device)
                    self.copy_to_model(new_model_param)
                else:
                    if self.concurrency < self.size:
                        for turn in range(2 * (self.size // self.concurrency + 1)):
                            self.comm.Barrier()

                self.training_loss = loss
                self.steps += 1
                if self.steps == self.total_step_number + self.warmup_steps:
                    return self.history

        return self.history

    def train_solo(self):
        # This method is used to train the model on a single worker
        assert self.size == 1
        for taken_steps in range((self.total_step_number + self.warmup_steps) // len(self.train_loader) + 1):
            for (data, target) in self.train_loader:
                if self.steps % self.log_period == 0:
                    self.evaluate()
                    if self.dataset_name == 'bracket' and self.history[-1]['eval/accuracy'] < 0.6 and self.steps > 200:
                        return self.history
                    if self.dataset_name == 'cifar10' and self.history[-1]['eval/accuracy'] < 0.2 and self.steps > 200:
                        return self.history

                loss = self.take_step(data, target)
                self.training_loss = loss

                if loss is None or loss is torch.nan or loss > 10 ** 6:  # Diverged!
                    return self.history
                self.steps += 1
                if self.steps == self.total_step_number + self.warmup_steps:
                    return self.history

        return self.history

    def evaluate(self):
        self.model.eval()
        # We evaluate each worker on the whole evaluation dataset
        for turn in range(self.size // self.concurrency + 1):
            if self.rank // self.concurrency == turn:
                if self.verbose:
                    print(f"Rank {self.rank} steps: {self.steps} evaluate")
                result = self.model.evaluate(self.test_loader, self.criterion)
                if self.concurrency < self.size:
                    torch.cuda.empty_cache()
            if self.concurrency < self.size:
                self.comm.Barrier()
        validation_loss = result['loss']
        validation_accuracy = result['accuracy']
        training_loss = self.training_loss if self.training_loss else 0
        if self.verbose:
            print(f"Rank {self.rank}: " +
                  "Steps: {:5.0f}, Training loss: {:.4f}, Validation loss: {:.4f}, Validation accuracy: {:.2f}"
                  .format(self.steps,
                          training_loss,
                          validation_loss,
                          validation_accuracy)
                  )
        # Aggregate the results from all workers
        validation_loss_cum = self.comm.reduce(validation_loss, op=MPI.SUM, root=0)
        validation_accuracy_cum = self.comm.reduce(validation_accuracy, op=MPI.SUM, root=0)
        training_loss_cum = self.comm.reduce(training_loss, op=MPI.SUM, root=0)
        validation_loss_mean = np.empty(1)
        if self.rank == 0:
            validation_loss_mean[0] = validation_loss_cum / self.size
        self.comm.Bcast(validation_loss_mean, root=0)
        var_i = (validation_loss - validation_loss_mean[0]) ** 2
        var_ = self.comm.reduce(var_i, op=MPI.SUM, root=0)
        if self.rank == 0:
            validation_loss = validation_loss_cum / self.size
            validation_accuracy = validation_accuracy_cum / self.size
            training_loss = training_loss_cum / self.size
            std_ = np.sqrt(var_ / self.size)

            result_dict = {'step': int(self.steps), 'train/loss': float(training_loss),
                           'eval/loss': float(validation_loss), 'eval/accuracy': float(validation_accuracy),
                           'eval/std': float(std_)}
            wandb.log(result_dict)
            self.history.append(result_dict)
            print(f"Rank {self.rank}: Aggregated results: " +
                  "Steps: {:5.0f}, Training loss: {:.4f}, Validation loss: {:.4f}, Validation accuracy: {:.2f}"
                  .format(self.steps,
                          training_loss,
                          validation_loss,
                          validation_accuracy)
                  )

        self.model.train()

    def copy_to_model(self, model_copy_tensor):
        # Copy the model parameters from the model_copy_tensor to the model
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
        # Copy the model parameters from the model to the model_copy_tensor
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

    def get_flat_model(self):
        # Get the model parameters as a flat tensor
        model_flat = []
        for param in self.model.parameters():
            model_flat.append(param.data.view(-1))
        return torch.cat(model_flat)
