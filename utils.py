import collections
import sys
import time
import os
from mpi4py import MPI
import gc
import pickle

import torch

from trainer import HybridSGDTrainer
from datasets.datasets import get_dataset
from models import get_temp_state_dict


def cast(lst, dtype=torch.float32):
    return list(map(lambda x: torch.tensor(x).to(dtype), lst))


def run(fn, dataset_name, steps, lr0, lr1, log_period, conv_number=2, hidden=128, num_layer=2, reps=1, path=None,
        file_name=None, model_name=None, freeze_model=False, random_vecs=200,
        num_workers=2, momentum0=0.0, momentum1=0.0, f_grad='first_order', z_grad='zeroth_order_cge', scheduler=False,
        scheduler_warmup_steps=0, warmup_steps=0, v_step=10.0, out_channels=8, f_batch_size=100, z_batch_size=100,
        is_cuda_aware=False, concurrency=1, device='cpu', config=None, exchange_period=0, verbose=True,
        clear_cache=False):
    results = {}
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    is_first = True if rank < fn else False
    is_vtransformer = True if model_name == 'vtransformer' else False
    if rank == 0:
        train_set, test_set, input_shape, n_class, args = get_dataset(dataset_name, path=path, is_vtransformer=is_vtransformer)
    comm.Barrier()
    if rank != 0:
        train_set, test_set, input_shape, n_class, args = get_dataset(dataset_name, path=path,
                                                                      is_vtransformer=is_vtransformer)
    if 'vocab_size' in args:
        vars(config)['vocab_size'] = args['vocab_size']
    lr = lr1 if is_first else lr0
    try:
        for run_number in range(1, reps + 1):
            # Set the data partition, the batch size and the gradient mode for the worker
            if is_first:
                grad_mode = f_grad
                sampler = torch.utils.data.DistributedSampler(train_set, fn, rank)
                batch_size = f_batch_size
            else:
                grad_mode = z_grad
                sampler = torch.utils.data.DistributedSampler(train_set, size - fn, rank - fn)
                batch_size = z_batch_size
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                                                       num_workers=num_workers)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size*2, num_workers=num_workers)

            # Sync the initial state dict among the workers
            initial_state_dict = None
            if rank == 0:
                initial_state_dict = get_temp_state_dict(input_shape, n_class, conv_number=conv_number,
                                                         hidden=hidden, num_layer=num_layer, model_name=model_name,
                                                         freeze_model=freeze_model, out_channels=out_channels,
                                                         device=device, config=config
                                                         )
                # We put the state dict on the CPU to avoid memory issues while broadcasting.
                initial_state_dict = collections.OrderedDict(
                    {key: value.to('cpu') for key, value in initial_state_dict.items()}
                )
                if concurrency < size:
                    torch.cuda.empty_cache()
            if size > 1:
                comm.barrier()
                initial_state_dict = comm.bcast(initial_state_dict, root=0)

            # Create the trainer
            trainer = HybridSGDTrainer(rank, size, comm, fn, grad_mode,
                                       dataset_name, train_loader, test_loader,
                                       initial_state_dict, lr,
                                       conv_number=conv_number, hidden=hidden,
                                       num_layer=num_layer, model_name=model_name,
                                       freeze_model=freeze_model, random_vecs=random_vecs,
                                       momentum0=momentum0, momentum1=momentum1, scheduler=scheduler,
                                       scheduler_warmup_steps=scheduler_warmup_steps, warmup_steps=warmup_steps,
                                       total_step_number=steps, log_period=log_period,
                                       v_step=v_step, out_channels=out_channels,
                                       is_cuda_aware=is_cuda_aware, device=device,
                                       config=config, concurrency=concurrency,
                                       exchange_period=exchange_period, verbose=verbose,
                                       clear_cache=clear_cache
                                       )
            if rank == 0:
                print(f"\n--- Run number: {run_number}")
            comm.Barrier()

            # Train the model
            start_time = time.time()
            history = trainer.train()
            comm.Barrier()
            end_time = time.time()

            if rank == 0:
                for key in history[0].keys():
                    results[key] = [x[key] for x in history]
                print("Running time: {:.4f}".format(float(end_time - start_time)))
            del trainer
            torch.cuda.empty_cache()
            gc.collect()
    except Exception as err:
        import traceback
        traceback.print_exc()
        if verbose:
            print(err)
        sys.stdout.flush()
        sys.exit()

    if file_name:
        os.makedirs(f'{path}/results/{dataset_name}', exist_ok=True)
        with open(f'{path}/results/{dataset_name}/{file_name}_rank_{rank}_size_{size}_fn_{fn}_warmup_{warmup_steps}_steps_{steps}.pkl', 'wb') as file:
            pickle.dump(results, file)

    return results
