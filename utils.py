import math
import sys
import time
import os
from mpi4py import MPI
import gc
import pickle

import torch
import matplotlib.pyplot as plt

from trainer import HybridSGDTrainer
from datasets.datasets import get_dataset
from models import get_temp_state_dict


def cast(lst, dtype=torch.float32):
    return list(map(lambda x: torch.tensor(x).to(dtype), lst))


def plot_trends(trends, x_axis, y_axis, start=0, path=None, end=float('inf'), dataset_folder=None, name=None):
    fig = plt.figure()
    fig.set_facecolor('white')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid(True)

    shapes = ["8", "s", "p", "P", "*", "h", "H", "x", "d", "D"]

    for i, case in enumerate(trends):
        trend = trends[case]
        X, Y = trend[x_axis], trend[y_axis]
        Z = zip(X, Y)
        X, Y = [], []
        for z in Z:
            if start <= z[0] <= end:
                X.append(z[0])
                Y.append(z[1])
        count = len(X)
        plt.plot(X, Y, marker=shapes[i], label=case, markevery=math.ceil(count * 0.1))

    plt.legend()
    if name is not None:
        plt.savefig(path + f'results/{dataset_folder}/{name}_{y_axis}_{x_axis}.pdf')
    plt.show()


def run(fn, dataset_name, steps, lr0, lr1, log_period, conv_number=2, hidden=128, num_layer=2, reps=1, path=None,
        file_name=None, model_name=None, freeze_model=False, plot=False, random_vecs=200,
        num_workers=2, momentum0=0.0, momentum1=0.0, f_grad='first_order', z_grad='zeroth_order_cge', scheduler=False,
        scheduler_warmup_steps=0, warmup_steps=0, v_step=10.0, out_channels=8, f_batch_size=100, z_batch_size=100,
        is_cuda_aware=False, concurrency=1, device='cpu', config=None, exchange_period=0, verbose=True):
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
            # test_loader = torch.utils.data.DataLoader(test_set, batch_size=4 * batch_size, num_workers=num_workers)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size*2, num_workers=num_workers)
            initial_state_dict = None
            if rank == 0:
                initial_state_dict = get_temp_state_dict(input_shape, n_class, conv_number=conv_number,
                                                         hidden=hidden, num_layer=num_layer, model_name=model_name,
                                                         freeze_model=freeze_model, out_channels=out_channels,
                                                         device=device, config=config
                                                         )
            if size > 1:
                comm.barrier()
                initial_state_dict = comm.bcast(initial_state_dict, root=0)

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
                                       exchange_period=exchange_period, verbose=verbose
                                       )
            if rank == 0:
                print(f"\n--- Run number: {run_number}")
            comm.Barrier()
            start_time = time.time()
            history = trainer.train()
            comm.Barrier()
            end_time = time.time()
            if size > 1:
                trainer.win.Free()

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
    # if rank == 0 and plot:
    #     name = 'test'
    #     os.makedirs(f'{path}/results/{dataset_name}', exist_ok=True)
    #     plot_trends(results, 'Steps', 'Training loss', 100, path=path, dataset_folder=dataset_name, name=name)
    return results
