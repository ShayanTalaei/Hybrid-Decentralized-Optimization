import math
import sys
import time
import os
from mpi4py import MPI
import gc

import torch
import matplotlib.pyplot as plt

from trainer import HybridSGDTrainer
from datasets import get_dataset
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
        file_name=None, batch_size=100, model_name=None, freeze_model=False, plot=False, random_vecs=200, num_workers=2):
    results = {}
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    is_first = True if rank < fn else False
    train_set, test_set, input_shape, n_class = get_dataset(dataset_name, path=path)
    lr = lr1 if is_first else lr0
    try:
        for run_number in range(1, reps + 1):
            if is_first:
                grad_mode = 'first order'
                sampler = torch.utils.data.DistributedSampler(train_set, fn, rank)
            else:
                grad_mode = 'zeroth order forward-mode AD'
                sampler = torch.utils.data.DistributedSampler(train_set, size - fn, rank - fn)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=4 * batch_size, shuffle=True, num_workers=num_workers)
            initial_state_dict = None
            if rank == 0:
                initial_state_dict = get_temp_state_dict(dataset_name, input_shape, n_class, conv_number=conv_number, hidden=hidden, num_layer=num_layer, model_name=model_name, freeze_model=freeze_model)
            if size > 1:
                comm.barrier()
                initial_state_dict = comm.bcast(initial_state_dict, root=0)

            trainer = HybridSGDTrainer(rank, size, comm, fn, grad_mode,
                                       dataset_name, train_loader, test_loader,
                                       initial_state_dict, lr,
                                       conv_number=conv_number, hidden=hidden,
                                       num_layer=num_layer, model_name=model_name,
                                       freeze_model=freeze_model, random_vecs=random_vecs
                                       )
            if rank == 0:
                print(f"\n--- Run number: {run_number}")
            comm.Barrier()
            start_time = time.time()
            history = trainer.train(steps, log_period)
            comm.Barrier()
            end_time = time.time()
            trainer.win.Free()

            for key in history[0].keys():
                results[key] = [x[key] for x in history]
            if rank == 0:
                print("Running time: {:.4f}".format(float(end_time - start_time)))
            del trainer
            torch.cuda.empty_cache()
            gc.collect()
    except Exception as err:
        import traceback
        traceback.print_exc()
        print(err)
        sys.stdout.flush()
        sys.exit()

    if file_name:
        torch.save(results, path + f"results/{dataset_name}/{file_name}_{rank}")
    if rank == 0 and plot:
        name = 'test'
        os.makedirs(f'{path}/results/{dataset_name}', exist_ok=True)
        plot_trends(results, 'Steps', 'Training loss', 100, path=path, dataset_folder=dataset_name, name=name)
    return results
