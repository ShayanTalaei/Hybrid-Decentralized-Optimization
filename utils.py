import math
import time

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


def run(setups, dataset_name, lr_schedule, conv_number=2, hidden=128, num_layer=2, reps=1, path=None, file_name=None, batch_size=100, model=None, freeze_model=False):
    results = {}
    for run_number in range(1, reps + 1):
        for case in setups:
            case_name = case if reps == 1 else case + f" run:{run_number}"
            results[case_name] = {}

    for run_number in range(1, reps + 1):
        train_set, test_set, input_shape, n_class = get_dataset(dataset_name, path=path)
        initial_state_dict = get_temp_state_dict(dataset_name, input_shape, n_class, conv_number=conv_number, hidden=hidden, num_layer=num_layer, model=model, freeze_model=freeze_model)

        for case, population_args in setups.items():
            print(f"\n--- Case: {case}, run number: {run_number}")
            start_time = time.time()
            trainer = HybridSGDTrainer(population_args,
                                       dataset_name, train_set, test_set,
                                       initial_state_dict, batch_size,
                                       conv_number=conv_number, hidden=hidden, num_layer=num_layer, model=model, freeze_model=freeze_model)
            history = trainer.train(lr_schedule)
            for key in history[0].keys():
                case_name = case if reps == 1 else case + f" run:{run_number}"
                results[case_name][key] = [x[key] for x in history]
            end_time = time.time()
            print("Running time: {:.4f}".format(float(end_time - start_time)))
    if file_name:
        torch.save(results, path + f"results/{dataset_name}/{file_name}")
    return results
