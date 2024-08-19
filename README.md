
# Hybrid Decentralized Optimization

## Overview

The source code for the paper "Hybrid Decentralized Optimization: Leveraging Both First- and Zeroth-Order Optimizers for Faster Convergence".

## Project Structure

- **main.py**: Entry point for running the optimization process.
- **models.py**: Contains the definitions of the machine learning models used in the experiments.
- **optimizers.py**: Implements the first-order and zeroth-order optimization algorithms.
- **trainer.py**: Handles the training loop and the interaction between nodes.
- **utils.py**: Contains utility functions to run the experiments.
- **models_bases/**: Contains the base classes for the models.
- **datasets/**: Contains scripts for loading and preprocessing datasets.
- **data/**: Contains "Brackets" train and evaluation datasets.
- **requirements.txt**: List of required packages.
- **hyperparams_opt/**: Configuration files and scripts for hyperparameter tuning.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd Hybrid-Decentralized-Optimization
   ```

2. **Install dependencies:**
   We recommend to use Python 3.8. Install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```
   If you have issues with installing `openmpi` and `mpi4py`, consider resolving the issue using conda installation:
   ```bash
    conda install -c conda-forge openmpi mpi4py
    ```

## Usage

1. **Login to your W&B account:**
   ```bash
   wandb login
   ```
2. **Run the optimization process:**
   ```bash
   mpiexec --oversubscribe --allow-run-as-root -np <num_nodes> python main.py <following_arguments>
   ```
    - `<num_nodes>`: Number of nodes to run the optimization process.
    - `<following_arguments>`: Arguments to specify the configuration, dataset, model, and optimizer.
    - `--oversubscribe`: Allow more processes than available cores.
    - `--allow-run-as-root`: Allow running the process as root.

   Example:
   ```bash
   mpiexec --oversubscribe --allow-run-as-root -np 20 python main.py --scheduler_warmup_steps 100 --warmup_steps 100 --seed 0 --steps 1000 --concurrency 100 --fn 4 --dataset bracket --model transformer --scheduler --z_grad zeroth_order_forward-mode_AD_sim --lr0=0.1 --momentum0=0.8 --rv=64 --z_batch_size=256 --f_batch_size=128 --lr1=0.05 --momentum1=0.8 --wandb_group brackets_16ZO_4FO
   ```

## Datasets

- **train_brackets_dataset.json**: Training data for the "Brackets" dataset.
- **test_brackets_dataset.json**: Test data for the "Brackets" dataset.
- Other datasets are loaded via the `datasets/` module.

## Models

The following models are supported and can be selected in `models.py`:

- **MLP (Customizable)**: For simple classification tasks.
- **CNN (Customizable)**: For image classification tasks on datasets like MNIST.
- **ResNet-18 (Pretrained on ImageNet-1K)**: For image classification tasks on datasets like CIFAR-10.
- **Transformer (Customizable)**: For sequence-based tasks such as the "Brackets" dataset.
