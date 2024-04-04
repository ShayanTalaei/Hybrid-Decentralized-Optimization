import numpy as np
import torch
import argparse
from utils import run
import os
import mpi4py

if __name__ == "__main__":
    """Main function to run the script."""
    # Create argument parser
    parser = argparse.ArgumentParser()
    # Add arguments to the parser
    parser.add_argument("--seed", default=0, type=int, help="The random seed to use for training.")
    parser.add_argument("--dataset", default="cifar10", help="The dataset to use for training and testing.")
    parser.add_argument("--hidden", default=128, type=int, help="The number of hidden units in the model.")
    parser.add_argument("--num_layer", default=2, type=int, help="The number of layers in the model.")
    parser.add_argument("--conv_number", default=2, type=int, help="The number of convolutional layers in the model.")
    parser.add_argument("--batch_size", default=100, type=int, help="The batch size for training.")
    parser.add_argument("--plot", action="store_true", help="Whether to plot the training and validation curves.")
    parser.add_argument("--lr0", default=0.0001, type=float, help="The learning rate for zero-order optimizers.")
    parser.add_argument("--lr1", default=0.0001, type=float, help="The learning rate for first-order optimizers.")
    parser.add_argument("--rv", default=200, type=int, help="The number of random vectors to use for zeroth-order "
                                                            "optimizers.")
    parser.add_argument("--steps", default=200, type=int, help="The learning steps for the optimizers.")
    parser.add_argument("--log_period", default=10, type=int, help="The log period.")
    parser.add_argument("--fn", default=3, type=int, help="The number of first-order optimizers.")
    parser.add_argument("--activation", default="relu", help="The activation function to use in the model.")
    parser.add_argument("--save", action="store_true", help="Whether to save the trained model.")
    parser.add_argument("--path", default="./", help="The directory where the trained model should be saved.")
    parser.add_argument("--criterion", default="cross_entropy", help="The loss function to use for training.")
    parser.add_argument("--verbose", action="store_true", help="Whether to print detailed training progress.")
    parser.add_argument("--model", default="resnet", help="The model to use for training. If None, a default model is "
                                                          "used based on the given arguments.")
    parser.add_argument("--freeze_model", action="store_true", help="Whether to freeze the model during training.")
    parser.add_argument("--scheduler", action="store_true", help="Whether to use a learning rate scheduler.")

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    mpi4py.rc.threads = False

    # Parse the arguments
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Check if CUDA is available and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Convert string arguments to appropriate data types
    if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Learning rates: Zero-order: {args.lr0}, First-order: {args.lr1}")
        print(f"Steps: {args.steps}")
        print(f"Number of layer: {args.num_layer}")
        print(f"Number of First-orders: {args.fn}")
        print(f"Number of Zero-orders: {mpi4py.MPI.COMM_WORLD.Get_size() - args.fn}")
        print(f"Number of convolutional layers: {args.conv_number}")
        print(f"Using {device}")

    # Run the training script

    logs = run(args.fn,
               args.dataset,
               args.steps,
               args.lr0,
               args.lr1,
               args.log_period,
               conv_number=args.conv_number,
               hidden=args.hidden,
               num_layer=args.num_layer,
               reps=1,
               path=args.path,
               file_name=None,
               batch_size=args.batch_size,
               model=args.model,
               freeze_model=args.freeze_model,
               plot=args.plot,
               random_vecs=args.rv,
               )
