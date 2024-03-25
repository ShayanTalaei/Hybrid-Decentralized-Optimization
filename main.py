import torch
import argparse
from utils import run, plot_trends
import os

if __name__ == "__main__":
    """Main function to run the script."""
    # Create argument parser
    parser = argparse.ArgumentParser()
    # Add arguments to the parser
    parser.add_argument("--dataset", default="cifar10", help="The dataset to use for training and testing.")
    parser.add_argument("--hidden", default=128, type=int, help="The number of hidden units in the model.")
    parser.add_argument("--num_layer", default=2, type=int, help="The number of layers in the model.")
    parser.add_argument("--conv_number", default=2, type=int, help="The number of convolutional layers in the model.")
    parser.add_argument("--batch_size", default=100, type=int, help="The batch size for training.")
    parser.add_argument("--plot", action="store_true", help="Whether to plot the training and validation curves.")
    parser.add_argument("--lr", default="0.0001", help="The list of learning rates for the optimizers.")
    parser.add_argument("--steps", default="200", help="The list of learning steps for the optimizers.")
    parser.add_argument("--log_period", default="10", help="The list of log periods.")
    parser.add_argument("--activation", default="relu", help="The activation function to use in the model.")
    parser.add_argument("--save", action="store_true", help="Whether to save the trained model.")
    parser.add_argument("--path", default="./", help="The directory where the trained model should be saved.")
    parser.add_argument("--criterion", default="cross_entropy", help="The loss function to use for training.")
    parser.add_argument("--verbose", action="store_true", help="Whether to print detailed training progress.")
    parser.add_argument("--model", default="resnet", help="The model to use for training. If None, a default model is used based on the given arguments.")
    parser.add_argument("--freeze_model", action="store_true", help="Whether to freeze the model during training.")

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # Parse the arguments
    args = parser.parse_args()

    # Convert string arguments to appropriate data types
    lrs = [float(lr) for lr in args.lr.split(",")]
    steps_ls = [int(steps) for steps in args.steps.split(",")]
    log_periods = [int(log_period) for log_period in args.log_period.split(",")]
    print(f"Learning rates: {lrs}")
    print(f"Number of layer: {args.num_layer}")
    print(f"Number of convolutional layers: {args.conv_number}")
    # Check if CUDA is available and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")

    setups = {'9 ZO': [{'grad_mode': 'zeroth order forward-mode AD', 'count': 9, 'random vecs': 100}],
              '3 FO': [{'grad_mode': 'first order', 'count': 3}],
              '3 FO 9 ZO': [{'grad_mode': 'first order', 'count': 3},
                            {'grad_mode': 'zeroth order forward-mode AD', 'count': 9, 'random vecs': 100}]}

    dataset_name = args.dataset
    lr_schedule = [(steps, lr, log_period) for steps, lr, log_period in zip(steps_ls, lrs, log_periods)]

    logs = run(setups,
               dataset_name,
               lr_schedule,
               conv_number=args.conv_number,
               hidden=args.hidden,
               num_layer=args.num_layer,
               reps=1, path=args.path,
               file_name=None,
               batch_size=args.batch_size,
               model=args.model,
               freeze_model=args.freeze_model,
               )

    name = 'test'
    os.makedirs(f'{args.path}/results/{dataset_name}', exist_ok=True)
    if args.plot:
        plot_trends(logs, 'Steps', 'Training loss', 100, path=args.path, dataset_folder=dataset_name, name=name)
