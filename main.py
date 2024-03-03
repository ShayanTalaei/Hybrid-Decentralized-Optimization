import torch
import argparse
from utils import run, plot_trends

# Import custom modules

# Main function
if __name__ == "__main__":
    """
    Main function to run the script. It parses command-line arguments, loads the dataset, splits it into training and validation sets,
    creates data loaders, sets the loss function, and performs cross-validation to find the best hyperparameters and optimizer.
    """
    # Create argument parser
    parser = argparse.ArgumentParser()
    # Add arguments to the parser
    parser.add_argument("--dataset", default="mnist", help="The dataset to use for training and testing.")
    parser.add_argument("--hidden", default=128, type=int, help="The number of hidden units in the model.")
    parser.add_argument("--num_layers", default="3", help="The list of numbers of layers in the model.")
    parser.add_argument("--conv_number", default="3", help="The list of numbers of convolutional layers in the model.")
    parser.add_argument("--batch_size", default=100, type=int, help="The batch size for training.")
    parser.add_argument("--epochs", default=2, type=int, help="The number of epochs to train for.")
    parser.add_argument("--plot", action="store_true", help="Whether to plot the training and validation curves.")
    parser.add_argument("--lr", default="0.001,0.001", help="The list of learning rates for the optimizers.")
    parser.add_argument("--optimizer", default="SGD,Adam",
                        help="The list of optimizers to use for training.")
    parser.add_argument("--activation", default="relu", help="The activation function to use in the model.")
    parser.add_argument("--save", action="store_true", help="Whether to save the trained model.")
    parser.add_argument("--path", default="./",
                        help="The directory where the trained model should be saved.")
    parser.add_argument("--criterion", default="cross_entropy", help="The loss function to use for training.")
    parser.add_argument("--verbose", action="store_true", help="Whether to print detailed training progress.")
    parser.add_argument("--scheduler", action="store_true", help="Whether to use a learning rate scheduler.")
    parser.add_argument("--num_iter", default=5, type=int, help="The number of different models to train.")

    # Parse the arguments
    args = parser.parse_args()

    # Convert string arguments to appropriate data types
    lrs = [float(lr) for lr in args.lr.split(",")]
    print(f"Learning rates: {lrs}")
    optimizers_ = args.optimizer.split(",")
    print(f"Optimizers: {optimizers_}")
    num_layers = [int(layer) for layer in args.num_layers.split(",")]
    print(f"Number of layers: {num_layers}")
    conv_numbers = [int(conv) for conv in args.conv_number.split(",")]
    print(f"Number of convolutional layers: {conv_numbers}")
    # Check if CUDA is available and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")
    print(f"Scheduler: {args.scheduler}")

    # Here is an example of how to initialize the setups.

    setups = {'2 FO': [{'grad_mode': 'first order', 'count': 2}],
              '6 ZO': [{'grad_mode': 'zeroth order forward-mode AD', 'count': 6, 'random vecs': 200}],
              '2 FO 6 ZO': [{'grad_mode': 'first order', 'count': 2},
                            {'grad_mode': 'zeroth order forward-mode AD', 'count': 6, 'random vecs': 200}]}

    dataset_name = 'mnist'
    lr_schedule = [(200, 0.001, 10)]

    logs = run(setups, dataset_name, lr_schedule, reps=1, path=args.path, file_name=None, batch_size=args.batch_size)

    name = None
    plot_trends(logs, 'Steps', 'Training loss', 100, dataset_folder=dataset_name, name=name)
