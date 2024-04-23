import numpy as np
import torch
import argparse
from utils import run
import os
import mpi4py
from mpi4py.util import dtlib
from mpi4py import MPI

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
    parser.add_argument("--f_batch_size", default=25, type=int, help="The first order optimizer batch size.")
    parser.add_argument("--z_batch_size", default=100, type=int, help="The zero order optimizer batch size.")
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
    parser.add_argument("--model", default="resnet", help="The model to use for training. If None, a default model is "
                                                          "used based on the given arguments.")
    parser.add_argument("--freeze_model", action="store_true", help="Whether to freeze the model during training.")
    parser.add_argument("--scheduler", action="store_true", help="Whether to use a learning rate scheduler.")
    parser.add_argument("--scheduler_warmup_steps", default=0, type=int, help="The number of warmup steps for the "
                                                                              "scheduler.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="The number of warmup steps before starting the "
                                                                    "communication.")
    parser.add_argument("--momentum", default=0.0, type=float, help="The momentum parameter for the optimizer.")
    parser.add_argument("--f_grad", default="first_order", help="The gradient mode for the first-order.")
    parser.add_argument("--z_grad", default="zeroth_order_cge", help="The gradient mode for the zeroth-order.")
    parser.add_argument("--v_step", default=10.0, type=float, help="The step size for the zeroth-order optimizer.")
    parser.add_argument("--out_channels", default=8, type=int, help="The number of output channels for the cnn model.")
    parser.add_argument("--file_name", default=None, help="The name of the file to save the trained model.")

    # mpi4py.rc.threads = False
    # MPI.Finalize()
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # required = mpi4py.MPI.THREAD_MULTIPLE
    # data = torch.zeros(10, dtype=torch.float64)  # Integer array of size 10
    # win = MPI.Win.Create(data, disp_unit=data.itemsize, comm=MPI.COMM_WORLD)
    # buf = MPI.memory.fromaddress(data.data_ptr(),
    #                              data.nelement() * data.element_size())
    # MPI.Win.Create(buf, comm=MPI.COMM_WORLD)
    # memory = torch.ones((1, 1))
    # buf = mpi4py.MPI.memory.fromaddress(memory.data_ptr(), memory.numel() * memory.element_size())
    # mpi4py.MPI.Win.Create(buf, comm=mpi4py.MPI.COMM_WORLD, disp_unit=memory.element_size())

    # print('rank:', mpi4py.MPI.COMM_WORLD.Get_rank(), 'required:', required)
    import numpy as np
    from mpi4py import MPI
    from mpi4py.util import dtlib

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    datatype = MPI.FLOAT
    np_dtype = dtlib.to_numpy_dtype(datatype)
    itemsize = datatype.Get_size()

    N = 10
    win_size = N * itemsize if rank == 0 else 0
    win = MPI.Win.Allocate(win_size, comm=comm)

    buf = np.empty(N, dtype=np_dtype)
    if rank == 0:
        buf.fill(42)
        win.Lock(rank=0)
        win.Put(buf, target_rank=0)
        win.Unlock(rank=0)
        comm.Barrier()
    else:
        comm.Barrier()
        win.Lock(rank=0)
        win.Get(buf, target_rank=0)
        win.Unlock(rank=0)
        assert np.all(buf == 42)

    # Parse the arguments
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Check if CUDA is available and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Convert string arguments to appropriate data types
    print('rank:', mpi4py.MPI.COMM_WORLD.Get_rank(), 'size:', mpi4py.MPI.COMM_WORLD.Get_size())
    if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Learning rates: Zero-order: {args.lr0}, First-order: {args.lr1}")
        print(f"Steps: {args.steps}")
        print(f"Number of layer: {args.num_layer}")
        print(f"Number of convolutional layers: {args.conv_number}")
        print(f"Out channels: {args.out_channels}")
        print(f"Dataset: {args.dataset}")
        print(f"Number of First-orders: {args.fn}")
        print(f"Number of Zero-orders: {mpi4py.MPI.COMM_WORLD.Get_size() - args.fn}")
        print(f"First-order gradient mode: {args.f_grad}")
        print(f"Zero-order gradient mode: {args.z_grad}")
        print(f"First-order batch size: {args.f_batch_size}")
        print(f"Zero-order batch size: {args.z_batch_size}")
        print(f"Hidden units: {args.hidden}")
        print(f"Random vectors: {args.rv}")
        print(f"Model: {args.model}")
        print(f"Freeze model: {args.freeze_model}")
        print(f"Learning rate scheduler: {args.scheduler}")
        print(f"Learning rate scheduler warmup steps: {args.scheduler_warmup_steps}")
        print(f"Warmup steps: {args.warmup_steps}")
        print(f"v_step: {args.v_step}")
        print(f"Momentum: {args.momentum}")
        print(f"Log period: {args.log_period}")
        print(f"Plot: {args.plot}")
        print(f"File name: {args.file_name}")
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
               file_name=args.file_name,
               model_name=args.model,
               freeze_model=args.freeze_model,
               plot=args.plot,
               random_vecs=args.rv,
               momentum=args.momentum,
               f_grad=args.f_grad,
               z_grad=args.z_grad,
               scheduler=args.scheduler,
               scheduler_warmup_steps=args.scheduler_warmup_steps,
               warmup_steps=args.warmup_steps,
               v_step=args.v_step,
               out_channels=args.out_channels,
               f_batch_size=args.f_batch_size,
               z_batch_size=args.z_batch_size
               )
