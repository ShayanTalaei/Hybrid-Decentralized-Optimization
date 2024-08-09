#!/bin/bash
#
#SBATCH --job-name=2l2c
#SBATCH --output=2l2c.txt
#
#number of CPUs to be used
##SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=2

#
#Define the number of hours the job should run.
#Maximum runtime is limited to 10 days, ie. 240 hours
#SBATCH --time=3:00:00
#
#Define the amount of system RAM used by your job in GigaBytes
#SBATCH --mem=100G

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=A10


#Send emails when a job starts, it is finished or it exits
#SBATCH --mail-user=matinansaripour@gmail.com
#SBATCH --mail-type=END,FAIL
#
#SBATCH --no-requeue
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV


#cd $HOME/Jupyter/Hybrid-Decentralized-Optimization
#source ./venv/bin/activate

source $HOME/.bashrc
conda activate hdo

module load openmpi/4.1.6
module load cuda/12
module load cudnn/8.9.5.30

#pip install -r requirements.txt

srun python main.py --seed 0 --steps 1500 --fn 1 --dataset cifar10 --model resnet --scheduler --z_grad zeroth_order_forward-mode_AD_sim --lr0=0.01 --momentum0=0.9 --rv=32 --z_batch_size=50 --f_batch_size=10 --lr1=0.001 --momentum1=0.9 --wandb_group resnet_5ZO_1FO
#srun python main.py --z_batch_size 500 --f_batch_size 100 --dataset mnist --lr0 0.05 --lr1 0.1\
# --plot --steps 200 --fn 2 --rv 100 --model cnn --momentum 0.9 --z_grad zeroth_order_rge --warmup_steps 0 --file_name zo\
# --scheduler --scheduler_warmup_steps 0 --v_step 0.005 --log_period 10 --conv_number 3 --hidden 128 --out_channels 32 --num_layer 3
#mpiexec --allow-run-as-root -n 2 python main.py --z_batch_size 500 --f_batch_size 100 --dataset mnist --lr0 0.05 --lr1 0.1\
# --plot --steps 200 --fn 2 --rv 100 --model cnn --momentum 0.9 --z_grad zeroth_order_rge --warmup_steps 0 --file_name zo\
 # --scheduler --scheduler_warmup_steps 0 --v_step 0.005 --log_period 10 --conv_number 3 --hidden 128 --out_channels 32 --num_layer 3