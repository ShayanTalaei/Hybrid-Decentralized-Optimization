#!/bin/bash
#
#SBATCH --job-name=2l2c
#SBATCH --output=2l2c.txt
#
#number of CPUs to be used
#SBATCH --ntasks=1
#SBATCH -c 4
#
#Define the number of hours the job should run.
#Maximum runtime is limited to 10 days, ie. 240 hours
#SBATCH --time=3:00:00
#
#Define the amount of system RAM used by your job in GigaBytes
#SBATCH --mem=40G

#SBATCH --partition=gpu


#Send emails when a job starts, it is finished or it exits
#SBATCH --mail-user=matinansaripour@gmail.com
#SBATCH --mail-type=END,FAIL
#
#SBATCH --no-requeue
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV


cd $HOME/Jupyter/Hybrid-Decentralized-Optimization
source ./venv/bin/activate

module load openmpi/4.1.6
module load python/3.10
module load cuda/12.3.1
module load cudnn/8.9.5.30


mpiexec --with-cuda --oversubscribe --allow-run-as-root -np 2 python main.py --z_batch_size 500 --f_batch_size 100 --dataset mnist --lr0 0.05 --lr1 0.1\
 --plot --steps 200 --fn 2 --rv 100 --model cnn --momentum 0.9 --z_grad zeroth_order_rge --warmup_steps 0 --file_name zo\
  --scheduler --scheduler_warmup_steps 0 --v_step 0.005 --log_period 10 --conv_number 3 --hidden 128 --out_channels 32 --num_layer 3