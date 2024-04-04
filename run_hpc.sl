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
#SBATCH --time=230:00:00
#
#Define the amount of system RAM used by your job in GigaBytes
#SBATCH --mem=128G

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --nodelist=gpu224
#SBATCH --constraint=A10

#Send emails when a job starts, it is finished or it exits
#SBATCH --mail-user=matinansaripour@gmail.com
#SBATCH --mail-type=END,FAIL
#
#SBATCH --no-requeue
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV


#module load python/3.8.5
#module load python/3.10.4
#module load cuda/11.8
#module load cuda/11.1
#module load cudnn/8.1.0.77
#module load cudnn/8.1

#module load cuda/11.2
#module load cudnn/8.1.0.77

module load cuda/11.6
module load cudnn/8.6

#source $HOME/Jupyter/venv/bin/activate
#source $HOME/.bashrc

cd $HOME/Jupyter/Hybrid-Decentralized-Optimization

conda activate venv

mpiexec --allow-run-as-root -n 2 python main.py --dataset cifar10 --lr0 0.001 --lr1 0.001 --plot --steps 200 --fn 1 --rv 50