#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=10gb
#SBATCH --output=output_files/slurm_%j.out
#SBATCH --gres=gpu:1
#SBATCH -C L40S

RUNPATH=/home/dmhofmann/Lomas/eval_lomas
cd $RUNPATH
source /home/dmhofmann/env_gsafeguard/bin/activate
python3 test_gpu.pys