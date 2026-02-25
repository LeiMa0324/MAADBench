#!/bin/bash
#SBATCH --job-name=Deepseek_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100-80G|H100|H200"
#SBATCH --time=23:59:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $CUDA_VISIBLE_DEVICES"
echo "Start:   $(date)"

# ── 路径 ──────────────────────────────────────────────────
WORK_DIR=~/Lomas/Sage

cd $WORK_DIR
mkdir -p logs

# ── 环境（按你集群实际情况选一种）────────────────────────
# conda:
# module load anaconda3 && conda activate base
# venv:
# source ~/envs/steering/bin/activate

source ~/miniconda3/etc/profile.d/conda.sh
conda activate Lomas311
# ── 安装依赖（首次运行取消注释）──────────────────────────
# pip install --user torch transformers accelerate huggingface_hub



# ── 运行实验 ──────────────────────────────────────────────
python -u main.py --config_file=configs/config.yaml

echo "End: $(date)"