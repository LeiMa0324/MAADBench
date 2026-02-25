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

set -euo pipefail

echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

module load cuda12.1
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate Lomas311

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "After export, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

PORT=8081
MODEL=~/llama.cpp/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf
LOG_DIR=~/Lomas/Sage/logs
mkdir -p $LOG_DIR

~/llama.cpp/build/bin/llama-server \
  -m "$MODEL" \
  --host 127.0.0.1 \
  --port $PORT \
  --ctx-size 4096 \
  --gpu-layers -1 \
  --cache-ram 0 \
  > $LOG_DIR/server.${SLURM_JOB_ID}.log 2>&1 &
SERVER_PID=$!
echo "llama-server PID=$SERVER_PID"

# 等待 server 就绪，最多 120 秒，server 崩了立即报错退出
echo "Waiting for server /health..."
for i in $(seq 1 120); do
  sleep 4
  if curl -sf http://127.0.0.1:${PORT}/health >/dev/null; then
    echo "Health OK!"
    break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server crashed! Last 50 lines of server log:"
    tail -50 $LOG_DIR/server.${SLURM_JOB_ID}.log
    exit 1
  fi
  echo "Waiting... ($((i*4))s)"
done

curl -s http://127.0.0.1:${PORT}/v1/models || true
echo

cd ~/Lomas/Sage

python -u main.py --config_file=configs/config.yaml \
  > $LOG_DIR/client.${SLURM_JOB_ID}.log 2>&1 || true

echo "Stopping server..."
kill $SERVER_PID || true
wait $SERVER_PID || true
echo "DONE"