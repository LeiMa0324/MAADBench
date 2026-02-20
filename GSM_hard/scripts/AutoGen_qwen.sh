#!/bin/bash
#!/bin/bash
#SBATCH --mem=64G
#SBATCH --job-name=QWEN_H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100-80G|H100|H200"
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load cuda12.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Lomas311

# 启动 DeepSeek server
~/llama.cpp/build/bin/llama-server \
  -m ~/llama.cpp/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --port 8080 \
  --gpu-layers 99 \
  --cache-ram 0 \
  &

# 等待 server 就绪
# 等待模型真正加载完毕
echo "Waiting for server..."
until curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' \
  2>/dev/null | grep -q "content"; do
    sleep 3
done
echo "Server ready!"

cd ~/Lomas/GSM_hard
# 运行你的 Python 脚本
python main.py --mas_arch=AutoGen --config_file=configs/default_config_qwen.yaml

# 结束后关闭 server
kill %1