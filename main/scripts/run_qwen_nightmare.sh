#!/bin/bash
#SBATCH --job-name=qwen-nightmare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen_nightmare_%j.out
#SBATCH --error=logs/qwen_nightmare_%j.err

set -e

# ── Environment ───────────────────────────────────────────────
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate Lomas311_forge

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNNER="$REPO_ROOT/main/runner.py"
CONFIG1="$REPO_ROOT/configs/qwen/config_0.0.yaml"
CONFIG2="$REPO_ROOT/configs/qwen/config_0.3.yaml"
CONFIG3="$REPO_ROOT/configs/qwen/config_0.6.yaml"
ROOMS_FILE="$REPO_ROOT/output/rooms/nightmare_rooms.jsonl"

VLLM_MODEL="Qwen/Qwen2.5-14B-Instruct"
VLLM_PORT=8000

mkdir -p logs

# ── Launch vLLM server ────────────────────────────────────────
echo "=========================================="
echo "  Starting vLLM: $VLLM_MODEL"
echo "  Port: $VLLM_PORT"
echo "=========================================="

python -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --dtype auto &

VLLM_PID=$!

# Wait for vLLM to be ready
echo "Waiting for vLLM server to start..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM server ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died"
        exit 1
    fi
    sleep 1
done

if ! curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start within 120s"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# ── Run benchmark ─────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Running Nightmare Benchmark (Qwen)"
echo "  Config: $CONFIG"
echo "  Rooms:  $ROOMS_FILE"
echo "=========================================="

python "$RUNNER" \
    --rooms-file "$ROOMS_FILE" \
    --config "$CONFIG1" \
    --output-dir "$REPO_ROOT/output"

python "$RUNNER" \
    --rooms-file "$ROOMS_FILE" \
    --config "$CONFIG2" \
    --output-dir "$REPO_ROOT/output"

python "$RUNNER" \
    --rooms-file "$ROOMS_FILE" \
    --config "$CONFIG3" \
    --output-dir "$REPO_ROOT/output"

# ── Cleanup ───────────────────────────────────────────────────
echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo ""
echo "=========================================="
echo "  Done."
echo "=========================================="
