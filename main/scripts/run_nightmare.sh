#!/bin/bash
# Run nightmare rooms across claude, deepseek, and gpt-5.4 configs (all temperatures)

set -e

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate Lomas311_forge

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOMS_FILE="$REPO_ROOT/output/rooms/nightmare_rooms.jsonl"
RUNNER="$REPO_ROOT/main/runner.py"

CONFIGS=(
    "$REPO_ROOT/configs/claude/config_0.0.yaml"
    "$REPO_ROOT/configs/claude/config_0.3.yaml"
    "$REPO_ROOT/configs/claude/config_0.6.yaml"
#    "$REPO_ROOT/configs/deepseek/config_0.0.yaml"
#    "$REPO_ROOT/configs/deepseek/config_0.3.yaml"
#    "$REPO_ROOT/configs/deepseek/config_0.6.yaml"
    "$REPO_ROOT/configs/gpt_54/config_0.0.yaml"
    "$REPO_ROOT/configs/gpt_54/config_0.3.yaml"
    "$REPO_ROOT/configs/gpt_54/config_0.6.yaml"
)

echo "=========================================="
echo "  Nightmare Room Benchmark"
echo "  Rooms file: $ROOMS_FILE"
echo "  Configs: ${#CONFIGS[@]}"
echo "=========================================="

for config in "${CONFIGS[@]}"; do
    echo ""
    echo "────────────────────────────────────────"
    echo "  Config: $(basename "$(dirname "$config")")/$(basename "$config")"
    echo "────────────────────────────────────────"
    python "$RUNNER" \
        --rooms-file "$ROOMS_FILE" \
        --config "$config" \
        --output-dir "$REPO_ROOT/output"
done

echo ""
echo "=========================================="
echo "  All runs complete."
echo "=========================================="
