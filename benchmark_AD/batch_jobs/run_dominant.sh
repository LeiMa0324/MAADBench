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

# parameters
SEEDS=(1 2 3 4 5)
METHODS=("dominant")
TRACE_DATASET_NAMES=("EscapeRoom_v3_gsm_hard" "EscapeRoom_v3_live_code_bench")
TRACE_TYPES=("tool" "no_tool")
PATH_TRACES=""
TEST_SIZES=(0.2)
TEMP_LLMS=("0")
DATASET_LEVELS=("action")
LLM_IN_MAS=(['claude-sonnet-4-20250514', 'deepseek-reasoner', 'gpt-4.1', 'gpt-5.4', 'Qwen2.5-14B-Instruct'])

# Run experiments
for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
        for trace_dataset in "${TRACE_DATASET_NAMES[@]}"; do
            for trace_type in "${TRACE_TYPES[@]}"; do
                for test_size in "${TEST_SIZES[@]}"; do
                    for temp_llm in "${TEMP_LLMS[@]}"; do
                        for dataset_level in "${DATASET_LEVELS[@]}"; do
                            for llm in "${LLM_IN_MAS[@]}"; do
                                output_file="output_files/${method}_${trace_dataset}_${trace_type}_seed${seed}_temp${temp_llm// /_}_${dataset_level}_${llm//./_}.out"
                                python3 main.py \
                                    --seed "$seed" \
                                    --method "$method" \
                                    --trace_dataset_name "$trace_dataset" \
                                    --trace_type "$trace_type" \
                                    --path_traces "$PATH_TRACES" \
                                    --test_size "$test_size" \
                                    --temp_llm $temp_llm \
                                    --dataset_level "$dataset_level" \
                                    --llm_in_mas $llm \
                                > "$output_file"
                            done
                        done
                    done
                done
            done
        done
    done
done
    