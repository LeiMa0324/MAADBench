#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=80gb
#SBATCH --output=output_files/slurm_%j.out
#SBATCH -C A100-80G
#SBATCH --gres=gpu:A100:1

RUNPATH=/home/dmhofmann/Lomas/eval_lomas
cd $RUNPATH
source /home/dmhofmann/env_test_llms_2/bin/activate

# parameters
SEEDS=(6 7 8 9)
METHODS=("gemma-4-E2B")
SUPERVISIONS=("few_shot" "zero_shot")
TRACE_DATASET_NAMES=("EscapeRoom_v3_gsm_hard" "EscapeRoom_v3_live_code_bench")
TRACE_TYPES=("tool" "no_tool")
TEMP_LLMS=("0")

for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for supervision in "${SUPERVISIONS[@]}"; do
            for trace_dataset in "${TRACE_DATASET_NAMES[@]}"; do
                for trace_type in "${TRACE_TYPES[@]}"; do
                    for temp_llm in "${TEMP_LLMS[@]}"; do
                        echo "Running method=$method seed=$seed supervision=$supervision trace_dataset=$trace_dataset trace_type=$trace_type temp_llm=$temp_llm"
                        output_file="output_files/${method}_${supervision}_${trace_dataset}_${trace_type}_seed${seed}_temp${temp_llm// /_}.out"
                        python3 methods/llm_as_judge/run_gemma_4.py \
                            --seed "$seed" \
                            --method "$method" \
                            --supervision "$supervision" \
                            --trace_dataset_name "$trace_dataset" \
                            --trace_type "$trace_type" \
                            --dataset_level "action" \
                            --temp_llm $temp_llm \
                        > "$output_file"
                    done
                done
            done
        done
    done
done
    