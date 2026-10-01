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
SEEDS=(1 2 3 4 5)
METHODS=("gemma-4-E2B")
SUPERVISIONS=("few_shot" "zero_shot")
for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for supervision in "${SUPERVISIONS[@]}"; do
            echo "Running method=$method seed=$seed supervision=$supervision"
            python3 methods/llm_as_judge/run_gemma_4.py \
                --seed "$seed" \
                --method "$method" \
                --supervision "$supervision"
        done
    done
done