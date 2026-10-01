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
METHODS=("ma2df_improved")
TRACE_DATASET_NAMES=("gsm_hard")
PATH_TRACES=""
TEST_SIZES=(0.2)

# Run experiments
for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
        for trace_dataset in "${TRACE_DATASET_NAMES[@]}"; do
            for test_size in "${TEST_SIZES[@]}"; do
                output_file="output_files/${method}_${trace_dataset}_seed${seed}.out"
                python3 main.py \
                    --seed "$seed" \
                    --method "$method" \
                    --trace_dataset_name "$trace_dataset" \
                    --path_traces "$PATH_TRACES" \
                    --test_size "$test_size" \
                    > "$output_file"
            done
        done
    done
done
    