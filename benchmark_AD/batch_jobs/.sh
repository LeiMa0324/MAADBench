#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=40gb
#SBATCH --output=output_files/slurm_%j.out
#SBATCH --gres=gpu:1
#SBATCH -C L40S

RUNPATH=/home/dmhofmann/grad_ascent_gen
cd $RUNPATH
source /home/dmhofmann/environment_folder_new/bin/activate

LR_VALUES=(0.0001)
LATENT_DIMS=(10)
MOMENTUM_VALUES=(75)
BATCH_SIZES=(256)
EPOCHS_VALUES=(100)
COL_NAME_LABELS="label"
W_B_VALUES=(0)
SEEDS=(1 2 3)
GRAD_ASCENT_STEP_SIZES=(5)
GRAD_ASCENT_NUM_STEPS=(10)
INLIERS_SELECT_AUGS=(1)
PERC_GEN_OUTLIERS_REMOVE=(0)
WARMUP_EPOCHS=(5)
GEN_EPOCHS=(10)
DATASETS=("fraud")



METHODS=(grad_ascent_gen)
EXPERIMENTS=("soft_labels adaptive_step")

# Run experiments
for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        for lr in "${LR_VALUES[@]}"; do
            for latent_dim in "${LATENT_DIMS[@]}"; do
                for momentum in "${MOMENTUM_VALUES[@]}"; do
                    for batch_size in "${BATCH_SIZES[@]}"; do
                        for epochs in "${EPOCHS_VALUES[@]}"; do
                            for w_b in "${W_B_VALUES[@]}"; do
                                for seed in "${SEEDS[@]}"; do
                                    for grad_step_size in "${GRAD_ASCENT_STEP_SIZES[@]}"; do
                                        for grad_num_steps in "${GRAD_ASCENT_NUM_STEPS[@]}"; do
                                            for inliers_select_aug in "${INLIERS_SELECT_AUGS[@]}"; do
                                                for perc_gen_outliers_remove in "${PERC_GEN_OUTLIERS_REMOVE[@]}"; do
                                                    for warmup_epochs in "${WARMUP_EPOCHS[@]}"; do
                                                        for gen_epochs in "${GEN_EPOCHS[@]}"; do
                                                           for experiment in "${EXPERIMENTS[@]}"; do
                                                               output_file="output_files/${method}_${dataset}_seed${seed}_lr${lr}_latent${latent_dim}.out"
                                                               python3 main.py \
                                                                   --lr "$lr" \
                                                                   --latent_dim "$latent_dim" \
                                                                   --momentum "$momentum" \
                                                                   --batch_size "$batch_size" \
                                                                   --epochs "$epochs" \
                                                                   --col_name_labels "$COL_NAME_LABELS" \
                                                                   --w_b "$w_b" \
                                                                   --seed "$seed" \
                                                                   --grad_ascent_step_size "$grad_step_size" \
                                                                   --grad_ascent_num_steps "$grad_num_steps" \
                                                                   --perc_gen_outliers_remove "$perc_gen_outliers_remove" \
                                                                   --warmup_epochs "$warmup_epochs" \
                                                                   --gen_epochs "$gen_epochs" \
                                                                   --inliers_select_aug "$inliers_select_aug" \
                                                                   --dataset "$dataset" \
                                                                   --method "$method" \
                                                                   --experiment $experiment \
                                                                   > "$output_file"
                                                           done
                                                       done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
