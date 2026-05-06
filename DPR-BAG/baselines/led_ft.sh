#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 40:00:00
#SBATCH --gpus=v100-32:1
#SBATCH -A YOUR_ALLOCATION


# echo commands to stdout - these are saved to a slurm log file in the same directory as your sh file
set -x

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${PROJECT_ROOT:-/path/to/dpr-bag}/baselines/"

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-led_fp16}"

python led_ft.py \
    --base_model patrickvonplaten/led-large-16384-pubmed \
    --output_dir ./checkpoints/led_pubmed_finetune

