#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 40:00:00
#SBATCH --gpus=v100-32:1
#SBATCH -A YOUR_ALLOCATION


# echo commands to stdout - these are saved to a slurm log file in the same directory as your sh file
set -x

cd "${PROJECT_ROOT:-/path/to/dpr-bag}/evaluation/"

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-dpr_bag}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# Optional: redirect HuggingFace cache to project space
# export HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf_cache}"
# mkdir -p "$HF_HOME"

nvidia-smi
export HF_DATASETS_CACHE="/ocean/projects/cis230089p/slin23/hf_cache"



# Example: evaluate DPR-BAG (BC) on PMC-MAD.
# Replace --pairs with the prediction file for any other configuration.
python eval_umls.py \
    --pairs ../outputs/pmc_mad/dpr_bag_bc.jsonl \
    --ref abstract \
    --pred generated_abstract