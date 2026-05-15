#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 40:00:00
#SBATCH --gpus=v100-32:1
#SBATCH -A YOUR_ALLOCATION

set -x


cd "${PROJECT_ROOT:-/path/to/dpr-bag}/evaluation/"
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-dpr_bag}"

# Optional: redirect HuggingFace cache to project space
# export HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf_cache}"
# mkdir -p "$HF_HOME"

nvidia-smi

# Example: evaluate DPR-BAG (BC) on PMC-MAD.
# Replace --pairs with the prediction file for any other configuration.
python eval_disco.py \
    --pairs ../outputs/pmc_mad/dpr_bag_bc.jsonl \
    --ref abstract \
    --pred generated_abstract
