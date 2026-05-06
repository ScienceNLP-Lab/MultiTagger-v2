#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 40:00:00
#SBATCH --gpus=v100-32:1
#SBATCH -A YOUR_ALLOCATION


set -x

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${PROJECT_ROOT:-/path/to/dpr-bag}/baselines/"

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-led_fp16}"


# Uncomment ONE of the following commands to run a specific LongT5 inference configuration.

# LongT5 (base) on PMC-MAD
# python longt5_inference.py --dataset pmc_mad --model_dir Stancld/longt5-tglobal-large-16384-pubmed-3k_steps

# LongT5 (base) on PubMedSum
# python longt5_inference.py --dataset pubmedsum --model_dir Stancld/longt5-tglobal-large-16384-pubmed-3k_steps