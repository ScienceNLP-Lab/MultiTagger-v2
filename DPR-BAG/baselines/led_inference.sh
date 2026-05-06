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


# Uncomment ONE of the following commands to run a specific LED inference configuration.

# LED-arXiv (base) on PMC-MAD
# python led_inference.py --dataset pmc_mad --model_dir allenai/led-large-16384-arxiv

# LED-arXiv (base) on PubMedSum
# python led_inference.py --dataset pubmedsum --model_dir allenai/led-large-16384-arxiv

# LED-PubMed (base) on PMC-MAD
# python led_inference.py --dataset pmc_mad --model_dir patrickvonplaten/led-large-16384-pubmed

# LED-PubMed (base) on PubMedSum
# python led_inference.py --dataset pubmedsum --model_dir patrickvonplaten/led-large-16384-pubmed

# LED-PubMed (FT) on PMC-MAD  (after fine-tuning with led_ft.sh)
# python led_inference.py --dataset pmc_mad --model_dir ./checkpoints/led_pubmed_finetune