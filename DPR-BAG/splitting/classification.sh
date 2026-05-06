#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 40:00:00
#SBATCH --gpus=v100-32:1
#SBATCH -A YOUR_ALLOCATION   # TODO: replace with your SLURM allocation account


# echo commands to stdout - these are saved to a slurm log file in the same directory as your sh file
set -x


cd "${PROJECT_ROOT:-/path/to/dpr-bag}/splitting"


source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-llm-ssc}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# Required: set HF_TOKEN in your environment before running, e.g.:
  #   export HF_TOKEN="your_token_here"
  # Or place it in ~/.bashrc / a .env file (do NOT commit it).


# --- FS: First Sentence Labeling (requires LLM-SSC checkpoint) ---
python fs.py \
    --dataset pubmedsum \
    --output ./output/fs.jsonl \
    --model-path ./checkpoints/llm_ssc/best_model.mdl

# --- NS: Naive Splitting (no model) ---
# python ns.py \
#     --dataset pmc_mad \
#     --output ./output/ns.jsonl

# --- SH: Section-Header Normalization (requires 2 models from Lin et al. 2025) ---
# python sh.py \
#     --dataset pmc_mad \
#     --output ./output/sh.jsonl \
#     --bert-model-path ./checkpoints/sh/sentence_bert \
#     --classifier-model-path ./checkpoints/sh/classifier.pth
