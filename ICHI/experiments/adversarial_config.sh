#!/bin/bash

# activate multitagger environment
source /username/miniconda3/bin/activate your_virtual_env

python train.py \
  --experiment_name="adversarial_training_dynamic" \
  --contrastive_loss="HeroCon" \
  --adnce_w1=1.0 \
  --adnce_w2=1.0 \
  --cl_alpha=0.01 \
  --cl_beta=0.1 \
  --cl_focusing=1 \
  --cl_clipping=1 \
  --cl_clipping_val=0.85 \
  --adv_lambda=0.5 \
  --adversarial="True" \
  --dynamic_adversarial="True"