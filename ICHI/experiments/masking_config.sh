#!/bin/bash

# activate multitagger environment
source /username/miniconda3/bin/activate your_virtual_env

python train.py \
  --experiment_name="masked_training" \
  --contrastive_loss="HeroCon" \
  --adnce_w1=1.0 \
  --adnce_w2=1.0 \
  --cl_alpha=0.01 \
  --cl_beta=0.1 \
  --cl_focusing=1 \
  --cl_clipping=1 \
  --cl_clipping_val=0.85 \
  --use_entity_type_marker="True" \
  --augment_with_masked="True"