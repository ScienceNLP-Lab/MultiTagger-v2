#!/bin/bash

# activate multitagger environment
source /username/miniconda3/bin/activate your_virtual_env

checkpoint=""

python train.py \
  --train_val_test="test" \
  --checkpoint="$checkpoint" \
  --contrastive_loss=HeroCon \
  --cl_alpha=0.01 \
  --adnce_w2=1.0 \
  --adnce_w1=1.0 \
  --cl_focusing=1 \
  --cl_clipping_val=0.85 \
  --cl_beta=0.1 \
  --cl_clipping=1 \
  --use_entity_type_marker="True" \
  --adv_lambda=0.5 \
  --adversarial="True" \
  --dynamic_adversarial="True" \

for perturb in 0.3_cui 0.5_cui 1.0_cui 0.3_semtype 0.5_semtype 1.0_semtype cui_eda semtype_eda; do
  echo "[$(date)] Running perturbation test: ${perturb}"

  python train.py \
    --train_val_test="test" \
    --checkpoint="$checkpoint" \
    --contrastive_loss=HeroCon \
    --cl_alpha=0.01 \
    --adnce_w2=1.0 \
    --adnce_w1=1.0 \
    --cl_focusing=1 \
    --cl_clipping_val=0.85 \
    --cl_beta=0.1 \
    --cl_clipping=1 \
    --perturbation_test_type="$perturb" \
    --use_entity_type_marker="True" \
    --adv_lambda=0.5 \
    --adversarial="True" \
    --dynamic_adversarial="True"
done
