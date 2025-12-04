#!/bin/bash

#echo commands to stdout
set -x

cd /path/to/project/dir

# activate multitagger environment
source /username/miniconda3/bin/activate your_virtual_env

# Model - smallest training set; all features w/ feature verbalization
python train.py --experiment_name="cl_v4_unsup_adnce_sup_herocon_25_cf_85" --contrastive_loss="HeroCon" --cl_alpha=0.01 --adnce_w1=1.0 --adnce_w2=1.0 --cl_beta=0.1 --cl_focusing=1 --cl_clipping=1 --cl_clipping_val=0.85

