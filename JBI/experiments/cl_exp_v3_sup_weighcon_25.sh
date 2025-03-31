#!/bin/bash

#echo commands to stdout
set -x

cd /ocean/projects/cis230089p/jmenke/multitagger_v3

# activate multitagger environment
source /jet/home/jmenke/miniconda3/bin/activate multitagger

# Model - smallest training set; all features w/ feature verbalization
python train.py --experiment_name="cl_03_sup_weighcon_25" --label_split="animals" --contrastive_loss="WeighCon" --cl_beta=0.1

