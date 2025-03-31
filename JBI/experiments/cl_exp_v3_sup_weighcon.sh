#!/bin/bash

#echo commands to stdout
set -x

cd /path/to/project/dir

# activate multitagger environment
source /username/miniconda3/bin/activate virtual env

# Model - smallest training set; all features w/ feature verbalization
python train.py --experiment_name="cl_03_sup_weighcon" --label_split="animals" --contrastive_loss="WeighCon" --cl_beta=0.1

