#!/bin/bash

#echo commands to stdout
set -x

cd /ocean/projects/cis230089p/jmenke/multitagger_v3

# activate multitagger environment
source /jet/home/jmenke/miniconda3/bin/activate multitagger

# Model - smallest training set; all features w/ feature verbalization
python train.py --experiment_name="00_amia_baseline" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --verbalize="original" --loss_function="bce" --lr_scheduler="linear" --optimizer="AdamW"

