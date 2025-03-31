#!/bin/bash

#echo commands to stdout
set -x

cd /path/to/project/dir

# activate multitagger environment
source /username/miniconda3/bin/activate your_virtual_env

# Model - baseline model using architecture and features from AMIA work
python train.py --experiment_name="00_amia_baseline" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --verbalize="original" --loss_function="bce" --lr_scheduler="linear" --optimizer="AdamW"

