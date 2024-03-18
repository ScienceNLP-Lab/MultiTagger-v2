# MultiTagger

This repository contains the code and links to the data, labels, and some pre-trained models.

## Getting Started

Download and install the environment using the requirements.txt file. Use the following command:
```bash
conda create -n <environment-name> --file requirements.txt
```

Before using the models, you will need to download and organize the data (https://uofi.box.com/v/multitagger-v2-data) and labels (https://uofi.box.com/v/multitagger-v2-labels). The data should go in a 'pubmed' directory, and the labels should go in a 'labels' directory. Both of these should be under the data directory within this repository.

## Training and Using the Models

Performing training and inference with the model is similar. A script (run_experiment_example.sh) is provided containing example commands for both fine-tuning and evaluation. As a general note, inference needs to be run on a validation subset before running on the test subset in order to use the optimized F1 thresholds. A full list of command arguments may be found in the train.py file. 

To avoid having to fine-tune, the best performing pretrained model (i.e., 20% unsampling rate with feature verbalization and unsupervised contrastive loss) is available here: https://uofi.box.com/v/multitagger-v2-model. Simply download it and add the filepath of the directory containing the model as the checkpoint argument. The model is available under "best_f1.mdl"; the model's predictions and performances on the validation set, test set, and MultiTagger-v1 restricted test set are also available there.
