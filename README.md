# MultiTagger

This repository contains the code and links to the data, labels, and some pre-trained models.

## Getting Started

Download and install the environment using the requirements.txt file. Use the following command:
```bash
conda create -n <environment-name> --file requirements.txt
```

## Training and Using the Models

Performing training and inference with the model is similar. A script (run_experiment_example.sh) is provided containing example commands for both fine-tuning and evaluation. As a general note, inference needs to be run on a validation subset before running on the test subset in order to use the optimized F1 thresholds. A full list of command arguments may be found in the train.py file. 
