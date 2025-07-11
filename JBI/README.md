# MultiTagger-JBI

This repository contains code and links to the data, labels, and some pre-trained models relating to our work available through medRxiv, "Enhancing automated indexing of publication types and study designs in biomedical literature using full-text features".

## Getting Started

Download and install the environment using the requirements.txt file. Use the following command:
```bash
conda create -n <environment-name> --file requirements.txt
```

Before using the models, you will need to download and organize the data (pubmed_data.csv and pmc_data.csv) and labels (split_stratified_data.csv) [available through box](https://uofi.box.com/s/lgvnrqqukab4b4izu7wr7dc4ood0z8w2). The data should go in a 'pubmed' directory, and the labels should go in a 'labels' directory. Both of these should be under the data directory within this repository.

## Training and Using the Models

Performing training and inference with the model is similar. A script (run_experiment_example.sh) is provided containing example commands for both fine-tuning and evaluation. A full list of command arguments may be found in the train.py file. 

To avoid having to fine-tune, the best performing pretrained model (i.e., asymmetric loss with label smoothing and WeighCon contrastive loss) is available [here](https://uofi.box.com/s/uspvg8s3hwzkp3zcd89jxrpj7ift4jqp). Simply download it and add the filepath of the directory containing the model as the checkpoint argument. The model is available under "best_model.pth"; the model's predictions and performances on the validation and test sets are also available there.
