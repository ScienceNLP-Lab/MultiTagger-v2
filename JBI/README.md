# MultiTagger-JBI

This repository contains code and links to the data, labels, and some pre-trained models relating to our work available through medRxiv, "Enhancing automated indexing of publication types and study designs in biomedical literature using full-text features".

## Getting Started

Download and install the environment using the requirements.txt file. Use the following command:
```bash
conda env create -n <environment-name> -f environment.yml
```

Before using the models, you will need to download and organize the data (pubmed_data.csv and pmc_data.csv) and labels (split_stratified_data.csv) [available through box](https://uofi.box.com/s/lgvnrqqukab4b4izu7wr7dc4ood0z8w2). The data should go in 'pubmed' and 'pmc' directories. Labels should go in a 'labels' directory. Both of these should be under the data directory within this repository.

## Training and Using the Models

Performing training and inference with the model is similar. Two scripts (experiments) are provided for the experiments using the previous architecture described in the AMIA work as well as the best performing experiment within this work. A full list of command arguments may be found in the train.py file. 

To avoid having to fine-tune, the best performing pretrained model (i.e., asymmetric loss with label smoothing and WeighCon contrastive loss) is available [here](https://uofi.box.com/s/uspvg8s3hwzkp3zcd89jxrpj7ift4jqp). Simply download it and add the filepath of the directory containing the model as the checkpoint argument. The model is available under "best_model.pth"; the model's predictions and performances on the validation and test sets are also available there.

## Error Analysis

The manual_review_sample_named_list.xlsx file contains 333 articles, NLM's labels, and their predictions using the best performing model. It contains the follwing information:
* preds: predictions from the best model
* gold_labels: gold labels from PubMed
* updated_gold_labels: updated labels that I created.
* notes: short notes
* TP|FP|FN: original counts
* updated_(TP|FP|FN): updated counts
