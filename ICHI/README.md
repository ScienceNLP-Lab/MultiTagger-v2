# MultiTagger-ICHI

This repository provides code, data links, and pretrained models for the paper `Robust Biomedical Publication Types and Study Design Classification with Knowledge-Guided Perturbations`
## Getting Started

Create the conda environment using the provided environment.yml file:
```bash
conda env create -n <environment-name> -f environment.yml
```
Before running the models, download the required data and label files:
PubMed data (pubmed_data.csv):
https://uofi.box.com/s/8qwcy64la136xhjh9rs1yvmrg4f5fdxg
Human-annotated labels (split_stratified_data.csv):
https://uofi.box.com/s/pb0ne5rgp5wifvn7tvn77zobh1cc5d7t

After downloading, organize the files as follows:
Place pubmed_data.csv in the pubmed/ directory
Place split_stratified_data.csv in the labels_human/ directory

## Training and Inference

Training and inference follow the same execution pattern. Example bash scripts for model training are provided in the `experiments/` directory. A complete list of configurable arguments can be found in `train.py`.

### Using Pretrained Models
To avoid fine-tuning from scratch, we provide the best-performing pretrained model (adversarial training with entity masking):
Pretrained model checkpoint:
https://uofi.box.com/s/477o240qpbvaj2i00pp8otkpy3re81p9 

Download the checkpoint and specify the directory path using the --checkpoint argument when running inference or evaluation.


### Entity Masking Ratio Experiments

To reproduce the masking + adversarial ablation experiments, run:
`train_ablation.py`, which will load data from `data_ablation.py` to ensure that the data loader randomly selects instances for entity masking during training.