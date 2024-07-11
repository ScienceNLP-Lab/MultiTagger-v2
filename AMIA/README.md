# MultiTagger-AMIA

This repository contains the code and links to the data, labels, and some pre-trained models.

## Getting Started

Download and install the environment using the requirements.txt file. Use the following command:
```bash
conda create -n <environment-name> --file requirements.txt
```

Before using the models, you will need to download and organize the data (https://uofi.box.com/v/multitagger-v2-data) and labels (https://uofi.box.com/v/multitagger-v2-labels). The data should go in a 'pubmed' directory, and the labels should go in a 'labels' directory. Both of these should be under the data directory within this repository.

## Training and Using the Models

Performing training and inference with the model is similar. A script (run_experiment_example.sh) is provided containing example commands for both fine-tuning and evaluation. As a general note, inference needs to be run on a validation subset before running on the test subset in order to use the optimized F1 thresholds. A full list of command arguments may be found in the train.py file. 

To avoid having to fine-tune, the best performing pretrained model (i.e., 80% unsampling rate with feature verbalization and unsupervised contrastive loss) is available here: https://uofi.box.com/v/multitagger-v2-model. Simply download it and add the filepath of the directory containing the model as the checkpoint argument. The model is available under "best_f1.mdl"; the model's predictions and performances on the validation set, test set, and MultiTagger-v1 restricted test set are also available there.

## Performance with the Best Performing Model
The best performing model featured 80% undersampling to achieve a more balanced dataset, as well as verbalizing extracted features for added context. Performance on the test set is reported in the table below.

| Publication Type | Precision | Recall | F1 | AUC |
| :---: | :---: | :---: | :---: | :---: |
| Autobiography | 0.461 | 0.392 | 0.423 | 0.992 |
| Bibliography | 0.868 | 0.492 | 0.628 | 0.992 |
| Biography | 0.751 | 0.741 | 0.746 | 0.990 |
| Case-Control Study | 0.565 | 0.706 | 0.628 | 0.972 |
| Case Report | 0.800 | 0.752 | 0.775 | 0.975 |
| Clinical Conference | 0.141 | 0.213 | 0.170 | 0.881 |
| Clinical Study as topic | 0.636 | 0.662 | 0.648 | 0.955 |
| Clinical Study | 0.792 | 0.705 | 0.746 | 0.950 |
| Clinical Trial | 0.626 | 0.700 | 0.661 | 0.952 |
| Clinical Trial Protocol | 0.727 | 0.852 | 0.785 | 0.999 |
| Cohort Study | 0.533 | 0.602 | 0.565 | 0.947 |
| Comment | 0.604 | 0.732 | 0.662 | 0.975 |
| Congress | 0.777 | 0.675 | 0.722 | 0.986 |
| Consensus Development Conference | 0.663 | 0.619 | 0.640 | 0.994 |
| Cross-Cultural Comparison | 0.668 | 0.574 | 0.617 | 0.984 |
| Cross-Over Study | 0.867 | 0.784 | 0.824 | 0.988 |
| Cross-Sectional Study | 0.867 | 0.667 | 0.754 | 0.970 |
| Double-Blind Method | 0.823 | 0.725 | 0.771 | 0.979 |
| Editorial | 0.515 | 0.654 | 0.576 | 0.970 |
| Evaluation Study | 0.403 | 0.464 | 0.431 | 0.911 |
| Evaluation Study as topic | 0.366 | 0.454 | 0.405 | 0.932 |
| Expression of Concern | 0.985 | 0.886 | 0.933 | 1.000 |
| Feasibility Study | 0.764 | 0.722 | 0.743 | 0.969 |
| Focus Groups | 0.857 | 0.766 | 0.809 | 0.991 |
| Follow-Up Study | 0.509 | 0.614 | 0.557 | 0.917 |
| Genome Wide Association Study | 0.834 | 0.809 | 0.821 | 0.997 |
| Historical Article | 0.794 | 0.733 | 0.762 | 0.979 |
| Human Experimentation | 0.651 | 0.612 | 0.631 | 0.989 |
| Interview | 0.853 | 0.736 | 0.790 | 0.992 |
| Interviews as topic | 0.545 | 0.643 | 0.590 | 0.977 |
| Lecture | 0.856 | 0.508 | 0.638 | 0.972 |
| Legal Case | 0.674 | 0.634 | 0.653 | 0.994 |
| Letter | 0.682 | 0.711 | 0.696 | 0.972 |
| Longitudinal Study | 0.726 | 0.610 | 0.663 | 0.957 |
| Matched-Pair Analysis | 0.339 | 0.415 | 0.373 | 0.964 |
| Meta-Analysis | 0.864 | 0.845 | 0.854 | 0.990 |
| Multicenter Study | 0.657 | 0.666 | 0.661 | 0.947 |
| News | 0.695 | 0.775 | 0.733 | 0.988 |
| Newspaper Article | 0.891 | 0.952 | 0.921 | 1.000 |
| Personal Narrative | 0.550 | 0.405 | 0.467 | 0.990 |
| Portrait | 0.470 | 0.559 | 0.511 | 0.983 |
| Practice Guideline | 0.709 | 0.749 | 0.729 | 0.995 |
| Predictive Value of tests | 0.516 | 0.664 | 0.581 | 0.945 |
| Prospective Study | 0.792 | 0.703 | 0.745 | 0.953 |
| Published Erratum | 0.971 | 0.912 | 0.940 | 0.996 |
| Random Allocation | 0.534 | 0.496 | 0.515 | 0.952 |
| Randomized Controlled Trial (RCT) | 0.640 | 0.688 | 0.664 | 0.971 |
| Reproducibility of results | 0.588 | 0.632 | 0.609 | 0.940 |
| Retraction of publication | 0.945 | 0.883 | 0.913 | 0.990 |
| Retrospective Study | 0.775 | 0.675 | 0.721 | 0.957 |
| Review | 0.714 | 0.702 | 0.708 | 0.960 |
| Scientific Integrity Review | 0.933 | 0.651 | 0.767 | 0.959 |
| Systematic Review | 0.940 | 0.903 | 0.921 | 0.999 |
| Systematic Review as topic | 0.829 | 0.853 | 0.841 | 0.997 |
| Twin Study | 0.880 | 0.875 | 0.877 | 0.996 |
| Validation Study | 0.634 | 0.669 | 0.651 | 0.974 |
| Veterinary Clinical Trial | 0.578 | 0.664 | 0.618 | 0.991 |
| Veterinary Observational Study | 0.785 | 0.718 | 0.750 | 0.995 |
| Veterinary RCT | 0.480 | 0.647 | 0.551 | 0.994 |
