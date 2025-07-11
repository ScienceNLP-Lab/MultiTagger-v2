# MultiTagger V2
This repository contains code and links to data, labels, and links to pre-trained models for work extending MultiTagger-v1, which can be found here: ["Fifty Ways to Tag your Pubtypes: Multi-Tagger, a Set of Probabilistic Publication Type and Study Design Taggers to Support Biomedical Indexing and Evidence-Based Medicine"](https://www.medrxiv.org/content/10.1101/2021.07.13.21260468v1).

The AMIA directory relates to work published in AMIA's 2024 annual symposium, which forumulates the task as multi-label and fine-tunes a PubMedBERT encoder for all classes. The JBI directory relates to work available through medRxiv that explores the use of full-text features for this task, a discussion of label noise (and the use of noise-aware training strategies), as well as a more exhaustive evaluation of BERT-based encoders and contrastive learning frameworks.

## Citation
If you find this work helpful, please consider citing our papers.

```
@inproceedings{menke2024tptindexing,
  title={Publication Type Tagging using Transformer Models and Multi-Label Classification},
  author={Menke, Joe D. and Kilicoglu, Halil and Smalheiser, Neil R.},
  booktitle={AMIA Annual Symposium Proceedings},
  publisher={American Medical Informatics Association},
  year={2024},
  doi={10.1101/2025.03.06.25323516}
}

@article{menke2025enhancing,
  title={Enhancing automated indexing of publication types and study designs in biomedical literature using full-text features},
  author={Menke, Joe D and Ming, Shufan and Radhakrishna, Shruthan and Kilicoglu, Halil and Smalheiser, Neil R},
  journal={medRxiv},
  pages={2025--04},
  year={2025},
  publisher={Cold Spring Harbor Laboratory Press}
}
```
