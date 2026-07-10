# DPR-BAG: Divide-Prompt-Refine: a Training-Free, Structure-Aware Framework for Biomedical Abstract Generation

DPR-BAG is a pipeline for generating abstracts for biomedical articles that lack them. Given a full-text article, the system first splits it into rhetorical sections (background, objective, methods, results, conclusions), summarizes each section using an LLM, and then refines the summaries into a coherent abstract.

---

## Repository Structure

```
dpr-bag/
├── splitting/          # Step 1 — split full-text into rhetorical sections
├── dpr-bag/            # Step 2 — generate abstracts from sections
├── baselines/          # LED and LongT5 baselines
├── evaluation/         # Evaluation scripts (factuality, extractiveness, ROUGE, BERTScore, etc.)
└── envs/               # Conda environment files
```

---

## Environment Setup

Multiple conda environments are used to isolate incompatible dependencies. Create any environment with:

```bash
conda env create -f envs/<env_file>.yml
conda activate <env_name>
```

| Environment file | Used for |
|---|---|
| `envs/gptoss20b.yml` | DPR-BAG pipeline (`dpr-bag/pipeline.py`) and most evaluation scripts (ROUGE, BERTScore, MiniCheck, DiscoScore, UMLS, extractiveness) |
| `envs/led_fp16.yml` | Baseline fine-tuning and inference (`baselines/led_*.py`, `baselines/longt5_inference.py`) |
| `envs/LLM-SSC.yml` | First Sentence (FS) splitting via the LLM-SSC classifier (`splitting/fs.py`) |
| `envs/env_alignscore.yml` | AlignScore evaluation (`evaluation/eval_alignscore.py`) |
| `envs/env_summac.yml` | SummaC evaluation (`evaluation/eval_summac.py`) |

AlignScore and SummaC are isolated because they pin incompatible `transformers` versions relative to the main pipeline.

The DPR-BAG pipeline additionally requires:

- **Ollama** ([install instructions](https://ollama.com/)) with the target model pulled (e.g. `ollama pull llama3.2:3b`).
- For UMLS-guided prompting (`di_trumls`): the scispaCy model `en_core_sci_md` and the UMLS entity linker. Install these inside the `gptoss20b` environment per [scispaCy's documentation](https://github.com/allenai/scispacy).

---

## Running Experiments

All `.sh` scripts are written for SLURM but can be adapted to other HPC schedulers or run locally. Before running, set the following environment variables or edit the scripts directly:

```bash
export PROJECT_ROOT=/path/to/dpr-bag
export CONDA_BASE=$HOME/miniconda3
export CONDA_ENV=dpr_bag      # adjust per script
```

If using SLURM: Replace `YOUR_ALLOCATION` in each `.sh` file with your SLURM account name before submitting with sbatch.

If using another scheduler: Replace the #SBATCH directives at the top of each .sh file with the equivalent directives for your system.

If running locally: The core commands inside each .sh file can be run directly in your terminal after setting the environment variables above.

---

## Pipeline Overview

### Step 1: Splitting

Three strategies are available for splitting a full-text article into labeled rhetorical sections:

| Strategy | Script | Description |
|---|---|---|
| **FS** (First Sentence) | `splitting/fs.py` | Uses the LLM-SSC classifier [1] to predict the rhetorical role of each paragraph from its first sentence |
| **NS** (Naive Splitting) | `splitting/ns.py` | Distributes paragraphs evenly into six equal-length facets without a model |
| **SH** (Section Header) | `splitting/sh.py` | Uses a SentenceTransformer + MLP classifier [2] to label sections by their header text |

All strategies output a JSONL file with six rhetorical facet columns: `background`, `objective`, `methods`, `results`, `conclusions`, `none`.

**Run splitting:**
```bash
cd splitting/
# See classification.sh for all options
bash classification.sh
```

**FS** requires the LLM-SSC checkpoint from Lan et al. (2024) [1]. **SH** requires two model checkpoints from Lin et al. (2025) [2]: a fine-tuned SentenceTransformer and a classifier `.pth` file.

---

### Step 2: Abstract Generation

`dpr-bag/pipeline.py` takes the split output and generates an abstract in two stages:

1. Each non-empty rhetorical section is independently summarized by an LLM.
2. The section summaries are concatenated into a draft and refined into a final abstract.

The pipeline uses [Ollama](https://ollama.com/) to serve local LLMs (tested with `llama3.2:3b`).

**Available prompt strategies** (`--paragraph_prompt_version`):

| Key | Name | Description |
|---|---|---|
| `bc` | Basic Concise | Minimal instruction with concise section-specific guidance |
| `di` | Detailed Instruction | Verbose instruction with detailed section-specific guidelines |
| `si` | Structural Instruction | Structured prompt with explicit role definition and output format |
| `bc_ns` | BC for Naive Split | Basic concise prompt without section-type information |
| `di_trumls` | DI + TR-UMLS | DI prompt augmented with top-N UMLS entities extracted via TextRank |
| `si_cot` | SI + Chain-of-Thought | Two-turn SI prompt with explicit element extraction before summarization |

**Run the pipeline:**
```bash
cd dpr-bag/

bash testing.sh
```

For UMLS-guided prompting (`di_trumls`), `en_core_sci_md` and the scispacy UMLS linker are required.

---

## Baselines

Two long-document baselines are provided.

### LED (Longformer Encoder-Decoder)

```bash
cd baselines/

# Run inference with a pretrained LED checkpoint
bash led_inference.sh

# Fine-tune LED on PMC-MAD, then run inference
bash led_ft.sh

```

### LongT5

```bash
cd baselines/

bash longt5_inference.sh
```

Both scripts accept `--dataset pubmedsum` or `--dataset pmc_mad` and write streamed predictions to a timestamped `.jsonl` file.

---

## Evaluation

All evaluation scripts read a JSONL `--pairs` file containing predictions and (where needed) a separate `--source_file` with full-text articles. Scripts write raw per-sample scores alongside aggregate results.

| Script | Metric | Notes |
|---|---|---|
| `eval_rouge.py` | ROUGE-1/2/L | Reference-based |
| `eval_bertscore.py` | BERTScore (SciBERT) | Reference-based |
| `eval_umls.py` | UMLS Entity Recall | Reference-based; requires scispacy |
| `eval_extract.py` | Coverage / Density / Novelty | Source-based extractiveness |
| `eval_alignscore.py` | AlignScore | Source-based factuality |
| `eval_summac.py` | SummaC | Source-based factuality |
| `eval_miniCheck.py` | MiniCheck (flan-t5-large) | Source-based factuality |
| `eval_disco.py` | DiscoScore | Coherence |

**Example:**
```bash
cd evaluation/
python eval_rouge.py \
    --pairs ../outputs/pmc_mad/dpr_bag_bc.jsonl \
    --ref abstract \
    --pred generated_abstract
```

---

## Datasets

| Name | HuggingFace ID | Notes |
|---|---|---|
| PubMedSum | `ccdv/pubmed-summarization` | Public benchmark |
| PMC-MAD | `TODO/pmc-mad` | Update with your dataset path |

---

## References

[1] Mengfei Lan, Lecheng Zheng, Shufan Ming, and Halil Kilicoglu. 2024. Multi-label sequential sentence classification via large language model. In *Findings of the Association for Computational Linguistics: EMNLP 2024*, pages 16086–16104, Miami, Florida, USA. Association for Computational Linguistics.

[2] Sylvey Lin, Joseph Menke, Arthur Holt, Halil Kilicoglu, and Neil Smalheiser. 2025. Section header normalization in biomedical articles using transformers. In *AMIA Annual Symposium Proceedings*. Poster P116.

---

## Citation
```
@inproceedings{lin-etal-2026-divide,
    title = "Divide-Prompt-Refine: a Training-Free, Structure-Aware Framework for Biomedical Abstract Generation",
    author = "Lin, Sylvey  and
      Menke, Joseph  and
      Ming, Shufan  and
      Nam, Dongin  and
      Smalheiser, Neil  and
      Kilicoglu, Halil",
    editor = "Demner-Fushman, Dina  and
      Ananiadou, Sophia  and
      Roberts, Kirk  and
      Tsujii, Junichi",
    booktitle = "{B}io{NLP} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.bionlp-1.64/",
    doi = "10.18653/v1/2026.bionlp-1.64",
    pages = "770--790",
    ISBN = "979-8-89176-434-7",
    abstract = "Biomedical abstracts play a critical role in downstream NLP applications, such as information retrieval, biocuration, and biomedical knowledge discovery. However, a non-trivial number of biomedical articles do not have abstracts, diminishing the utility of these articles for downstream tasks. We propose DPR-BAG (Divide, Prompt, and Refine for Biomedical Abstract Generation), a training-free, zero-shot framework that generates coherent and factually grounded abstracts for biomedical articles with full text but no abstract. DPR-BAG decomposes full-text documents into structured rhetorical facets following the Background-Objective-Methods-Results-Conclusions (BOMRC) schema, performs parallel LLM-based summarization for each facet, and applies a final refinement stage to restore global discourse coherence. On PMC-MAD, a distribution-aligned dataset of 46,309 biomedical articles, DPR-BAG improves abstractive novelty over strong extractive and fine-tuned baselines, while maintaining factual consistency. Our ablation study reveals a counterintuitive finding: increasing prompt complexity or explicitly injecting entity-level guidance can degrade factual alignment, highlighting the importance of controlled prompting strategies. These findings underscore the potential of training-free, structure-aware frameworks for scalable biomedical abstract generation in low-resource settings. Our data and code are available at https://huggingface.co/datasets/pmc-mad/PMC-MAD and https://github.com/ScienceNLP-Lab/MultiTagger-v2/tree/main/DPR-BAG."
}
```
