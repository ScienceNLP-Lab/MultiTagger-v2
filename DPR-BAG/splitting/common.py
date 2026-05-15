"""Shared utilities for splitting strategies (FS / NS / SH)."""

import json
import os
import sys
import numpy as np
import pandas as pd
from datasets import load_dataset


LABELS = ["background", "objective", "methods", "results", "conclusions", "none"]

DATASET_REGISTRY = {
    "pubmedsum": ("ccdv/pubmed-summarization", "document"),
    "pmc_mad": ("sylvey/PMC-MAD", None),
}


def load_hf_dataset(dataset_name, split="test", cache_dir=None):
    """Load a dataset from HuggingFace via DATASET_REGISTRY lookup."""
    hf_name, hf_config = DATASET_REGISTRY[dataset_name]
    print(f"Loading dataset from Hugging Face: {dataset_name} ({split})")
    return load_dataset(hf_name, hf_config, cache_dir=cache_dir, split=split)


def count_lines(filename):
    """Count rows in an existing output file (for resumable execution)."""
    if not os.path.exists(filename):
        return 0
    with open(filename, 'rb') as f:
        return sum(1 for _ in f)


def clean_for_json(data_dict):
    """Convert NaN / numpy types to JSON-serializable values."""
    cleaned = {}
    for k, v in data_dict.items():
        if pd.isna(v): 
            cleaned[k] = "" 
        elif isinstance(v, (np.int64, np.int32)): 
            cleaned[k] = int(v)
        elif isinstance(v, (np.float64, np.float32)): 
            cleaned[k] = float(v)
        else:
            cleaned[k] = v
    return cleaned


def empty_facets():
    """Return a fresh dict of 6 empty facet strings."""
    return {label: "" for label in LABELS}


