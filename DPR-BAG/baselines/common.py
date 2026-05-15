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

