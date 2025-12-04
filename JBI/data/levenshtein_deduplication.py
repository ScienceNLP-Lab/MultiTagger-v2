#!/usr/bin/env python3
"""
Find near-duplicate articles (title + abstract) using Levenshtein ratio >= 85%.

Outputs: matches_lev90.csv with columns:
pmid_1, pmid_2, ratio, text_1, text_2, len_1, len_2
"""
"""
Hybrid MinHash + Levenshtein duplicate detection
------------------------------------------------

Step 1: Use MinHash LSH to find likely duplicate title+abstract pairs.
Step 2: Verify those candidate pairs using Levenshtein ratio (>85%).

"""

import pandas as pd
from datasketch import MinHash, MinHashLSH
from rapidfuzz import fuzz
import re

from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.utils import indexable, _safe_indexing
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _validate_shuffle_split
from itertools import chain

from tqdm import tqdm

# --------------------- CONFIG ---------------------
INPUT_DATA_CSV = "pubmed_data.csv"          # Must have columns: pmid, title, abstract
INPUT_LABEL_CSV = "split_stratified_data.csv"
OUTPUT_CSV = "split_stratified_data_dup_corrected.csv"

MIN_LEN = 30                        # Skip very short title+abstracts (usually missing abstracts)
LSH_THRESHOLD = 0.7                 # Candidate similarity cutoff (Jaccard)
LEV_RATIO_THRESHOLD = 85.0          # Final Levenshtein % threshold
NUM_PERM = 64                       # MinHash permutations (64–128 typical)
# --------------------------------------------------


def normalize_text(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def text_to_minhash(tokens: list[str], num_perm=NUM_PERM) -> MinHash:
    """Convert list of tokens to MinHash signature."""
    mh = MinHash(num_perm=num_perm)
    for token in set(tokens):  # unique tokens only
        mh.update(token.encode("utf8"))
    return mh


def correct_potential_duplicates():
    print("Loading data...")
    df = pd.read_csv(INPUT_DATA_CSV, dtype=str).fillna("")
    df['pmid'] = df['pmid'].astype(str)
    preprint_ids = df[df['pub_type'].str.contains('Preprint')]['pmid'].to_list()
    print(preprint_ids)
    
    if not {"pmid", "title", "abstract", "pub_date"}.issubset(df.columns):
        raise ValueError("Input CSV must contain 'pmid', 'title', and 'abstract' columns.")

    # Combine title + abstract
    df["text"] = (df["title"].str.strip() + " " + df["abstract"].str.strip() + " " + df["pub_date"].str.strip()).str.strip()
    df["text_len"] = df["text"].str.len()
    df = df[df["text_len"] >= MIN_LEN].reset_index(drop=True)

    print(f"Records kept (len ≥ {MIN_LEN}): {len(df)}")

    # Build MinHash index 
    print("Building MinHash signatures...")
    minhashes = {}
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)

    for i, text in tqdm(enumerate(df["text"]), total=len(df), desc="MinHash"):
        tokens = normalize_text(text)
        mh = text_to_minhash(tokens)
        minhashes[i] = mh
        lsh.insert(str(i), mh)

    # Query candidates 
    print("Querying LSH for candidate pairs...")
    candidate_pairs = []
    for i, mh in tqdm(minhashes.items(), total=len(minhashes), desc="LSH query"):
        similar = lsh.query(mh)
        for j in similar:
            j = int(j)
            if j > i:
                candidate_pairs.append((i, j))

    print(f"Candidate pairs found: {len(candidate_pairs)}")

    # Verify with Levenshtein ratio
    print("Verifying candidates with Levenshtein ratio ≥", LEV_RATIO_THRESHOLD)
    matches = []
    for i, j in tqdm(candidate_pairs, total=len(candidate_pairs), desc="Levenshtein verify"):
        text_a = df.at[i, "text"]
        text_b = df.at[j, "text"]

        # Quick check for exact match
        if text_a == text_b:
            ratio = 100.0
        else:
            ratio = fuzz.ratio(text_a, text_b, score_cutoff=LEV_RATIO_THRESHOLD)

        if ratio >= LEV_RATIO_THRESHOLD:
            matches.append({
                "pmid_1": df.at[i, "pmid"],
                "pmid_2": df.at[j, "pmid"],
                "ratio": ratio,
                "title_abstract_1": text_a,
                "title_abstract_2": text_b,
                "len_1": df.at[i, "text_len"],
                "len_2": df.at[j, "text_len"]
            })

    # Add cluster number
    if matches:
        print(f"Verified duplicates: {len(matches)}")
        cluster_map = dict()
        cluster_ids = []
        c = 0
        out_df = pd.DataFrame(matches)
        for index, row in out_df.iterrows():
            if row['title_abstract_1'] in cluster_map:
                cluster_ids.append(cluster_map[row['title_abstract_1']])
            elif row['title_abstract_2'] in cluster_map:
                cluster_ids.append(cluster_map[row['title_abstract_2']])
            else:
                cluster_ids.append(c)
                cluster_map[row['title_abstract_1']] = c
                cluster_map[row['title_abstract_2']] = c
                c += 1

        out_df['cluster_id'] = cluster_ids

        labels = pd.read_csv(INPUT_LABEL_CSV, low_memory=True)
        labels['ids'] = labels['ids'].astype(str)
        duplicated_label_splits = pd.Series(labels['split'].values, index=labels['ids']).to_dict()

        out_df['pmid_1_split'] = out_df['pmid_1'].map(duplicated_label_splits)
        out_df['pmid_2_split'] = out_df['pmid_2'].map(duplicated_label_splits)

        # Determine majority split for each cluster
        cluster_majority_split = (
            pd.concat([
                out_df[['cluster_id', 'pmid_1_split']].rename(columns={'pmid_1_split': 'split'}),
                out_df[['cluster_id', 'pmid_2_split']].rename(columns={'pmid_2_split': 'split'})
            ])
            .dropna(subset=['split'])
            .groupby('cluster_id')['split']
            .agg(lambda x: x.value_counts().idxmax())
            .to_dict()
        )

        # Map cluster majority splits back to each PMID
        pmid_to_cluster = {}
        for _, row in out_df.iterrows():
            pmid_to_cluster[row['pmid_1']] = row['cluster_id']
            pmid_to_cluster[row['pmid_2']] = row['cluster_id']

        # Update label splits based on cluster-majority split

        labels['cluster_id'] = labels['ids'].map(pmid_to_cluster)
        labels['majority_split'] = labels['cluster_id'].map(cluster_majority_split)
        labels['split'] = labels['majority_split'].combine_first(labels['split'])

        # Clean up unnecessary columns
        labels = labels.drop(columns=['cluster_id', 'majority_split'])

        # Remove preprint articles
        deduplicated_labels = labels[~labels['ids'].isin(preprint_ids)]

        # Save results
        deduplicated_labels.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved verified matches to: {OUTPUT_CSV}")

    else:
        print("No verified duplicates found.")
    
    return deduplicated_labels


if __name__ == "__main__":
    dedup_labs = correct_potential_duplicates()
