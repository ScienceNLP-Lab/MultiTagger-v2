import argparse
import json
import numpy as np

from tqdm import tqdm

import spacy
from minicheck.minicheck import MiniCheck
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="outputs/pairs.jsonl",
                    help="Path to the JSONL file containing predictions")
    parser.add_argument("--source_file", required=True,
                        help="Path to the JSONL file containing source full texts, keyed by article_id")
    parser.add_argument("--source_field", default="article",
                        help="Field name for source text in --source_file")
    parser.add_argument("--id", default="article_id",
                        help="JSONL field name for article ID (in both --pairs and --source_file)")
    parser.add_argument("--pred", default="pred",
                        help="JSONL field name for predicted abstract")

    
    args = parser.parse_args()

    print(f"Loading source texts from {args.source_file}...")
    id_to_source = {}
    with open(args.source_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            id_to_source[record[args.id]] = record[args.source_field]
    print(f"Loaded {len(id_to_source)} source texts.")

    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])

    preds = []
    sources = []
    ids = []

    with open(args.pairs, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            article_id = record[args.id]
            p = str(record[args.pred]).replace("\n", " ").strip()
            s = str(id_to_source[article_id]).replace("\n", " ").strip()
            
            if p and s:
                ids.append(article_id)
                preds.append(p)
                sources.append(s)

    # ---------------------------------------------------------
    #  MiniCheck 
    # ---------------------------------------------------------
    
    print("Initializing flan-t5-large...")
    scorer = MiniCheck(model_name='flan-t5-large')

    all_abstract_scores = []
    # all_raw_probs = []


    for abstract, document in tqdm(zip(preds, sources), total=len(preds), desc="Evaluating MiniCheck"):

        doc_spacy = nlp(abstract)
        sentences = [sent.text.strip() for sent in doc_spacy.sents]

        if not sentences:
            all_abstract_scores.append(0.0)
            continue

        
        docs_batch = [document] * len(sentences)
        pred_label, raw_prob, _, _ = scorer.score(docs=docs_batch, claims=sentences)
        abstract_score = float(np.mean(pred_label)) if pred_label else 0.0
        
        all_abstract_scores.append(abstract_score)


    final_dataset_score = np.mean(all_abstract_scores)
    print(f"\nEvaluation Complete!")
    print(f"Total pairs evaluated: {len(preds)}")
    print(f"Average MiniCheck Factuality Score: {final_dataset_score:.4f}")

    with open(args.pairs.replace(".jsonl", "_minicheck_results.json"), "w") as f:
        json.dump({"ids": ids, "raw_minicheck": all_abstract_scores}, f, indent=4)
    print(f"Detailed scores saved to {args.pairs.replace('.jsonl', '_minicheck_results.json')}")



if __name__ == "__main__":
    main()