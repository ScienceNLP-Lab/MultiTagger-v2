import argparse
import json
import numpy as np
import torch

from summac.model_summac import SummaCConv


def evaluate_factuality_batch(summaries, sources, device="cuda" if torch.cuda.is_available() else "cpu"):
   
    summac_model = SummaCConv(
        models=["vitc"], 
        bins='percentile', 
        granularity="sentence", 
        nli_labels="e", 
        device=device, 
        start_file="default", 
        agg="mean"
    )
    
    print("Calculating SummaC...")
    summac_results = summac_model.score(sources, summaries)
    summac_scores = summac_results["scores"]

    results = {
        "avg_summac": np.mean(summac_scores),
        "raw_summac": summac_scores,
    }
    
    return results

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

    results = evaluate_factuality_batch(preds, sources)

    print(f"\n[Evaluation Finished] {len(preds)} pairs evaluated.")
    print(f"Average SummaC Score: {results['avg_summac']:.4f}")

    with open(args.pairs.replace(".jsonl", "_summac_results.json"), "w") as f:
        json.dump({"ids": ids, "raw_summac": [float(s) for s in results['raw_summac']]}, f, indent=4)
    print(f"Detailed scores saved to {args.pairs.replace('.jsonl', '_summac_results.json')}")

if __name__ == "__main__":
    main()