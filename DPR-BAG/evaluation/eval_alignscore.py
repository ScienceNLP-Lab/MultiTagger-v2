import argparse
import json
import numpy as np
import torch

from alignscore import AlignScore

def evaluate_factuality_batch(summaries, sources, batch_size, device="cuda" if torch.cuda.is_available() else "cpu"):
    
    align_model = AlignScore(model='roberta-base', batch_size=batch_size, device=device, ckpt_path="AlignScore/AlignScore-base.ckpt")

    print(f"Starting evaluation on {len(summaries)} pairs using {device}...")


    print("Calculating AlignScore...")
    align_scores = align_model.score(sources, summaries)

    results = {
        "avg_alignscore": np.mean(align_scores),
        "raw_alignscore": align_scores
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
    parser.add_argument("--ckpt_path", default="AlignScore/AlignScore-base.ckpt",
                        help="Path to AlignScore checkpoint")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for AlignScore computation")
    args = parser.parse_args()

    # Load source full texts into id -> source_text mapping
    print(f"Loading source texts from {args.source_file}...")
    id_to_source = {}
    with open(args.source_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            id_to_source[record[args.id]] = record[args.source_field]
    print(f"Loaded {len(id_to_source)} source texts.")

    print(f"Loading predictions from {args.pairs}...")
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

    print(f"Aligned {len(preds)} prediction-source pairs.")
    results = evaluate_factuality_batch(preds, sources, batch_size=args.batch_size)

    print(f"\n[Evaluation Finished] {len(preds)} pairs evaluated.")
    print(f"Average AlignScore: {results['avg_alignscore']:.4f}")

    with open(args.pairs.replace(".jsonl", "_alignscore_results.json"), "w") as f:
        json.dump({"ids": ids, "raw_alignscore": [float(s) for s in results['raw_alignscore']]}, f, indent=4)
    print(f"Detailed scores saved to {args.pairs.replace('.jsonl', '_alignscore_results.json')}")

if __name__ == "__main__":
    main()