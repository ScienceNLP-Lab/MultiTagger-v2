#!/usr/bin/env python3
import argparse
import json
from evaluate import load as load_metric

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="outputs/pairs.jsonl",
                    help="Path to the JSONL file containing prediction and reference pairs")
    parser.add_argument("--ref", default="ref", help="JSONL field name for reference abstract")
    parser.add_argument("--id", default="article_id", help="JSONL field name for article ID")
    parser.add_argument("--pred", default="pred", help="JSONL field name for predicted abstract")
    parser.add_argument("--limit", type=int, default=None,
                    help="Optional cap on records to evaluate, useful for debugging (default: all)")
    
    args = parser.parse_args()

    
    preds = []
    refs = []
    ids = []
    count = 0

    try:
        with open(args.pairs, "r", encoding="utf-8") as f:
            for line in f:
                if args.limit is not None and count >= args.limit:
                    break 
                
                try:
                    record = json.loads(line)
                    preds.append(str(record[args.pred]).rstrip("\n"))
                    refs.append(str(record[args.ref]).rstrip("\n"))
                    ids.append(record[args.id])
                    count += 1
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Skipping malformed line/record in {args.pairs}. Error: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: {args.pairs} not found. Please check the file path.")
        return

    if not preds:
        print("Error: No valid prediction data was loaded.")
        return
        
    assert len(preds) == len(refs), "Error: Data extraction from pairs.jsonl failed to match lengths."
    
    print(f"[Loaded] Successfully loaded {len(preds)} pairs from {args.pairs}")
 
    rouge = load_metric("rouge")
    types = ["rouge1","rouge2","rougeL"]

    res = rouge.compute(
        predictions=preds,
        references=refs,
        rouge_types=types,
        use_aggregator=False,
        use_stemmer=True
    )

    print("== ROUGE ==")
    raw_scores = {}
    for t in types:
        scores = res[t]
    
        try:
            f1_list = [float(s.fmeasure) for s in scores]
            p_list = [float(s.precision) for s in scores]
            r_list = [float(s.recall) for s in scores]
        except AttributeError:
            f1_list = [float(s) for s in scores]
            p_list = []
            r_list = []
        
        avg_f1 = sum(f1_list) / len(f1_list)
        if p_list:
            avg_p = sum(p_list) / len(p_list)
            avg_r = sum(r_list) / len(r_list)
            print(f"{t.upper():<10} F1={avg_f1:.4f} (P={avg_p:.4f}, R={avg_r:.4f})")
        else:
            print(f"{t.upper():<10} {avg_f1:.4f}")
        
        raw_scores[f"raw_{t}_f1"] = f1_list
        if p_list:
            raw_scores[f"raw_{t}_precision"] = p_list
            raw_scores[f"raw_{t}_recall"] = r_list
    
    out_path = args.pairs.replace(".jsonl", "_rouge_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "ids": [str(i) for i in ids],
            **raw_scores,
        }, f, indent=4)
    print(f"Detailed scores saved to {out_path}")

if __name__ == "__main__":
    main()
