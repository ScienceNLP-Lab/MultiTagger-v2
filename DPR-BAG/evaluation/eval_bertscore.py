#!/usr/bin/env python3
import argparse
import torch
import numpy as np
import json # 導入 json 庫用於讀取 jsonl 檔案
import sys
# 導入 AutoTokenizer，用於精確截斷
from transformers import AutoTokenizer
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="outputs/pairs.jsonl",
                    help="Path to the JSONL file containing prediction and reference pairs")

    parser.add_argument("--ref", default="ref", help="JSONL field name for reference abstract")
    parser.add_argument("--id", default="article_id", help="JSONL field name for article ID")
    parser.add_argument("--pred", default="pred", help="JSONL field name for predicted abstract")


    parser.add_argument("--limit", type=int, default=None,
                    help="Optional cap on records to evaluate, useful for debugging (default: all)")
    parser.add_argument("--model_type", default="../led/models/scibert",
                        help="HF model name or local path (e.g., SciBERT)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for BERTScore computation")
    parser.add_argument("--num_layers", type=int, default=9,
                        help="Number of transformer layers to use for BERTScore")
    parser.add_argument("--no_idf", action="store_true",
                        help="Disable IDF weighting")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to run on: auto, cpu, or cuda")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:0" if args.device == "cuda" else "cpu"


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
                    
                    ids.append(str(record[args.id]))
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

    # =============================================================
    MAX_LENGTH = 510 
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_type, use_fast=False)
        
        def truncate_text(text_list):
            tokens = tokenizer(
                text_list, 
                max_length=MAX_LENGTH, 
                truncation=True, 
                padding=False,
                add_special_tokens=False 
            )['input_ids']
            
            return [tokenizer.decode(ids, skip_special_tokens=False) for ids in tokens]

        preds = truncate_text(preds)
        refs = truncate_text(refs)
        
        print(f"INFO: Successfully truncated all sequences to max token length {MAX_LENGTH} using {args.model_type} tokenizer.")
        
    except Exception as e:
        sys.stderr.write(f"WARNING: Manual truncation failed. Proceeding with original long texts. Error: {e}\n")
    # =============================================================
    

    
    from evaluate import load as load_metric
    bertscore = load_metric("bertscore")

    res = bertscore.compute(
        predictions=preds,
        references=refs,
        model_type=args.model_type,
        lang="en",
        idf=not args.no_idf,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        device=device,
    )

    p = float(np.mean(res["precision"]))
    r = float(np.mean(res["recall"]))
    f1 = float(np.mean(res["f1"]))

    print("== BERTScore (SciBERT) ==")
    print(f"P={p:.4f}  R={r:.4f}  F1={f1:.4f}")

    with open(args.pairs.replace(".jsonl", "_bertscore_results.json"), "w") as f:
        json.dump({"ids": ids, 
                "raw_bertscore_f1": [float(s) for s in res['f1']],
                "raw_bertscore_recall": [float(s) for s in res['recall']],
                "raw_bertscore_precision": [float(s) for s in res['precision']]
                }, 
                f, indent=4)
    print(f"Detailed scores saved to {args.pairs.replace('.jsonl', '_bertscore_results.json')}")


if __name__ == "__main__":
    main()