#!/usr/bin/env python3
import os
import json, time
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from evaluate import load
from transformers import AutoTokenizer
from transformers.models.led.modeling_led import LEDForConditionalGeneration
import pickle
import argparse
import re
from datasets import load_dataset
from common import DATASET_REGISTRY, load_hf_dataset


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if not hasattr(F, "_orig_softmax"):
    F._orig_softmax = F.softmax

@torch.no_grad()
def stable_softmax(input, dim=None, **kwargs):
    x = input.to(torch.float32)
    x = torch.nan_to_num(x, nan=float("-inf"), posinf=float("-inf"), neginf=float("-inf"))
    out = F._orig_softmax(x, dim=dim, **kwargs)
    return out.to(input.dtype)

F.softmax = stable_softmax

def generate_answer(batch, output_path):
    try:
        with torch.no_grad():
            inputs_dict = tokenizer(
                batch["article"],
                padding=True,
                max_length=8192,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = inputs_dict["input_ids"].to(device)
            attention_mask = inputs_dict["attention_mask"].to(device)

            global_attention_mask = torch.zeros_like(attention_mask)
            global_attention_mask[:, 0] = 1  # 讓第一個 token (通常是 BOS) 有 global attention

            predicted_ids = led.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                max_length=512,
                num_beams=4,
            )

            decoded = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            batch["predicted_abstract"] = decoded

            with open(output_path, "a", encoding="utf-8") as f:
                ids = batch.get("article_id", [None]*len(decoded)) 
                
                for idx, pred, ref in zip(ids, decoded, batch["abstract"]):
                    rec = {"article_id": idx, "prediction": pred, "reference": ref}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    except torch.cuda.OutOfMemoryError:
        print("⚠️ Skipping example due to OOM in generate_answer.")
        torch.cuda.empty_cache()
        with open(output_path, "a", encoding="utf-8") as f:
            for _ in batch["article"]:
                 f.write(json.dumps({"error": "OOM"}, ensure_ascii=False) + "\n")
        batch["predicted_abstract"] = ["[OOM ERROR]"] * len(batch["article"])

    except RuntimeError as e:
        print(f"⚠️ RuntimeError in generate_answer: {e}")
        torch.cuda.empty_cache()
        batch["predicted_abstract"] = ["[Runtime ERROR]"] * len(batch["article"])

    return batch

class LEDForConditionalGenerationWithGlobalMask(LEDForConditionalGeneration):
    def forward(self, *args, **kwargs):
        if "global_attention_mask" not in kwargs and "attention_mask" in kwargs:
            attn = kwargs["attention_mask"]
            kwargs["global_attention_mask"] = torch.zeros_like(attn)
            kwargs["global_attention_mask"][:, 0] = 1
        return super().forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        kwargs.pop("labels", None)
        return super().generate(*args, **kwargs)

def main():
    global tokenizer, led, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--split", default="test")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--model_dir", type=str, default="patrickvonplaten/led-large-16384-pubmed",
                    help="HuggingFace model name or local checkpoint path")
    
    parser.add_argument("--output_dir", default="./outputs/baselines/led",
                    help="Directory to save generations and metadata")
    
    args = parser.parse_args()

    pubmed_test = load_hf_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    pubmed_test   = pubmed_test.select(range(len(pubmed_test)))

    model_dir = args.model_dir
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    led = LEDForConditionalGenerationWithGlobalMask.from_pretrained(
        model_dir,
        use_cache=True
    )
    led.to(device)
    led.eval()

    led.config.num_beams = 4
    led.config.max_length = 512
    led.config.min_length = 100
    led.config.length_penalty = 2.0
    led.config.early_stopping = True
    led.config.no_repeat_ngram_size = 3

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = os.path.basename(args.model_dir)
    gen_path = out_dir / f"test_generations_{stamp}_{name}.jsonl"
    meta_path = out_dir / f"test_generations_{stamp}_{name}.meta.json"

    print(f"📂 Output will be streamed to: {gen_path}")
    
    with open(gen_path, 'w', encoding='utf-8') as f:
        pass

    result = pubmed_test.map(generate_answer, batched=True, batch_size=4, fn_kwargs={"output_path": gen_path})


    meta = {
        "time": stamp,
        "model_name_or_path": args.model_dir,
        "num_beams": getattr(led.config, "num_beams", None),
        "max_length": getattr(led.config, "max_length", None),
        "min_length": getattr(led.config, "min_length", None),
        "length_penalty": getattr(led.config, "length_penalty", None),
        "early_stopping": getattr(led.config, "early_stopping", None),
        "no_repeat_ngram_size": getattr(led.config, "no_repeat_ngram_size", None),
        "tokenizer": tokenizer.name_or_path if hasattr(tokenizer, "name_or_path") else "unknown",
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved generations to: {gen_path}")
    print(f"📝 Saved metadata to:    {meta_path}")

if __name__ == "__main__":
    main()
