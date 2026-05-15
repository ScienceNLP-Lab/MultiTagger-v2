import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
import os
import json, time
from pathlib import Path
import pickle
#!/usr/bin/env python3
from datasets import load_dataset, load_metric
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import torch
from transformers import DataCollatorForSeq2Seq

from evaluate import load
from transformers import LEDTokenizer, LEDForConditionalGeneration
import shutil

from transformers import Seq2SeqTrainer

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
from torch.cuda.amp import autocast

from contextlib import nullcontext

from transformers import Seq2SeqTrainer
import torch
from torch.cuda.amp import autocast

from transformers import Seq2SeqTrainer
import torch

from transformers.trainer_callback import TrainerCallback
import argparse



class IgnoreGradException(Exception):
    pass

class IgnoreGradCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if hasattr(state, "log_history") and isinstance(state.log_history, list):
            if state.log_history and isinstance(state.log_history[-1], dict):
                state.log_history[-1]["skipped_batch"] = True

from transformers.models.led.modeling_led import LEDForConditionalGeneration




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
        
class SafeTrainer(Seq2SeqTrainer):
    def training_step(self, model, inputs):
        print("🔥 Actually running a training step...")
        model.train()
        inputs = self._prepare_inputs(inputs)

        try:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError("NaN or Inf loss detected.")

            if self.args.n_gpu > 1 and hasattr(loss, "mean"):
                loss = loss.mean()

            self.accelerator.backward(loss)

            return loss.detach()

        except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as e:
            print(f"⚠️ Skipping batch due to error: {e}")
            torch.cuda.empty_cache()

            dummy = torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
            self.accelerator.backward(dummy)

            return dummy.detach()


import torch
import torch.nn.functional as F

# 只做一次 monkey-patch，並把原始 softmax 綁成預設參數，避免後續被覆蓋
if not hasattr(F, "_orig_softmax"):
    F._orig_softmax = F.softmax

@torch.no_grad()
def stable_softmax(input, dim=None, **kwargs):
    # 先轉成 float32 做 softmax，再轉回原 dtype
    x = input.to(torch.float32)
    # 將 NaN/Inf 處理掉，避免傳到 softmax
    x = torch.nan_to_num(x, nan=float("-inf"), posinf=float("-inf"), neginf=float("-inf"))
    out = F._orig_softmax(x, dim=dim, **kwargs)   # 用備份，不會遞迴
    return out.to(input.dtype)

F.softmax = stable_softmax

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    
    global tokenizer, led 

    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="patrickvonplaten/led-large-16384-pubmed",
                        help="HuggingFace model name or local path to base LED model.")
    parser.add_argument("--output_dir", default="./checkpoints/led_pubmed_finetune",
                        help="Directory to save fine-tuning checkpoints.")
    parser.add_argument("--cache_dir", default=None,
                        help="Optional HuggingFace datasets cache directory.")
    args = parser.parse_args()

    
    print("Loading PMC-MAD from HuggingFace...")
    ds = load_dataset("sylvey/PMC-MAD", cache_dir=args.cache_dir)
    pubmed_train = ds["train"]
    pubmed_val = ds["validation"]
    

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    encoder_max_length = 8192
    decoder_max_length = 512

    def process_data_to_model_inputs(example):
        inputs = tokenizer(
            example["article"],
            padding=False,
            truncation=True,
            max_length=encoder_max_length,
        )
        outputs = tokenizer(
            example["abstract"],
            padding=False,
            truncation=True,
            max_length=decoder_max_length,
        )
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        labels = outputs.input_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "global_attention_mask": [1] + [0] * (len(input_ids) - 1),
            "labels": [-100 if token == tokenizer.pad_token_id else token for token in labels]
        }

    pubmed_train = pubmed_train.map(
        process_data_to_model_inputs,
        batched=False,
        remove_columns=["article", "abstract"],
        load_from_cache_file=False,
        desc="🧪 Mapping training data"
    )
    pubmed_val = pubmed_val.map(
        process_data_to_model_inputs,
        batched=False,
        remove_columns=["article", "abstract"],
        load_from_cache_file=False,
        desc="🧪 Mapping val data"
    )


    print("✅ pubmed_train size:", len(pubmed_train))
    print("✅ pubmed_val size:", len(pubmed_val))
    
    print(pubmed_train[0])

    pubmed_train = pubmed_train.flatten_indices()
    pubmed_val = pubmed_val.flatten_indices()
    

    pubmed_train = pubmed_train.select(range(len(pubmed_train)))
    pubmed_val   = pubmed_val.select(range(len(pubmed_val)))


    eval_dataset_small = pubmed_val.select(range(min(len(pubmed_val), 100)))

    print(len(pubmed_train))

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        fp16=True,
        output_dir=args.output_dir,
        logging_steps=250,
        eval_steps=500,                 # ← 讓訓練中真的會 eval
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,     # ← 重要
        metric_for_best_model="eval_rouge2_fmeasure",
        greater_is_better=True,
        max_grad_norm=0.5,          # originally 1.0
        warmup_ratio=0.06,               # ← 用比例，避免不同資料量失衡
        gradient_accumulation_steps=8,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        group_by_length=False,
        dataloader_drop_last=True,

        learning_rate=2e-5, # added for longer prompts

        num_train_epochs=1,

        generation_max_length=128,
        generation_num_beams=1,
    )



    rouge = load("rouge")
    def compute_metrics(eval_preds):

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.array(labels)  
        labels = labels.copy()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
        labels[labels == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

        try:
            out = rouge.compute(
                predictions=pred_str,
                references=label_str,
                rouge_types=["rouge2"],
                use_aggregator=False,
            )["rouge2"]

            if isinstance(out, list) and len(out) > 0 and hasattr(out[0], "precision"):
                p = float(np.mean([s.precision for s in out]))
                r = float(np.mean([s.recall    for s in out]))
                f = float(np.mean([s.fmeasure  for s in out]))
                return {
                    "rouge2_precision": round(p, 4),
                    "rouge2_recall":    round(r, 4),
                    "rouge2_fmeasure":  round(f, 4),
                }

            if isinstance(out, (list, tuple, np.ndarray)):
                f = float(np.mean(out))
                return {
                    "rouge2_precision": round(f, 4),
                    "rouge2_recall":    round(f, 4),
                    "rouge2_fmeasure":  round(f, 4),
                }

            if isinstance(out, (float, np.floating)):
                f = float(out)
                return {
                    "rouge2_precision": round(f, 4),
                    "rouge2_recall":    round(f, 4),
                    "rouge2_fmeasure":  round(f, 4),
                }
        except Exception:
            pass

        agg = rouge.compute(
            predictions=pred_str,
            references=label_str,
            rouge_types=["rouge2"],
            use_aggregator=True,
        )["rouge2"]

        if isinstance(agg, (float, np.floating)):
            f = float(agg)
            return {
                "rouge2_precision": round(f, 4),
                "rouge2_recall":    round(f, 4),
                "rouge2_fmeasure":  round(f, 4),
            }
        try:
            p = float(agg.mid.precision)
            r = float(agg.mid.recall)
            f = float(agg.mid.fmeasure)
        except Exception:
            f = float(getattr(agg, "fmeasure", getattr(agg, "f1", 0.0)))
            p = float(getattr(agg, "precision", f))
            r = float(getattr(agg, "recall", f))

        return {
            "rouge2_precision": round(p, 4),
            "rouge2_recall":    round(r, 4),
            "rouge2_fmeasure":  round(f, 4),
        }


    led = LEDForConditionalGenerationWithGlobalMask.from_pretrained(
        args.base_model,
        use_cache=False,
        # torch_dtype=torch.float16,
    )


    led.gradient_checkpointing_enable()
    led.led.decoder.gradient_checkpointing = False
    led.config.num_beams = 4
    led.config.max_length = 512
    led.config.min_length = 100
    led.config.length_penalty = 2.0
    led.config.early_stopping = True
    led.config.no_repeat_ngram_size = 3

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=led, label_pad_token_id=-100)

    # instantiate trainer
    trainer = SafeTrainer(
        model=led,
        args=training_args,
        train_dataset=pubmed_train,
        eval_dataset=eval_dataset_small,
        tokenizer=tokenizer,
        callbacks=[IgnoreGradCallback()],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print("Trainer train_dataset size:", len(trainer.train_dataset)) 

    dl = trainer.get_train_dataloader()
    num_batches = len(dl)  
    updates_per_epoch = (num_batches + training_args.gradient_accumulation_steps - 1) // training_args.gradient_accumulation_steps
    total_steps = updates_per_epoch * int(training_args.num_train_epochs)
    print(f"num_batches={num_batches}, updates_per_epoch={updates_per_epoch}, total_steps={total_steps}")
    print(f"current_global_step={trainer.state.global_step}")

    checkpoint_dir = training_args.output_dir

    def extract_step(path):
        name = os.path.basename(path)
        try:
            return int(name.replace("checkpoint-", ""))
        except:
            return -1

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        checkpoints = [
            os.path.join(training_args.output_dir, d)
            for d in os.listdir(training_args.output_dir)
            if d.startswith("checkpoint")
        ]
        if checkpoints:
            def extract_step(path):
                name = os.path.basename(path)
                try:
                    return int(name.replace("checkpoint-", ""))
                except:
                    return -1
            last_checkpoint = max(checkpoints, key=extract_step)
            print(f"⏪ Resuming from checkpoint: {last_checkpoint}")
        else:
            print("🚀 Starting training from scratch.")
    else:
        print("🚀 No checkpoint directory found. Starting from scratch.")


    try:
        if last_checkpoint is not None:
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            trainer.train()
    except IgnoreGradException:
        print("⚠️ Skipped a batch but training continued.")
        


    checkpoints = [
        os.path.join(training_args.output_dir, d)
        for d in os.listdir(training_args.output_dir)
        if d.startswith("checkpoint")
    ]

    checkpoints_sorted = sorted(checkpoints, key=extract_step)

    to_delete = checkpoints_sorted[:-2]

    for path in to_delete:
        print(f"🧹 Removing old checkpoint: {path}")
        shutil.rmtree(path)

    

if __name__ == "__main__":
    main()