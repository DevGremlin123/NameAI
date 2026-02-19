"""Fine-tuning Flan-T5-Small for brand name generation.

Uses HuggingFace Trainer for simplicity and reliability.

Usage:
    python -m nameai.training.trainer [--config configs/training.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from nameai.config import load_training_config


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataset(
    records: list[dict],
    tokenizer: T5Tokenizer,
    prompt_prefix: str,
    max_input_length: int,
    max_target_length: int,
) -> Dataset:
    """Convert JSONL records into a tokenized HuggingFace Dataset."""
    inputs = []
    targets = []
    for r in records:
        desc = r.get("description", "")
        name = r.get("name", "")
        if desc and name:
            inputs.append(prompt_prefix + desc)
            targets.append(name)

    ds = Dataset.from_dict({"input_text": inputs, "target_text": targets})

    def tokenize(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_input_length,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            batch["target_text"],
            max_length=max_target_length,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return ds.map(tokenize, batched=True, remove_columns=["input_text", "target_text"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Flan-T5 for name generation")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    args = parser.parse_args()

    cfg = load_training_config(args.config)
    tc = cfg.training
    dc = cfg.data
    cc = cfg.checkpointing
    lc = cfg.logging

    print(f"Loading base model: {cfg.base_model}")
    tokenizer = T5Tokenizer.from_pretrained(cfg.base_model)
    model = T5ForConditionalGeneration.from_pretrained(cfg.base_model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Load and tokenize data
    print(f"Loading training data from {dc.train_path}")
    train_records = load_jsonl(dc.train_path)
    train_ds = build_dataset(train_records, tokenizer, dc.prompt_prefix, dc.max_input_length, dc.max_target_length)
    print(f"Train examples: {len(train_ds)}")

    val_ds = None
    if Path(dc.val_path).exists():
        val_records = load_jsonl(dc.val_path)
        val_ds = build_dataset(val_records, tokenizer, dc.prompt_prefix, dc.max_input_length, dc.max_target_length)
        print(f"Val examples: {len(val_ds)}")

    # Data collator handles dynamic padding
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=cc.output_dir,
        num_train_epochs=tc.epochs,
        per_device_train_batch_size=tc.batch_size,
        per_device_eval_batch_size=tc.batch_size,
        learning_rate=tc.learning_rate,
        weight_decay=tc.weight_decay,
        warmup_ratio=tc.warmup_ratio,
        bf16=tc.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        max_grad_norm=tc.max_grad_norm,
        seed=tc.seed,
        # Checkpointing
        save_strategy=cc.save_strategy,
        save_steps=cc.save_steps,
        save_total_limit=cc.save_total_limit,
        load_best_model_at_end=cc.load_best_model_at_end,
        # Logging
        logging_steps=lc.logging_steps,
        eval_strategy=lc.eval_strategy if val_ds else "no",
        eval_steps=lc.eval_steps if val_ds else None,
        report_to="wandb" if lc.wandb_project else "none",
        run_name=lc.wandb_project,
        # Performance
        dataloader_num_workers=4,
        remove_unused_columns=True,
        metric_for_best_model="eval_loss" if val_ds else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    # Save best model
    best_path = Path(cc.output_dir) / "best"
    model.save_pretrained(best_path)
    tokenizer.save_pretrained(best_path)
    print(f"Best model saved to {best_path}")


if __name__ == "__main__":
    main()
