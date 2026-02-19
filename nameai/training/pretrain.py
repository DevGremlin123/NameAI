"""Phase 1: Pre-train encoder on Wikipedia with Masked Language Modeling.

Randomly masks 15% of BPE tokens, trains encoder to predict them.
This teaches the encoder English before we fine-tune on name generation.

Usage:
    python -m nameai.training.pretrain [--wiki-text data/wikipedia/wiki_text.txt]
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nameai.model.nameformer import NameFormer
from nameai.tokenizer.bpe_tokenizer import BPETokenizer


class WikiMLMDataset(Dataset):
    """Wikipedia text dataset for masked language modeling."""

    def __init__(self, text_path: str, tokenizer: BPETokenizer, max_len: int = 512,
                 mask_prob: float = 0.15) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_id = tokenizer.mask_id

        # Load and tokenize all lines
        print(f"Loading and tokenizing {text_path}...")
        self.samples: list[list[int]] = []
        with open(text_path, encoding="utf-8") as f:
            for line in tqdm(f, desc="Tokenizing"):
                line = line.strip()
                if not line:
                    continue
                ids = tokenizer.encode(line, add_special=True)
                if len(ids) > 10:
                    self.samples.append(ids[:max_len])
        print(f"Loaded {len(self.samples):,} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = list(self.samples[idx])
        labels = list(ids)

        # Mask 15% of tokens (skip special tokens at positions 0 and -1)
        for i in range(1, len(ids) - 1):
            if random.random() < self.mask_prob:
                labels[i] = ids[i]  # Keep original as label
                if random.random() < 0.8:
                    ids[i] = self.mask_id  # 80%: replace with [MASK]
                elif random.random() < 0.5:
                    ids[i] = random.randint(5, self.tokenizer.vocab_size - 1)  # 10%: random token
                # 10%: keep original
            else:
                labels[i] = -100  # Don't compute loss on unmasked tokens

        labels[0] = -100   # BOS
        labels[-1] = -100  # EOS

        # Pad
        pad_len = self.max_len - len(ids)
        padding_mask = [False] * len(ids) + [True] * pad_len
        ids = ids + [0] * pad_len
        labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train encoder on Wikipedia (MLM)")
    parser.add_argument("--wiki-text", type=str, default="data/wikipedia/wiki_text.txt")
    parser.add_argument("--bpe-model", type=str, default="data/bpe_tokenizer.model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--save-path", type=str, default="checkpoints/pretrained.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop after N steps (for quick tests)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    # Load tokenizer
    tokenizer = BPETokenizer(args.bpe_model)
    print(f"BPE vocab size: {tokenizer.vocab_size}")

    # Build model
    from nameai.tokenizer.char_tokenizer import CharTokenizer
    char_tok = CharTokenizer()

    model = NameFormer(
        enc_vocab_size=tokenizer.vocab_size,
        dec_vocab_size=char_tok.vocab_size,
    ).to(device)

    params = model.count_parameters()
    print(f"Model: {params['total']:,} params (encoder: {params['encoder']:,})")

    # Dataset
    dataset = WikiMLMDataset(args.wiki_text, tokenizer, max_len=args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=args.lr,
                                   betas=(0.9, 0.98), weight_decay=0.01)

    # LR scheduler: warmup + cosine
    total_steps = len(loader) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = min(2000, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        import math
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    # Training
    print(f"\nPre-training on {device} ({'BF16' if use_bf16 else 'FP32'})")
    print(f"Steps per epoch: {len(loader):,}, Total steps: {total_steps:,}")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}\n")

    global_step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            if args.max_steps and global_step >= args.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model.forward_mlm(input_ids, padding_mask)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )

            loss.backward()
            nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 100 == 0:
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}", step=global_step)

        if args.max_steps and global_step >= args.max_steps:
            break

        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch + 1}: avg_loss={avg_loss:.4f}, elapsed={elapsed:.0f}s")

    # Save
    torch.save(model.state_dict(), args.save_path)
    total_time = time.time() - start_time
    print(f"\nPre-training done in {total_time:.0f}s. Saved to {args.save_path}")


if __name__ == "__main__":
    main()
