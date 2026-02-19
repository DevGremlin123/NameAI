"""Phase 2: Fine-tune the pre-trained NameFormer on (description → brand name) pairs.

Usage:
    python -m nameai.training.trainer [--pretrained checkpoints/pretrained.pt]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nameai.model.nameformer import NameFormer
from nameai.tokenizer.bpe_tokenizer import BPETokenizer
from nameai.tokenizer.char_tokenizer import CharTokenizer


class NameDataset(Dataset):
    """(description, name) pairs for fine-tuning."""

    def __init__(self, data_path: str, bpe_tok: BPETokenizer, char_tok: CharTokenizer,
                 max_src_len: int = 512, max_tgt_len: int = 64) -> None:
        self.bpe_tok = bpe_tok
        self.char_tok = char_tok
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.records = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        src_ids = self.bpe_tok.encode(r["description"], add_special=True)
        src_ids = self.bpe_tok.pad_sequence(src_ids, self.max_src_len)

        char_ids = self.char_tok.encode(r["name"], add_special=True)
        tgt_input = char_ids[:-1]
        tgt_label = char_ids[1:]

        tgt_input = self.char_tok.pad_sequence(tgt_input, self.max_tgt_len)
        tgt_label = self.char_tok.pad_sequence(tgt_label, self.max_tgt_len)

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "src_mask": torch.tensor([t == 0 for t in src_ids], dtype=torch.bool),
            "tgt_ids": torch.tensor(tgt_input, dtype=torch.long),
            "tgt_labels": torch.tensor(tgt_label, dtype=torch.long),
            "quality": torch.tensor(r.get("quality", 0.5), dtype=torch.float),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune NameFormer on name generation")
    parser.add_argument("--pretrained", type=str, default="checkpoints/pretrained.pt")
    parser.add_argument("--bpe-model", type=str, default="data/bpe_tokenizer.model")
    parser.add_argument("--train-data", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--val-data", type=str, default="data/processed/val.jsonl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    # Tokenizers
    bpe_tok = BPETokenizer(args.bpe_model)
    char_tok = CharTokenizer()

    # Model
    model = NameFormer(
        enc_vocab_size=bpe_tok.vocab_size,
        dec_vocab_size=char_tok.vocab_size,
    ).to(device)

    # Load pre-trained weights
    if Path(args.pretrained).exists():
        print(f"Loading pre-trained weights from {args.pretrained}")
        state = torch.load(args.pretrained, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
    else:
        print("WARNING: No pre-trained weights found. Training from scratch.")

    params = model.count_parameters()
    print(f"Model: {params['total']:,} params")

    # Data
    train_ds = NameDataset(args.train_data, bpe_tok, char_tok)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)

    val_loader = None
    if Path(args.val_data).exists():
        val_ds = NameDataset(args.val_data, bpe_tok, char_tok)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds):,} examples, {len(train_loader)} batches/epoch")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.98), weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        import math
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print(f"\nFine-tuning on {device} ({'BF16' if use_bf16 else 'FP32'})")
    print(f"Batch: {args.batch_size}, LR: {args.lr}, Epochs: {args.epochs}\n")

    start_time = time.time()
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            src_ids = batch["src_ids"].to(device)
            src_mask = batch["src_mask"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            tgt_labels = batch["tgt_labels"].to(device)
            quality = batch["quality"].to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(src_ids, tgt_ids, src_mask)
                # Per-token loss
                loss_flat = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    tgt_labels.view(-1),
                    ignore_index=0,
                    label_smoothing=0.1,
                    reduction="none",
                ).view(logits.size(0), -1)

                # Quality-weighted mean
                mask = tgt_labels != 0
                per_example = (loss_flat * mask).sum(1) / mask.sum(1).clamp(min=1)
                loss = (per_example * quality.pow(2.0)).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)

        # Validation
        val_msg = ""
        if val_loader:
            val_loss = evaluate(model, val_loader, device, use_bf16)
            val_msg = f", val_loss={val_loss:.4f}"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save(str(save_dir / "best.pt"))

        elapsed = time.time() - start_time
        print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}{val_msg}, elapsed={elapsed:.0f}s")

    model.save(str(save_dir / "final.pt"))
    print(f"\nFine-tuning done in {time.time() - start_time:.0f}s")


@torch.no_grad()
def evaluate(model, loader, device, use_bf16):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        src_ids = batch["src_ids"].to(device)
        src_mask = batch["src_mask"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)
        tgt_labels = batch["tgt_labels"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(src_ids, tgt_ids, src_mask)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   tgt_labels.view(-1), ignore_index=0)
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


if __name__ == "__main__":
    main()
