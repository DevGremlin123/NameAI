"""Phase 2: Fine-tune the pre-trained NameFormer on (description -> brand name) pairs.

Supports multi-GPU via torchrun:
    torchrun --nproc_per_node=4 -m nameai.training.trainer
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
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
    args = parser.parse_args()

    # DDP setup
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    bpe_tok = BPETokenizer(args.bpe_model)
    char_tok = CharTokenizer()

    model = NameFormer(
        enc_vocab_size=bpe_tok.vocab_size,
        dec_vocab_size=char_tok.vocab_size,
    ).to(device)

    # Load pre-trained weights
    if Path(args.pretrained).exists():
        if rank == 0:
            print(f"Loading pre-trained weights from {args.pretrained}")
        state = torch.load(args.pretrained, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
    elif rank == 0:
        print("WARNING: No pre-trained weights found. Training from scratch.")

    if distributed:
        model = DDP(model, device_ids=[rank])
    raw_model = model.module if distributed else model

    if rank == 0:
        params = raw_model.count_parameters()
        print(f"Model: {params['total']:,} params")
        print(f"GPUs: {world_size}, Batch/GPU: {args.batch_size}, Effective batch: {args.batch_size * world_size}")

    train_ds = NameDataset(args.train_data, bpe_tok, char_tok)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)

    val_loader = None
    if Path(args.val_data).exists():
        val_ds = NameDataset(args.val_data, bpe_tok, char_tok)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                sampler=val_sampler, num_workers=4, pin_memory=True)

    if rank == 0:
        print(f"Train: {len(train_ds):,} examples, {len(train_loader)} batches/epoch")

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.98), weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    if rank == 0:
        print(f"\nFine-tuning on {world_size}x GPU ({'BF16' if use_bf16 else 'FP32'})")
        print(f"Batch: {args.batch_size}, LR: {args.lr}, Epochs: {args.epochs}\n")

    start_time = time.time()
    global_step = 0

    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", disable=(rank != 0))

        for batch in pbar:
            src_ids = batch["src_ids"].to(device)
            src_mask = batch["src_mask"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            tgt_labels = batch["tgt_labels"].to(device)
            quality = batch["quality"].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = raw_model(src_ids, tgt_ids, src_mask)
                loss_flat = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    tgt_labels.view(-1),
                    ignore_index=0,
                    label_smoothing=0.1,
                    reduction="none",
                ).view(logits.size(0), -1)

                mask = tgt_labels != 0
                per_example = (loss_flat * mask).sum(1) / mask.sum(1).clamp(min=1)
                loss = (per_example * quality.pow(2.0)).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1
            if rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)

        val_msg = ""
        if val_loader:
            val_loss = evaluate(raw_model, val_loader, device, use_bf16)
            val_msg = f", val_loss={val_loss:.4f}"
            if rank == 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                raw_model.save(str(save_dir / "best.pt"))

        if rank == 0:
            print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}{val_msg}, elapsed={time.time() - start_time:.0f}s")

    if rank == 0:
        raw_model.save(str(save_dir / "final.pt"))
        print(f"\nFine-tuning done in {time.time() - start_time:.0f}s")

    if distributed:
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(model, loader, device, use_bf16):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        src_ids = batch["src_ids"].to(device)
        src_mask = batch["src_mask"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)
        tgt_labels = batch["tgt_labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(src_ids, tgt_ids, src_mask)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   tgt_labels.view(-1), ignore_index=0)
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


if __name__ == "__main__":
    main()
