"""Phase 1: Pre-train encoder on Wikipedia with Masked Language Modeling.

Supports multi-GPU via torchrun:
    torchrun --nproc_per_node=4 -m nameai.training.pretrain
"""

from __future__ import annotations

import argparse
import math
import os
import random
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


class WikiMLMDataset(Dataset):
    """Wikipedia text dataset for masked language modeling."""

    def __init__(self, text_path: str, tokenizer: BPETokenizer, max_len: int = 512,
                 mask_prob: float = 0.15) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_id = tokenizer.mask_id

        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print(f"Loading and tokenizing {text_path}...")
        self.samples: list[list[int]] = []
        with open(text_path, encoding="utf-8") as f:
            for line in tqdm(f, desc="Tokenizing", disable=(rank != 0)):
                line = line.strip()
                if not line:
                    continue
                ids = tokenizer.encode(line, add_special=True)
                if len(ids) > 10:
                    self.samples.append(ids[:max_len])
        if rank == 0:
            print(f"Loaded {len(self.samples):,} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = list(self.samples[idx])
        labels = list(ids)

        for i in range(1, len(ids) - 1):
            if random.random() < self.mask_prob:
                labels[i] = ids[i]
                if random.random() < 0.8:
                    ids[i] = self.mask_id
                elif random.random() < 0.5:
                    ids[i] = random.randint(5, self.tokenizer.vocab_size - 1)
            else:
                labels[i] = -100

        labels[0] = -100
        labels[-1] = -100

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
    parser.add_argument("--max-steps", type=int, default=None)
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

    tokenizer = BPETokenizer(args.bpe_model)
    from nameai.tokenizer.char_tokenizer import CharTokenizer
    char_tok = CharTokenizer()

    model = NameFormer(
        enc_vocab_size=tokenizer.vocab_size,
        dec_vocab_size=char_tok.vocab_size,
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[rank])
    raw_model = model.module if distributed else model

    if rank == 0:
        params = raw_model.count_parameters()
        print(f"Model: {params['total']:,} params (encoder: {params['encoder']:,})")
        print(f"GPUs: {world_size}, Batch/GPU: {args.batch_size}, Effective batch: {args.batch_size * world_size}")

    dataset = WikiMLMDataset(args.wiki_text, tokenizer, max_len=args.max_len)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(sampler is None),
                        sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = torch.optim.AdamW(raw_model.encoder.parameters(), lr=args.lr,
                                   betas=(0.9, 0.98), weight_decay=0.01)

    total_steps = len(loader) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = min(2000, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(f"\nPre-training on {world_size}x GPU ({'BF16' if use_bf16 else 'FP32'})")
        print(f"Steps/epoch: {len(loader):,}, Total steps: {total_steps:,}\n")

    global_step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", disable=(rank != 0))

        for batch in pbar:
            if args.max_steps and global_step >= args.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = (model.module if distributed else model).forward_mlm(input_ids, padding_mask) if not distributed else model.module.forward_mlm(input_ids, padding_mask)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

            loss.backward()
            nn.utils.clip_grad_norm_(raw_model.encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1
            if rank == 0 and global_step % 100 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}", step=global_step)

        if args.max_steps and global_step >= args.max_steps:
            break

        if rank == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"  Epoch {epoch + 1}: avg_loss={avg_loss:.4f}, elapsed={time.time() - start_time:.0f}s")

    if rank == 0:
        torch.save(raw_model.state_dict(), args.save_path)
        print(f"\nPre-training done in {time.time() - start_time:.0f}s. Saved to {args.save_path}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
