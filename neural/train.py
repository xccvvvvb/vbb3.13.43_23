#!/usr/bin/env python3
"""Train Hyperion from scratch. Requires Python 3.11+, PyTorch 2.2+. Run: torchrun --nproc_per_node=8 -m neural.train --preset 7b --data data/corpus.txt"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .config import PRESETS
from .model import HyperionTransformer
from .tokenizer import ByteTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preentrenamiento local de Hyperion")
    parser.add_argument("--data", type=Path, required=True, help="Corpus UTF-8 local")
    parser.add_argument("--preset", choices=PRESETS, default="tiny")
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


class CorpusSampler:
    def __init__(self, path: Path, sequence_length: int, seed: int) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Corpus not found: {path}")
        self.data = path.read_bytes()
        self.length = sequence_length
        self.random = random.Random(seed)
        if len(self.data) < sequence_length + 2:
            raise ValueError(f"Corpus requires at least {sequence_length + 2} UTF-8 bytes")

    def batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        rows = []
        for _ in range(batch_size):
            start = self.random.randrange(0, len(self.data) - self.length - 1)
            rows.append(list(self.data[start:start + self.length + 1]))
        values = torch.tensor(rows, dtype=torch.long, device=device) + ByteTokenizer.OFFSET
        return values[:, :-1], values[:, 1:]


def learning_rate(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.learning_rate * (step + 1) / max(1, args.warmup_steps)
    progress = min(1.0, (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps))
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_learning_rate + coefficient * (args.learning_rate - args.min_learning_rate)


def save_checkpoint(path: Path, model: HyperionTransformer, optimizer: torch.optim.Optimizer, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
        "config": asdict(model.config), "metadata": model.metadata(),
    }, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, local_rank, world_size = dist.get_rank(), int(os.environ["LOCAL_RANK"]), dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)
        torch.set_float32_matmul_precision("high")

    config = PRESETS[args.preset]
    model = HyperionTransformer(config).to(device)
    raw_model = model
    if distributed:
        model = DistributedDataParallel(model, device_ids=[device.index], gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=device.type == "cuda")
    sampler = CorpusSampler(args.data, config.max_seq_len, args.seed + rank * 100_003)
    use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bfloat16 else contextlib.nullcontext()

    if rank == 0:
        print(f"Hyperion {args.preset}: {raw_model.parameter_count():,} parameters | {world_size} process(es) | {device}")
    model.train()
    started = time.monotonic()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for micro_step in range(args.gradient_accumulation):
            inputs, targets = sampler.batch(args.batch_size, device)
            sync_context = model.no_sync() if distributed and micro_step < args.gradient_accumulation - 1 else contextlib.nullcontext()
            with sync_context, autocast:
                _, loss = model(inputs, targets)
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            accumulated += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        if rank == 0 and (step % 10 == 0 or step == args.steps - 1):
            elapsed = max(time.monotonic() - started, 1e-6)
            tokens = (step + 1) * args.gradient_accumulation * args.batch_size * config.max_seq_len * world_size
            print(f"step={step + 1} loss={accumulated / args.gradient_accumulation:.4f} lr={lr:.3e} tokens/s={tokens / elapsed:,.0f}")
        if rank == 0 and ((step + 1) % args.save_every == 0 or step == args.steps - 1):
            save_checkpoint(args.output / f"hyperion-{args.preset}-step-{step + 1}.pt", raw_model, optimizer, step + 1)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
