#!/usr/bin/env python3
"""Alineación SFT de Hyperion con el dataset escrito para Soup.

El preentrenamiento (`neural.train`) aprende a continuar texto. Este script da el
siguiente paso: enseña el formato de diálogo del dataset alpaca y solo calcula la
pérdida sobre los tokens de la respuesta, que es lo que hace que el modelo
"sea" Hyperion y no un simple autocompletado.

No introduce datos externos: lee `soup/data/train.jsonl`, elaborado en este
repositorio. Requiere PyTorch 2.2+ (solo runtime de tensores, sin API de IA).

    python -m neural.sft --dataset soup/data/train.jsonl --steps 400
    python -m neural.chat --checkpoint checkpoints/sft-hyperion.pt
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import PRESETS
from .model import HyperionTransformer
from .tokenizer import ByteTokenizer

from .sft_format import IGNORE_INDEX, encode_dialogue, load_dataset

def collate(
    examples: list[dict[str, str]], max_length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [encode_dialogue(example, max_length) for example in examples]
    width = max(len(tokens) for tokens, _ in pairs)
    inputs, labels = [], []
    for tokens, tags in pairs:
        padding = width - len(tokens)
        inputs.append(tokens + [ByteTokenizer.PAD] * padding)
        labels.append(tags + [IGNORE_INDEX] * padding)
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alineación SFT local de Hyperion")
    parser.add_argument("--dataset", type=Path, default=Path("soup/data/train.jsonl"))
    parser.add_argument("--preset", choices=PRESETS, default="tiny")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint preentrenado")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/sft-hyperion.pt"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    args = parser.parse_args()
    args.max_length = min(args.max_length, PRESETS[args.preset].max_seq_len)
    return args


def learning_rate(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.learning_rate * (step + 1) / max(1, args.warmup_steps)
    progress = min(1.0, (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_learning_rate + cosine * (args.learning_rate - args.min_learning_rate)


def build_model(args: argparse.Namespace, device: torch.device) -> HyperionTransformer:
    config = PRESETS[args.preset]
    model = HyperionTransformer(config)
    if args.resume:
        if not args.resume.is_file():
            raise SystemExit(f"Checkpoint no encontrado: {args.resume}")
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        print(f"Reanudado desde {args.resume} (paso {payload.get('step', '?')})")
    return model.to(device)


def masked_loss(model: HyperionTransformer, inputs: torch.Tensor, labels: torch.Tensor):
    """Pérdida media solo sobre los tokens de respuesta (etiquetas != -100).

    El modelo promedia sobre todas las posiciones, así que aquí se recalcula la
    suma y se divide por el número real de tokens supervisados. Sin esto, las
    respuestas cortas pesarían menos que el prompt enmascarado.
    """
    logits, _ = model(inputs)
    supervised = int((labels != IGNORE_INDEX).sum())
    if supervised == 0:
        raise RuntimeError("Lote sin tokens de respuesta: revisa el dataset")
    total = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    loss = total / supervised
    if not torch.isfinite(loss):
        raise RuntimeError("Pérdida no finita: revisa el dataset o baja el learning rate")
    return loss


@torch.no_grad()
def evaluate(
    model: HyperionTransformer, rows: list[dict[str, str]], args: argparse.Namespace,
    device: torch.device, batch_size: int = 8,
) -> float:
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        inputs, labels = collate(chunk, args.max_length, device)
        total += float(masked_loss(model, inputs, labels)) * len(chunk)
        count += len(chunk)
    model.train()
    return total / max(1, count)


def save_checkpoint(
    path: Path, model: HyperionTransformer, optimizer: torch.optim.Optimizer,
    step: int, args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": asdict(model.config),
        "metadata": model.metadata(),
        "dataset": str(args.dataset),
    }, temporary)
    os.replace(temporary, path)
    print(f"  checkpoint -> {path}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        rows = load_dataset(args.dataset)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    random.Random(args.seed).shuffle(rows)
    split = int(len(rows) * args.val_split)
    validation, training = rows[:split], rows[split:] or rows
    if not validation:
        validation = training[:1]

    print(f"Dataset: {len(training)} ejemplos de entrenamiento, {len(validation)} de validación")
    model = build_model(args, device)
    print(f"Hyperion {args.preset}: {model.parameter_count():,} parámetros | dispositivo {device.type}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.learning_rate, betas=(0.9, 0.95),
    )

    model.train()
    started = time.monotonic()
    for step in range(args.steps):
        rate = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(args.gradient_accumulation):
            batch = [random.choice(training) for _ in range(args.batch_size)]
            inputs, labels = collate(batch, args.max_length, device)
            loss = masked_loss(model, inputs, labels) / args.gradient_accumulation
            loss.backward()
            accumulated += float(loss.detach()) * args.gradient_accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_value = accumulated / args.gradient_accumulation
        if step % 10 == 0 or step == args.steps - 1:
            seconds = time.monotonic() - started
            perplexity = math.exp(min(loss_value, 20.0))
            print(
                f"paso {step + 1:>5}/{args.steps} | pérdida {loss_value:.4f} | "
                f"ppl {perplexity:8.2f} | lr {rate:.2e} | {seconds:.1f}s"
            )
        if args.eval_every and (step + 1) % args.eval_every == 0:
            value = evaluate(model, validation, args, device)
            print(f"  validación: pérdida {value:.4f} | ppl {math.exp(min(value, 20.0)):.2f}")
        if args.save_every and (step + 1) % args.save_every == 0:
            save_checkpoint(args.output, model, optimizer, step + 1, args)

    save_checkpoint(args.output, model, optimizer, args.steps, args)
    final = evaluate(model, validation, args, device)
    print(f"Validación final: pérdida {final:.4f} | ppl {math.exp(min(final, 20.0)):.2f}")
    print(f"Listo. Prueba: python -m neural.chat --checkpoint {args.output}")


if __name__ == "__main__":
    main()
