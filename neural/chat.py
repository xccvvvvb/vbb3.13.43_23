#!/usr/bin/env python3
"""Offline CLI inference for a trained Hyperion checkpoint. Run: python -m neural.chat --checkpoint checkpoints/model.pt"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import ModelConfig
from .model import HyperionTransformer
from .sft_format import TURN_END, TURN_SYSTEM, TURN_USER, build_prompt
from .tokenizer import ByteTokenizer


def load_model(path: Path, device: torch.device) -> HyperionTransformer:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint or "model" not in checkpoint:
        raise ValueError("Invalid Hyperion checkpoint")
    config = ModelConfig(**checkpoint["config"])
    model = HyperionTransformer(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat local con Hyperion neuronal")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ByteTokenizer()
    model = load_model(args.checkpoint, device)
    print(f"Hyperion neuronal activo en {device}. Escribe /salir para terminar.")
    while True:
        try:
            prompt = input("\nTú> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.lower() in {"/salir", "/exit", "/quit"}:
            break
        if not prompt:
            continue
        formatted = build_prompt(prompt)
        encoded = tokenizer.encode(formatted)
        tokens = torch.tensor([encoded], dtype=torch.long, device=device)
        generated = model.generate(tokens, args.max_tokens, args.temperature, args.top_p)
        answer_tokens = generated[0, len(encoded):].tolist()
        answer = tokenizer.decode(answer_tokens)
        for stop in (TURN_END, TURN_USER, TURN_SYSTEM):
            answer = answer.split(stop)[0]
        print(f"Hyperion> {answer.strip()}")


if __name__ == "__main__":
    main()
