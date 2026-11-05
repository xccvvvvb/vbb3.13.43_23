"""Formato de diálogo del ajuste fino supervisado (sin dependencias externas).

Vive separado de `neural.sft` para poder probar el formato y validar el dataset
sin PyTorch instalado. Los tokens de la pregunta se enmascaran con
`IGNORE_INDEX`, de modo que la pérdida solo se calcula sobre la respuesta.
"""

from __future__ import annotations

import json
from pathlib import Path

from .tokenizer import ByteTokenizer

IGNORE_INDEX = -100
SYSTEM = (
    "Eres Hyperion, un motor neuro-simbólico local, auditable y sin APIs externas. "
    "Respondes en español, con precisión, sin inventar datos y admitiendo tus límites."
)

TURN_SYSTEM = "<|sistema|>"
TURN_USER = "<|usuario|>"
TURN_ASSISTANT = "<|asistente|>"
TURN_END = "<|fin|>"


def build_prompt(question: str) -> str:
    return f"{TURN_SYSTEM}{SYSTEM}{TURN_USER}{question.strip()}{TURN_ASSISTANT}"


def encode_dialogue(example: dict[str, str], max_length: int) -> tuple[list[int], list[int]]:
    """Devuelve (tokens, etiquetas) con la pregunta enmascarada."""
    tokenizer = ByteTokenizer()
    question = (example.get("instruction") or "").strip()
    if (example.get("input") or "").strip():
        question += "\n" + example["input"].strip()
    prompt_ids = tokenizer.encode(build_prompt(question))
    answer_ids = tokenizer.encode((example.get("output") or "").strip() + TURN_END)
    tokens = (prompt_ids + answer_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + answer_ids)[:max_length]
    return tokens, labels


def load_dataset(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido en {path}:{number}: {exc}") from exc
            if not row.get("instruction") or not row.get("output"):
                raise ValueError(f"Fila {number} sin 'instruction' u 'output'")
            rows.append(row)
    if not rows:
        raise ValueError(f"Dataset vacío: {path}")
    return rows
