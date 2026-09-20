#!/usr/bin/env python3
"""Build one local UTF-8 training corpus. Run: python -m neural.prepare_corpus ./documents --output data/corpus.txt"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ALLOWED = {".txt", ".md", ".rst", ".py", ".js", ".ts", ".json", ".jsonl"}


def extract(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="strict").replace("\x00", "")
    if path.suffix == ".json":
        data = json.loads(text)
        return json.dumps(data, ensure_ascii=False, indent=2)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Preparar corpus local de Hyperion")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/corpus.txt"))
    parser.add_argument("--max-file-mb", type=int, default=128)
    args = parser.parse_args()
    files: list[Path] = []
    for source in args.inputs:
        files.extend(source.rglob("*") if source.is_dir() else [source])
    selected = sorted({path.resolve() for path in files if path.is_file() and path.suffix.lower() in ALLOWED})
    if not selected:
        raise SystemExit("No compatible UTF-8 files found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as destination:
        for path in selected:
            if path.stat().st_size > args.max_file_mb * 1024 * 1024:
                print(f"skip oversized: {path}")
                continue
            try:
                content = extract(path).strip()
            except (UnicodeError, OSError, json.JSONDecodeError) as exc:
                print(f"skip unreadable: {path} ({exc})")
                continue
            if content:
                destination.write(f"<|documento|>\n{content}\n<|fin|>\n")
                written += 1
    if not written:
        args.output.unlink(missing_ok=True)
        raise SystemExit("No valid documents were written")
    print(f"Corpus ready: {written} documents -> {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
