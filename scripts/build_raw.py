"""Склеивает real_labeled.jsonl и synthetic.jsonl в raw.jsonl
в формате chat-completions для fine-tuning.

Usage:
  python scripts/build_raw.py                          # ROOT/data
  python scripts/build_raw.py --data-dir data/zeal     # подпапка
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT = (
    "You are a classifier for developer requests to an AI coding assistant. "
    "The request can be in Russian, English, or mixed. "
    "Classify the developer's intent and reply with exactly one word from: "
    "search, understand, describe, modify. "
    "No explanations, prefixes, quotes, or punctuation."
)

ALLOWED = {"search", "understand", "describe", "modify"}


def load(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def to_chat(ex: dict) -> dict:
    label = ex["label"]
    if label not in ALLOWED:
        raise ValueError(f"bad label: {label!r} in {ex}")
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["text"]},
            {"role": "assistant", "content": label},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    data = ROOT / args.data_dir
    real = list(load(data / "real_labeled.jsonl"))
    synth = list(load(data / "synthetic.jsonl"))
    combined = real + synth
    out = data / "raw.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for ex in combined:
            f.write(json.dumps(to_chat(ex), ensure_ascii=False) + "\n")
    print(f"wrote {len(combined)} examples to {out}")
    # Распределение
    counts = {c: 0 for c in ALLOWED}
    for ex in combined:
        counts[ex["label"]] += 1
    for c, n in counts.items():
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
