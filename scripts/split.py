"""Стратифицированный сплит raw.jsonl → train.jsonl (80%) + eval.jsonl (20%).

Стратификация: пропорция классов сохраняется в train и eval.
Seed: 42 (воспроизводимость).

Usage:
  python scripts/split.py
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
EVAL_RATIO = 0.2


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def label_of(example: dict) -> str:
    return example["messages"][2]["content"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    data_dir = ROOT / args.data_dir
    rng = random.Random(SEED)
    data = load(data_dir / "raw.jsonl")

    # Группируем по label
    by_label: dict[str, list[dict]] = defaultdict(list)
    for ex in data:
        by_label[label_of(ex)].append(ex)

    train: list[dict] = []
    eval_: list[dict] = []

    for label, items in sorted(by_label.items()):
        rng.shuffle(items)
        n_eval = max(1, round(len(items) * EVAL_RATIO))
        eval_.extend(items[:n_eval])
        train.extend(items[n_eval:])

    rng.shuffle(train)
    rng.shuffle(eval_)

    write(data_dir / "train.jsonl", train)
    write(data_dir / "eval.jsonl", eval_)

    print(f"train: {len(train)} examples")
    print(f"eval:  {len(eval_)} examples")
    print(f"ratio: {len(eval_) / len(data):.0%} eval")
    print("\nDistribution:")
    print(f"  {'label':<12} {'train':>6} {'eval':>6}")
    for label in sorted(by_label):
        t = sum(1 for ex in train if label_of(ex) == label)
        e = sum(1 for ex in eval_ if label_of(ex) == label)
        print(f"  {label:<12} {t:>6} {e:>6}")


if __name__ == "__main__":
    main()
