"""Замер baseline: прогон eval.jsonl через gpt-4o-mini (без ФТ).

Считает метрики из results/criteria.md:
  1. Accuracy (общая и per-class)
  2. Формат (strict match)
  3. Покрытие классов в предсказаниях

Сохраняет ответы модели в results/baseline_results.json —
это точка отсчёта, которую нельзя перезаписывать после ФТ.

Usage:
  python scripts/baseline.py
  python scripts/baseline.py --model ft:gpt-4o-mini:...  # прогон после ФТ
  python scripts/baseline.py --dry-run                   # без реальных API-вызовов
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ALLOWED = ("search", "understand", "describe", "modify")
STRICT_FORMAT = re.compile(r"^(search|understand|describe|modify)$")


def load_holdout(data_dir: Path) -> list[dict]:
    with (data_dir / "eval.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(reply: str) -> str:
    return reply.strip().strip(".!?:;\"'`").lower()


def classify(client, model: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=10,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def run(items: list[dict], model: str, dry_run: bool) -> list[dict]:
    if dry_run:
        # Имитация: всегда возвращаем "search" — для проверки логики метрик
        return [
            {
                "user": ex["messages"][1]["content"],
                "expected": ex["messages"][2]["content"],
                "raw_reply": "search",
                "normalized": "search",
                "format_ok": True,
                "correct": ex["messages"][2]["content"] == "search",
            }
            for ex in items
        ]

    try:
        from openai import OpenAI
    except ImportError:
        print("error: install openai>=1.0 (pip install openai python-dotenv)", file=sys.stderr)
        sys.exit(2)

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    if not os.getenv("OPENAI_API_KEY"):
        print("error: OPENAI_API_KEY not set (see .env.example)", file=sys.stderr)
        sys.exit(2)

    client = OpenAI()
    out = []
    for i, ex in enumerate(items, 1):
        system = ex["messages"][0]["content"]
        user = ex["messages"][1]["content"]
        expected = ex["messages"][2]["content"]
        raw = classify(client, model, system, user)
        norm = normalize(raw)
        out.append(
            {
                "user": user,
                "expected": expected,
                "raw_reply": raw,
                "normalized": norm,
                "format_ok": bool(STRICT_FORMAT.match(raw.strip())),
                "correct": norm == expected,
            }
        )
        mark = "OK" if norm == expected else "XX"
        print(f"  [{i:2d}/{len(items)}] {mark} expected={expected:<10} got={norm:<15} raw={raw!r}")
    return out


def metrics(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(r["correct"] for r in results)
    format_ok = sum(r["format_ok"] for r in results)

    per_class_total = Counter(r["expected"] for r in results)
    per_class_correct = Counter(r["expected"] for r in results if r["correct"])
    per_class_acc = {
        cls: {
            "correct": per_class_correct[cls],
            "total": per_class_total[cls],
            "accuracy": (per_class_correct[cls] / per_class_total[cls]) if per_class_total[cls] else None,
        }
        for cls in ALLOWED
    }

    predicted_classes = sorted({r["normalized"] for r in results if r["normalized"] in ALLOWED})

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "format_strict_ok": format_ok,
        "format_strict_ratio": format_ok / total if total else 0.0,
        "per_class": per_class_acc,
        "predicted_classes_present": predicted_classes,
        "predicted_classes_count": len(predicted_classes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--output", default=None, help="override output json path")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dry-run", action="store_true", help="simulate without API calls")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    items = load_holdout(data_dir)
    print(f"Loaded {len(items)} holdout examples from {data_dir}")
    print(f"Model: {args.model}{' (DRY RUN)' if args.dry_run else ''}\n")

    results = run(items, args.model, args.dry_run)
    m = metrics(results)

    print("\n--- Metrics ---")
    print(f"Accuracy:         {m['correct']}/{m['total']} = {m['accuracy']:.0%}")
    print(f"Format strict OK: {m['format_strict_ok']}/{m['total']} = {m['format_strict_ratio']:.0%}")
    print(f"Classes predicted: {m['predicted_classes_count']}/4 ({', '.join(m['predicted_classes_present'])})")
    print("\nPer-class:")
    for cls, info in m["per_class"].items():
        if info["total"] == 0:
            continue
        acc = f"{info['accuracy']:.0%}" if info["accuracy"] is not None else "n/a"
        print(f"  {cls:<12} {info['correct']}/{info['total']} = {acc}")

    out_path = Path(args.output) if args.output else RESULTS / "baseline_results.json"
    RESULTS.mkdir(exist_ok=True)
    payload = {"model": args.model, "metrics": m, "results": results}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
