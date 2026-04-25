"""Routing между моделями — Day 8.

3 режима в одном скрипте для честного сравнения:
  --mode cheap   — только gpt-4o-mini, 1 вызов
  --mode strong  — только gpt-4o, 1 вызов
  --mode router  — gpt-4o-mini сначала, эскалация на gpt-4o если conf < threshold

Все режимы используют одинаковый JSON-формат запроса {label, confidence}.

Usage:
  python scripts/router.py --mode router
  python scripts/router.py --mode cheap   --output results/router_cheap.json
  python scripts/router.py --mode strong  --output results/router_strong.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = ("search", "understand", "describe", "modify")

CHEAP_MODEL = "gpt-4o-mini"
STRONG_MODEL = "gpt-4o"

# Цены за 1M токенов (input/output) на момент написания
PRICING = {
    "gpt-4o-mini":  {"in": 0.150,  "out": 0.600},
    "gpt-4o":       {"in": 2.500,  "out": 10.000},
}

THRESHOLD = 0.85
TEMPERATURE = 0.0  # детерминизм для воспроизводимости

SYSTEM_JSON = (
    "You are a classifier for developer requests to an AI coding assistant. "
    "The request can be in Russian, English, or mixed. "
    "Classify the developer's intent into one of: search, understand, describe, modify. "
    "Respond ONLY with a compact JSON object: "
    '{"label": "<one of the 4>", "confidence": <0.0-1.0>}. '
    "Confidence must reflect how sure you are. Use 1.0 only when the intent is unambiguous. "
    "Use values below 0.6 if the request is ambiguous, off-topic, or could fit multiple labels. "
    "No prose, no markdown, just the JSON."
)


def normalize(reply: str) -> str:
    return reply.strip().strip(".!?:;\"'`").lower()


def get_client():
    try:
        from openai import OpenAI
    except ImportError:
        print("error: pip install openai python-dotenv", file=sys.stderr)
        sys.exit(2)
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    if not os.getenv("OPENAI_API_KEY"):
        print("error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(2)
    return OpenAI()


def call_model(client, model: str, user: str) -> dict:
    """Один вызов с JSON-схемой. Возвращает dict с label/confidence/latency/tokens."""
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        max_tokens=40,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_JSON},
            {"role": "user", "content": user},
        ],
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    raw = resp.choices[0].message.content or "{}"
    in_tok = resp.usage.prompt_tokens
    out_tok = resp.usage.completion_tokens

    try:
        parsed = json.loads(raw)
        label = normalize(str(parsed.get("label", "")))
        conf_raw = parsed.get("confidence", 0.0)
        confidence = float(conf_raw) if isinstance(conf_raw, (int, float, str)) else 0.0
    except (json.JSONDecodeError, ValueError, TypeError):
        label = ""
        confidence = 0.0

    return {
        "model": model,
        "label": label,
        "confidence": confidence,
        "raw": raw,
        "latency_ms": latency_ms,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }


def cost_of(call: dict) -> float:
    p = PRICING[call["model"]]
    return (call["input_tokens"] / 1_000_000) * p["in"] + (call["output_tokens"] / 1_000_000) * p["out"]


def process_router(client, item: dict) -> dict:
    """Cheap → escalate если confidence < THRESHOLD."""
    user = item["user"]
    cheap = call_model(client, CHEAP_MODEL, user)

    escalated = cheap["confidence"] < THRESHOLD or cheap["label"] not in ALLOWED

    if not escalated:
        return {
            "user": user,
            "expected": item["expected"],
            "kind": item["kind"],
            "route": "cheap",
            "final_label": cheap["label"],
            "final_confidence": cheap["confidence"],
            "calls": [cheap],
            "total_cost": cost_of(cheap),
            "total_latency_ms": cheap["latency_ms"],
        }

    strong = call_model(client, STRONG_MODEL, user)
    return {
        "user": user,
        "expected": item["expected"],
        "kind": item["kind"],
        "route": "escalated",
        "final_label": strong["label"],
        "final_confidence": strong["confidence"],
        "cheap_label": cheap["label"],
        "cheap_confidence": cheap["confidence"],
        "calls": [cheap, strong],
        "total_cost": cost_of(cheap) + cost_of(strong),
        "total_latency_ms": cheap["latency_ms"] + strong["latency_ms"],
    }


def process_single(client, item: dict, model: str) -> dict:
    user = item["user"]
    call = call_model(client, model, user)
    return {
        "user": user,
        "expected": item["expected"],
        "kind": item["kind"],
        "route": model,
        "final_label": call["label"],
        "final_confidence": call["confidence"],
        "calls": [call],
        "total_cost": cost_of(call),
        "total_latency_ms": call["latency_ms"],
    }


def load_holdout(data_dir: Path) -> list[dict]:
    items = []
    with (data_dir / "eval.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            items.append({
                "user": obj["messages"][1]["content"],
                "expected": obj["messages"][2]["content"],
                "kind": "holdout",
            })
    return items


def load_adversarial(data_dir: Path) -> list[dict]:
    path = data_dir / "adversarial.jsonl"
    if not path.exists():
        return []
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            items.append({
                "user": obj["text"],
                "expected": obj["label"],
                "kind": f"adv-{obj.get('kind', 'unknown')}",
            })
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cheap", "strong", "router"), required=True)
    parser.add_argument("--data-dir", default="data/hard")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-workers", type=int, default=8, help="parallelism across examples")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    items = load_holdout(data_dir) + load_adversarial(data_dir)
    print(f"Mode: {args.mode}, total examples: {len(items)}")

    client = get_client()

    if args.mode == "cheap":
        process = lambda it: process_single(client, it, CHEAP_MODEL)
    elif args.mode == "strong":
        process = lambda it: process_single(client, it, STRONG_MODEL)
    else:
        process = lambda it: process_router(client, it)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        results = list(ex.map(process, items))
    wall = time.perf_counter() - t0

    correct = sum(1 for r in results if r["final_label"] == r["expected"])
    holdout = [r for r in results if r["kind"] == "holdout"]
    holdout_correct = sum(1 for r in holdout if r["final_label"] == r["expected"])
    adv = [r for r in results if r["kind"].startswith("adv-")]

    routes = Counter(r["route"] for r in results)
    total_cost = sum(r["total_cost"] for r in results)
    total_calls = sum(len(r["calls"]) for r in results)
    latencies = sorted(r["total_latency_ms"] for r in results)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    print(f"\n--- Mode: {args.mode} ---")
    print(f"Wall clock: {wall:.1f}s")
    print(f"API calls:  {total_calls}")
    print(f"Routes:     {dict(routes)}")
    print(f"Holdout accuracy:  {holdout_correct}/{len(holdout)} = "
          f"{holdout_correct / len(holdout):.0%}")
    if adv:
        adv_correct = sum(1 for r in adv if r["final_label"] == r["expected"])
        print(f"Adversarial label match: {adv_correct}/{len(adv)}")
    print(f"Total accuracy:    {correct}/{len(results)} = {correct / len(results):.0%}")
    print(f"Cost (estimated):  ${total_cost:.4f}")
    print(f"Latency p50:       {p50:.0f}ms")
    print(f"Latency p95:       {p95:.0f}ms")

    payload = {
        "mode": args.mode,
        "config": {
            "cheap_model": CHEAP_MODEL,
            "strong_model": STRONG_MODEL,
            "threshold": THRESHOLD,
            "temperature": TEMPERATURE,
        },
        "summary": {
            "total": len(results),
            "holdout_total": len(holdout),
            "adv_total": len(adv),
            "correct_total": correct,
            "holdout_correct": holdout_correct,
            "routes": dict(routes),
            "api_calls": total_calls,
            "wall_clock_sec": round(wall, 2),
            "total_cost_usd": round(total_cost, 6),
            "latency_p50_ms": round(p50),
            "latency_p95_ms": round(p95),
        },
        "results": results,
    }

    out_path = ROOT / (args.output or f"results/router_{args.mode}.json")
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
