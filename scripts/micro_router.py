"""Micro-router: embed → LogReg → если max_proba < threshold → gpt-4o-mini fallback.

Pipeline на 1 пример:
  1. Embed input через text-embedding-3-small  (~50ms, ~$0.00001)
  2. classifier.predict + predict_proba         (~1ms CPU)
  3. Если max_proba >= THRESHOLD → принять
     Иначе → escalate to gpt-4o-mini             (~1000ms, ~$0.001)

Тестовый набор: 40 holdout (data/hard/eval.jsonl) + 10 adversarial.

Usage:
  python scripts/micro_router.py
  python scripts/micro_router.py --threshold 0.5
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

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
ALLOWED = ("search", "understand", "describe", "modify")

PRICING = {
    "embed":  {"in": 0.020 / 1_000_000},
    "gpt-4o-mini": {"in": 0.150 / 1_000_000, "out": 0.600 / 1_000_000},
}

SYSTEM_LLM = (
    "You are a classifier for developer requests to an AI coding assistant. "
    "The request can be in Russian, English, or mixed. "
    "Classify the developer's intent into one of: search, understand, describe, modify. "
    "Respond ONLY with a compact JSON object: "
    '{"label": "<one of the 4>", "confidence": <0.0-1.0>}. '
    "No prose, no markdown, just the JSON."
)


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


def embed_one(client, text: str) -> dict:
    t0 = time.perf_counter()
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    latency_ms = (time.perf_counter() - t0) * 1000
    emb = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    return {
        "vector": emb,
        "latency_ms": latency_ms,
        "tokens": resp.usage.total_tokens,
        "cost": resp.usage.total_tokens * PRICING["embed"]["in"],
    }


def call_llm(client, user: str) -> dict:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.0,
        max_tokens=40,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_LLM},
            {"role": "user", "content": user},
        ],
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
        label = str(parsed.get("label", "")).strip().lower()
        confidence = float(parsed.get("confidence", 0.0))
    except (json.JSONDecodeError, ValueError, TypeError):
        label = ""
        confidence = 0.0
    cost = (resp.usage.prompt_tokens * PRICING["gpt-4o-mini"]["in"]
            + resp.usage.completion_tokens * PRICING["gpt-4o-mini"]["out"])
    return {
        "label": label if label in ALLOWED else None,
        "confidence": confidence,
        "raw": raw,
        "latency_ms": latency_ms,
        "tokens_in": resp.usage.prompt_tokens,
        "tokens_out": resp.usage.completion_tokens,
        "cost": cost,
    }


def process_one(client, classifier, classes, item: dict, threshold: float) -> dict:
    user = item["user"]
    expected = item["expected"]
    kind = item["kind"]

    embed_call = embed_one(client, user)
    proba = classifier.predict_proba(embed_call["vector"])[0]
    top_idx = int(np.argmax(proba))
    micro_label = classes[top_idx]
    max_proba = float(proba[top_idx])
    proba_dict = {c: float(p) for c, p in zip(classes, proba)}

    out = {
        "user": user,
        "expected": expected,
        "kind": kind,
        "embed_call": {k: v for k, v in embed_call.items() if k != "vector"},
        "micro": {
            "label": micro_label,
            "max_proba": max_proba,
            "proba": proba_dict,
        },
    }

    if max_proba >= threshold:
        out["route"] = "stayed"
        out["final_label"] = micro_label
        out["final_confidence"] = max_proba
        out["llm_call"] = None
        out["total_cost"] = embed_call["cost"]
        out["total_latency_ms"] = embed_call["latency_ms"]
    else:
        llm = call_llm(client, user)
        out["route"] = "escalated"
        out["final_label"] = llm["label"]
        out["final_confidence"] = llm["confidence"]
        out["llm_call"] = llm
        out["total_cost"] = embed_call["cost"] + llm["cost"]
        out["total_latency_ms"] = embed_call["latency_ms"] + llm["latency_ms"]

    out["correct"] = out["final_label"] == expected
    return out


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
    parser.add_argument("--data-dir", default="data/hard")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--output", default="results/micro_router.json")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--classifier", default="results/micro_classifier.joblib")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    items = load_holdout(data_dir) + load_adversarial(data_dir)
    print(f"Loaded {len(items)} examples (40 holdout + 10 adversarial)")

    bundle = joblib.load(ROOT / args.classifier)
    classifier = bundle["classifier"]
    classes = bundle["classes"]
    print(f"Loaded classifier from {args.classifier}, classes={classes}, train_acc={bundle['train_accuracy']:.0%}")
    print(f"Threshold: max_proba >= {args.threshold} → STAY, else → ESCALATE to {LLM_MODEL}\n")

    client = get_client()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        results = list(ex.map(
            lambda it: process_one(client, classifier, classes, it, args.threshold),
            items,
        ))
    wall = time.perf_counter() - t0

    routes = Counter(r["route"] for r in results)
    correct = sum(1 for r in results if r["correct"])
    holdout = [r for r in results if r["kind"] == "holdout"]
    holdout_correct = sum(1 for r in holdout if r["correct"])
    adv = [r for r in results if r["kind"].startswith("adv-")]
    adv_correct = sum(1 for r in adv if r["correct"])
    total_cost = sum(r["total_cost"] for r in results)
    embed_cost = sum(r["embed_call"]["cost"] for r in results)
    llm_cost = sum(r["llm_call"]["cost"] for r in results if r["llm_call"])
    n_llm_calls = sum(1 for r in results if r["llm_call"])

    latencies = sorted(r["total_latency_ms"] for r in results)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    # Stayed accuracy (на тех что micro обработал самостоятельно)
    stayed = [r for r in results if r["route"] == "stayed"]
    stayed_correct = sum(1 for r in stayed if r["correct"])

    # Escalated accuracy
    escalated = [r for r in results if r["route"] == "escalated"]
    escalated_correct = sum(1 for r in escalated if r["correct"])

    print("--- Routing distribution ---")
    print(f"  Stayed on micro:    {routes['stayed']}/{len(items)} = {routes['stayed'] / len(items):.0%}")
    print(f"  Escalated to LLM:   {routes['escalated']}/{len(items)} = {routes['escalated'] / len(items):.0%}")

    print("\n--- Accuracy ---")
    print(f"  Holdout total:      {holdout_correct}/{len(holdout)} = {holdout_correct / len(holdout):.0%}")
    print(f"  Adversarial:        {adv_correct}/{len(adv)} = {adv_correct / len(adv):.0%}")
    print(f"  Total:              {correct}/{len(items)} = {correct / len(items):.0%}")
    if stayed:
        print(f"  On stayed:          {stayed_correct}/{len(stayed)} = {stayed_correct / len(stayed):.0%}")
    if escalated:
        print(f"  On escalated:       {escalated_correct}/{len(escalated)} = {escalated_correct / len(escalated):.0%}")

    print("\n--- Cost ---")
    print(f"  Embeddings:         ${embed_cost:.6f}")
    print(f"  LLM (fallback):     ${llm_cost:.6f}")
    print(f"  Total:              ${total_cost:.6f}")
    print(f"  LLM calls used:     {n_llm_calls}/{len(items)}")

    print("\n--- Latency ---")
    print(f"  Wall clock:         {wall:.1f}s (parallel {args.max_workers} workers)")
    print(f"  p50:                {p50:.0f}ms")
    print(f"  p95:                {p95:.0f}ms")

    payload = {
        "config": {
            "embed_model": EMBED_MODEL,
            "llm_model": LLM_MODEL,
            "threshold": args.threshold,
            "n_train": bundle["n_train"],
            "train_accuracy": bundle["train_accuracy"],
        },
        "summary": {
            "total": len(items),
            "holdout_total": len(holdout),
            "adv_total": len(adv),
            "stayed": routes["stayed"],
            "escalated": routes["escalated"],
            "correct_total": correct,
            "holdout_correct": holdout_correct,
            "adv_correct": adv_correct,
            "stayed_correct": stayed_correct,
            "escalated_correct": escalated_correct,
            "total_cost_usd": round(total_cost, 6),
            "embed_cost_usd": round(embed_cost, 6),
            "llm_cost_usd": round(llm_cost, 6),
            "n_llm_calls": n_llm_calls,
            "wall_clock_sec": round(wall, 2),
            "latency_p50_ms": round(p50),
            "latency_p95_ms": round(p95),
        },
        "results": results,
    }

    out_path = ROOT / args.output
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
