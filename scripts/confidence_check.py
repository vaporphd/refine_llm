"""Inference с оценкой уверенности и gating (Constraint + Redundancy + Scoring).

Pipeline на 1 пример:
  A. Constraint: формат ответа должен соответствовать regex
  B. Redundancy: 3 параллельных вызова при T=0.7 → majority vote
     (один из них с JSON-схемой, чтобы заодно получить self-confidence)
  C. Scoring: confidence из JSON-вызова (B-3) сравнивается с порогом

Финальный статус:
  OK     — все 3 голоса совпали, confidence >= 0.85, format ok
  UNSURE — 2/3 голоса, или confidence < 0.85, или disagreement vote/scoring
  FAIL   — формат битый, или 0-1 голос за топ-класс, или confidence < 0.5

Usage:
  python scripts/confidence_check.py
  python scripts/confidence_check.py --data-dir data/hard --include-adversarial
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ALLOWED = ("search", "understand", "describe", "modify")
STRICT_FORMAT = re.compile(r"^(search|understand|describe|modify)$")
MODEL = "gpt-4o-mini"
TEMP_REDUNDANCY = 0.7
THRESHOLD_OK = 0.85
THRESHOLD_FAIL = 0.5

SYSTEM_PLAIN = (
    "You are a classifier for developer requests to an AI coding assistant. "
    "The request can be in Russian, English, or mixed. "
    "Classify the developer's intent and reply with exactly one word from: "
    "search, understand, describe, modify. "
    "No explanations, prefixes, quotes, or punctuation."
)

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


def call_plain(client, user: str):
    """Один вызов с T=0.7. Возвращает (raw_reply, latency_ms, total_tokens)."""
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMP_REDUNDANCY,
        max_tokens=10,
        messages=[
            {"role": "system", "content": SYSTEM_PLAIN},
            {"role": "user", "content": user},
        ],
    )
    latency = (time.perf_counter() - t0) * 1000
    return resp.choices[0].message.content or "", latency, resp.usage.total_tokens


def call_json(client, user: str):
    """Один вызов с JSON-схемой. Возвращает (parsed_dict, latency_ms, tokens)."""
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMP_REDUNDANCY,
        max_tokens=40,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_JSON},
            {"role": "user", "content": user},
        ],
    )
    latency = (time.perf_counter() - t0) * 1000
    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
        label = normalize(str(parsed.get("label", "")))
        conf_raw = parsed.get("confidence", 0.0)
        confidence = float(conf_raw) if isinstance(conf_raw, (int, float, str)) else 0.0
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"label": "", "confidence": 0.0, "raw": raw, "parse_error": True}, latency, resp.usage.total_tokens
    return {"label": label, "confidence": confidence, "raw": raw, "parse_error": False}, latency, resp.usage.total_tokens


def gate(plain1: str, plain2: str, json_resp: dict) -> dict:
    """Применить gating-логику по результатам 3 вызовов."""
    p1 = normalize(plain1)
    p2 = normalize(plain2)
    p3 = json_resp["label"]
    confidence = json_resp["confidence"]

    votes = [v for v in (p1, p2, p3) if v in ALLOWED]
    counts = Counter(votes)

    format_ok = (
        bool(STRICT_FORMAT.match(plain1.strip()))
        and bool(STRICT_FORMAT.match(plain2.strip()))
        and not json_resp.get("parse_error")
        and p3 in ALLOWED
    )

    if not votes:
        return {"status": "FAIL", "predicted": None, "confidence": confidence,
                "votes": [p1, p2, p3], "vote_top_count": 0, "format_ok": format_ok,
                "reason": "no valid votes"}

    top, top_count = counts.most_common(1)[0]

    if top_count == 3 and confidence >= THRESHOLD_OK and format_ok:
        status = "OK"
        reason = "3/3 agree, confidence >= 0.85, format ok"
    elif top_count == 3 and confidence >= THRESHOLD_FAIL:
        status = "UNSURE"
        reason = f"3/3 agree but confidence={confidence:.2f} < 0.85"
    elif top_count == 2:
        status = "UNSURE"
        reason = "2/3 agree (split vote)"
    elif top_count == 1 and len(votes) == 3:
        status = "FAIL"
        reason = "all 3 different"
    elif confidence < THRESHOLD_FAIL:
        status = "FAIL"
        reason = f"confidence={confidence:.2f} < 0.5"
    elif not format_ok:
        status = "FAIL"
        reason = "format invalid"
    else:
        status = "UNSURE"
        reason = "fallback"

    if json_resp["label"] in ALLOWED and json_resp["label"] != top and top_count < 3:
        if status == "OK":
            status = "UNSURE"
            reason = f"vote={top}, scoring label={json_resp['label']} disagree"

    return {"status": status, "predicted": top, "confidence": confidence,
            "votes": [p1, p2, p3], "vote_top_count": top_count,
            "format_ok": format_ok, "reason": reason}


def process_example(client, item: dict, idx: int, total: int) -> dict:
    user = item["user"]
    expected = item["expected"]
    kind = item.get("kind", "holdout")

    with ThreadPoolExecutor(max_workers=3) as ex:
        f1 = ex.submit(call_plain, client, user)
        f2 = ex.submit(call_plain, client, user)
        f3 = ex.submit(call_json, client, user)
        plain1, lat1, tok1 = f1.result()
        plain2, lat2, tok2 = f2.result()
        json_resp, lat3, tok3 = f3.result()

    decision = gate(plain1, plain2, json_resp)
    total_latency_ms = max(lat1, lat2, lat3)
    total_tokens = tok1 + tok2 + tok3

    out = {
        "idx": idx,
        "user": user,
        "expected": expected,
        "kind": kind,
        **decision,
        "raw": {"plain1": plain1, "plain2": plain2, "json": json_resp},
        "latency_ms_max": total_latency_ms,
        "latency_ms_each": [lat1, lat2, lat3],
        "tokens_total": total_tokens,
    }

    mark = {"OK": "v", "UNSURE": "?", "FAIL": "x"}[decision["status"]]
    print(
        f"  [{idx:2d}/{total}] {mark} {decision['status']:<7} "
        f"pred={str(decision['predicted']):<10} conf={decision['confidence']:.2f} "
        f"expected={expected:<10} kind={kind:<20} | {decision['reason']}",
        flush=True,
    )
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
    parser.add_argument("--output", default="results/confidence_hard.json")
    parser.add_argument("--include-adversarial", action="store_true", default=True)
    parser.add_argument("--no-adversarial", dest="include_adversarial", action="store_false")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    items = load_holdout(data_dir)
    if args.include_adversarial:
        adv = load_adversarial(data_dir)
        items.extend(adv)
        print(f"Loaded {len(items) - len(adv)} holdout + {len(adv)} adversarial = {len(items)} total")
    else:
        print(f"Loaded {len(items)} holdout examples")

    client = get_client()
    print(f"Model: {MODEL}, T={TEMP_REDUNDANCY}, threshold OK>={THRESHOLD_OK}\n")

    t_start = time.perf_counter()
    results = []
    for i, item in enumerate(items, 1):
        results.append(process_example(client, item, i, len(items)))
    t_end = time.perf_counter()

    statuses = Counter(r["status"] for r in results)
    total_tokens = sum(r["tokens_total"] for r in results)
    total_calls = len(results) * 3
    latencies = [r["latency_ms_max"] for r in results]
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    print("\n--- Aggregate ---")
    print(f"Total examples: {len(results)}")
    print(f"Statuses: {dict(statuses)}")
    print(f"  OK:     {statuses['OK']}  ({statuses['OK'] / len(results):.0%})")
    print(f"  UNSURE: {statuses['UNSURE']}  ({statuses['UNSURE'] / len(results):.0%})")
    print(f"  FAIL:   {statuses['FAIL']}  ({statuses['FAIL'] / len(results):.0%})")
    print(f"\nAPI calls total: {total_calls}")
    print(f"Tokens total:    {total_tokens}")
    print(f"Wall clock:      {t_end - t_start:.1f}s")
    print(f"Latency per example (max of 3 parallel): p50={p50:.0f}ms, p95={p95:.0f}ms")

    cost_input_per_1m = 0.150
    cost_output_per_1m = 0.600
    estimated_cost = total_tokens / 1_000_000 * (cost_input_per_1m + cost_output_per_1m) / 2
    print(f"Estimated cost:  ~${estimated_cost:.4f}")

    payload = {
        "model": MODEL,
        "config": {
            "temperature_redundancy": TEMP_REDUNDANCY,
            "threshold_ok": THRESHOLD_OK,
            "threshold_fail": THRESHOLD_FAIL,
        },
        "summary": {
            "total": len(results),
            "statuses": dict(statuses),
            "api_calls": total_calls,
            "tokens": total_tokens,
            "wall_clock_sec": round(t_end - t_start, 2),
            "latency_p50_ms": round(p50),
            "latency_p95_ms": round(p95),
            "estimated_cost_usd": round(estimated_cost, 4),
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
