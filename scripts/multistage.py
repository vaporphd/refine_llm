"""Multi-stage inference: monolithic vs decomposed (Day 9).

Извлекает 6 полей из запроса разработчика:
  primary_intent, secondary_intent, target_type, target_name, urgency, negation

Два режима:
  --mode mono   — 1 вызов с большим JSON-промптом (все 6 полей)
  --mode multi  — 4 стадии:
                   Stage 1: Normalize (sequential)
                   Stages 2-4: Classify, Extract, Sentiment (parallel)
                   Final composition в локальном коде

Модель одна — gpt-4o-mini — для честного сравнения архитектуры.

Usage:
  python scripts/multistage.py --mode mono   --output results/multistage_mono.json
  python scripts/multistage.py --mode multi  --output results/multistage_multi.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
PRICING = {"in": 0.150 / 1_000_000, "out": 0.600 / 1_000_000}

ALLOWED_INTENTS = {"search", "understand", "describe", "modify"}
ALLOWED_INTENTS_NONE = ALLOWED_INTENTS | {"none"}
ALLOWED_TARGETS = {"file", "function", "module", "project", "feature", "bug", "config", "test", "other"}
ALLOWED_URGENCY = {"low", "normal", "high"}


# ===== MONOLITHIC =====

SYSTEM_MONO = """You are a structured intent extractor for developer requests to an AI coding assistant.
The request can be in Russian, English, mixed, or other languages.

Extract a JSON object with EXACTLY these 6 fields:

{
  "primary_intent":   "search" | "understand" | "describe" | "modify",
  "secondary_intent": "search" | "understand" | "describe" | "modify" | "none",
  "target_type":      "file" | "function" | "module" | "project" | "feature" | "bug" | "config" | "test" | "other",
  "target_name":      "<short identifier extracted verbatim from input, or null if not present>",
  "urgency":          "low" | "normal" | "high",
  "negation":         true | false
}

Definitions:
- primary_intent = the dominant action requested. If two intents present, the FINAL action wins.
  (e.g. "explain why and then fix" -> primary=modify, secondary=understand)
- target_type:
  - file: a specific file (e.g. "utils.py", "Makefile")
  - function: a function or method or symbol (e.g. "validateConfig", "PacketKind")
  - module: a subsystem or component (e.g. "scheduler", "auth module")
  - project: whole project or repo (e.g. "scrumban project")
  - feature: a high-level feature (e.g. "AI naming", "task flow")
  - bug: error, crash, broken behavior
  - config: settings, env, tokens, credentials, model names
  - test: unit/integration tests
  - other: doesn't fit
- target_name: extract verbatim from input. Use null if no concrete identifier.
- urgency: high if rude, panic, "ASAP", "fix that shit". low if vague, single-char, no real intent.
- negation: true if there is an explicit "don't / не трогай / не меняй" constraint.

Respond ONLY with the JSON object. No markdown, no prose."""


# ===== MULTI-STAGE PROMPTS =====

SYSTEM_NORMALIZE = """Extract surface-level features from a developer request to an AI assistant.
The text can be in any language.

Return ONLY this JSON:
{
  "verbs": ["<imperative verbs found, lowercase, max 3>"],
  "nouns": ["<concrete identifiers / file names / function names found, max 3>"],
  "tone": "neutral" | "rude" | "urgent" | "casual",
  "language": "ru" | "en" | "mixed" | "other"
}

Verbs are action words: fix, find, explain, describe, refactor, add, удали, найди, etc.
Nouns are technical identifiers: file names, function names, module names. NOT generic words.
"""

SYSTEM_CLASSIFY = """Classify developer intent. Given the request and pre-extracted features, output JSON:
{
  "primary_intent":   "search" | "understand" | "describe" | "modify",
  "secondary_intent": "search" | "understand" | "describe" | "modify" | "none"
}

primary_intent = dominant action. If two are present, the FINAL action wins.
("explain why and then fix" -> primary=modify, secondary=understand)

search = locate / where is / find / check status
understand = explain / why / how does X work
describe = high-level overview / architecture / what we have
modify = fix / add / refactor / change / write / create
"""

SYSTEM_TARGET = """Extract the target of a developer request. Output JSON:
{
  "target_type": "file" | "function" | "module" | "project" | "feature" | "bug" | "config" | "test" | "other",
  "target_name": "<verbatim identifier from input, or null>"
}

target_type:
- file: specific file (utils.py, Makefile)
- function: function/method/symbol/type/enum
- module: subsystem (scheduler, auth)
- project: whole repo/project
- feature: high-level feature
- bug: error/crash/broken behavior
- config: settings/env/tokens/model names
- test: tests
- other: none of the above

target_name: extract verbatim from text, or null if no concrete identifier present.
"""

SYSTEM_SENTIMENT = """Analyze sentiment and constraints of a developer request. Output JSON:
{
  "urgency": "low" | "normal" | "high",
  "negation": true | false
}

urgency:
- high: rude/panic/swearing/explicit "ASAP" or "срочно" or "fix that shit"
- low: vague single-char / "lol idk" / no real intent
- normal: everything else

negation: true if there is an explicit "don't X / не трогай / не меняй / без X" constraint
that limits scope of action. Negation about a property (e.g. "no tests yet") is NOT negation.
"""


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


def call_json(client, system: str, user: str, max_tokens: int = 200) -> dict:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
    return {
        "parsed": parsed,
        "raw": raw,
        "latency_ms": latency_ms,
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }


def cost_of(call: dict) -> float:
    return call["input_tokens"] * PRICING["in"] + call["output_tokens"] * PRICING["out"]


def normalize_value(value, allowed: set, default=None):
    if not isinstance(value, str):
        return default
    v = value.strip().lower()
    return v if v in allowed else default


def normalize_target_name(value):
    if value is None or value == "" or (isinstance(value, str) and value.lower() in ("null", "none")):
        return None
    if isinstance(value, str):
        return value.strip()
    return None


# ===== MONO PIPELINE =====

def run_mono(client, item: dict) -> dict:
    user = item["text"]
    call = call_json(client, SYSTEM_MONO, user, max_tokens=200)
    p = call["parsed"]

    final = {
        "primary_intent":   normalize_value(p.get("primary_intent"),   ALLOWED_INTENTS, default="modify"),
        "secondary_intent": normalize_value(p.get("secondary_intent"), ALLOWED_INTENTS_NONE, default="none"),
        "target_type":      normalize_value(p.get("target_type"),      ALLOWED_TARGETS, default="other"),
        "target_name":      normalize_target_name(p.get("target_name")),
        "urgency":          normalize_value(p.get("urgency"),          ALLOWED_URGENCY, default="normal"),
        "negation":         bool(p.get("negation")) if isinstance(p.get("negation"), bool) else (
                                 str(p.get("negation", "")).lower() == "true"),
    }

    return {
        "text": user,
        "expected": {k: item[k] for k in final.keys()},
        "predicted": final,
        "calls": [{"stage": "mono", **call}],
        "total_cost": cost_of(call),
        "total_latency_ms": call["latency_ms"],
        "wall_latency_ms": call["latency_ms"],
        "n_calls": 1,
    }


# ===== MULTI PIPELINE =====

def run_multi(client, item: dict) -> dict:
    user = item["text"]

    # Stage 1: Normalize (sequential, blocks others)
    norm_call = call_json(client, SYSTEM_NORMALIZE, user, max_tokens=120)
    norm = norm_call["parsed"]
    norm_summary = json.dumps(norm, ensure_ascii=False)

    enriched_user = f"REQUEST: {user}\nFEATURES: {norm_summary}"

    # Stages 2, 3, 4 in parallel
    with ThreadPoolExecutor(max_workers=3) as ex:
        f2 = ex.submit(call_json, client, SYSTEM_CLASSIFY, enriched_user, 60)
        f3 = ex.submit(call_json, client, SYSTEM_TARGET, enriched_user, 80)
        f4 = ex.submit(call_json, client, SYSTEM_SENTIMENT, enriched_user, 40)
        c_call = f2.result()
        t_call = f3.result()
        s_call = f4.result()

    classify = c_call["parsed"]
    target = t_call["parsed"]
    sentiment = s_call["parsed"]

    final = {
        "primary_intent":   normalize_value(classify.get("primary_intent"),   ALLOWED_INTENTS, default="modify"),
        "secondary_intent": normalize_value(classify.get("secondary_intent"), ALLOWED_INTENTS_NONE, default="none"),
        "target_type":      normalize_value(target.get("target_type"),        ALLOWED_TARGETS, default="other"),
        "target_name":      normalize_target_name(target.get("target_name")),
        "urgency":          normalize_value(sentiment.get("urgency"),         ALLOWED_URGENCY, default="normal"),
        "negation":         bool(sentiment.get("negation")) if isinstance(sentiment.get("negation"), bool) else (
                                 str(sentiment.get("negation", "")).lower() == "true"),
    }

    all_calls = [norm_call, c_call, t_call, s_call]
    total_cost = sum(cost_of(c) for c in all_calls)
    total_latency = sum(c["latency_ms"] for c in all_calls)
    wall_latency = norm_call["latency_ms"] + max(c_call["latency_ms"], t_call["latency_ms"], s_call["latency_ms"])

    return {
        "text": user,
        "expected": {k: item[k] for k in final.keys()},
        "predicted": final,
        "calls": [
            {"stage": "normalize", **norm_call},
            {"stage": "classify",  **c_call},
            {"stage": "target",    **t_call},
            {"stage": "sentiment", **s_call},
        ],
        "stage_outputs": {"normalize": norm, "classify": classify, "target": target, "sentiment": sentiment},
        "total_cost": total_cost,
        "total_latency_ms": total_latency,
        "wall_latency_ms": wall_latency,
        "n_calls": 4,
    }


# ===== MAIN =====

def load_dataset(data_dir: Path) -> list[dict]:
    items = []
    with (data_dir / "rich_eval.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def field_match(pred: dict, exp: dict) -> dict:
    return {
        "primary_intent":   pred["primary_intent"]   == exp["primary_intent"],
        "secondary_intent": pred["secondary_intent"] == exp["secondary_intent"],
        "target_type":      pred["target_type"]      == exp["target_type"],
        "target_name":      (pred["target_name"] or "").lower() == (exp["target_name"] or "").lower(),
        "urgency":          pred["urgency"]          == exp["urgency"],
        "negation":         pred["negation"]         == exp["negation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("mono", "multi"), required=True)
    parser.add_argument("--data-dir", default="data/hard")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    items = load_dataset(ROOT / args.data_dir)
    print(f"Mode: {args.mode}, examples: {len(items)}, model: {MODEL}\n")

    client = get_client()
    process = run_mono if args.mode == "mono" else run_multi

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        results = list(ex.map(lambda it: process(client, it), items))
    wall = time.perf_counter() - t0

    field_correct = {f: 0 for f in ("primary_intent", "secondary_intent", "target_type",
                                     "target_name", "urgency", "negation")}
    composite_correct = 0
    for r in results:
        m = field_match(r["predicted"], r["expected"])
        r["field_match"] = m
        for f, ok in m.items():
            field_correct[f] += int(ok)
        if all(m.values()):
            composite_correct += 1

    total = len(results)
    total_cost = sum(r["total_cost"] for r in results)
    total_calls = sum(r["n_calls"] for r in results)
    wall_latencies = sorted(r["wall_latency_ms"] for r in results)
    p50 = wall_latencies[len(wall_latencies) // 2]
    p95 = wall_latencies[int(len(wall_latencies) * 0.95)]

    print(f"--- {args.mode.upper()} results ---")
    print(f"Total examples:  {total}")
    print(f"API calls:       {total_calls}")
    print(f"Wall (parallel): {wall:.1f}s")
    print(f"Cost:            ${total_cost:.4f}")
    print(f"Latency p50:     {p50:.0f}ms (per example)")
    print(f"Latency p95:     {p95:.0f}ms")
    print()
    print("Field accuracy:")
    for f, n in field_correct.items():
        print(f"  {f:<18} {n:>3}/{total} = {n / total:.0%}")
    print(f"\nComposite (all 6 right): {composite_correct}/{total} = {composite_correct / total:.0%}")

    payload = {
        "mode": args.mode,
        "model": MODEL,
        "summary": {
            "total": total,
            "api_calls": total_calls,
            "wall_clock_sec": round(wall, 2),
            "total_cost_usd": round(total_cost, 6),
            "latency_p50_ms": round(p50),
            "latency_p95_ms": round(p95),
            "field_accuracy": {f: {"correct": n, "total": total, "rate": round(n / total, 4)}
                                for f, n in field_correct.items()},
            "composite_correct": composite_correct,
            "composite_rate": round(composite_correct / total, 4),
        },
        "results": results,
    }

    out_path = ROOT / (args.output or f"results/multistage_{args.mode}.json")
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
