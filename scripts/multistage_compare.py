"""Сравнение mono vs multi-stage инференса (Day 9).

Печатает:
  - Field-level accuracy для обоих режимов рядом
  - Composite accuracy (все 6 полей одновременно)
  - Cost / latency / API calls / wall clock
  - Куда расходятся: примеры где один режим прав, другой нет
  - Per-stage failure analysis (где multi теряет точность)

Usage:
  python scripts/multistage_compare.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mono",  default="results/multistage_mono.json")
    parser.add_argument("--multi", default="results/multistage_multi.json")
    args = parser.parse_args()

    mono  = load(ROOT / args.mono)
    multi = load(ROOT / args.multi)

    print("=" * 78)
    print("MULTI-STAGE vs MONOLITHIC")
    print("=" * 78)
    print(f"Model: {mono['model']} (same in both modes)")

    # Headline
    print("\n--- Headline ---")
    print(f"{'metric':<22} {'mono':>10} {'multi':>10} {'delta':>10}")
    rows = [
        ("examples",          mono["summary"]["total"],          multi["summary"]["total"],          None),
        ("api_calls",         mono["summary"]["api_calls"],      multi["summary"]["api_calls"],      None),
        ("wall_clock_sec",    mono["summary"]["wall_clock_sec"], multi["summary"]["wall_clock_sec"], None),
        ("cost_usd",          f"${mono['summary']['total_cost_usd']:.4f}", f"${multi['summary']['total_cost_usd']:.4f}", None),
        ("latency_p50_ms",    mono["summary"]["latency_p50_ms"], multi["summary"]["latency_p50_ms"], None),
        ("latency_p95_ms",    mono["summary"]["latency_p95_ms"], multi["summary"]["latency_p95_ms"], None),
        ("composite_correct", mono["summary"]["composite_correct"], multi["summary"]["composite_correct"], None),
        ("composite_rate",    f"{mono['summary']['composite_rate']:.0%}", f"{multi['summary']['composite_rate']:.0%}", None),
    ]
    for name, mv, vv, _ in rows:
        print(f"  {name:<22} {str(mv):>10} {str(vv):>10}")

    cost_ratio = multi["summary"]["total_cost_usd"] / mono["summary"]["total_cost_usd"]
    print(f"\n  cost ratio multi/mono: {cost_ratio:.2f}x")
    print(f"  composite delta: {(multi['summary']['composite_rate'] - mono['summary']['composite_rate']) * 100:+.1f} pp")

    # Field-level
    print("\n--- Field-level accuracy ---")
    print(f"  {'field':<20} {'mono':>10} {'multi':>10} {'delta':>8}")
    fields = ["primary_intent", "secondary_intent", "target_type", "target_name", "urgency", "negation"]
    for f in fields:
        m_acc = mono["summary"]["field_accuracy"][f]["rate"]
        v_acc = multi["summary"]["field_accuracy"][f]["rate"]
        delta = (v_acc - m_acc) * 100
        delta_str = f"{delta:+.1f}pp"
        print(f"  {f:<20} {m_acc:>9.0%} {v_acc:>9.0%}  {delta_str:>8}")

    # Where they diverge
    print("\n--- Cases where modes disagree ---")
    mono_results = {r["text"]: r for r in mono["results"]}
    multi_results = {r["text"]: r for r in multi["results"]}

    mono_only_correct = []
    multi_only_correct = []
    both_wrong = []
    for text in mono_results:
        m_r = mono_results[text]
        v_r = multi_results[text]
        m_ok = all(m_r["field_match"].values())
        v_ok = all(v_r["field_match"].values())
        if m_ok and not v_ok:
            mono_only_correct.append((text, m_r, v_r))
        elif v_ok and not m_ok:
            multi_only_correct.append((text, m_r, v_r))
        elif not m_ok and not v_ok:
            both_wrong.append((text, m_r, v_r))

    print(f"  Both right (composite):     {sum(1 for r in mono['results'] if all(r['field_match'].values())) - len(mono_only_correct)}")
    print(f"  Mono right only:            {len(mono_only_correct)}")
    print(f"  Multi right only:           {len(multi_only_correct)}")
    print(f"  Both wrong:                 {len(both_wrong)}")

    if multi_only_correct:
        print("\n  Examples where multi rescued mono:")
        for text, _, _ in multi_only_correct[:5]:
            print(f"    - {text[:70]}")

    if mono_only_correct:
        print("\n  Examples where mono won, multi failed:")
        for text, _, v in mono_only_correct[:5]:
            wrong = [f for f, ok in v["field_match"].items() if not ok]
            print(f"    - {text[:50]} (multi failed: {wrong})")

    # Per-stage failure analysis (multi only)
    print("\n--- Where multi-stage breaks down (per-field error attribution) ---")
    error_fields = Counter()
    for r in multi["results"]:
        for f, ok in r["field_match"].items():
            if not ok:
                error_fields[f] += 1
    print("  Field errors in multi:")
    for f in fields:
        if error_fields[f]:
            print(f"    {f:<20} {error_fields[f]:>3}")

    # Tokens breakdown
    print("\n--- Token usage breakdown ---")
    mono_in = sum(c["input_tokens"] for r in mono["results"] for c in r["calls"])
    mono_out = sum(c["output_tokens"] for r in mono["results"] for c in r["calls"])
    multi_in = sum(c["input_tokens"] for r in multi["results"] for c in r["calls"])
    multi_out = sum(c["output_tokens"] for r in multi["results"] for c in r["calls"])
    print(f"  mono:  in={mono_in}  out={mono_out}  total={mono_in + mono_out}")
    print(f"  multi: in={multi_in}  out={multi_out}  total={multi_in + multi_out}")
    print(f"  multi/mono input ratio:  {multi_in / mono_in:.2f}x")
    print(f"  multi/mono output ratio: {multi_out / mono_out:.2f}x")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
