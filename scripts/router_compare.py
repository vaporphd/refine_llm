"""Сравнение трёх прогонов router.py: cheap-only / strong-only / router.

Считает:
  - Accuracy holdout, adversarial, total для каждого режима
  - Cost / latency / saving ratio
  - Распределение route в router-режиме (cheap vs escalated)
  - Какие запросы эскалировались — по классам, по типам adv
  - Match с strong: насколько часто router совпадает с strong-only

Usage:
  python scripts/router_compare.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def acc(results: list[dict], filt=lambda r: True) -> tuple[int, int]:
    sel = [r for r in results if filt(r)]
    n = len(sel)
    correct = sum(1 for r in sel if r["final_label"] == r["expected"])
    return correct, n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cheap", default="results/router_cheap.json")
    parser.add_argument("--strong", default="results/router_strong.json")
    parser.add_argument("--router", default="results/router_results.json")
    args = parser.parse_args()

    cheap = load(ROOT / args.cheap)
    strong = load(ROOT / args.strong)
    router = load(ROOT / args.router)

    print("=" * 78)
    print("ROUTING COMPARISON: cheap-only vs strong-only vs router")
    print("=" * 78)

    print("\n--- Configuration ---")
    print(f"Cheap model:  {router['config']['cheap_model']}")
    print(f"Strong model: {router['config']['strong_model']}")
    print(f"Threshold:    {router['config']['threshold']}")
    print(f"Temperature:  {router['config']['temperature']}")

    print("\n--- Headline numbers ---")
    print(f"{'mode':<10} {'holdout':>10} {'adv':>8} {'total':>8} {'calls':>6} {'cost':>10} {'p50ms':>7} {'p95ms':>7}")
    for label, run in (("cheap", cheap), ("strong", strong), ("router", router)):
        s = run["summary"]
        h_acc = f"{s['holdout_correct']}/{s['holdout_total']}={s['holdout_correct'] / s['holdout_total']:.0%}"
        adv_correct = s["correct_total"] - s["holdout_correct"]
        adv_total = s["total"] - s["holdout_total"]
        adv_acc = f"{adv_correct}/{adv_total}" if adv_total else "n/a"
        tot_acc = f"{s['correct_total'] / s['total']:.0%}"
        cost = f"${s['total_cost_usd']:.4f}"
        print(f"{label:<10} {h_acc:>10} {adv_acc:>8} {tot_acc:>8} {s['api_calls']:>6} {cost:>10} {s['latency_p50_ms']:>7} {s['latency_p95_ms']:>7}")

    print("\n--- Router routing distribution ---")
    routes = router["summary"]["routes"]
    total = router["summary"]["total"]
    print(f"  Stayed on cheap:   {routes.get('cheap', 0)}/{total} = {routes.get('cheap', 0) / total:.0%}")
    print(f"  Escalated to strong: {routes.get('escalated', 0)}/{total} = {routes.get('escalated', 0) / total:.0%}")

    # Cost saving vs strong-only
    saving = (strong["summary"]["total_cost_usd"] - router["summary"]["total_cost_usd"]) / strong["summary"]["total_cost_usd"]
    print(f"\n--- Cost saving vs strong-only ---")
    print(f"  Strong-only cost: ${strong['summary']['total_cost_usd']:.4f}")
    print(f"  Router cost:      ${router['summary']['total_cost_usd']:.4f}")
    print(f"  Saving:           {saving:.0%}")
    print(f"  (router cost / strong cost: {router['summary']['total_cost_usd'] / strong['summary']['total_cost_usd']:.2f}x)")

    cheap_to_router_ratio = router["summary"]["total_cost_usd"] / cheap["summary"]["total_cost_usd"]
    print(f"  Router cost / cheap cost: {cheap_to_router_ratio:.2f}x (cost premium for accuracy gain)")

    # Per-class анализ эскалаций
    print("\n--- Where escalations happened (router mode) ---")
    by_class = defaultdict(lambda: Counter())
    for r in router["results"]:
        if r["kind"] == "holdout":
            by_class[r["expected"]][r["route"]] += 1
    print(f"  {'class':<12} {'cheap':>6} {'esc':>6} {'esc%':>6}")
    for cls in ("search", "understand", "describe", "modify"):
        c = by_class[cls]
        total_cls = c["cheap"] + c["escalated"]
        pct = c["escalated"] / total_cls if total_cls else 0
        print(f"  {cls:<12} {c['cheap']:>6} {c['escalated']:>6} {pct:>5.0%}")

    by_kind_adv = defaultdict(lambda: Counter())
    for r in router["results"]:
        if r["kind"].startswith("adv-"):
            by_kind_adv[r["kind"]][r["route"]] += 1
    print(f"\n  Adversarial:")
    print(f"  {'kind':<28} {'cheap':>6} {'esc':>6}")
    for kind, c in sorted(by_kind_adv.items()):
        print(f"  {kind:<28} {c['cheap']:>6} {c['escalated']:>6}")

    # Где cheap ошибся, но router (после эскалации) спас
    print("\n--- Where router rescued cheap (cheap wrong, router right) ---")
    cheap_results = {r["user"]: r for r in cheap["results"]}
    rescued = 0
    rescued_examples = []
    for r in router["results"]:
        if r["kind"] != "holdout":
            continue
        c = cheap_results.get(r["user"])
        if not c:
            continue
        cheap_wrong = c["final_label"] != r["expected"]
        router_right = r["final_label"] == r["expected"]
        if cheap_wrong and router_right and r["route"] == "escalated":
            rescued += 1
            rescued_examples.append({
                "user": r["user"],
                "expected": r["expected"],
                "cheap_predicted": c["final_label"],
                "router_predicted": r["final_label"],
            })
    print(f"  Cases rescued by escalation: {rescued}")
    for ex in rescued_examples:
        print(f"    - expected={ex['expected']:<10} cheap={ex['cheap_predicted']:<10} "
              f"router={ex['router_predicted']:<10} | {ex['user'][:60]}")

    # И обратное — где cheap был прав, но router сломал
    print("\n--- Where router broke cheap (cheap right, router wrong) ---")
    broken = 0
    broken_examples = []
    for r in router["results"]:
        if r["kind"] != "holdout":
            continue
        c = cheap_results.get(r["user"])
        if not c:
            continue
        cheap_right = c["final_label"] == r["expected"]
        router_wrong = r["final_label"] != r["expected"]
        if cheap_right and router_wrong and r["route"] == "escalated":
            broken += 1
            broken_examples.append({
                "user": r["user"],
                "expected": r["expected"],
                "cheap_predicted": c["final_label"],
                "router_predicted": r["final_label"],
            })
    print(f"  Cases broken by escalation: {broken}")
    for ex in broken_examples:
        print(f"    - expected={ex['expected']:<10} cheap={ex['cheap_predicted']:<10} "
              f"router={ex['router_predicted']:<10} | {ex['user'][:60]}")

    # Match между router и strong-only
    print("\n--- Router vs strong-only agreement ---")
    strong_results = {r["user"]: r for r in strong["results"]}
    agree = sum(
        1 for r in router["results"]
        if (s := strong_results.get(r["user"])) and r["final_label"] == s["final_label"]
    )
    print(f"  Same final label: {agree}/{len(router['results'])} = "
          f"{agree / len(router['results']):.0%}")
    print(f"  (router gets the strong answer ~{agree / len(router['results']):.0%} of the time, "
          f"often without paying for strong)")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
