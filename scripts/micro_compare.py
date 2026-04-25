"""Сравнение micro-router с cheap-only / strong-only / router (Day 8).

Ожидает что в results/ есть:
  - router_cheap.json   (Day 8 cheap-only)
  - router_strong.json  (Day 8 strong-only)
  - router_results.json (Day 8 mini→4o router)
  - micro_router.json   (Day 10 embed+LogReg→mini)

Печатает headline сравнение, разбор где micro победил/проиграл cheap-only,
адверсариальный анализ, cost/latency.

Usage:
  python scripts/micro_compare.py
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cheap",  default="results/router_cheap.json")
    parser.add_argument("--strong", default="results/router_strong.json")
    parser.add_argument("--router", default="results/router_results.json")
    parser.add_argument("--micro",  default="results/micro_router.json")
    args = parser.parse_args()

    cheap  = load(ROOT / args.cheap)
    strong = load(ROOT / args.strong)
    router = load(ROOT / args.router)
    micro  = load(ROOT / args.micro)

    print("=" * 84)
    print("MICRO-MODEL FIRST: comparison with Day 8 baselines")
    print("=" * 84)

    print(f"\nMicro: {micro['config']['embed_model']} + LogReg @ threshold={micro['config']['threshold']}")
    print(f"Train: {micro['config']['n_train']} examples, train_acc={micro['config']['train_accuracy']:.0%}")

    print("\n--- Headline numbers (50 examples = 40 holdout + 10 adv) ---")
    print(f"{'mode':<14} {'hold_acc':>10} {'adv':>8} {'total_acc':>10} {'LLM_calls':>10} {'cost':>10} {'p50ms':>7} {'p95ms':>7}")
    rows = [
        ("cheap-only",  cheap["summary"]),
        ("strong-only", strong["summary"]),
        ("8-router",    router["summary"]),
    ]
    for label, s in rows:
        adv_correct = s["correct_total"] - s["holdout_correct"]
        adv_total = s["total"] - s["holdout_total"]
        h_acc = f"{s['holdout_correct']}/{s['holdout_total']}={s['holdout_correct'] / s['holdout_total']:.0%}"
        adv = f"{adv_correct}/{adv_total}"
        tot = f"{s['correct_total'] / s['total']:.0%}"
        cost = f"${s['total_cost_usd']:.4f}"
        print(f"{label:<14} {h_acc:>10} {adv:>8} {tot:>10} {s['api_calls']:>10} {cost:>10} {s['latency_p50_ms']:>7} {s['latency_p95_ms']:>7}")

    ms = micro["summary"]
    h_acc = f"{ms['holdout_correct']}/{ms['holdout_total']}={ms['holdout_correct'] / ms['holdout_total']:.0%}"
    adv = f"{ms['adv_correct']}/{ms['adv_total']}"
    tot = f"{ms['correct_total'] / ms['total']:.0%}"
    cost = f"${ms['total_cost_usd']:.4f}"
    n_calls = f"{ms['n_llm_calls']}+50e"  # 50 embeds, n_llm_calls escalations
    print(f"{'micro-router':<14} {h_acc:>10} {adv:>8} {tot:>10} {n_calls:>10} {cost:>10} {ms['latency_p50_ms']:>7} {ms['latency_p95_ms']:>7}")

    # Routing
    print("\n--- Micro-router routing ---")
    print(f"  Stayed on embed+LogReg: {ms['stayed']}/{ms['total']} = {ms['stayed'] / ms['total']:.0%}")
    print(f"  Escalated to LLM:       {ms['escalated']}/{ms['total']} = {ms['escalated'] / ms['total']:.0%}")
    print(f"  Stayed accuracy:        {ms['stayed_correct']}/{ms['stayed']} = {ms['stayed_correct'] / max(ms['stayed'], 1):.0%}")
    print(f"  Escalated accuracy:     {ms['escalated_correct']}/{ms['escalated']} = {ms['escalated_correct'] / max(ms['escalated'], 1):.0%}")

    # Cost saving
    print("\n--- Cost savings ---")
    cheap_cost = cheap["summary"]["total_cost_usd"]
    strong_cost = strong["summary"]["total_cost_usd"]
    router_cost = router["summary"]["total_cost_usd"]
    micro_cost = ms["total_cost_usd"]
    print(f"  vs cheap-only:  ${micro_cost:.5f} / ${cheap_cost:.5f} = {micro_cost / cheap_cost:.2f}x  "
          f"(saving {(cheap_cost - micro_cost) / cheap_cost * 100:.0f}%)")
    print(f"  vs 8-router:    ${micro_cost:.5f} / ${router_cost:.5f} = {micro_cost / router_cost:.3f}x  "
          f"(saving {(router_cost - micro_cost) / router_cost * 100:.0f}%)")
    print(f"  vs strong-only: ${micro_cost:.5f} / ${strong_cost:.5f} = {micro_cost / strong_cost:.4f}x  "
          f"(saving {(strong_cost - micro_cost) / strong_cost * 100:.0f}%)")

    # Where micro vs cheap diverge
    print("\n--- Where micro vs cheap-only diverge (holdout) ---")
    cheap_by_text = {r["user"]: r for r in cheap["results"]}
    micro_correct_cheap_wrong = []
    cheap_correct_micro_wrong = []
    both_wrong = []
    for mr in micro["results"]:
        if mr["kind"] != "holdout":
            continue
        cr = cheap_by_text.get(mr["user"])
        if not cr:
            continue
        m_ok = mr["correct"]
        c_ok = (cr["final_label"] == cr["expected"])
        if m_ok and not c_ok:
            micro_correct_cheap_wrong.append((mr, cr))
        elif c_ok and not m_ok:
            cheap_correct_micro_wrong.append((mr, cr))
        elif not m_ok and not c_ok:
            both_wrong.append((mr, cr))

    print(f"  Both correct:                {sum(1 for r in micro['results'] if r['kind'] == 'holdout' and r['correct']) - len(micro_correct_cheap_wrong)}")
    print(f"  Micro right, cheap wrong:    {len(micro_correct_cheap_wrong)}")
    print(f"  Cheap right, micro wrong:    {len(cheap_correct_micro_wrong)}")
    print(f"  Both wrong:                  {len(both_wrong)}")

    if cheap_correct_micro_wrong:
        print("\n  Cases where micro lost (was confident on stayed but wrong, or escalation went wrong):")
        for mr, cr in cheap_correct_micro_wrong[:8]:
            note = f"route={mr['route']} micro_label={mr['micro']['label']} (proba={mr['micro']['max_proba']:.2f})"
            print(f"    expected={mr['expected']:<10} micro_final={mr['final_label']:<10}  | {note}")
            print(f"        text: {mr['user'][:80]}")

    # Adversarial breakdown
    print("\n--- Adversarial detail (10 noisy inputs) ---")
    print(f"  {'kind':<28} {'route':<10} {'final':<10} {'expected':<11} {'proba':<6} {'ok':<3}")
    for r in micro["results"]:
        if not r["kind"].startswith("adv-"):
            continue
        ok = "OK" if r["correct"] else "XX"
        print(f"  {r['kind']:<28} {r['route']:<10} {str(r['final_label']):<10} "
              f"{r['expected']:<11} {r['micro']['max_proba']:<.2f}   {ok}")

    # Threshold analysis: показать как менялось бы accuracy при разных threshold
    print("\n--- Counterfactual: stayed if we change threshold ---")
    for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        n_stay = sum(1 for r in micro["results"] if r["micro"]["max_proba"] >= t)
        # accuracy at this threshold = correct on stayed (micro label) + correct on escalated (llm label)
        # For escalated, we'd need to query LLM — but we already have escalated results
        n_correct = 0
        for r in micro["results"]:
            if r["micro"]["max_proba"] >= t:
                if r["micro"]["label"] == r["expected"]:
                    n_correct += 1
            else:
                if r["llm_call"] and r["llm_call"]["label"] == r["expected"]:
                    n_correct += 1
                elif not r["llm_call"]:
                    # this example was actually stayed in our run; we can't know LLM output
                    # so skip — but realistically threshold > 0.6 only ESCALATES MORE, never less
                    # so this case happens only for t < 0.6. For our run we have llm_call data only when escalated
                    pass
        print(f"    t={t:.1f}  stay={n_stay:>2}/50  acc≈{n_correct}/50 = {n_correct / 50:.0%}")

    print("\n" + "=" * 84)


if __name__ == "__main__":
    main()
