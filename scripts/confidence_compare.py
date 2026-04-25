"""Анализ результатов confidence_check.py против ground truth.

Считает:
  - Accuracy на принятых (status=OK)
  - Pickup rate / rejection rate / catch rate
  - Калибровка confidence (Spearman)
  - Per-status breakdown vs truth
  - Какие baseline-ошибки поймал механизм

Usage:
  python scripts/confidence_compare.py
  python scripts/confidence_compare.py --input results/confidence_hard.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation без scipy."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    n = len(xs)

    def rank(arr):
        s = sorted(range(n), key=lambda i: arr[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and arr[s[j + 1]] == arr[s[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[s[k]] = avg
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    num = sum((rx[i] - mean_x) * (ry[i] - mean_y) for i in range(n))
    den_x = sum((r - mean_x) ** 2 for r in rx) ** 0.5
    den_y = sum((r - mean_y) ** 2 for r in ry) ** 0.5
    if den_x * den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/confidence_hard.json")
    parser.add_argument("--baseline", default="results/baseline_hard.json")
    args = parser.parse_args()

    with (ROOT / args.input).open(encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    summary = data["summary"]
    config = data["config"]

    print("=" * 78)
    print("CONFIDENCE GATING ANALYSIS")
    print("=" * 78)
    print(f"Model: {data['model']}, T={config['temperature_redundancy']}, threshold OK>={config['threshold_ok']}")
    print(f"Total: {summary['total']} examples, {summary['api_calls']} API calls")
    print(f"Wall: {summary['wall_clock_sec']}s, latency p50={summary['latency_p50_ms']}ms p95={summary['latency_p95_ms']}ms")
    print(f"Tokens: {summary['tokens']}, est cost: ${summary['estimated_cost_usd']:.4f}")
    print()

    # Разделим на holdout и adversarial
    holdout = [r for r in results if r["kind"] == "holdout"]
    adv = [r for r in results if r["kind"].startswith("adv-")]

    def status_breakdown(items, label):
        c = Counter(r["status"] for r in items)
        n = len(items)
        if n == 0:
            return
        print(f"\n--- {label} ({n} examples) ---")
        for s in ("OK", "UNSURE", "FAIL"):
            print(f"  {s:<7} {c[s]:>3} ({c[s] / n:.0%})")

    status_breakdown(holdout, "HOLDOUT (data/hard/eval.jsonl)")
    status_breakdown(adv, "ADVERSARIAL (noisy/edge inputs)")

    # Accuracy на принятых (OK)
    print("\n--- Accuracy on accepted (status=OK) ---")
    ok_holdout = [r for r in holdout if r["status"] == "OK"]
    if ok_holdout:
        correct = sum(1 for r in ok_holdout if r["predicted"] == r["expected"])
        print(f"  Holdout OK: {correct}/{len(ok_holdout)} = {correct / len(ok_holdout):.0%}")
    ok_adv = [r for r in adv if r["status"] == "OK"]
    if ok_adv:
        correct_adv = sum(1 for r in ok_adv if r["predicted"] == r["expected"])
        print(f"  Adversarial OK: {correct_adv}/{len(ok_adv)} = {correct_adv / len(ok_adv):.0%}")
    else:
        print("  Adversarial OK: 0 examples accepted (good — they should be rejected)")

    # Pickup / rejection / catch rates
    print("\n--- Quality metrics ---")
    pickup = sum(1 for r in holdout if r["status"] == "OK") / len(holdout) if holdout else 0
    rejection_adv = sum(1 for r in adv if r["status"] in ("UNSURE", "FAIL")) / len(adv) if adv else 0
    print(f"  Pickup rate (holdout accepted as OK):       {pickup:.0%}")
    print(f"  Rejection rate (adversarial UNSURE+FAIL):   {rejection_adv:.0%}")

    # Catch rate: из тех holdout где модель ошиблась — сколько помечено UNSURE/FAIL?
    holdout_wrong = [r for r in holdout if r["predicted"] != r["expected"]]
    holdout_caught = [r for r in holdout_wrong if r["status"] in ("UNSURE", "FAIL")]
    print(f"  Holdout misclassifications: {len(holdout_wrong)}")
    if holdout_wrong:
        print(f"  Catch rate (wrong predictions caught as UNSURE/FAIL): "
              f"{len(holdout_caught)}/{len(holdout_wrong)} = {len(holdout_caught) / len(holdout_wrong):.0%}")

    # Сравнение с baseline (если файл есть)
    baseline_path = ROOT / args.baseline
    if baseline_path.exists():
        with baseline_path.open(encoding="utf-8") as f:
            base = json.load(f)
        baseline_acc = base["metrics"]["accuracy"]
        gated_holdout_correct = sum(
            1 for r in holdout if r["status"] == "OK" and r["predicted"] == r["expected"]
        )
        gated_acc_on_accepted = (
            gated_holdout_correct / sum(1 for r in holdout if r["status"] == "OK")
            if any(r["status"] == "OK" for r in holdout) else 0
        )
        print(f"\n--- Baseline vs Gated (holdout) ---")
        print(f"  Baseline single-call accuracy:         {baseline_acc:.0%}")
        print(f"  Gated accuracy on accepted (OK only):  {gated_acc_on_accepted:.0%}")
        print(f"  Trade-off: baseline serves {len(holdout)}, gated serves {len(ok_holdout)} "
              f"({len(holdout) - len(ok_holdout)} flagged for review)")

    # Калибровка scoring
    print("\n--- Scoring calibration ---")
    confidences = [r["confidence"] for r in holdout if r["confidence"] > 0]
    correctness = [1.0 if r["predicted"] == r["expected"] else 0.0 for r in holdout if r["confidence"] > 0]
    if len(confidences) >= 2:
        rho = spearman(confidences, correctness)
        print(f"  Spearman(confidence, correct): {rho:+.3f}")
        if rho > 0.3:
            print("  -> Confidence is positively correlated with correctness (useful signal)")
        elif rho > 0.0:
            print("  -> Weak positive correlation (confidence is barely informative)")
        else:
            print("  -> No useful correlation (confidence is noise)")

    avg_conf_correct = (
        sum(r["confidence"] for r in holdout if r["predicted"] == r["expected"])
        / max(1, sum(1 for r in holdout if r["predicted"] == r["expected"]))
    )
    avg_conf_wrong = (
        sum(r["confidence"] for r in holdout if r["predicted"] != r["expected"])
        / max(1, sum(1 for r in holdout if r["predicted"] != r["expected"]))
    )
    print(f"  Avg confidence on correct:   {avg_conf_correct:.2f}")
    print(f"  Avg confidence on wrong:     {avg_conf_wrong:.2f}")
    print(f"  Gap:                         {avg_conf_correct - avg_conf_wrong:+.2f}")

    # Изоляция вкладов отдельных подходов
    print("\n--- Approach isolation (which signal caught the wrong predictions) ---")
    caught_by = {"vote_split": 0, "low_confidence": 0, "format": 0, "vote_disagree_scoring": 0}
    for r in holdout_caught:
        votes = r["votes"]
        valid_votes = [v for v in votes if v in ("search", "understand", "describe", "modify")]
        unique_votes = set(valid_votes)
        if not r["format_ok"]:
            caught_by["format"] += 1
        elif len(unique_votes) > 1:
            caught_by["vote_split"] += 1
        if r["confidence"] < config["threshold_ok"]:
            caught_by["low_confidence"] += 1
    print("  (a holdout error can be caught by multiple signals at once):")
    for sig, n in caught_by.items():
        print(f"  {sig:<25} {n}")

    # Per-class breakdown
    print("\n--- Per-class status (holdout) ---")
    by_class = defaultdict(lambda: Counter())
    for r in holdout:
        by_class[r["expected"]][r["status"]] += 1
    print(f"  {'class':<12} {'OK':>5} {'UNSURE':>7} {'FAIL':>5}")
    for cls in ("search", "understand", "describe", "modify"):
        c = by_class[cls]
        print(f"  {cls:<12} {c['OK']:>5} {c['UNSURE']:>7} {c['FAIL']:>5}")

    # Adversarial detailed
    print("\n--- Adversarial details ---")
    print(f"  {'kind':<28} {'status':<7} {'pred':<11} {'expected':<10} {'conf':<5}")
    for r in adv:
        print(f"  {r['kind']:<28} {r['status']:<7} {str(r['predicted']):<11} "
              f"{r['expected']:<10} {r['confidence']:.2f}")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
