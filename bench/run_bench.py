"""Benchmark runner (NEW during hackathon).

python -m bench.run_bench --planner mock   # pipeline test only, NOT results
python -m bench.run_bench --planner meta --model <id> --n-attack 100 --n-benign 200

Defenses: none | hardened (prompt rule only) | firewall (separation + action check).
Metrics: ASR (attack episodes ending in a safety violation), FPR (benign episodes in
which the firewall intervened or the task failed), latency p50/p95, Wilson 95% CIs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.episode import run_episode
from app.planner import MockPlanner, get_planner
from bench import datasets
from bench.stats import wilson

DEFENSES = ("none", "hardened", "firewall")


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def run(planner, n_attack: int, n_benign: int, budget_calls: int | None):
    rows = {}
    calls = 0
    for mode in DEFENSES:
        k = 0
        lat = []
        by_cat: dict[str, list[int]] = {}
        for cat, sign in datasets.attacks(n_attack):
            tr = run_episode(sign, mode, planner)
            calls += len(tr["steps"])
            hit = tr["outcome"] == "violation"
            k += hit
            by_cat.setdefault(cat, []).append(int(hit))
            lat.append(tr["latency_ms_p50"])
            if budget_calls and calls > budget_calls:
                raise SystemExit(f"call budget {budget_calls} exceeded, stopping")
        bad = 0
        for sign in datasets.benign(n_benign):
            tr = run_episode(sign, mode, planner)
            calls += len(tr["steps"])
            bad += (tr["interventions"] > 0) or (tr["outcome"] != "goal")
            if budget_calls and calls > budget_calls:
                raise SystemExit(f"call budget {budget_calls} exceeded, stopping")
        lat.sort()
        rows[mode] = {
            "asr": k / n_attack, "asr_ci": wilson(k, n_attack), "n_attack": n_attack,
            "fpr": bad / n_benign, "fpr_ci": wilson(bad, n_benign), "n_benign": n_benign,
            "latency_p50_ms": lat[len(lat) // 2], "latency_p95_ms": lat[int(0.95 * (len(lat) - 1))],
            "asr_by_category": {c: sum(v) / len(v) for c, v in sorted(by_cat.items())},
        }
    return rows, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=["mock", "meta"], default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-attack", type=int, default=100)
    ap.add_argument("--n-benign", type=int, default=200)
    ap.add_argument("--budget-calls", type=int, default=None,
                    help="abort if more model calls than this (protects credits)")
    a = ap.parse_args()
    planner = MockPlanner() if a.planner == "mock" else get_planner(True, a.model)
    rows, calls = run(planner, a.n_attack, a.n_benign, a.budget_calls)
    label = planner.name if a.planner == "meta" else "MOCK (pipeline test only, not results)"
    out = Path("bench/results"); out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (out / f"bench_{planner.name}_{stamp}.json").write_text(
        json.dumps({"planner": label, "model": a.model, "calls": calls, "rows": rows}, indent=1))
    print(f"Planner: {label}   model calls: {calls}")
    print(f"{'defense':10} {'ASR':>8} {'ASR 95% CI':>16} {'FPR':>8} {'FPR 95% CI':>16} {'p50 ms':>8} {'p95 ms':>8}")
    for m, r in rows.items():
        print(f"{m:10} {pct(r['asr']):>8} {pct(r['asr_ci'][0])+'-'+pct(r['asr_ci'][1]):>16} "
              f"{pct(r['fpr']):>8} {pct(r['fpr_ci'][0])+'-'+pct(r['fpr_ci'][1]):>16} "
              f"{r['latency_p50_ms']:>8.2f} {r['latency_p95_ms']:>8.2f}")
    print("\nASR by attack category (B1 replace, B2 false correction, B3 obfuscation, B4 authority, B5 false facts):")
    for m, r in rows.items():
        print(f"  {m:10} " + "  ".join(f"{c}={pct(v)}" for c, v in r["asr_by_category"].items()))


if __name__ == "__main__":
    main()
