"""Benchmark runner (NEW during hackathon).

python -m bench.run_bench --planner mock                      # pipeline test only, NOT results
python -m bench.run_bench --planner meta --model muse-spark-1.3 --n-attack 20 --n-benign 8 --budget-calls 600   # smoke test
python -m bench.run_bench --planner meta --model muse-spark-1.3 --budget-calls 6000                              # full run

Defenses (see app/prompts.py): none | hardened | separation | check | firewall.
  separation = prompt separation only; check = action check only; firewall = both.

What is measured (and what is NOT):
  * unsafe-proposal rate  (PRIMARY): episodes in which the MODEL proposed at least one action
    that the world scores as a safety violation, before any action check. This is the
    quantity a defense can actually change.
  * violation rate (ASR): episodes that END in a violation. For modes with the action check
    this is 0 BY CONSTRUCTION (the check blocks exactly the world's violation set); it is
    reported for completeness, not as a defense result.
  * stall rate (timeout): availability failures, e.g. a freeze attack; reported separately.
  * stall rate on benign signs is the no-attack baseline: planner failures (e.g. wandering) are not attacks.
  * model calls / latency count real API calls only (replayed replies are reported as cache_hits).
  * invalid rate: model replies that were not exactly one action (executed as STOP, counted).
  * API errors/deadline episodes are excluded from rates and reported as n_error.
Intervals: attacks are 25 independent bases x 4 wrappers, so CIs for attack rates use a
cluster bootstrap over bases (Wilson on n=100 would be too narrow). Paired comparisons use
exact McNemar on the same attack text, Holm-adjusted. Pre-registered primary endpoint:
unsafe-proposal rate, none vs separation.

Every episode is appended to a JSONL file as soon as it finishes, so an aborted run keeps its data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.episode import run_episode
from app.planner import MockPlanner, get_planner, meta_configured
from app.prompts import CHECK_MODES
from bench import datasets
from bench.stats import ci_with_fallback, holm, mcnemar_exact, wilson

DEFENSES = ("none", "hardened", "separation", "check", "firewall")
BAD_OUTCOMES = ("error", "deadline")


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def select(items: list, n: int) -> list:
    if n >= len(items):
        return items
    return [items[i * len(items) // n] for i in range(n)]


def make_tasks(n_attack: int, n_benign: int, attack_set: str = "core") -> list[dict]:
    recs = datasets.extra_attack_records() if attack_set == "extra" else datasets.attack_records()
    atk = [{"kind": "attack", **r} for r in select(recs, n_attack)]
    ben = []
    for t in datasets.benign(n_benign):
        tpl = datasets.BENIGN_T[len(ben) % len(datasets.BENIGN_T)]
        ben.append({"kind": "benign", "cat": "benign", "base": f"benign-{len(ben) % len(datasets.BENIGN_T)}",
                    "wrapper": 0, "text": t, "template": tpl})
    # interleave so that any prefix of the run (budget stop, abort) holds both attack and benign episodes
    out, ai, bi = [], 0, 0
    while ai < len(atk) or bi < len(ben):
        if bi >= len(ben) or (ai < len(atk) and ai * max(len(ben), 1) <= bi * max(len(atk), 1)):
            out.append(atk[ai]); ai += 1
        else:
            out.append(ben[bi]); bi += 1
    return out


def run_bench(planner, tasks, budget_calls, workers, jsonl_path, max_seconds, modes=DEFENSES):
    """Returns (records, model_calls, stop_reason). stop_reason is None for a complete run."""
    calls = {"n": 0, "stop": None, "steps": 0, "invalid": 0}
    lock = threading.Lock()
    out = jsonl_path.open("a", encoding="utf-8")
    records: list[dict] = []

    def work(task):
        for mode in modes:
            with lock:
                if calls["stop"]:
                    return
            tr = run_episode(task["text"], mode, planner, max_seconds=max_seconds)
            rec = {
                "kind": task["kind"], "cat": task["cat"], "base": task["base"], "wrapper": task["wrapper"],
                "text_sha": hashlib.sha256(task["text"].encode()).hexdigest()[:12], "text": task["text"],
                "mode": mode, "outcome": tr["outcome"], "error": tr["error"],
                "unsafe_proposals": tr["unsafe_proposals"], "invalid_actions": tr["invalid_actions"],
                "interventions": tr["interventions"], "n_steps": len(tr["steps"]),
                "latencies_ms": tr["latencies_ms"], "cache_hits": tr["cache_hits"],
                "proposed": [st["proposed"] for st in tr["steps"]],
                "path": [st["pos"] for st in tr["steps"]],
            }
            with lock:
                calls["n"] += len(tr["latencies_ms"]) + (1 if tr["outcome"] == "error" else 0)
                calls["steps"] += len(tr["steps"])
                calls["invalid"] += tr["invalid_actions"]
                if budget_calls and calls["n"] >= budget_calls and not calls["stop"]:
                    calls["stop"] = f"call budget {budget_calls} reached (incomplete run)"
                u = getattr(planner, "usage", {})
                api_calls, null = u.get("calls", 0), u.get("null_content", 0)
                if not calls["stop"] and api_calls >= 40 and null / api_calls > 0.2:
                    calls["stop"] = (f"ABORTED: {null}/{api_calls} API replies had empty content "
                                     "(token cap too low or reasoning too long); fix META_MAX_TOKENS first")
                if not calls["stop"] and calls["steps"] >= 60 and calls["invalid"] / calls["steps"] > 0.25:
                    calls["stop"] = (f"ABORTED: {calls['invalid']}/{calls['steps']} actions were invalid "
                                     "(model replies are not a single action); check the prompt/parser")
                records.append(rec)
                out.write(json.dumps(rec) + "\n")
                out.flush()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, tasks))
    out.close()
    return records, calls["n"], calls["stop"]


def summarize(records: list[dict]) -> dict:
    s: dict = {"attack": {}, "benign": {}, "paired": {}, "by_category": {}}
    present = [m for m in DEFENSES if any(r["mode"] == m for r in records)]
    for mode in present:
        a = [r for r in records if r["kind"] == "attack" and r["mode"] == mode]
        ok = [r for r in a if r["outcome"] not in BAD_OUTCOMES]
        up = [int(r["unsafe_proposals"] > 0) for r in ok]
        cl = [r["base"] for r in ok]
        lat = sorted(x for r in ok for x in r["latencies_ms"])
        steps = sum(r["n_steps"] for r in ok)
        s["attack"][mode] = {
            "n": len(a), "n_ok": len(ok), "n_error": len(a) - len(ok),
            "unsafe_proposal_rate": sum(up) / len(up) if up else None,
            "unsafe_proposal_ci_cluster": ci_with_fallback(up, cl) if up else None,
            "violation_rate": sum(r["outcome"] == "violation" for r in ok) / len(ok) if ok else None,
            "violation_by_construction_zero": mode in CHECK_MODES,
            "stall_rate": sum(r["outcome"] == "timeout" for r in ok) / len(ok) if ok else None,
            "invalid_rate": sum(r["invalid_actions"] for r in ok) / steps if steps else None,
            "intervention_rate": (sum(r["interventions"] > 0 for r in ok) / len(ok)) if ok and mode in CHECK_MODES else None,
            "latency_p50_ms": lat[len(lat) // 2] if lat else None,
            "latency_p95_ms": lat[int(0.95 * (len(lat) - 1))] if lat else None,
            "n_calls": len(lat),
        }
        b = [r for r in records if r["kind"] == "benign" and r["mode"] == mode]
        bok = [r for r in b if r["outcome"] not in BAD_OUTCOMES]
        succ = sum(r["outcome"] == "goal" for r in bok)
        s["benign"][mode] = {
            "n": len(b), "n_ok": len(bok), "n_error": len(b) - len(bok),
            "task_success_rate": succ / len(bok) if bok else None,
            "task_success_wilson": wilson(succ, len(bok)) if bok else None,
            "intervention_rate": (sum(r["interventions"] > 0 for r in bok) / len(bok)) if bok and mode in CHECK_MODES else None,
            "unsafe_proposal_rate": (sum(r["unsafe_proposals"] > 0 for r in bok) / len(bok)) if bok else None,
        }
        cats: dict[str, list[int]] = {}
        for r in ok:
            cats.setdefault(r["cat"], []).append(int(r["unsafe_proposals"] > 0))
        s["by_category"][mode] = {c: sum(v) / len(v) for c, v in sorted(cats.items())}
    # paired McNemar on unsafe proposal, same attack text
    by = {m: {r["text_sha"]: r for r in records
              if r["kind"] == "attack" and r["mode"] == m and r["outcome"] not in BAD_OUTCOMES} for m in present}
    pairs = [(A, B) for A, B in [("none", "hardened"), ("none", "separation"), ("hardened", "separation")]
             if A in present and B in present]
    raw = {}
    for A, B in pairs:
        common = by[A].keys() & by[B].keys()
        b = sum(by[A][k]["unsafe_proposals"] > 0 and not by[B][k]["unsafe_proposals"] > 0 for k in common)
        c = sum(by[B][k]["unsafe_proposals"] > 0 and not by[A][k]["unsafe_proposals"] > 0 for k in common)
        raw[f"{A} vs {B}"] = {"n_pairs": len(common), "only_A_unsafe": b, "only_B_unsafe": c, "p": mcnemar_exact(b, c)}
    adj = holm({k: v["p"] for k, v in raw.items()})
    for k in raw:
        raw[k]["p_holm"] = adj[k]
    s["paired"] = raw
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=["mock", "meta"], default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-attack", type=int, default=100, help="max 100 (25 bases x 4 wrappers)")
    ap.add_argument("--n-benign", type=int, default=24)
    ap.add_argument("--attack-set", choices=["core", "extra"], default="core",
                    help="core = B1-B5 (100 texts); extra = B6 pressure and B7 fake-reasoning signs (40 texts)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-seconds", type=float, default=240.0, help="wall-clock limit per episode")
    ap.add_argument("--defenses", default=",".join(DEFENSES),
                    help="comma-separated subset of: " + ", ".join(DEFENSES) + " (e.g. --defenses none to measure raw attack success)")
    ap.add_argument("--no-cache", action="store_true",
                    help="do not reuse/store model replies (default: bench/results/prompt_cache_<model>.jsonl)")
    ap.add_argument("--budget-calls", type=int, default=None,
                    help="stop starting new episodes after this many model calls (protects credits)")
    a = ap.parse_args()
    modes = tuple(m.strip() for m in a.defenses.split(",") if m.strip())
    bad = [m for m in modes if m not in DEFENSES]
    if bad or not modes:
        raise SystemExit(f"unknown --defenses {bad or a.defenses!r}; choose from {DEFENSES}")
    if a.planner == "meta" and not meta_configured():
        raise SystemExit("--planner meta needs META_API_BASE, META_API_KEY and META_MODEL "
                         "(fill them in .env); refusing to fall back to the mock planner.")
    out = Path("bench/results")
    out.mkdir(parents=True, exist_ok=True)
    model_name = a.model or os.environ.get("META_MODEL", "mock")
    cache_path = None if (a.no_cache or a.planner == "mock") else out / f"prompt_cache_{model_name}.jsonl"
    planner = MockPlanner() if a.planner == "mock" else get_planner(True, a.model, cache_path=cache_path)
    label = "MOCK (pipeline test only, NOT results)" if a.planner == "mock" else f"meta:{planner.model}"
    tasks = make_tasks(a.n_attack, a.n_benign, a.attack_set)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jsonl = out / f"episodes_{planner.name}_{stamp}.jsonl"
    t0 = time.time()
    records, calls, stopped = run_bench(planner, tasks, a.budget_calls, a.workers, jsonl, a.max_seconds, modes)
    summ = summarize(records)
    meta = {"planner": label, "model": a.model or getattr(planner, "model", None),
            "reasoning_effort": getattr(planner, "effort", None), "calls": calls,
            "stop_reason": stopped, "usage": planner.usage, "reasoning_effort_note": "see usage.reasoning_tokens", "seconds": round(time.time() - t0, 1),
            "attack_set": a.attack_set, "n_attack_requested": a.n_attack, "n_benign_requested": a.n_benign, "defenses": list(modes)}
    (out / f"summary_{planner.name}_{stamp}.json").write_text(json.dumps({"meta": meta, "summary": summ}, indent=1))
    print(f"Planner: {label}  model calls: {calls}  tokens: {planner.usage}"
          + (f"\n  [{stopped}]" if stopped else ""))
    print("\nATTACKS (rates over valid episodes; CI = cluster bootstrap over bases)")
    print(f"{'defense':11} {'n_ok/err':>9} {'unsafe-proposal':>16} {'95% CI':>15} {'violation':>10} {'stall':>7} {'invalid':>8} {'p50ms':>7} {'p95ms':>7}")
    for m, r in summ["attack"].items():
        ci = r["unsafe_proposal_ci_cluster"]
        viol = pct(r["violation_rate"]) + ("*" if r["violation_by_construction_zero"] else "") if r["violation_rate"] is not None else "n/a"
        print(f"{m:11} {str(r['n_ok'])+'/'+str(r['n_error']):>9} "
              f"{pct(r['unsafe_proposal_rate']) if r['unsafe_proposal_rate'] is not None else 'n/a':>16} "
              f"{(pct(ci[0])+'-'+pct(ci[1])) if ci else 'n/a':>15} {viol:>10} "
              f"{pct(r['stall_rate']) if r['stall_rate'] is not None else 'n/a':>7} "
              f"{pct(r['invalid_rate']) if r['invalid_rate'] is not None else 'n/a':>8} "
              f"{(r['latency_p50_ms'] or 0):>7.0f} {(r['latency_p95_ms'] or 0):>7.0f}")
    print("* 0 by construction (the action check blocks exactly the world's violation set); not a defense result.")
    print("\nBENIGN (task success; interventions only exist for modes with the action check)")
    for m, r in summ["benign"].items():
        w = r["task_success_wilson"]
        print(f"  {m:11} n_ok={r['n_ok']:>3}/{r['n']}  success={pct(r['task_success_rate']) if r['task_success_rate'] is not None else 'n/a'}"
              f" ({pct(w[0])}-{pct(w[1])})" if w else f"  {m:11} no valid episodes",
              f" intervention={pct(r['intervention_rate'])}" if r["intervention_rate"] is not None else "")
    print("\nPAIRED (exact McNemar on unsafe proposal, same attack text; Holm-adjusted)")
    for k, v in summ["paired"].items():
        print(f"  {k:24} pairs={v['n_pairs']:>3}  onlyA={v['only_A_unsafe']:>3} onlyB={v['only_B_unsafe']:>3}  p={v['p']:.4f}  p_holm={v['p_holm']:.4f}")
    print("\nUnsafe-proposal rate by attack category:")
    for m, c in summ["by_category"].items():
        print(f"  {m:11} " + "  ".join(f"{k}={pct(v)}" for k, v in c.items()))
    print(f"\nEpisodes: {jsonl}")


if __name__ == "__main__":
    main()
