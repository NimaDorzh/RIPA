"""Episode runner (NEW during hackathon)."""
from __future__ import annotations

import time

from . import firewall, world


def run_episode(sign_text: str | None, mode: str, planner) -> dict:
    """mode in {"none","hardened","firewall"}. Returns a full trace."""
    pos = world.START
    steps = []
    outcome = "timeout"
    interventions = 0
    latencies = []
    for t in range(world.MAX_STEPS):
        obs = world.observe(pos, sign_text)
        t0 = time.perf_counter()
        proposed = planner.plan(obs, mode)
        verdict = None
        executed = proposed
        if mode == "firewall":
            verdict = firewall.check_action(obs, proposed)
            if not verdict["allowed"]:
                executed = firewall.fallback_action(obs)
                interventions += 1
        latencies.append((time.perf_counter() - t0) * 1000)
        new_pos, event = world.execute(pos, executed)
        steps.append({
            "t": t, "pos": list(pos), "sign": obs["sign"], "detections": obs["detections"],
            "proposed": proposed, "verdict": verdict, "executed": executed,
            "event": event, "new_pos": list(new_pos),
        })
        pos = new_pos
        if event == "violation":
            outcome = "violation"
            break
        if pos == world.GOAL:
            outcome = "goal"
            break
    return {
        "outcome": outcome,
        "planner": planner.name,
        "mode": mode,
        "interventions": interventions,
        "latency_ms_p50": sorted(latencies)[len(latencies) // 2] if latencies else 0,
        "steps": steps,
    }
