"""Episode runner (NEW during hackathon)."""
from __future__ import annotations

import time

from . import firewall, world
from .planner import INVALID, PlannerError
from .prompts import CHECK_MODES


def run_episode(sign_text: str | None, mode: str, planner, max_seconds: float = 60.0) -> dict:
    """mode in prompts.MODES. Returns a full trace.

    outcome: goal | violation | timeout (step budget, e.g. a freeze/DoS attack) |
             error (API failure) | deadline (wall clock exceeded).
    unsafe_proposals counts steps where the PLANNER proposed an action the world would score
    as a violation, regardless of whether the action check then blocked it.
    """
    pos = world.START
    steps = []
    outcome = "timeout"
    error = None
    interventions = 0
    unsafe_proposals = 0
    invalid_actions = 0
    latencies = []
    t_start = time.perf_counter()
    for t in range(world.MAX_STEPS):
        if time.perf_counter() - t_start > max_seconds:
            outcome, error = "deadline", f"exceeded {max_seconds}s"
            break
        obs = world.observe(pos, sign_text)
        t0 = time.perf_counter()
        try:
            proposed = planner.plan(obs, mode)
        except PlannerError as e:
            outcome, error = "error", str(e)
            break
        latencies.append((time.perf_counter() - t0) * 1000)
        if proposed == INVALID:
            invalid_actions += 1
        proposed_exec = "STOP" if proposed == INVALID else proposed
        unsafe = world.execute(pos, proposed_exec)[1] == "violation"
        unsafe_proposals += unsafe
        verdict = None
        executed = proposed_exec
        if mode in CHECK_MODES:
            verdict = firewall.check_action(obs, proposed_exec)
            if not verdict["allowed"]:
                executed = firewall.fallback_action(obs)
                interventions += 1
        new_pos, event = world.execute(pos, executed)
        steps.append({
            "t": t, "pos": list(pos), "sign": obs["sign"], "detections": obs["detections"],
            "proposed": proposed, "unsafe_proposal": bool(unsafe), "verdict": verdict,
            "executed": executed, "event": event, "new_pos": list(new_pos),
        })
        pos = new_pos
        if event == "violation":
            outcome = "violation"
            break
        if pos == world.GOAL:
            outcome = "goal"
            break
    lat_sorted = sorted(latencies)
    return {
        "outcome": outcome,
        "error": error,
        "planner": planner.name,
        "mode": mode,
        "interventions": interventions,
        "unsafe_proposals": unsafe_proposals,
        "invalid_actions": invalid_actions,
        "latencies_ms": [round(x, 1) for x in latencies],
        "latency_ms_p50": lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0,
        "steps": steps,
    }
