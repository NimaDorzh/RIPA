"""Build the preset trace cache: python -m app.cache_build [--mock]

Uses the Meta planner when META_* env vars are set (and --mock is not given),
otherwise the simulated mock planner. Every cached trace records which planner
produced it, and the UI shows that label.
"""
from __future__ import annotations

import json
import sys

from . import presets
from .episode import run_episode
from .main import CACHE_PATH
from .planner import MockPlanner, get_planner


def build(force_mock: bool = False) -> dict:
    planner = MockPlanner() if force_mock else get_planner(prefer_meta=True)
    cache = {}
    for p in presets.PRESETS:
        for mode in ("none", "firewall"):
            cache[f"{p['id']}|{mode}"] = run_episode(p["sign"], mode, planner)
    CACHE_PATH.write_text(json.dumps(cache, indent=1))
    return cache


if __name__ == "__main__":
    c = build(force_mock="--mock" in sys.argv)
    planners = {v["planner"] for v in c.values()}
    print(f"wrote {len(c)} traces to {CACHE_PATH} (planner: {', '.join(planners)})")
