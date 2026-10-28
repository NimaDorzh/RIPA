"""Build the preset trace cache.

python -m app.cache_build                    # all presets (needs META_* env vars)
python -m app.cache_build --only map-errata,cam-spoof,cam-honest   # only these; the rest of the cache is kept
python -m app.cache_build --mock             # offline stand-in traces (UI development only, labelled as such)

Uses the Meta planner when META_* env vars are set (and --mock is not given). Every cached trace records which
planner produced it, and the UI shows that label. Traces with outcome error/deadline are never written: the
previous cache is kept instead.

The trace shown for a preset is always the one recorded with decoding seed 1 (chosen in advance, not picked after
the fact). Presets with `replicates: N` are additionally rerun with seeds 1..N and the per-seed results are stored
in trace["replicates"], so the UI can state how reproducible the shown run is.
"""
from __future__ import annotations

import json
import sys

from . import presets
from .episode import run_episode
from .main import CACHE_PATH
from .planner import MockPlanner, get_planner, meta_configured


def _run(p: dict, side: dict, planner, seed: int | None = None) -> dict:
    if seed is not None and hasattr(planner, "seed"):
        planner.seed = seed          # part of the reply-memo key, so seeds never share replies
    return run_episode(p["sign"], side["mode"], planner, scenario=presets.scenario_for(p, side))


def build(force_mock: bool = False, only: set[str] | None = None) -> dict:
    if not force_mock and not meta_configured():
        raise SystemExit("META_API_BASE / META_API_KEY / META_MODEL are not set (see .env.example). "
                         "Use --mock only for offline stand-in traces.")
    planner = MockPlanner() if force_mock else get_planner(prefer_meta=True)
    base_seed = getattr(planner, "seed", None)
    cache: dict = {}
    if only:
        unknown = only - set(presets.BY_ID)
        if unknown:
            raise SystemExit(f"unknown preset id(s): {sorted(unknown)}; known: {sorted(presets.BY_ID)}")
        try:
            cache = json.loads(CACHE_PATH.read_text())
        except (OSError, ValueError):
            cache = {}
    for p in presets.PRESETS:
        if only and p["id"] not in only:
            continue
        for right in (False, True):
            side = presets.side_for(p, right)
            if base_seed is not None:
                planner.seed = base_seed
            tr = _run(p, side, planner)
            if tr["outcome"] in ("error", "deadline"):
                raise SystemExit(f"{p['id']}|{side['mode']}: {tr['outcome']} ({tr['error']}); cache NOT overwritten")
            n_rep = int(p.get("replicates", 0))
            if n_rep and not force_mock:
                rep = {"seeds": [], "unsafe": [], "violation": []}
                for s in range(1, n_rep + 1):
                    r = _run(p, side, planner, seed=s)
                    if r["outcome"] in ("error", "deadline"):
                        raise SystemExit(f"{p['id']}|{side['mode']} seed {s}: {r['outcome']} ({r['error']}); cache NOT overwritten")
                    rep["seeds"].append(s)
                    rep["unsafe"].append(int(r["unsafe_proposals"] > 0))
                    rep["violation"].append(int(r["outcome"] == "violation"))
                tr["replicates"] = rep
                if base_seed is not None:
                    planner.seed = base_seed
            cache[f"{p['id']}|{side['mode']}"] = tr
            print(f"  {p['id']:18} {side['label']:24} outcome={tr['outcome']:9} unsafe_proposals={tr['unsafe_proposals']}"
                  + (f"  replicates unsafe={tr['replicates']['unsafe']}" if "replicates" in tr else ""), flush=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=1))
    return cache


if __name__ == "__main__":
    only = None
    if "--only" in sys.argv:
        only = {x.strip() for x in sys.argv[sys.argv.index("--only") + 1].split(",") if x.strip()}
    c = build(force_mock="--mock" in sys.argv, only=only)
    planners = {v["planner"] for v in c.values()}
    print(f"wrote {len(c)} traces to {CACHE_PATH} (planner: {', '.join(planners)})")
