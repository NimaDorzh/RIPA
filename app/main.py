"""FastAPI backend (NEW during hackathon)."""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import presets, world
from .episode import run_episode
from .planner import MetaNotConfigured, MockPlanner, get_planner, meta_configured

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = Path(__file__).resolve().parent / "preset_cache.json"
RATE_PER_MIN = int(os.getenv("RATE_PER_MIN", "10"))
DAILY_CAP = int(os.getenv("DAILY_CAP", "500"))

app = FastAPI(title="Hack the Robot")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

_hits: dict[str, deque] = defaultdict(deque)
_daily = {"day": time.strftime("%Y-%m-%d"), "n": 0}


def _mode(firewall: bool) -> str:
    return "firewall" if firewall else "none"


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _limit(ip: str) -> None:
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_PER_MIN:
        raise HTTPException(429, "Too many live runs, try the preset buttons or wait a minute.")
    today = time.strftime("%Y-%m-%d")
    if _daily["day"] != today:
        _daily.update(day=today, n=0)
    if _daily["n"] >= DAILY_CAP:
        raise HTTPException(429, "Daily live-run budget reached. Presets still work (recorded runs).")
    q.append(now)
    _daily["n"] += 1


class RunRequest(BaseModel):
    preset_id: str | None = None
    sign_text: str | None = None
    firewall: bool = False
    live: bool = False  # force a live (non-cached) run


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "meta_configured": meta_configured()}


@app.get("/api/world")
def get_world():
    return {
        "w": world.W, "h": world.H, "start": world.START, "goal": world.GOAL,
        "shelves": sorted(world.SHELVES), "human_zone": sorted(world.HUMAN_ZONE),
        "people": world.PEOPLE, "sign_pos": world.SIGN_POS,
    }


@app.get("/api/presets")
def get_presets():
    return presets.PRESETS


@app.post("/api/run")
def run(req: RunRequest, request: Request):
    mode = _mode(req.firewall)
    if req.preset_id:
        p = presets.BY_ID.get(req.preset_id)
        if not p:
            raise HTTPException(404, "unknown preset")
        cache = _load_cache()
        key = f"{req.preset_id}|{mode}"
        if not req.live and key in cache:
            return {**cache[key], "cached": True}
        sign = p["sign"]
    elif req.sign_text is not None:
        sign = req.sign_text[:300]
        req.live = True
    else:
        raise HTTPException(400, "preset_id or sign_text required")

    _limit(request.client.host if request.client else "anon")
    try:
        planner = get_planner(prefer_meta=True)
    except MetaNotConfigured:
        planner = MockPlanner()
    trace = run_episode(sign, mode, planner)
    return {**trace, "cached": False}
