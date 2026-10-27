"""FastAPI backend (NEW during hackathon).

Public-playground protections: presets are served from recorded traces (no model call);
live runs (free-text signs, or presets missing from the cache) are limited per IP per
minute, per IP per day, globally per day, and by a concurrency cap. The Meta key lives
only in the server environment. Live sign text is logged as a SHA-256 hash only.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
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

log = logging.getLogger("hack-the-robot")

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = Path(__file__).resolve().parent / "preset_cache.json"
RATE_PER_MIN = int(os.getenv("RATE_PER_MIN", "10"))
PER_IP_DAILY = int(os.getenv("PER_IP_DAILY", "30"))
DAILY_CAP = int(os.getenv("DAILY_CAP", "500"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "4"))
LIVE_MAX_SECONDS = float(os.getenv("LIVE_MAX_SECONDS", "25"))
# Behind Heroku's router the LAST X-Forwarded-For entry is the address the router saw.
# Only trust it when explicitly enabled; otherwise every visitor would share the proxy IP.
TRUST_PROXY = os.getenv("TRUST_PROXY", "0") == "1"

app = FastAPI(title="Hack the Robot")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

_lock = threading.Lock()
_hits: dict[str, deque] = defaultdict(deque)
_ip_daily: dict[str, int] = defaultdict(int)
_daily = {"day": time.strftime("%Y-%m-%d"), "n": 0}
_slots = threading.BoundedSemaphore(MAX_CONCURRENT)
_cache_state = {"mtime": None, "data": {}}


def _mode(firewall: bool) -> str:
    return "firewall" if firewall else "none"


def _load_cache() -> dict:
    try:
        m = CACHE_PATH.stat().st_mtime
    except OSError:
        return {}
    if _cache_state["mtime"] != m:
        try:
            _cache_state["data"] = json.loads(CACHE_PATH.read_text())
        except (OSError, ValueError):
            log.warning("preset cache unreadable; falling back to live runs")
            _cache_state["data"] = {}
        _cache_state["mtime"] = m
    return _cache_state["data"]


def _client_ip(request: Request) -> str:
    if TRUST_PROXY:
        xff = request.headers.get("x-forwarded-for", "")
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            return parts[-1]
    return request.client.host if request.client else "anon"


def _limit(ip: str) -> None:
    now = time.time()
    today = time.strftime("%Y-%m-%d")
    with _lock:
        if _daily["day"] != today:
            _daily.update(day=today, n=0)
            _ip_daily.clear()
        if len(_hits) > 5000:  # bound memory against spoofed/rotating addresses
            for k in [k for k, q in _hits.items() if not q or now - q[-1] > 60]:
                del _hits[k]
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_PER_MIN:
            raise HTTPException(429, "Too many live runs, try the preset buttons or wait a minute.")
        if _ip_daily[ip] >= PER_IP_DAILY:
            raise HTTPException(429, "Daily live-run limit for your address reached. Presets still work.")
        if _daily["n"] >= DAILY_CAP:
            raise HTTPException(429, "Daily live-run budget reached. Presets still work (recorded runs).")
        q.append(now)
        _ip_daily[ip] += 1
        _daily["n"] += 1


class RunRequest(BaseModel):
    preset_id: str | None = None
    sign_text: str | None = None
    firewall: bool = False
    live: bool = False  # kept for API compatibility; ignored (presets are always served from recordings)


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    cache = _load_cache()
    return {"ok": True, "meta_configured": meta_configured(),
            "cache_planners": sorted({v.get("planner", "?") for v in cache.values()})}


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
        cached = _load_cache().get(f"{req.preset_id}|{mode}")
        if cached:
            return {**cached, "cached": True}
        sign = p["sign"]
    elif req.sign_text is not None:
        sign = req.sign_text[:300]
    else:
        raise HTTPException(400, "preset_id or sign_text required")

    _limit(_client_ip(request))
    if not _slots.acquire(blocking=False):
        raise HTTPException(503, "Server busy, please retry in a few seconds.")
    try:
        log.info("live run sha256=%s mode=%s", hashlib.sha256(sign.encode()).hexdigest()[:16], mode)
        try:
            planner = get_planner(prefer_meta=True)
        except MetaNotConfigured:
            planner = MockPlanner()
        trace = run_episode(sign, mode, planner, max_seconds=LIVE_MAX_SECONDS)
    finally:
        _slots.release()
    return {**trace, "cached": False}
