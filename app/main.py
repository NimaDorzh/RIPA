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

from . import gate, presets, world
from .episode import run_episode
from .planner import MetaNotConfigured, MockPlanner, get_planner, meta_configured

log = logging.getLogger("hack-the-robot")

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = Path(__file__).resolve().parent / "preset_cache.json"
RATE_PER_MIN = int(os.getenv("RATE_PER_MIN", "10"))
PER_IP_DAILY = int(os.getenv("PER_IP_DAILY", "10"))
DAILY_CAP = int(os.getenv("DAILY_CAP", "100"))
# A live episode is ~12 model calls of 3-7 s each (about a minute) and ~12k reasoning tokens per mode,
# and Heroku's router cuts requests at 30 s. Live runs are therefore OFF unless ALLOW_LIVE=1.
ALLOW_LIVE = os.getenv("ALLOW_LIVE", "0") == "1"
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


@app.get("/play")
def play():
    return FileResponse(ROOT / "static" / "game.html")


@app.get("/api/health")
def health():
    cache = _load_cache()
    return {"ok": True, "meta_configured": meta_configured(), "live_enabled": ALLOW_LIVE,
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
    # "firewall" selects the right-hand side of the comparison. For text presets that is the full firewall; for
    # the camera presets it is camera + lidar fusion (see presets.py).
    side = presets.side_for(presets.BY_ID.get(req.preset_id) or {"sides": presets.TEXT_SIDES}, req.firewall)
    mode = side["mode"]
    scenario = None
    live_off = HTTPException(403, "Live runs are disabled on this deployment. Use the recorded presets.")
    if req.preset_id:
        p = presets.BY_ID.get(req.preset_id)
        if not p:
            raise HTTPException(404, "unknown preset")
        cached = _load_cache().get(f"{req.preset_id}|{mode}")
        if cached and cached.get("sign_text", p["sign"]) != p["sign"]:
            log.warning("cached trace for %s|%s was recorded with a different sign text; ignoring it", req.preset_id, mode)
            cached = None   # stale recording (the preset text changed): never show it as this preset
        if cached:
            return {**cached, "cached": True}
        if not ALLOW_LIVE:
            raise live_off
        sign = p["sign"]
        scenario = presets.scenario_for(p, side)
    elif req.sign_text is not None:
        if not ALLOW_LIVE:
            raise live_off
        sign = req.sign_text[:300]
    else:
        raise HTTPException(400, "preset_id or sign_text required")

    _limit(_client_ip(request))
    if not _slots.acquire(blocking=False):
        raise HTTPException(503, "Server busy, please retry in a few seconds.")
    try:
        log.info("live run sha256=%s mode=%s", hashlib.sha256((sign or "").encode()).hexdigest()[:16], mode)
        try:
            planner = get_planner(prefer_meta=True)
        except MetaNotConfigured:
            planner = MockPlanner()
        trace = run_episode(sign, mode, planner, max_seconds=LIVE_MAX_SECONDS, scenario=scenario)
    finally:
        _slots.release()
    return {**trace, "cached": False}


# ---------- "Break Muse" mini-game (patch 14): every answer is a recorded model reply, so clicks are free ----------
GATE_CACHE_PATH = Path(__file__).resolve().parent / "gate_cache.json"
_gate_state = {"mtime": None, "data": {}}


def _load_gate() -> dict:
    try:
        m = GATE_CACHE_PATH.stat().st_mtime
    except OSError:
        return {}
    if _gate_state["mtime"] != m:
        try:
            _gate_state["data"] = json.loads(GATE_CACHE_PATH.read_text())
        except (OSError, ValueError):
            log.warning("gate cache unreadable")
            _gate_state["data"] = {}
        _gate_state["mtime"] = m
    return _gate_state["data"]


AUTOPILOT_CACHE_PATH = Path(__file__).resolve().parent / "autopilot_cache.json"
DRIVE_CACHE_PATH = Path(__file__).resolve().parent / "drive_cache.json"
_auto_state = {"mtime": None, "data": {}}
_drive_state = {"mtime": None, "data": {}}


def _load_drive() -> dict:
    try:
        m = DRIVE_CACHE_PATH.stat().st_mtime
    except OSError:
        return {}
    if _drive_state["mtime"] != m:
        try:
            _drive_state["data"] = json.loads(DRIVE_CACHE_PATH.read_text())
        except (OSError, ValueError):
            _drive_state["data"] = {}
        _drive_state["mtime"] = m
    return _drive_state["data"]


def _load_auto() -> dict:
    try:
        m = AUTOPILOT_CACHE_PATH.stat().st_mtime
    except OSError:
        return {}
    if _auto_state["mtime"] != m:
        try:
            _auto_state["data"] = json.loads(AUTOPILOT_CACHE_PATH.read_text())
        except (OSError, ValueError):
            _auto_state["data"] = {}
        _auto_state["mtime"] = m
    return _auto_state["data"]


@app.get("/api/gate/config")
def gate_config():
    auto = _load_auto()
    current = {gate.key(*r): gate.drive_sha(*r) for r in gate.requests()}
    data = {k: v for k, v in _load_drive().items() if v.get("prompt_sha") == current.get(k)}   # the game uses driver mode
    any_rec = next(iter(data.values()), {})
    planners = {v.get("planner") for v in data.values()} | {v.get("planner") for v in auto.values()}
    return {**gate.public_config(), "recorded": len(data), "requests": len(gate.requests()),
            "autopilot_recorded": len(auto), "autopilot_starts": len(gate.autopilot_starts()),
            "planner": "mock" if "mock" in planners else any_rec.get("planner"),
            "model": any_rec.get("model"), "seed": any_rec.get("seed")}


class GateAuto(BaseModel):
    pos: list[int]


@app.post("/api/gate/autopilot")
def gate_autopilot(req: GateAuto):
    pos = tuple(req.pos)
    if pos not in gate.autopilot_starts():
        raise HTTPException(400, "invalid position")
    tr = _load_auto().get(gate.autopilot_key(pos))
    if not tr or tr.get("sign_text") != gate.SIGN:
        raise HTTPException(404, "The autopilot from this cell has not been recorded yet.")
    steps = [{k: s[k] for k in ("pos", "new_pos", "proposed", "executed", "event", "raw")} for s in tr["steps"]]
    return {"outcome": tr["outcome"], "broke": tr["outcome"] == "violation", "steps": steps,
            "planner": tr.get("planner"), "unsafe_proposals": tr.get("unsafe_proposals", 0)}


class GateMove(BaseModel):
    pos: list[int]
    dir: str


@app.post("/api/gate/move")
def gate_move(req: GateMove):
    pos = tuple(req.pos)
    if pos not in gate.floor_cells() or req.dir not in gate.DIRS:
        raise HTTPException(400, "invalid position or direction")
    target = world.next_cell(pos, req.dir)
    if not world.in_bounds(target) or target in world.SHELVES:
        return {"decision": "BLOCKED", "pos": list(pos), "target": list(target)}
    rec = _load_gate().get(gate.key(pos, req.dir))
    if rec and rec.get("prompt_sha") != gate.prompt_sha(pos, req.dir):
        rec = None   # recorded with an older checker prompt: never shown as this game's answer
    if not rec:
        raise HTTPException(404, "This move has not been recorded yet.")
    executed = rec["decision"] == "EXECUTE"
    return {"decision": rec["decision"], "raw": rec.get("raw"), "unsafe": rec["unsafe"],
            "broke": executed and rec["unsafe"], "pos": list(target if executed else pos), "target": list(target),
            "goal": executed and target == world.GOAL, "planner": rec.get("planner")}


@app.post("/api/gate/drive")
def gate_drive(req: GateMove):
    """The game's move: the player suggests a direction, Muse (driver, no defense) picks the actual move."""
    pos = tuple(req.pos)
    if pos not in gate.floor_cells() or req.dir not in gate.DIRS:
        raise HTTPException(400, "invalid position or direction")
    rec = _load_drive().get(gate.key(pos, req.dir))
    if rec and rec.get("prompt_sha") != gate.drive_sha(pos, req.dir):
        rec = None
    if not rec:
        raise HTTPException(404, "This move has not been recorded yet.")
    action = "STOP" if rec["action"] == "INVALID" else rec["action"]
    new_pos, event = world.execute(pos, action)
    return {"action": rec["action"], "suggested": req.dir, "followed": rec["action"] == req.dir, "raw": rec.get("raw"),
            "event": event, "pos": list(new_pos), "broke": event == "violation", "goal": new_pos == world.GOAL,
            "unsafe_suggestion": gate.unsafe(pos, req.dir), "target": list(world.next_cell(pos, req.dir)),
            "planner": rec.get("planner")}
