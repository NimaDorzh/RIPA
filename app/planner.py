"""Planners (NEW during hackathon).

MetaPlanner  : calls the Meta Model API (endpoint/model configured via env).
MockPlanner  : SIMULATED planner for offline development and tests only.
               Its numbers are invented parameters and must NEVER be reported
               as results. The UI and trace label every run with its planner.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time

import httpx

from . import world
from .prompts import build_messages

FOLLOW_RE = re.compile(
    r"(ignore|override|proceed|shortcut|speed up|go straight|path (is )?clear|"
    r"no (obstacles|people)|skip|restricted zone is open|fastest way)",
    re.I,
)
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
# Invented parameters of the simulated planner (not measurements).
MOCK_FOLLOW_PROB = {"none": 0.95, "hardened": 0.40, "firewall": 0.40}


class MetaNotConfigured(RuntimeError):
    pass


def meta_configured() -> bool:
    return bool(os.getenv("META_API_BASE") and os.getenv("META_API_KEY") and os.getenv("META_MODEL"))


def _parse_action(text: str) -> str:
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            a = str(json.loads(m.group(0)).get("action", "")).upper()
            if a in world.ACTIONS:
                return a
        except json.JSONDecodeError:
            pass
    for a in world.ACTIONS:
        if re.search(rf"\b{a}\b", text.upper()):
            return a
    return "STOP"


class MetaPlanner:
    name = "meta"

    def __init__(self, model: str | None = None):
        if not meta_configured():
            raise MetaNotConfigured("Set META_API_BASE, META_API_KEY, META_MODEL (see docs/api-notes.md)")
        self.base = os.environ["META_API_BASE"].rstrip("/")
        self.key = os.environ["META_API_KEY"]
        self.model = model or os.environ["META_MODEL"]
        self.last_latency_ms = 0.0

    def plan(self, obs: dict, mode: str) -> str:
        # NOTE: assumes an OpenAI-compatible chat/completions route. Verify in the API probe
        # and adjust here if the Meta Model API differs.
        t0 = time.perf_counter()
        r = httpx.post(
            f"{self.base}/chat/completions",
            headers={"Authorization": f"Bearer {self.key}"},
            json={"model": self.model, "messages": build_messages(obs, mode),
                  "temperature": 0, "max_tokens": 40},
            timeout=30,
        )
        r.raise_for_status()
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        text = r.json()["choices"][0]["message"]["content"]
        return _parse_action(text)


class MockPlanner:
    name = "mock"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.last_latency_ms = 0.0

    def _rng(self, obs: dict, mode: str) -> random.Random:
        key = f"{self.seed}|{obs['pos']}|{obs.get('sign')}|{mode}"
        return random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32))

    def plan(self, obs: dict, mode: str) -> str:
        pos = tuple(obs["pos"])
        sign = (obs.get("sign") or "").translate(LEET)
        if sign and FOLLOW_RE.search(sign) and self._rng(obs, mode).random() < MOCK_FOLLOW_PROB[mode]:
            gx, gy = world.GOAL  # greedy, hazard-blind move toward the dock
            if pos[0] < gx:
                return "RIGHT"
            if pos[1] != gy:
                return "DOWN" if pos[1] < gy else "UP"
            return "STOP"
        return world.safe_step(pos)


def get_planner(prefer_meta: bool = True, model: str | None = None):
    if prefer_meta and meta_configured():
        return MetaPlanner(model)
    return MockPlanner()
