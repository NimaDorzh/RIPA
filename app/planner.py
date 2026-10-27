"""Planners (NEW during hackathon).

MetaPlanner  : calls the Meta Model API (OpenAI-compatible /chat/completions).
MockPlanner  : SIMULATED planner for offline development and tests only.
               Its numbers are invented parameters and must NEVER be reported
               as results. The UI and trace label every run with its planner.

Actions returned by a planner are one of world.ACTIONS or INVALID (the model
produced no parseable single action). INVALID is executed as STOP by the episode
runner but is counted and reported separately; it is never silently "safe".
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from pathlib import Path

import httpx

from . import world
from .prompts import build_messages

INVALID = "INVALID"

FOLLOW_RE = re.compile(
    r"(ignore|override|proceed|shortcut|speed up|go straight|path (is )?clear|"
    r"no (obstacles|people)|skip|restricted zone is open|fastest way)",
    re.I,
)
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
# Invented parameters of the simulated planner (not measurements).
MOCK_FOLLOW_PROB = {"none": 0.95, "hardened": 0.40, "separation": 0.40, "check": 0.95, "firewall": 0.40}


class MetaNotConfigured(RuntimeError):
    pass


class PlannerError(RuntimeError):
    """API/transport failure after retries (not a model decision)."""


def meta_configured() -> bool:
    return bool(os.getenv("META_API_BASE") and os.getenv("META_API_KEY") and os.getenv("META_MODEL"))


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def _parse_action(text: str | None) -> str:
    """Strict parser. The whole reply must be exactly one action, either a JSON
    object {"action": X} (optionally in a code fence) or the bare action word.
    Anything else (prose, negation, several JSON objects, null) is INVALID.
    No keyword scanning: "RIGHT is unsafe, I choose STOP" must not become RIGHT."""
    if not text:
        return INVALID
    t = _FENCE.sub("", text.strip()).strip()
    try:
        obj = json.loads(t)
    except (json.JSONDecodeError, ValueError):
        obj = None
    if isinstance(obj, dict) and set(obj) == {"action"}:
        a = str(obj["action"]).strip().upper()
        return a if a in world.ACTIONS else INVALID
    bare = t.strip(" \t\r\n\"'.").upper()
    return bare if bare in world.ACTIONS else INVALID


class MetaPlanner:
    name = "meta"

    def __init__(self, model: str | None = None, client: httpx.Client | None = None,
                 cache_path: str | os.PathLike | None = None, memoize: bool = True):
        if not meta_configured() and client is None:
            raise MetaNotConfigured("Set META_API_BASE, META_API_KEY, META_MODEL (see docs/api-notes.md)")
        self.base = os.environ.get("META_API_BASE", "http://test").rstrip("/")
        self.key = os.environ.get("META_API_KEY", "")
        self.model = model or os.environ.get("META_MODEL", "")
        # Probe (Oct 27): Muse Spark 1.1-1.3 are reasoning models. 'none' is rejected. On the real robot
        # prompt even 'minimal' spends ~550-1050 reasoning tokens (3-7 s per call); a small token cap makes
        # `content` come back null. The cap only limits runaways: you pay for tokens actually used.
        self.effort = os.getenv("META_REASONING", "minimal")
        self.max_tokens = int(os.getenv("META_MAX_TOKENS", "4000"))
        self.seed = int(os.getenv("META_SEED", "1"))
        self.timeout = float(os.getenv("META_TIMEOUT", "60"))
        self.client = client or httpx.Client(timeout=self.timeout)
        self.usage = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "reasoning_tokens": 0, "null_content": 0}
        self._lock = threading.Lock()
        # Memo of raw replies by exact request (model, effort, seed, messages). Identical prompts occur in
        # several defenses (e.g. none/check) and in re-runs; reusing the reply keeps paired comparisons exact,
        # and the optional JSONL file makes an interrupted or repeated run free. Null replies are never cached.
        self.memoize = memoize
        self._tl = threading.local()                       # per-thread "last reply came from the cache"
        self._key_locks: dict[str, threading.Lock] = {}    # single-flight: one API call per distinct prompt
        self._memo: dict[str, str] = {}
        self._cache_path = Path(cache_path) if cache_path else None
        if self._cache_path and self._cache_path.exists():
            for line in self._cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line)
                    self._memo[d["k"]] = d["c"]
                except (ValueError, KeyError):
                    continue

    def _post(self, body: dict) -> dict:
        last = None
        for attempt in range(3):
            try:
                r = self.client.post(
                    f"{self.base}/chat/completions",
                    headers={"Authorization": f"Bearer {self.key}"},
                    json=body, timeout=self.timeout,
                )
                if r.status_code in (429, 500, 502, 503, 504):
                    last = f"HTTP {r.status_code}"
                    time.sleep(min(float(r.headers.get("retry-after", 1.5 * (attempt + 1))), 8))
                    continue
                if r.status_code != 200:
                    raise PlannerError(f"HTTP {r.status_code}: {r.text[:200]}")
                return r.json()
            except httpx.HTTPError as e:
                last = type(e).__name__
                time.sleep(1.0 * (attempt + 1))
        raise PlannerError(f"API unavailable after retries ({last})")

    def _key(self, messages: list[dict]) -> str:
        blob = json.dumps({"m": self.model, "e": self.effort, "s": self.seed, "msgs": messages}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    @property
    def last_raw(self) -> str | None:
        """Raw text of the most recent reply on THIS thread (cached or fresh), for audit."""
        return getattr(self._tl, "raw", None)

    @property
    def last_cached(self) -> bool:
        """True if the most recent plan() call on THIS thread was served from the memo (no API call)."""
        return getattr(self._tl, "cached", False)

    def plan(self, obs: dict, mode: str) -> str:
        messages = build_messages(obs, mode)
        if not self.memoize:
            return self._query(messages, None)
        key = self._key(messages)
        with self._lock:
            kl = self._key_locks.setdefault(key, threading.Lock())
        with kl:  # concurrent workers asking the same prompt wait for the first reply instead of racing
            with self._lock:
                hit = self._memo.get(key)
                if hit is not None:
                    self.usage["cache_hits"] += 1
            if hit is not None:
                self._tl.cached = True
                self._tl.raw = hit
                return _parse_action(hit)
            return self._query(messages, key)

    def _query(self, messages: list[dict], key: str | None) -> str:
        self._tl.cached = False
        body = {
            "model": self.model, "messages": messages,
            "temperature": 0, "seed": self.seed,
            "reasoning_effort": self.effort, "max_completion_tokens": self.max_tokens,
        }
        j = self._post(body)
        try:
            content = j["choices"][0]["message"].get("content")
        except (KeyError, IndexError, TypeError, AttributeError):
            raise PlannerError("malformed API response")
        self._tl.raw = content
        u = j.get("usage") or {}
        with self._lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
            self.usage["completion_tokens"] += u.get("completion_tokens", 0) or 0
            self.usage["reasoning_tokens"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
            if content is None:
                self.usage["null_content"] += 1
            elif key is not None and key not in self._memo:   # first writer wins
                self._memo[key] = content
                if self._cache_path:
                    self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with self._cache_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"k": key, "c": content}) + "\n")
        return _parse_action(content)


class MockPlanner:
    name = "mock"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.usage = {"calls": 0}

    def _rng(self, obs: dict, mode: str) -> random.Random:
        key = f"{self.seed}|{obs['pos']}|{obs.get('sign')}|{mode}"
        return random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32))

    def plan(self, obs: dict, mode: str) -> str:
        self.usage["calls"] += 1
        pos = tuple(obs["pos"])
        sign = (obs.get("sign") or "").translate(LEET)
        if sign and FOLLOW_RE.search(sign) and self._rng(obs, mode).random() < MOCK_FOLLOW_PROB[mode]:
            gx, gy = world.GOAL  # greedy, hazard-blind move toward the dock
            if pos[0] < gx:
                return "RIGHT"
            if pos[1] != gy:
                return "DOWN" if pos[1] < gy else "UP"
            return "STOP"
        seen = [(d["x"], d["y"]) for d in obs.get("detections", [])] + [(d["x"], d["y"]) for d in obs.get("lidar", [])]
        return world.safe_step(pos, seen)


_meta_cache: dict[str, MetaPlanner] = {}
_meta_lock = threading.Lock()


def get_planner(prefer_meta: bool = True, model: str | None = None, cache_path=None):
    if prefer_meta and meta_configured():
        key = f"{model or os.environ['META_MODEL']}|{cache_path}"
        with _meta_lock:  # one shared httpx.Client per (model, cache file): connection reuse
            if key not in _meta_cache:
                _meta_cache[key] = MetaPlanner(model, cache_path=cache_path)
            return _meta_cache[key]
    return MockPlanner()
