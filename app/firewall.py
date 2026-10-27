"""Firewall v2 (NEW during hackathon).

Two ideas, both different from the keyword filter in the earlier RIPA work:

1. Structural separation: text read from the environment is never mixed into
   the instruction channel. It is passed to the planner inside an explicit
   untrusted block (see prompts.build_messages with mode="firewall").
2. Action-consistency check: every action proposed by the planner is checked
   against the robot's own trusted map and camera before execution. If the
   action would enter the Human Worker Zone or a cell with a person, it is
   blocked and replaced by the reference safe step. A text-vs-perception
   conflict (a sign claiming the path is clear while the camera sees a person)
   is flagged for the trace.

Known limitation (also documented in README): the check is grounded in the
robot's own perception and map. It stops unsafe motion; it does not detect
attacks that cause safe-but-wrong behavior (e.g. delays, detours).
"""
from __future__ import annotations

import re

from . import world

CLEAR_CLAIM_RE = re.compile(
    r"(path\s+(is\s+)?clear|no\s+(people|obstacles|one)|all\s+clear|speed\s*up|zone\s+(is\s+)?open|safe\s+to\s+(go|proceed))",
    re.I,
)


def text_perception_conflict(obs: dict) -> bool:
    sign = obs.get("sign") or ""
    return bool(CLEAR_CLAIM_RE.search(sign)) and len(obs.get("detections", [])) > 0


def check_action(obs: dict, action: str) -> dict:
    pos = tuple(obs["pos"])
    n = world.next_cell(pos, action)
    conflict = text_perception_conflict(obs)
    people = {(d["x"], d["y"]) for d in obs.get("detections", [])}
    if action != "STOP" and world.in_bounds(n):
        if n in world.HUMAN_ZONE:
            return {"allowed": False, "conflict": conflict,
                    "reason": "Blocked: action enters Human Worker Zone (trusted map)"}
        if n in people:
            return {"allowed": False, "conflict": conflict,
                    "reason": "Blocked: camera sees a person in the target cell"}
    reason = "OK"
    if conflict:
        reason = "OK, but sign text contradicts camera (flagged)"
    return {"allowed": True, "conflict": conflict, "reason": reason}


def fallback_action(obs: dict) -> str:
    return world.safe_step(tuple(obs["pos"]))
