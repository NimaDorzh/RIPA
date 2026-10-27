# Hack the Robot

An interactive red-teaming playground for AI-controlled robots. A warehouse robot is driven by a Meta
model; you plant a malicious sign and watch whether it enters the Human Worker Zone. Toggle the
firewall to see the same attack get blocked. A reproducible benchmark backs it up.

Built for the Meta Global AI Developer Hackathon (Oct 26 - Nov 4, 2026), Indie track.
**What is new vs. reused from earlier work: see [`HACKATHON.md`](HACKATHON.md).**

## Run
```
pip install -r requirements.txt
python -m app.cache_build --mock      # offline stand-in traces (labelled SIMULATED in the UI)
uvicorn app.main:app --reload         # open http://localhost:8000
pytest -q
python -m bench.run_bench --planner mock   # pipeline test only, NOT results
```
Real model: copy `.env.example` to `.env`, fill the three META_* values, rebuild the cache, run the bench.

## Defenses compared
`none` | `hardened` (system-prompt rule) | `firewall` (untrusted-block separation + action check against the
robot's trusted map and camera).

## Limitations (honest)
- Simulation only; one task and one map. External validity is limited.
- The firewall is grounded in the robot's own perception/map: it stops unsafe motion but does not detect
  attacks that cause safe-but-wrong behavior (delays, detours).
- Results marked MOCK come from a simulated planner with invented parameters and are never reported as results.

## Earlier work (before the hackathon)
This project builds on my published research on prompt injection through the sensory channels of
LLM-controlled ROS 2 robots: *RIPA: Sensory-Vector Prompt Injection Attacks on LLM-Controlled ROS 2
Robots* (arXiv:2606.28649). The original code, data and README are preserved in git tag
`pre-hackathon-2026-10-26`. Nothing from that work (figures, CSVs, Gazebo material) is presented as
hackathon work.

```bibtex
@misc{dorzhiev2026ripa,
  title         = {RIPA: Sensory-Vector Prompt Injection Attacks on LLM-Controlled ROS 2 Robots},
  author        = {Dorzhiev, Nima},
  year          = {2026},
  eprint        = {2606.28649},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR}
}
```