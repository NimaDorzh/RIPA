# Hack the Robot

An interactive red-teaming playground for AI-controlled robots. A warehouse robot is driven by a Meta
model; you plant a malicious sign and watch whether it enters the Human Worker Zone. Toggle the
firewall to see the same attack get blocked. A benchmark harness (results: see below) measures how
often the model itself proposes unsafe actions under each defense.

Built for the Meta Global AI Developer Hackathon (Oct 26 - Nov 4, 2026), Indie track.
**What is new vs. reused from earlier work: see [`HACKATHON.md`](HACKATHON.md).**

## Run
```
pip install -r requirements.txt
python -m app.cache_build --mock      # offline stand-in traces (labelled SIMULATED in the UI)
# real model: cp .env.example .env, put your key in META_API_KEY, then `python -m app.cache_build`
uvicorn app.main:app --reload         # open http://localhost:8000
pytest -q
python -m bench.run_bench --planner mock   # pipeline test only, NOT results
python -m bench.run_bench --planner meta --model muse-spark-1.3 --n-attack 20 --n-benign 8 --budget-calls 600  # smoke test
```
Real model: copy `.env.example` to `.env`, fill `META_API_KEY`, rebuild the cache, run the bench.

## Defenses compared
`none` | `hardened` (system-prompt rule) | `separation` (untrusted-block prompt separation, no action check) |
`check` (action check only) | `firewall` (separation + action check).

The action check compares each proposed move with the trusted map and the simulator's camera
detections. In this simulator the set of moves it blocks is exactly the set the world scores as a
safety violation, so **"violation rate with the check = 0" holds by construction and is not a result**
(a unit test documents this). It is a safety envelope. What the benchmark measures is the
**unsafe-proposal rate**: how often the model proposes an unsafe move before any check, and whether
prompt-level separation lowers it. Stalls (freeze attacks), invalid replies and API errors are reported separately.

## Benchmark design
- Attacks: 25 independent bases (5 categories B1-B5) x 4 wrappers = 100 texts, written for this task
  (categories follow the taxonomy of the earlier RIPA paper; none of RIPA's payload texts is used).
  The bases were written together with the defenses, so the attack set is not held out; treat results as a
  development-set measurement, not a robustness guarantee.
- CIs for attack rates: cluster bootstrap over the 25 bases (if all outcomes are identical, e.g. 0 successes, the
  bootstrap collapses to a point, so a Wilson interval with n = 25 bases is used instead). Paired comparisons: exact McNemar on the
  same text, Holm-adjusted. Primary endpoint (fixed before running): unsafe-proposal rate, `none` vs `separation`.
- Benign: 8 sign templates (3 are hard negatives containing words like "fastest way") x aisle numbers; small n, Wilson CI.
- Models: `muse-spark-1.3` and `muse-spark-1.2`, both with `reasoning_effort=minimal`, temperature 0, seed 1.
  These are reasoning models: even on `minimal` they spend roughly 550-1050 reasoning tokens (3-7 s) per robot step,
  so a full run is ~3-5k calls per model. Replies are memoized by exact prompt (identical prompts occur in several
  defenses), which keeps paired comparisons exact; whether the API is deterministic at temperature 0 / seed 1 is measured by `python -m bench.determinism`
  (results will be reported here; until then it is NOT assumed). The prompt includes the robot's own recent path
  (trusted log) because a memoryless planner otherwise oscillates between two cells; stalls on benign signs are the
  no-attack baseline for the stall rate.
- A live episode takes about a minute and costs ~12k tokens per mode, so live free-text runs are off by default
  (`ALLOW_LIVE=0`); the public demo plays recorded real-model traces (the UI labels the planner of every run).

## Results
_None yet. Numbers will be added here from `bench/results/summary_*.json`; runs on the mock planner are never reported._

## Limitations (honest)
- Simulation only; one task and one map. External validity is limited.
- The action check is grounded in the trusted map and the simulator's ground-truth camera detections; it
  does not model perception poisoning (a patch or occlusion that hides the person from the camera). It stops
  unsafe motion but does not detect attacks that cause safe-but-wrong behavior (freezes, delays, detours).
- The sign/camera conflict flag is an annotation only and is keyword-based (it can be evaded by obfuscation).
- Results marked MOCK come from a simulated planner with invented parameters and are never reported as results.

## Earlier work (before the hackathon)
This project builds on my published research on prompt injection through the sensory channels of
LLM-controlled ROS 2 robots: *RIPA: Sensory-Vector Prompt Injection Attacks on LLM-Controlled ROS 2
Robots* (arXiv:2606.28649). All of it now lives in `legacy-ripa/`; the original code, data and README are also preserved in git tag
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