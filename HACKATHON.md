# Old vs new (Meta Global AI Developer Hackathon)

Pre-hackathon work: RIPA, arXiv:2606.28649, github.com/NimaDorzh/RIPA (ROS 2 + Gazebo,
sensory-channel prompt injection on LLM-controlled robots). Pre-hackathon snapshot: git tag
`pre-hackathon-2026-10-26`. (Checked against RIPA commit 991f0b1; confirm it is at or before the tag.)

| Path | Status | Notes |
|---|---|---|
| `reused/ripa_payloads.py` | **reused** | 19 payloads + judge prompt copied verbatim from RIPA `firewall_bypass_test.py`. Used only as the source of attack categories B1-B5. Not counted as new work. |
| Attack taxonomy B1-B5, ASR/FPR metrics | **reused idea** | Concepts from the paper; all code and texts below are new. |
| `app/world.py`, `app/episode.py` | new | Warehouse simulation (not ROS 2 / Gazebo). |
| `app/firewall.py`, `app/prompts.py` | new | Firewall v2: structural separation + perception-grounded action check (replaces the keyword filter of RIPA). |
| `app/planner.py` | new | Meta Model API client + simulated offline planner (for tests only). |
| `app/main.py`, `static/index.html`, `app/presets.py`, `app/cache_build.py` | new | Public playground, presets, cached traces, rate limiting. |
| `bench/*` | new | New datasets (warehouse task), 3 defenses, Wilson CIs, latency. |
| `tests/*` | new | |

Nothing from RIPA's figures, CSVs or Gazebo material is used in the demo.
