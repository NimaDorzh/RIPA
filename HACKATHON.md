# Old vs new (Meta Global AI Developer Hackathon)

Pre-hackathon work: RIPA, arXiv:2606.28649, github.com/NimaDorzh/RIPA (ROS 2 + Gazebo, sensory-channel
prompt injection on LLM-controlled robots). Pre-hackathon snapshot: git tag `pre-hackathon-2026-10-26`
(original README, code, data and figures are preserved there).

| Path | Status | Notes |
|---|---|---|
| `legacy-ripa/` | **old, not part of the submission** | All pre-hackathon RIPA material: ROS 2 package, results, test media, scripts, phase reports. It also references non-Meta models (DeepSeek, Together). Moved here on Oct 27 so the repository root contains only hackathon work. |
| `reused/ripa_payloads.py` | **reused** | 19 payloads + judge prompt copied verbatim from RIPA `firewall_bypass_test.py` (commit 991f0b1). Not used by the benchmark; kept only as a record of the source of the attack taxonomy B1-B5. Not counted as new work. |
| Attack taxonomy B1-B5, ASR/FPR-style metrics | **reused idea** | Concepts from the paper; all code and texts outside `legacy-ripa/` and `reused/` are new. |
| `README.md`, `.env.example`, `.gitignore` | **replaced / modified** | Were RIPA's files; rewritten for this project. Originals: `git show pre-hackathon-2026-10-26:<file>`. |
| `app/world.py`, `app/episode.py` | new | Warehouse simulation (not ROS 2 / Gazebo). |
| `app/firewall.py`, `app/prompts.py` | new | Firewall v2: structural separation + action check against the trusted map and simulator detections (replaces the keyword filter of RIPA). |
| `app/planner.py` | new | Meta Model API client (reply cache, strict parser) + simulated offline planner (for tests only). |
| `app/main.py`, `static/index.html`, `app/presets.py`, `app/cache_build.py` | new | Public playground, recorded presets, rate limiting. |
| `bench/*` | new | Warehouse-task datasets, 5 defense configurations, cluster-bootstrap CIs, McNemar, determinism check. |
| `tests/*`, `docs/*`, `scripts/*`, `Procfile`, `requirements.txt`, `pytest.ini` | new | |

Nothing from RIPA's figures, CSVs or Gazebo material is used in the demo. The hackathon pipeline
(`app/`, `bench/`, `static/`) uses Meta models only.
