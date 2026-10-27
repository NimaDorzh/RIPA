# legacy-ripa: pre-hackathon work (NOT part of the hackathon submission)

Code, data and figures of RIPA ("Sensory-Vector Prompt Injection Attacks on LLM-Controlled ROS 2 Robots",
arXiv:2606.28649): the ROS 2 package (`src/`), result CSVs/figures (`results/`), test media, helper scripts and
the phase reports (`docs/Phase*.md`). It references non-Meta models (DeepSeek, Together) and is kept only for
provenance. The original repository state, including the original README, is preserved in git tag
`pre-hackathon-2026-10-26`.

To use it as a ROS 2 workspace, treat this folder as the workspace root: `cd legacy-ripa && colcon build`.
