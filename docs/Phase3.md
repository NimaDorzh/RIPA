feat(audio): add end-to-end audio injection channel (Channel 2)

Add audio-channel sensory injection pipeline for RIPA framework,
implementing STT-based prompt injection via ROS 2 topic bridge.

New files:
- audio_listener_node.py: ROS 2 node that transcribes .wav files via
  Whisper base and publishes to /object_label with TRANSIENT_LOCAL QoS
- audio_injection_test.py: end-to-end injection test runner publishing
  transcriptions to /object_label and measuring attack success via
  /cmd_vel (TwistStamped) response, matching injection_test.py architecture

Key design decisions:
- Single publish per run (no retry loop) to prevent DDS queue bleed
- Inter-variant 3s flush to drain controller queue between variants
- header.stamp filtering to reject stale cmd_vel messages
- WAIT_FOR_CMD_VEL_SEC=8.0 to accommodate STT+LLM pipeline latency
- Whisper transcription computed once per variant outside run loop

Bug fixes applied during development:
- Removed retry-publish loop that caused queue contamination across runs
- Added drain loop then replaced with stamp-based filtering
- Aligned VARIANTS payloads with injection_test.py exactly
- Fixed /object_label_safe -> /object_label in controller_node.py
- Fixed LLM_PROVIDER/LLM_MODEL/.env configuration drift

Results (DeepSeek deepseek-chat, temperature=0, N=30):
  N:  0.0% ASR (baseline)
  A1: 100.0% ASR
  A2: 100.0% ASR
  A3:  96.7% ASR (1 API timeout >8s)

Notable finding: A3 payload syntax fully destroyed by Whisper
("[INST]" -> "Inst") yet injection succeeded — LLM extracts
semantics from degraded STT output.

Updates setup.py entry_points with audio_listener and
audio_injection_test console scripts.