# Meta Model API probe notes (probe run Oct 27, 2026)

- [x] Base URL: `https://api.meta.ai/v1`; auth header `Authorization: Bearer <key>`.
- [x] `POST {base}/chat/completions` works (OpenAI-compatible, HTTP 200). `GET {base}/models` works.
- [x] Models listed for this account: muse-spark-1.3, muse-spark-1.3-contributor, muse-spark-1.2,
      muse-spark-1.2-contributor, muse-spark-1.1, muse-wonder-1.0, muse-image-1.0, muse-voice-transcribe-1.0, sam-3.1.
      No Llama Guard / Prompt Guard in the list. `-contributor` models: discounted, provider may train on inputs.
- [x] muse-spark-1.3 is a **reasoning model**. With `max_tokens=50` the whole budget went to reasoning and
      `content` came back `null` (`finish_reason: length`). Use `max_completion_tokens` >= 200-300.
- [x] `reasoning_effort`: `none` is rejected (400). Supported: minimal, low, medium, high, xhigh, max.
      minimal: ~7 reasoning tokens for a trivial prompt; low: ~110-160. The planner uses `minimal`.
- [x] `temperature=0` and `seed=1` are accepted (HTTP 200). Whether they make output deterministic is NOT verified.
- [x] Image input (data-URL PNG) works on muse-spark-1.3 (a red square was described as "Red").
- [x] `response_format={"type":"json_object"}` is accepted, but at effort=low the whole budget went to reasoning
      (content null). Not used: the prompt asks for JSON and the parser is strict.
- [x] Latency: ~0.6-1.2 s per small call.
- [x] **Real robot prompt (Oct 27):** reasoning tokens per call: muse-spark-1.3 minimal ~1040, low ~1160, default ~2640;
      muse-spark-1.2 minimal ~545, low ~1090, default ~1640; muse-spark-1.1 minimal ~920, low ~1090, default ~1855.
      Latency 3-7 s per call. `max_completion_tokens=300` made every reply empty (240/240 null); use 4000.
- [ ] SAM 3.1 input/output format, latency, limits:
- [ ] Rate limits / TPM:
- [ ] Muse Code: included in hackathon access? (not enabled)

Set META_API_KEY in `.env` (base URL and model are in `.env.example`), then:
`python -m app.cache_build` (record presets with the real model) and
`python -m bench.run_bench --planner meta --model muse-spark-1.3 --n-attack 20 --n-benign 8 --budget-calls 600` (smoke test first).
