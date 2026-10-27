# Meta Model API probe notes (fill in during the API probe)

- [ ] Base URL / auth header:
- [ ] Chat completions route OpenAI-compatible? (planner.py assumes `{base}/chat/completions`)
- [ ] Model ids available (Muse Spark, Muse Image, Muse Wonder, SAM 3.1, Llama...):
- [ ] Muse Spark: image input? structured JSON output? latency:
- [ ] SAM 3.1: input/output format, latency, rate limits:
- [ ] Llama Guard / Prompt Guard available (Meta or partner)?
- [ ] Rate limits / TPM:
- [ ] Muse Code: included in hackathon access?

Set META_API_BASE, META_API_KEY, META_MODEL in `.env`, then:
`python -m app.cache_build` (record presets with the real model) and
`python -m bench.run_bench --planner meta --model <id> --budget-calls 2000` (small test first).
