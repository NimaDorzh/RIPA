import base64, json, os, struct, time, zlib
import httpx

def load_env(path=".env"):
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

load_env()
BASE = os.environ["META_API_BASE"].rstrip("/")
H = {"Authorization": f"Bearer {os.environ['META_API_KEY']}", "Content-Type": "application/json"}
MODEL = os.environ["META_MODEL"]

def png(w=64, h=64, rgb=(255, 0, 0)):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

IMG = base64.b64encode(png()).decode()

def call(name, content, **extra):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], **extra}
    t = time.time()
    try:
        r = httpx.post(BASE + "/chat/completions", headers=H, json=body, timeout=60)
    except Exception as e:
        print(f"\n=== {name}: ERROR {type(e).__name__}: {e}"); return
    dt = time.time() - t
    try:
        j = r.json()
    except Exception:
        print(f"\n=== {name}: HTTP {r.status_code} {dt:.2f}s  {r.text[:300]}"); return
    if r.status_code != 200:
        print(f"\n=== {name}: HTTP {r.status_code} {dt:.2f}s  {json.dumps(j)[:400]}"); return
    c = j["choices"][0]
    u = j.get("usage", {})
    print(f"\n=== {name}: HTTP 200 {dt:.2f}s finish={c['finish_reason']} "
          f"tokens={u.get('completion_tokens')} reasoning={u.get('completion_tokens_details', {}).get('reasoning_tokens')}")
    print("   content:", repr(c["message"]["content"])[:200])

Q = "Reply with the single word OK."
call("default effort, 200 tok", Q, max_completion_tokens=200)
for eff in ("none", "minimal", "low"):
    call(f"reasoning_effort={eff}", Q, reasoning_effort=eff, max_completion_tokens=200)
call("effort=low + temperature=0 + seed=1", Q, reasoning_effort="low", temperature=0, seed=1, max_completion_tokens=200)
call("json mode, effort=low", 'Return JSON {"action":"STOP"} and nothing else.',
     reasoning_effort="low", response_format={"type": "json_object"}, max_completion_tokens=200)
call("image (red square)", [
    {"type": "text", "text": "What color is this image? One word."},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{IMG}"}},
], reasoning_effort="low", max_completion_tokens=200)