"""Hack the Robot (NEW during hackathon)."""
import os
from pathlib import Path


def _load_dotenv(path: str | Path | None = None) -> None:
    """Minimal .env loader (no extra dependency). Real, non-empty environment variables win;
    empty values in the file are ignored. On Heroku there is no .env: config vars are used."""
    p = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and not os.environ.get(k):
            os.environ[k] = v


_load_dotenv()
