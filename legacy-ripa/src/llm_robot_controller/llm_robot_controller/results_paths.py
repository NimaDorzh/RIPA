import csv
from datetime import datetime
from pathlib import Path
import re


RESULTS_ROOT_DIR = Path("/home/amin/robotics_ws/results")
CSV_RESULTS_DIR = RESULTS_ROOT_DIR / "csv"
PNG_RESULTS_DIR = RESULTS_ROOT_DIR / "png"


def sanitize_filename_part(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    sanitized = normalized.strip("_")
    return sanitized or "unknown"


def extract_model_short_name(model_name: str) -> str:
    normalized_model_name = model_name.strip()
    if not normalized_model_name:
        return "unknown"

    tail = normalized_model_name.split("/")[-1]
    model_parts = [part for part in re.split(r"[-_]+", tail) if part]
    return model_parts[-1] if model_parts else tail


def build_experiment_results_path(
    provider: str,
    model_name: str,
    num_runs: int,
    started_at: datetime | None = None,
) -> Path:
    timestamp = (started_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    provider_token = sanitize_filename_part(provider)
    model_token = sanitize_filename_part(extract_model_short_name(model_name))
    return CSV_RESULTS_DIR / f"experiment_{provider_token}_{model_token}_{num_runs}runs_{timestamp}.csv"


def csv_matches_provider(path: Path, provider: str) -> bool:
    normalized_provider = provider.strip().lower()
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            first_row = next(reader, None)
    except (OSError, UnicodeDecodeError, csv.Error):
        return False

    if first_row is None:
        return False

    return (first_row.get("provider") or "").strip().lower() == normalized_provider


def find_latest_experiment_csv(provider: str | None = None) -> Path | None:
    pattern = "experiment_*.csv"
    if provider:
        pattern = f"experiment_{sanitize_filename_part(provider)}_*.csv"

    candidates = sorted(CSV_RESULTS_DIR.glob(pattern), key=lambda path: path.stat().st_mtime)
    if candidates:
        return candidates[-1]

    legacy_candidates = sorted(RESULTS_ROOT_DIR.glob(pattern), key=lambda path: path.stat().st_mtime)
    if legacy_candidates:
        return legacy_candidates[-1]

    if provider is not None:
        scanned_candidates = sorted(
            [
                *CSV_RESULTS_DIR.glob("experiment_*.csv"),
                *RESULTS_ROOT_DIR.glob("experiment_*.csv"),
            ],
            key=lambda path: path.stat().st_mtime,
        )
        matching_candidates = [path for path in scanned_candidates if csv_matches_provider(path, provider)]
        if not matching_candidates:
            return None

        return matching_candidates[-1]

    candidates = sorted(CSV_RESULTS_DIR.glob("experiment_*.csv"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        legacy_candidates = sorted(RESULTS_ROOT_DIR.glob("experiment_*.csv"), key=lambda path: path.stat().st_mtime)
        if not legacy_candidates:
            return None

        return legacy_candidates[-1]

    return candidates[-1]