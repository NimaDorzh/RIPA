import csv
import argparse
from pathlib import Path
import sys
from typing import TypedDict

import matplotlib.pyplot as plt


RESULTS_ROOT_DIR = Path("/home/amin/robotics_ws/results")
CSV_RESULTS_DIR = RESULTS_ROOT_DIR / "csv"
PNG_RESULTS_DIR = RESULTS_ROOT_DIR / "png"


class ResultRow(TypedDict):
    sequence: int
    test_id: str
    provider: str
    timestamp: str
    actual_cmd_vel_x: float
    expected_match: bool
    attack_success: bool
    is_normal: bool


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def sanitize_filename_part(value: str) -> str:
    normalized = "".join(character if character.isalnum() else "_" for character in value.strip().lower())
    sanitized = normalized.strip("_")
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize experiment CSV results.")
    parser.add_argument("csv_path", nargs="?", help="Path to a specific experiment CSV file")
    parser.add_argument("--provider", help="Use the latest CSV for the specified provider")
    return parser.parse_args(sys.argv[1:])


def csv_matches_provider(path: Path, provider: str) -> bool:
    normalized_provider = provider.strip().lower()
    try:
        with path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
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
    if candidates:
        return candidates[-1]

    legacy_candidates = sorted(RESULTS_ROOT_DIR.glob("experiment_*.csv"), key=lambda path: path.stat().st_mtime)
    if not legacy_candidates:
        return None

    return legacy_candidates[-1]


def resolve_csv_path(args: argparse.Namespace) -> Path:
    if args.csv_path:
        return Path(args.csv_path).expanduser().resolve()

    latest_path = find_latest_experiment_csv(args.provider)
    if latest_path is None:
        if args.provider:
            raise FileNotFoundError(f"No experiment CSV files found for provider '{args.provider}' in {CSV_RESULTS_DIR}")

        raise FileNotFoundError(f"No experiment CSV files found in {CSV_RESULTS_DIR}")

    return latest_path


def build_output_path(csv_path: Path) -> Path:
    PNG_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return PNG_RESULTS_DIR / f"{csv_path.stem}.png"


def build_title(rows: list[ResultRow]) -> str:
    provider = rows[0]["provider"] or "unknown"
    started_at = rows[0]["timestamp"] or "unknown"
    return f"RIPA: Prompt Injection Attack Results - {provider} - {started_at}"


def load_rows(csv_path: Path) -> list[ResultRow]:
    rows: list[ResultRow] = []

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for index, row in enumerate(reader, start=1):
            actual_cmd_vel_x_raw = (row.get("actual_cmd_vel_x") or "").strip()
            actual_cmd_vel_x = float(actual_cmd_vel_x_raw) if actual_cmd_vel_x_raw else 0.0
            test_id = (row.get("test_id") or "").strip()
            is_normal = test_id.startswith("NORMAL_")

            rows.append(
                {
                    "sequence": index,
                    "test_id": test_id,
                    "provider": (row.get("provider") or "").strip(),
                    "timestamp": (row.get("timestamp") or "").strip(),
                    "actual_cmd_vel_x": actual_cmd_vel_x,
                    "expected_match": parse_bool(row.get("expected_match") or "false"),
                    "attack_success": parse_bool(row.get("attack_success") or "false"),
                    "is_normal": is_normal,
                }
            )

    return rows


def mean_percentage(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values) * 100.0


def build_bar_values(rows: list[ResultRow]) -> tuple[list[str], list[float], list[str]]:
    groups = [
        ("Normal", "NORMAL_", "expected_match"),
        ("A1 (Direct)", "INJECTION_A1_", "attack_success"),
        ("A2 (Newline)", "INJECTION_A2_", "attack_success"),
        ("A3 (Template)", "INJECTION_A3_", "attack_success"),
    ]

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []

    for label, prefix, metric_key in groups:
        group_rows = [row for row in rows if str(row["test_id"]).startswith(prefix)]
        metric_values = [bool(row[metric_key]) for row in group_rows]

        labels.append(label)
        values.append(mean_percentage(metric_values))
        colors.append("#2e8b57" if label == "Normal" else "#c0392b")

    return labels, values, colors


def plot_results(rows: list[ResultRow], output_path: Path) -> None:
    labels, bar_values, bar_colors = build_bar_values(rows)
    sequences = [row["sequence"] for row in rows]
    cmd_values = [row["actual_cmd_vel_x"] for row in rows]
    point_colors = ["#1f77b4" if row["is_normal"] else "#c0392b" for row in rows]

    figure, (bar_ax, timeline_ax) = plt.subplots(1, 2, figsize=(14, 6))
    figure.suptitle(build_title(rows), fontsize=14, fontweight="bold", y=0.98)

    bars = bar_ax.bar(labels, bar_values, color=bar_colors, width=0.6)
    bar_ax.set_ylabel("Success rate (%)")
    bar_ax.set_xlabel("Test type")
    bar_ax.set_ylim(0, 110)
    bar_ax.set_title("Attack Success Rate by Injection Type")
    bar_ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, bar_values):
        bar_ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 2,
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    timeline_ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1)
    timeline_ax.axhline(-0.5, color="#888888", linestyle="--", linewidth=1)
    timeline_ax.plot(sequences, cmd_values, color="#666666", linewidth=1, alpha=0.6)
    timeline_ax.scatter(sequences, cmd_values, c=point_colors, s=55)
    timeline_ax.set_title("cmd_vel_x over Test Sequence")
    timeline_ax.set_xlabel("Test number")
    timeline_ax.set_ylabel("linear.x")
    timeline_ax.set_xlim(1, max(sequences) if sequences else 20)
    timeline_ax.set_ylim(-0.55, 0.55)
    timeline_ax.set_xticks(sequences)
    timeline_ax.grid(True, linestyle="--", alpha=0.4)

    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    csv_path = resolve_csv_path(args)
    output_path = build_output_path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows = load_rows(csv_path)
    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")

    plot_results(rows, output_path)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()