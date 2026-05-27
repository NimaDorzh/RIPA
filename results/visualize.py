import csv
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path("/home/amin/robotics_ws/results")
CSV_PATH = RESULTS_DIR / "experiment_1.csv"
OUTPUT_PATH = RESULTS_DIR / "experiment_1.png"
TITLE = "RIPA: Prompt Injection Attack Results - DeepSeek-chat"


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    with CSV_PATH.open("r", encoding="utf-8", newline="") as csv_file:
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


def build_bar_values(rows: list[dict[str, object]]) -> tuple[list[str], list[float], list[str]]:
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


def plot_results(rows: list[dict[str, object]]) -> None:
    labels, bar_values, bar_colors = build_bar_values(rows)
    sequences = [int(row["sequence"]) for row in rows]
    cmd_values = [float(row["actual_cmd_vel_x"]) for row in rows]
    point_colors = ["#1f77b4" if bool(row["is_normal"]) else "#c0392b" for row in rows]

    figure, (bar_ax, timeline_ax) = plt.subplots(1, 2, figsize=(14, 6))
    figure.suptitle(TITLE, fontsize=14, fontweight="bold", y=0.98)

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
    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    rows = load_rows()
    if not rows:
        raise ValueError(f"CSV file is empty: {CSV_PATH}")

    plot_results(rows)
    print(f"Saved visualization to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()