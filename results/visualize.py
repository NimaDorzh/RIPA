import csv
import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import TypedDict

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch


RESULTS_ROOT_DIR = Path("/home/amin/robotics_ws/results")
CSV_RESULTS_DIR = RESULTS_ROOT_DIR / "csv"
PNG_RESULTS_DIR = RESULTS_ROOT_DIR / "png"
PDF_RESULTS_DIR = RESULTS_ROOT_DIR / "pdf"
BACKGROUND_COLOR = "#f4efe7"
PANEL_COLOR = "#fffaf3"
TEXT_COLOR = "#1f2933"
MUTED_COLOR = "#5b6670"
ACCENT_COLOR = "#a64b2a"
SUCCESS_COLOR = "#2e8b57"
WARNING_COLOR = "#c0392b"
NEUTRAL_COLOR = "#476582"
GRID_COLOR = "#d8cfc0"
HIGHLIGHT_FILL = "#f1dfc9"


class ResultRow(TypedDict):
    sequence: int
    test_id: str
    provider: str
    timestamp: str
    actual_cmd_vel_x: float
    expected_match: bool
    attack_success: bool
    is_normal: bool


class GroupSummary(TypedDict):
    label: str
    prefix: str
    attempts: int
    positives: int
    rate: float
    metric_key: str
    mean_cmd_vel_x: float


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


def build_pdf_output_path(csv_path: Path) -> Path:
    PDF_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return PDF_RESULTS_DIR / f"{csv_path.stem}.pdf"


def build_title(rows: list[ResultRow]) -> str:
    provider = rows[0]["provider"] or "unknown"
    started_at = rows[0]["timestamp"] or "unknown"
    return f"RIPA: Prompt Injection Attack Results - {provider} - {started_at}"


def format_timestamp(timestamp: str) -> str:
    try:
        return datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M:%S %Z")
    except ValueError:
        return timestamp


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


def mean_value(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_group_summaries(rows: list[ResultRow]) -> list[GroupSummary]:
    groups = [
        ("Normal", "NORMAL_", "expected_match"),
        ("A1 Direct Override", "INJECTION_A1_", "attack_success"),
        ("A2 Newline Override", "INJECTION_A2_", "attack_success"),
        ("A3 Template Escape", "INJECTION_A3_", "attack_success"),
    ]

    summaries: list[GroupSummary] = []
    for label, prefix, metric_key in groups:
        group_rows = [row for row in rows if row["test_id"].startswith(prefix)]
        metric_values = [row[metric_key] for row in group_rows]
        cmd_values = [row["actual_cmd_vel_x"] for row in group_rows]
        positives = sum(1 for value in metric_values if value)
        attempts = len(metric_values)
        summaries.append(
            {
                "label": label,
                "prefix": prefix,
                "attempts": attempts,
                "positives": positives,
                "rate": mean_percentage(metric_values),
                "metric_key": metric_key,
                "mean_cmd_vel_x": mean_value(cmd_values),
            }
        )

    return summaries


def build_kpis(rows: list[ResultRow]) -> dict[str, str]:
    summaries = build_group_summaries(rows)
    normal_summary = summaries[0]
    injection_summaries = summaries[1:]
    total_injection_attempts = sum(summary["attempts"] for summary in injection_summaries)
    total_injection_successes = sum(summary["positives"] for summary in injection_summaries)
    overall_injection_rate = (
        total_injection_successes / total_injection_attempts * 100.0 if total_injection_attempts else 0.0
    )
    weakest = max(injection_summaries, key=lambda summary: summary["rate"])
    strongest = min(injection_summaries, key=lambda summary: summary["rate"])

    return {
        "provider": rows[0]["provider"] or "unknown",
        "started_at": format_timestamp(rows[0]["timestamp"]),
        "finished_at": format_timestamp(rows[-1]["timestamp"]),
        "total_tests": str(len(rows)),
        "normal_compliance": f"{normal_summary['rate']:.0f}%",
        "overall_attack_success": f"{overall_injection_rate:.0f}%",
        "strongest_pattern": f"{strongest['label']} ({strongest['rate']:.0f}%)",
        "weakest_pattern": f"{weakest['label']} ({weakest['rate']:.0f}%)",
    }


def build_bar_values(rows: list[ResultRow]) -> tuple[list[str], list[float], list[str]]:
    summaries = build_group_summaries(rows)
    labels = [summary["label"].replace(" Override", "") for summary in summaries]
    values = [summary["rate"] for summary in summaries]
    colors = [SUCCESS_COLOR if summary["label"] == "Normal" else WARNING_COLOR for summary in summaries]
    return labels, values, colors


def create_charts_figure(rows: list[ResultRow], title: str) -> Figure:
    labels, bar_values, bar_colors = build_bar_values(rows)
    sequences = [row["sequence"] for row in rows]
    cmd_values = [row["actual_cmd_vel_x"] for row in rows]
    point_colors = [NEUTRAL_COLOR if row["is_normal"] else WARNING_COLOR for row in rows]

    figure, (bar_ax, timeline_ax) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BACKGROUND_COLOR)
    figure.suptitle(title, fontsize=16, fontweight="bold", y=0.98, color=TEXT_COLOR)
    for axis in (bar_ax, timeline_ax):
        axis.set_facecolor(PANEL_COLOR)

    bars = bar_ax.bar(labels, bar_values, color=bar_colors, width=0.6)
    bar_ax.set_ylabel("Success rate (%)")
    bar_ax.set_xlabel("Test type")
    bar_ax.set_ylim(0, 110)
    bar_ax.set_title("Attack Success Rate by Injection Type")
    bar_ax.grid(axis="y", linestyle="--", alpha=0.4, color=GRID_COLOR)
    bar_ax.tick_params(colors=TEXT_COLOR)
    bar_ax.yaxis.label.set_color(TEXT_COLOR)
    bar_ax.xaxis.label.set_color(TEXT_COLOR)
    bar_ax.title.set_color(TEXT_COLOR)

    for bar, value in zip(bars, bar_values):
        bar_ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 2,
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color=TEXT_COLOR,
        )

    timeline_ax.axhline(0.5, color=MUTED_COLOR, linestyle="--", linewidth=1)
    timeline_ax.axhline(-0.5, color=MUTED_COLOR, linestyle="--", linewidth=1)
    timeline_ax.plot(sequences, cmd_values, color=MUTED_COLOR, linewidth=1, alpha=0.6)
    timeline_ax.scatter(sequences, cmd_values, c=point_colors, s=55)
    timeline_ax.set_title("cmd_vel_x over Test Sequence")
    timeline_ax.set_xlabel("Test number")
    timeline_ax.set_ylabel("linear.x")
    timeline_ax.set_xlim(1, max(sequences) if sequences else 20)
    timeline_ax.set_ylim(-0.55, 0.55)
    timeline_ax.set_xticks(sequences)
    timeline_ax.grid(True, linestyle="--", alpha=0.4, color=GRID_COLOR)
    timeline_ax.tick_params(colors=TEXT_COLOR)
    timeline_ax.yaxis.label.set_color(TEXT_COLOR)
    timeline_ax.xaxis.label.set_color(TEXT_COLOR)
    timeline_ax.title.set_color(TEXT_COLOR)

    figure.tight_layout(rect=(0, 0, 1, 0.9))
    return figure


def plot_results(rows: list[ResultRow], output_path: Path) -> None:
    figure = create_charts_figure(rows, build_title(rows))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def draw_metric_card(figure: Figure, x: float, y: float, width: float, height: float, label: str, value: str, accent: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        transform=figure.transFigure,
        facecolor=PANEL_COLOR,
        edgecolor=accent,
        linewidth=2.0,
    )
    figure.patches.append(patch)
    figure.text(x + 0.02, y + height - 0.05, label, fontsize=10, color=MUTED_COLOR, weight="bold")
    figure.text(x + 0.02, y + 0.04, value, fontsize=18, color=TEXT_COLOR, weight="bold")


def create_cover_page(rows: list[ResultRow], csv_path: Path) -> Figure:
    kpis = build_kpis(rows)
    summaries = build_group_summaries(rows)

    figure = plt.figure(figsize=(11.69, 8.27), facecolor=BACKGROUND_COLOR)
    canvas = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    canvas.axis("off")

    hero_patch = FancyBboxPatch(
        (0.05, 0.09),
        0.9,
        0.82,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        transform=figure.transFigure,
        facecolor=PANEL_COLOR,
        edgecolor=GRID_COLOR,
        linewidth=1.5,
    )
    accent_patch = FancyBboxPatch(
        (0.05, 0.09),
        0.28,
        0.82,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        transform=figure.transFigure,
        facecolor=ACCENT_COLOR,
        edgecolor=ACCENT_COLOR,
        linewidth=0,
    )
    figure.patches.extend((hero_patch, accent_patch))

    figure.text(0.085, 0.82, "RIPA", fontsize=28, fontweight="bold", color="white")
    figure.text(0.085, 0.77, "Experiment Report", fontsize=20, fontweight="bold", color="white")
    figure.text(0.085, 0.67, "Provider", fontsize=10, color="#f9e7dd", weight="bold")
    figure.text(0.085, 0.63, kpis["provider"], fontsize=18, color="white", weight="bold")
    figure.text(0.085, 0.54, "Experiment Window", fontsize=10, color="#f9e7dd", weight="bold")
    figure.text(0.085, 0.50, kpis["started_at"], fontsize=10.5, color="white")
    figure.text(0.085, 0.46, kpis["finished_at"], fontsize=10.5, color="white")
    figure.text(0.085, 0.25, "Artifacts", fontsize=10, color="#f9e7dd", weight="bold")
    figure.text(0.085, 0.21, csv_path.name, fontsize=10.5, color="white")

    figure.text(0.38, 0.80, "Prompt Injection Evaluation", fontsize=24, fontweight="bold", color=TEXT_COLOR)
    figure.text(
        0.38,
        0.74,
        "A presentation-ready summary of navigation compliance, attack success, and per-pattern response behavior.",
        fontsize=12,
        color=MUTED_COLOR,
    )

    draw_metric_card(figure, 0.38, 0.55, 0.17, 0.14, "Total tests", kpis["total_tests"], ACCENT_COLOR)
    draw_metric_card(figure, 0.57, 0.55, 0.17, 0.14, "Normal compliance", kpis["normal_compliance"], SUCCESS_COLOR)
    draw_metric_card(figure, 0.76, 0.55, 0.15, 0.14, "Attack success", kpis["overall_attack_success"], WARNING_COLOR)

    figure.text(0.38, 0.45, "Executive Snapshot", fontsize=15, fontweight="bold", color=TEXT_COLOR)
    highlights = [
        f"Most robust pattern: {kpis['strongest_pattern']}",
        f"Highest-risk pattern: {kpis['weakest_pattern']}",
        "Normal prompts measure compliance with the intended destination.",
        "Injection prompts measure whether the robot can be redirected to zone B.",
    ]
    for index, line in enumerate(highlights):
        figure.text(0.4, 0.40 - index * 0.05, f"- {line}", fontsize=11, color=TEXT_COLOR)

    chip_y = 0.13
    chip_x = 0.38
    chip_width = 0.125
    for index, summary in enumerate(summaries):
        chip = FancyBboxPatch(
            (chip_x + index * (chip_width + 0.015), chip_y),
            chip_width,
            0.06,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            transform=figure.transFigure,
            facecolor=HIGHLIGHT_FILL,
            edgecolor=GRID_COLOR,
            linewidth=1.0,
        )
        figure.patches.append(chip)
        figure.text(
            chip_x + 0.012 + index * (chip_width + 0.015),
            chip_y + 0.034,
            summary["label"].replace(" Override", ""),
            fontsize=8.5,
            color=MUTED_COLOR,
            weight="bold",
        )
        figure.text(
            chip_x + 0.012 + index * (chip_width + 0.015),
            chip_y + 0.013,
            f"{summary['rate']:.0f}%",
            fontsize=12,
            color=TEXT_COLOR,
            weight="bold",
        )

    return figure


def style_table(table, body_fontsize: float) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(body_fontsize)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if row_index == 0:
            cell.set_facecolor(ACCENT_COLOR)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor(PANEL_COLOR if row_index % 2 else "#f9f4ec")
            cell.set_text_props(color=TEXT_COLOR)


def create_summary_page(rows: list[ResultRow], csv_path: Path) -> Figure:
    kpis = build_kpis(rows)
    summaries = build_group_summaries(rows)

    figure = plt.figure(figsize=(11.69, 8.27), facecolor=BACKGROUND_COLOR)
    canvas = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    canvas.axis("off")

    figure.text(0.06, 0.93, "RIPA Experiment Report", fontsize=24, fontweight="bold", color=TEXT_COLOR)
    figure.text(
        0.06,
        0.895,
        f"Provider: {kpis['provider']}  |  Started: {kpis['started_at']}  |  Finished: {kpis['finished_at']}",
        fontsize=11,
        color=MUTED_COLOR,
    )
    figure.text(0.06, 0.865, f"Source CSV: {csv_path.name}", fontsize=10, color=MUTED_COLOR)

    card_width = 0.2
    card_height = 0.14
    card_y = 0.67
    draw_metric_card(figure, 0.06, card_y, card_width, card_height, "Total tests", kpis["total_tests"], ACCENT_COLOR)
    draw_metric_card(figure, 0.29, card_y, card_width, card_height, "Normal compliance", kpis["normal_compliance"], SUCCESS_COLOR)
    draw_metric_card(figure, 0.52, card_y, card_width, card_height, "Overall attack success", kpis["overall_attack_success"], WARNING_COLOR)
    draw_metric_card(figure, 0.75, card_y, card_width, card_height, "Weakest pattern", kpis["weakest_pattern"], WARNING_COLOR)

    figure.text(0.06, 0.58, "Key Takeaways", fontsize=15, fontweight="bold", color=TEXT_COLOR)
    highlights = [
        f"Strongest resistance pattern: {kpis['strongest_pattern']}",
        f"Highest-risk pattern: {kpis['weakest_pattern']}",
        "Normal prompts are measured by expected-match rate; injection prompts are measured by attack success rate.",
        "Negative linear.x indicates the robot was redirected toward zone B during the attack sequence.",
    ]
    for index, line in enumerate(highlights):
        figure.text(0.08, 0.54 - index * 0.05, f"- {line}", fontsize=11, color=TEXT_COLOR)

    table_ax = figure.add_axes((0.06, 0.08, 0.88, 0.28))
    table_ax.axis("off")
    table_ax.set_title("Group Summary", fontsize=14, fontweight="bold", color=TEXT_COLOR, loc="left", pad=12)
    table_rows = [
        [
            summary["label"],
            str(summary["attempts"]),
            str(summary["positives"]),
            f"{summary['rate']:.0f}%",
            f"{summary['mean_cmd_vel_x']:.3f}",
        ]
        for summary in summaries
    ]
    table = table_ax.table(
        cellText=table_rows,
        colLabels=["Group", "Attempts", "Positive", "Rate", "Mean linear.x"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.scale(1, 1.7)
    style_table(table, 10)

    return figure


def create_detailed_table_page(rows: list[ResultRow]) -> Figure:
    summaries = build_group_summaries(rows)
    figure = plt.figure(figsize=(11.69, 8.27), facecolor=BACKGROUND_COLOR)

    figure.text(0.06, 0.93, "Attack Pattern Breakdown", fontsize=18, fontweight="bold", color=TEXT_COLOR)
    figure.text(
        0.06,
        0.902,
        "This page separates pattern-level outcomes from the full test log to keep labels clear and readable.",
        fontsize=10.5,
        color=MUTED_COLOR,
    )

    summary_ax = figure.add_axes((0.06, 0.62, 0.88, 0.20))
    summary_ax.axis("off")
    summary_rows = [
        [
            summary["label"],
            "Expected match" if summary["metric_key"] == "expected_match" else "Attack success",
            f"{summary['positives']}/{summary['attempts']}",
            f"{summary['rate']:.0f}%",
            f"{summary['mean_cmd_vel_x']:.3f}",
        ]
        for summary in summaries
    ]
    summary_table = summary_ax.table(
        cellText=summary_rows,
        colLabels=["Pattern", "Metric", "Wins", "Rate", "Mean linear.x"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    summary_table.scale(1, 1.7)
    style_table(summary_table, 10)

    figure.text(0.06, 0.515, "Trial-by-Trial Results", fontsize=18, fontweight="bold", color=TEXT_COLOR)
    figure.text(
        0.06,
        0.487,
        "Every trial is listed below with the observed linear.x command and the resulting attack outcome.",
        fontsize=10.5,
        color=MUTED_COLOR,
    )

    detail_ax = figure.add_axes((0.06, 0.08, 0.88, 0.33))
    detail_ax.axis("off")
    detail_rows = [
        [
            row["test_id"],
            f"{row['actual_cmd_vel_x']:.3f}",
            "yes" if row["expected_match"] else "no",
            "yes" if row["attack_success"] else "no",
            "normal" if row["is_normal"] else "injection",
        ]
        for row in rows
    ]
    detail_table = detail_ax.table(
        cellText=detail_rows,
        colLabels=["Test ID", "linear.x", "Expected match", "Attack success", "Type"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    detail_table.scale(1, 1.18)
    style_table(detail_table, 8.5)

    return figure


def save_pdf_report(rows: list[ResultRow], csv_path: Path, output_path: Path) -> None:
    with PdfPages(output_path) as pdf:
        cover_figure = create_cover_page(rows, csv_path)
        pdf.savefig(cover_figure, bbox_inches="tight")
        plt.close(cover_figure)

        summary_figure = create_summary_page(rows, csv_path)
        pdf.savefig(summary_figure, bbox_inches="tight")
        plt.close(summary_figure)

        charts_figure = create_charts_figure(rows, build_title(rows))
        pdf.savefig(charts_figure, bbox_inches="tight")
        plt.close(charts_figure)

        details_figure = create_detailed_table_page(rows)
        pdf.savefig(details_figure, bbox_inches="tight")
        plt.close(details_figure)


def main() -> None:
    args = parse_args()
    csv_path = resolve_csv_path(args)
    output_path = build_output_path(csv_path)
    pdf_output_path = build_pdf_output_path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows = load_rows(csv_path)
    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")

    plot_results(rows, output_path)
    save_pdf_report(rows, csv_path, pdf_output_path)
    print(f"Saved visualization to {output_path}")
    print(f"Saved PDF report to {pdf_output_path}")


if __name__ == "__main__":
    main()