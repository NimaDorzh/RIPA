import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path("/home/amin/robotics_ws/results")
PNG_DIR = RESULTS_DIR / "png"
TITLE = "RIPA Phase 3 — Visual Injection Test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a visual injection summary chart from a CSV file.")
    parser.add_argument("csv_path", help="Path to the visual injection CSV file")
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


def build_output_path(csv_path: Path) -> Path:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    return PNG_DIR / f"{csv_path.stem}.png"


def decision_to_value(decision: str) -> int:
    return 1 if decision.strip().upper() == "BLOCK" else 0


def decision_to_color(decision: str) -> str:
    return "#d62728" if decision.strip().upper() == "BLOCK" else "#2ca02c"


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()
    rows = load_rows(csv_path)

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    image_names = [row.get("image_file", "") for row in rows]
    decisions = [row.get("firewall_decision", "") for row in rows]
    values = [decision_to_value(decision) for decision in decisions]
    colors = [decision_to_color(decision) for decision in decisions]

    width = max(8, len(image_names) * 1.4)
    figure, axis = plt.subplots(figsize=(width, 5))
    axis.bar(image_names, values, color=colors)
    axis.set_title(TITLE)
    axis.set_xlabel("Image")
    axis.set_ylabel("Decision")
    axis.set_ylim(-0.1, 1.1)
    axis.set_yticks([0, 1], labels=["ALLOW", "BLOCK"])
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()

    output_path = build_output_path(csv_path)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

    print(output_path)


if __name__ == "__main__":
    main()