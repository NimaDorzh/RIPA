import argparse
import ast
import csv
import json
import re
from datetime import datetime
from pathlib import Path


CSV_FIELDS = [
    "timestamp",
    "image_file",
    "ocr_text",
    "firewall_stage1",
    "firewall_stage2",
    "firewall_decision",
    "controller_action",
]
FIREWALL_MARKER = "FIREWALL_DECISION "
OCR_PATTERN = re.compile(r"Received /object_label text for (?P<image_file>[^:]+): (?P<ocr_text>'.*')")
CONTROLLER_PATTERN = re.compile(r"LLM returned command: (?P<controller_action>.+)$")
TIMESTAMP_PATTERN = re.compile(r"\[(?P<timestamp>\d+(?:\.\d+)?)\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse a visual injection log into CSV results.")
    parser.add_argument("log_path", help="Path to the combined visual injection log file")
    parser.add_argument("csv_path", help="Path to the output CSV file")
    return parser.parse_args()


def extract_timestamp(line: str) -> str:
    for match in TIMESTAMP_PATTERN.finditer(line):
        timestamp_text = match.group("timestamp")
        try:
            return datetime.fromtimestamp(float(timestamp_text)).isoformat()
        except ValueError:
            return timestamp_text
    return ""


def parse_ocr_text(value: str) -> str:
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value.strip("'\"")

    return parsed if isinstance(parsed, str) else str(parsed)


def load_records(log_path: Path) -> list[dict[str, str]]:
    ocr_records: list[dict[str, str]] = []
    firewall_records: list[dict[str, str]] = []
    controller_actions: list[str] = []

    with log_path.open("r", encoding="utf-8") as log_file:
        for raw_line in log_file:
            line = raw_line.strip()
            if not line:
                continue

            ocr_match = OCR_PATTERN.search(line)
            if ocr_match:
                ocr_records.append(
                    {
                        "timestamp": extract_timestamp(line),
                        "image_file": ocr_match.group("image_file").strip(),
                        "ocr_text": parse_ocr_text(ocr_match.group("ocr_text")),
                    }
                )
                continue

            if FIREWALL_MARKER in line:
                payload_text = line.split(FIREWALL_MARKER, maxsplit=1)[1].strip()
                payload = json.loads(payload_text)
                firewall_records.append(
                    {
                        "firewall_stage1": str(payload.get("stage1_result", "")),
                        "firewall_stage2": str(payload.get("stage2_result", "")),
                        "firewall_decision": str(payload.get("final_decision", "")),
                    }
                )
                continue

            controller_match = CONTROLLER_PATTERN.search(line)
            if controller_match:
                controller_actions.append(controller_match.group("controller_action").strip())

    row_count = max(len(ocr_records), len(firewall_records))
    controller_index = 0
    rows: list[dict[str, str]] = []

    for index in range(row_count):
        ocr_record = ocr_records[index] if index < len(ocr_records) else {}
        firewall_record = firewall_records[index] if index < len(firewall_records) else {}

        controller_action = ""
        if firewall_record.get("firewall_decision", "").upper() == "ALLOW":
            if controller_index < len(controller_actions):
                controller_action = controller_actions[controller_index]
                controller_index += 1

        rows.append(
            {
                "timestamp": str(ocr_record.get("timestamp", "")),
                "image_file": str(ocr_record.get("image_file", "")),
                "ocr_text": str(ocr_record.get("ocr_text", "")),
                "firewall_stage1": str(firewall_record.get("firewall_stage1", "")),
                "firewall_stage2": str(firewall_record.get("firewall_stage2", "")),
                "firewall_decision": str(firewall_record.get("firewall_decision", "")),
                "controller_action": controller_action,
            }
        )

    return rows


def write_csv(rows: list[dict[str, str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_path).expanduser().resolve()
    csv_path = Path(args.csv_path).expanduser().resolve()
    rows = load_records(log_path)
    write_csv(rows, csv_path)
    print(csv_path)


if __name__ == "__main__":
    main()