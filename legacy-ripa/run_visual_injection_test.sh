#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_DIR="$SCRIPT_DIR"
RESULTS_DIR="$WORKSPACE_DIR/results"
CSV_DIR="$RESULTS_DIR/csv"
PNG_DIR="$RESULTS_DIR/png"
INSTALL_SETUP="$WORKSPACE_DIR/install/setup.bash"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_PATH="$RESULTS_DIR/visual_injection_${TIMESTAMP}.log"
CSV_PATH="$CSV_DIR/visual_injection_${TIMESTAMP}.csv"
PNG_PATH="$PNG_DIR/visual_injection_${TIMESTAMP}.png"
PIDS=()
STARTUP_DELAY_SECONDS="${STARTUP_DELAY_SECONDS:-5}"

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM

    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" >/dev/null 2>&1; then
            kill "$pid" >/dev/null 2>&1 || true
        fi
    done

    wait >/dev/null 2>&1 || true
    exit "$exit_code"
}

prefix_to_log() {
    local node_name=$1
    while IFS= read -r line || [[ -n "$line" ]]; do
        printf '[%s] %s\n' "$node_name" "$line"
    done >> "$LOG_PATH"
}

start_node() {
    local node_name=$1
    shift

    if command -v stdbuf >/dev/null 2>&1; then
        stdbuf -oL -eL "$@" \
            > >(prefix_to_log "$node_name") \
            2> >(prefix_to_log "$node_name") &
    else
        "$@" \
            > >(prefix_to_log "$node_name") \
            2> >(prefix_to_log "$node_name") &
    fi

    PIDS+=("$!")
}

run_with_log() {
    local node_name=$1
    shift

    if command -v stdbuf >/dev/null 2>&1; then
        stdbuf -oL -eL "$@" \
            > >(prefix_to_log "$node_name") \
            2> >(prefix_to_log "$node_name")
    else
        "$@" \
            > >(prefix_to_log "$node_name") \
            2> >(prefix_to_log "$node_name")
    fi
}

trap cleanup EXIT INT TERM

mkdir -p "$CSV_DIR" "$PNG_DIR"
: > "$LOG_PATH"

if [[ -f "$WORKSPACE_DIR/venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$WORKSPACE_DIR/venv/bin/activate"
fi

if [[ ! -f "$INSTALL_SETUP" ]]; then
    echo "Missing ROS workspace setup: $INSTALL_SETUP" >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required to generate CSV and PNG outputs" >&2
    exit 1
fi

echo "Writing log to $LOG_PATH"

# shellcheck disable=SC1091
source "$INSTALL_SETUP"

start_node ocr_node ros2 run llm_robot_controller ocr_node
start_node firewall_node ros2 run llm_robot_controller firewall_node
start_node controller_node ros2 run llm_robot_controller controller_node

sleep "$STARTUP_DELAY_SECONDS"

for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
        echo "A background node exited during startup. See $LOG_PATH" >&2
        exit 1
    fi
done

run_with_log ocr_test ros2 run llm_robot_controller ocr_test

python3 "$RESULTS_DIR/parse_visual_injection_log.py" "$LOG_PATH" "$CSV_PATH" >/dev/null
python3 "$RESULTS_DIR/generate_visual_chart.py" "$CSV_PATH" >/dev/null

if [[ ! -f "$CSV_PATH" ]]; then
    echo "CSV output was not created: $CSV_PATH" >&2
    exit 1
fi

if [[ ! -f "$PNG_PATH" ]]; then
    echo "PNG output was not created: $PNG_PATH" >&2
    exit 1
fi

printf 'Test complete.\n'
printf 'Log:  results/visual_injection_%s.log\n' "$TIMESTAMP"
printf 'CSV:  results/csv/visual_injection_%s.csv\n' "$TIMESTAMP"
printf 'PNG:  results/png/visual_injection_%s.png\n' "$TIMESTAMP"