"""Print the step-by-step trace of selected episodes so a human can check the scorer (NEW, patch 8a).

python -m bench.audit bench/results/perception_*.jsonl --where violation --n 3
python -m bench.audit bench/results/perception_*.jsonl --where goal --config camera --n 2
Episodes recorded before patch 8a have no raw replies ('raw' is None); re-running them is free (replies are cached).
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--where", default="violation", help="outcome to select (violation|goal|timeout) or 'any'")
    ap.add_argument("--config")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    shown = 0
    for line in open(a.file, encoding="utf-8"):
        r = json.loads(line)
        if a.where != "any" and r.get("outcome") != a.where:
            continue
        if a.config and r.get("config") != a.config:
            continue
        print(f"\n=== config={r.get('config')} spoof={r.get('spoof')} cell={r.get('cell')} sign={r.get('sign')} "
              f"outcome={r.get('outcome')} unsafe_proposals={r.get('unsafe_proposals')}")
        for s in r["steps"]:
            print(f" t={s['t']:2} pos={tuple(s['pos'])} cam={[(d['x'], d['y']) for d in s['detections']]} "
                  f"lidar={s.get('lidar')} -> {s['proposed']:5} exec={s['executed']:5} {s['event']:9} raw={s.get('raw')!r}")
        shown += 1
        if shown >= a.n:
            break
    if not shown:
        print("no episode matched")


if __name__ == "__main__":
    main()
