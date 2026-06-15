#!/usr/bin/env python3
"""Append a manual reviewed outcome label for a routed-vs-frontier AB pair."""

from __future__ import annotations

import argparse
import getpass
import json
import time
from pathlib import Path


OUTCOMES = ("same_quality", "routed_worse", "routed_better", "inconclusive")


def load_pair(path: str) -> dict:
    pair_path = Path(path).expanduser()
    if not pair_path.exists():
        raise SystemExit(f"Pair JSON not found: {pair_path}")
    try:
        data = json.loads(pair_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid pair JSON {pair_path}: {exc}") from exc
    if not isinstance(data, dict) or "routed" not in data or "opus" not in data:
        raise SystemExit(f"Not a goose_pair_report JSON artifact: {pair_path}")
    data["_path"] = str(pair_path)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="/tmp/router-eval/labels.jsonl")
    parser.add_argument("--pair-json", required=True)
    parser.add_argument("--outcome", required=True, choices=OUTCOMES)
    parser.add_argument("--notes", default="")
    parser.add_argument("--reviewer", default=getpass.getuser())
    args = parser.parse_args()

    pair = load_pair(args.pair_json)
    labels = Path(args.labels).expanduser()
    labels.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "ts": time.time(),
        "reviewer": args.reviewer,
        "pair_json": str(Path(args.pair_json).expanduser()),
        "routed_session_id": (pair.get("routed") or {}).get("id"),
        "opus_session_id": (pair.get("opus") or {}).get("id"),
        "routed_name": (pair.get("routed") or {}).get("name"),
        "opus_name": (pair.get("opus") or {}).get("name"),
        "outcome": args.outcome,
        "notes": args.notes,
    }
    with labels.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"Appended {args.outcome} label -> {labels}")


if __name__ == "__main__":
    main()
