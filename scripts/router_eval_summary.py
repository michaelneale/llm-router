#!/usr/bin/env python3
"""Aggregate routed trial and routed-vs-frontier AB JSON evidence.

This is local/provider-free. It reads JSON artifacts produced by:

  - scripts/router_trial_report.py / scripts/goose-routed-task.sh
  - scripts/goose_pair_report.py / scripts/goose-ab-task.sh

The goal is an overall readout: are we seeing real routed rows, meaningful
savings, and no obvious local quality/friction regression versus Opus?
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


OUTCOMES = {"same_quality", "routed_worse", "routed_better", "inconclusive"}


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        data["_path"] = str(path)
        return data
    return None


def is_routed_trial(data: dict[str, Any]) -> bool:
    return "route" in data and "calibration" in data and "trial_tolerance" in data


def is_pair_report(data: dict[str, Any]) -> bool:
    return "routed" in data and "opus" in data and "delta" in data


def discover(paths: list[str], dirs: list[str]) -> tuple[list[dict], list[dict]]:
    routed: list[dict] = []
    pairs: list[dict] = []
    candidates: list[Path] = []
    for item in paths:
        candidates.append(Path(item).expanduser())
    for item in dirs:
        root = Path(item).expanduser()
        if root.is_file():
            candidates.append(root)
        elif root.is_dir():
            candidates.extend(sorted(root.rglob("*.json")))

    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        data = load_json(path)
        if not data:
            continue
        if is_routed_trial(data):
            routed.append(data)
        elif is_pair_report(data):
            pairs.append(data)
    return routed, pairs


def load_labels(paths: list[str]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for item in paths:
        path = Path(item).expanduser()
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    record["_label_path"] = str(path)
                    labels.append(record)
        else:
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            records = data if isinstance(data, list) else [data]
            for record in records:
                if isinstance(record, dict):
                    record["_label_path"] = str(path)
                    labels.append(record)
    return labels


def pct(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def gate_passed(data: dict) -> bool | None:
    gate = data.get("gate") or {}
    passed = gate.get("passed")
    return passed if isinstance(passed, bool) else None


def failure_counts(items: list[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in items:
        for failure in (item.get("gate") or {}).get("failures") or []:
            counts[str(failure)] += 1
    return counts


def _path_keys(path: str | None) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    keys = {str(p), p.name}
    try:
        keys.add(str(p.expanduser().resolve()))
    except OSError:
        pass
    return keys


def label_matches_pair(label: dict[str, Any], pair: dict[str, Any]) -> bool:
    pair_paths = _path_keys(pair.get("_path"))
    for key in ("artifact", "pair_json", "path"):
        if _path_keys(label.get(key)) & pair_paths:
            return True

    routed_id = str((pair.get("routed") or {}).get("id") or "")
    opus_id = str((pair.get("opus") or {}).get("id") or "")
    label_routed = str(label.get("routed_session_id") or label.get("routed_id") or "")
    label_opus = str(label.get("opus_session_id") or label.get("opus_id") or "")
    if label_routed and label_routed == routed_id:
        return not label_opus or label_opus == opus_id
    return False


def summarize_labels(labels: list[dict[str, Any]], pairs: list[dict]) -> dict:
    matched: list[dict] = []
    unmatched = 0
    for label in labels:
        outcome = str(label.get("outcome") or "").strip().lower()
        if outcome not in OUTCOMES:
            unmatched += 1
            continue
        if any(label_matches_pair(label, pair) for pair in pairs):
            matched.append({**label, "outcome": outcome})
        else:
            unmatched += 1
    counts = Counter(label["outcome"] for label in matched)
    return {
        "labels": len(labels),
        "matched": len(matched),
        "unmatched": unmatched,
        "outcomes": dict(counts),
    }


def summarize_routed(trials: list[dict]) -> dict:
    tolerances = Counter()
    real_rows = 0
    proxy_savings: list[float] = []
    cache_savings: list[float] = []
    verified_loss: list[float] = []
    verified_savings: list[float] = []
    baseline_share: list[float] = []
    gate_counts = Counter(gate_passed(trial) for trial in trials)

    for trial in trials:
        rows = int(trial.get("real_rows") or 0)
        real_rows += rows
        tolerances[f"{float(trial.get('trial_tolerance') or 0):.6f}"] += 1
        route = trial.get("route") or {}
        calibration = trial.get("calibration") or {}
        point = calibration.get("point") or {}
        for target, source in (
            (proxy_savings, route.get("proxy_savings_pct")),
            (cache_savings, route.get("cache_savings_pct")),
            (baseline_share, route.get("baseline_share_pct")),
            (verified_loss, point.get("loss_pp")),
            (verified_savings, point.get("savings_pct")),
        ):
            value = pct(source)
            if value is not None:
                target.append(value)
        if point.get("loss_pp") is None and point.get("delta_pp") is not None:
            verified_loss.append(max(0.0, -float(point["delta_pp"])))

    return {
        "trials": len(trials),
        "real_rows": real_rows,
        "gate_passed": gate_counts.get(True, 0),
        "gate_failed": gate_counts.get(False, 0),
        "gate_not_run": gate_counts.get(None, 0),
        "tolerances": dict(tolerances),
        "avg_proxy_savings_pct": mean(proxy_savings) if proxy_savings else None,
        "avg_cache_savings_pct": mean(cache_savings) if cache_savings else None,
        "avg_verified_loss_pp": mean(verified_loss) if verified_loss else None,
        "avg_verified_savings_pct": mean(verified_savings) if verified_savings else None,
        "avg_baseline_share_pct": mean(baseline_share) if baseline_share else None,
        "failures": dict(failure_counts(trials)),
    }


def summarize_pairs(pairs: list[dict]) -> dict:
    gate_counts = Counter(gate_passed(pair) for pair in pairs)
    deltas = {
        "score100": [],
        "tool_errors": [],
        "user_problem_turns": [],
        "assistant_problem_turns": [],
        "ends_on_user": [],
    }
    for pair in pairs:
        delta = pair.get("delta") or {}
        for key, values in deltas.items():
            value = pct(delta.get(key))
            if value is not None:
                values.append(value)
    return {
        "pairs": len(pairs),
        "gate_passed": gate_counts.get(True, 0),
        "gate_failed": gate_counts.get(False, 0),
        "gate_not_run": gate_counts.get(None, 0),
        "avg_delta": {
            key: mean(values) if values else None for key, values in deltas.items()
        },
        "failures": dict(failure_counts(pairs)),
    }


def fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}{suffix}"


def verdict(routed: dict, pairs: dict, labels: dict) -> str:
    if routed["trials"] == 0:
        return "missing: no routed trial JSON artifacts were found."
    if routed["real_rows"] == 0:
        return "missing: routed trial artifacts exist, but no real route rows were counted."
    if routed["gate_failed"]:
        return "not ready: at least one routed savings/calibration gate failed."
    if pairs["pairs"] == 0:
        return "savings-only: routed evidence exists, but no Opus AB pair report was found."
    if pairs["gate_failed"]:
        return "quality-risk: at least one routed-vs-Opus friction gate failed."
    outcomes = labels.get("outcomes") or {}
    if outcomes.get("routed_worse", 0) > 0:
        return "quality-risk: manual review marked at least one routed run worse."
    if outcomes.get("same_quality", 0) > 0 and routed["gate_passed"] and pairs["gate_passed"]:
        return "reviewed-promising: savings gates, AB friction gates, and manual same-quality labels exist."
    if routed["gate_passed"] and pairs["gate_passed"]:
        return "promising: savings gates and local AB friction gates passed; manual outcome review still required."
    return "partial: evidence exists, but some gates were not run."


def print_summary(routed: dict, pairs: dict, labels: dict) -> None:
    print("Router Evaluation Summary")
    print()
    print("Routed trial evidence:")
    print(
        f"  trials={routed['trials']} real_rows={routed['real_rows']} "
        f"gate_pass={routed['gate_passed']} gate_fail={routed['gate_failed']} "
        f"gate_missing={routed['gate_not_run']}"
    )
    print(
        "  avg: "
        f"proxy_savings={fmt(routed['avg_proxy_savings_pct'], '%')}, "
        f"cache_savings={fmt(routed['avg_cache_savings_pct'], '%')}, "
        f"verified_loss={fmt(routed['avg_verified_loss_pp'], 'pp')}, "
        f"verified_savings={fmt(routed['avg_verified_savings_pct'], '%')}, "
        f"baseline_share={fmt(routed['avg_baseline_share_pct'], '%')}"
    )
    if routed["tolerances"]:
        print(
            "  tolerances: "
            + ", ".join(
                f"{float(tol):.6g}={count}"
                for tol, count in sorted(routed["tolerances"].items())
            )
        )
    if routed["failures"]:
        print("  routed gate failures:")
        for failure, count in sorted(routed["failures"].items()):
            print(f"    {count}x {failure}")

    print()
    print("Routed-vs-Opus AB friction evidence:")
    print(
        f"  pairs={pairs['pairs']} gate_pass={pairs['gate_passed']} "
        f"gate_fail={pairs['gate_failed']} gate_missing={pairs['gate_not_run']}"
    )
    delta = pairs["avg_delta"]
    print(
        "  avg routed-minus-opus delta: "
        f"score100={fmt(delta['score100'])}, "
        f"tool_errors={fmt(delta['tool_errors'])}, "
        f"user_flags={fmt(delta['user_problem_turns'])}, "
        f"assistant_flags={fmt(delta['assistant_problem_turns'])}, "
        f"ends_on_user={fmt(delta['ends_on_user'])}"
    )
    if pairs["failures"]:
        print("  AB gate failures:")
        for failure, count in sorted(pairs["failures"].items()):
            print(f"    {count}x {failure}")

    print()
    print("Manual reviewed outcome labels:")
    print(
        f"  labels={labels['labels']} matched={labels['matched']} "
        f"unmatched={labels['unmatched']}"
    )
    if labels["outcomes"]:
        print(
            "  outcomes: "
            + ", ".join(
                f"{outcome}={count}"
                for outcome, count in sorted(labels["outcomes"].items())
            )
        )

    print()
    print("Verdict: " + verdict(routed, pairs, labels))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        action="append",
        default=[],
        help="Explicit routed trial or AB pair JSON artifact. Can be repeated.",
    )
    parser.add_argument(
        "--dir",
        action="append",
        default=[],
        help="Directory to scan recursively for routed trial and AB pair JSON artifacts.",
    )
    parser.add_argument(
        "--labels",
        action="append",
        default=[],
        help="JSONL/JSON manual reviewed outcome labels. Can be repeated.",
    )
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    routed_trials, pair_reports = discover(args.json, args.dir)
    labels_raw = load_labels(args.labels)
    routed = summarize_routed(routed_trials)
    pairs = summarize_pairs(pair_reports)
    labels = summarize_labels(labels_raw, pair_reports)
    print_summary(routed, pairs, labels)

    if args.json_out:
        out = Path(args.json_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "routed": routed,
                    "pairs": pairs,
                    "labels": labels,
                    "verdict": verdict(routed, pairs, labels),
                    "routed_artifacts": [trial["_path"] for trial in routed_trials],
                    "pair_artifacts": [pair["_path"] for pair in pair_reports],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"Wrote JSON summary -> {out}")


if __name__ == "__main__":
    main()
