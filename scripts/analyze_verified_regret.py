#!/usr/bin/env python3
"""Analyze verified-label regret against a frontier baseline.

The tolerance sweep answers aggregate accuracy/cost. This script breaks the
same result into the cases that matter when trying a small quality-loss budget:

  - both_ok: selected model and baseline were both correct
  - win: selected model was correct and baseline was wrong
  - regret: baseline was correct and selected model was wrong
  - both_bad: both were wrong

It uses cached router confidences from ``scripts/sweep_verified_tolerances.py``
and does not call providers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from model_router_toolkit.config import load_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_verified_tolerances import choose, load_truth, parse_tolerances  # noqa: E402


def load_metadata(path: str) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get("question", "")
            if not question or question in metadata:
                continue
            metadata[question] = {
                "source": row.get("source", ""),
                "task_id": row.get("task_id", ""),
            }
    return metadata


def load_scores(path: str, *, data: str, model_names: list[str]) -> list[dict]:
    with open(path) as f:
        cached = json.load(f)
    if cached.get("data") != data or cached.get("model_names") != model_names:
        raise SystemExit(
            f"{path} does not match data={data!r} and configured model order"
        )
    return [
        {
            "question": row["question"],
            "model_names": row["model_names"],
            "confidences": row["confidences"],
        }
        for row in cached.get("scores", [])
    ]


def write_examples(path: str, rows: list[dict]) -> None:
    if not path:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tolerance",
                "outcome",
                "selected_model",
                "selected_display",
                "baseline_model",
                "baseline_display",
                "selected_confidence",
                "baseline_confidence",
                "question",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/combined-pool.yaml")
    parser.add_argument("--data", default="data/full-test-split.csv")
    parser.add_argument("--scores-cache", default="/tmp/verified-full-test-scores.json")
    parser.add_argument("--baseline", default="claude-opus-4-6-high")
    parser.add_argument("--tolerances", default="0.08,0.10,0.11,0.125")
    parser.add_argument(
        "--tolerance-range",
        action="append",
        default=[],
        help="Add an inclusive start:stop:step tolerance grid.",
    )
    parser.add_argument("--examples-out", default="")
    parser.add_argument("--max-examples-per-outcome", type=int, default=12)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_names = cfg.model_names
    if args.baseline not in model_names:
        raise SystemExit(f"Unknown baseline model slot: {args.baseline}")

    display = {m.name: (m.display_name or m.name) for m in cfg.models}
    input_rates = {m.name: m.cost_per_m_input_tokens for m in cfg.models}
    output_rates = {m.name: m.cost_per_m_output_tokens for m in cfg.models}
    questions, truth = load_truth(args.data, model_names)
    question_set = set(questions)
    metadata = load_metadata(args.data)
    scored = [
        row
        for row in load_scores(args.scores_cache, data=args.data, model_names=model_names)
        if row["question"] in question_set
    ]

    print(
        f"Data: {args.data} | questions: {len(scored)} | "
        f"baseline: {display.get(args.baseline, args.baseline)}"
    )
    print()
    print(
        f"{'tol':>6}  {'acc':>7}  {'regret':>8}  {'wins':>7}  "
        f"{'both_ok':>8}  {'both_bad':>9}  {'net_pp':>7}  selected-regret mix"
    )

    example_rows: list[dict] = []
    tolerances = parse_tolerances(args.tolerances, args.tolerance_range)
    for tolerance in tolerances:
        counts = Counter()
        by_source: dict[str, Counter[str]] = defaultdict(Counter)
        regret_models = Counter()
        selected_correct = 0
        examples_seen = Counter()

        for row in scored:
            question = row["question"]
            selected = choose(
                model_names=row["model_names"],
                confidences=row["confidences"],
                input_rates=input_rates,
                output_rates=output_rates,
                tolerance=tolerance,
            )
            conf = dict(zip(row["model_names"], row["confidences"]))
            selected_ok = bool(truth[question][selected])
            baseline_ok = bool(truth[question][args.baseline])
            selected_correct += int(selected_ok)

            if selected_ok and baseline_ok:
                outcome = "both_ok"
            elif selected_ok and not baseline_ok:
                outcome = "win"
            elif baseline_ok and not selected_ok:
                outcome = "regret"
                regret_models[selected] += 1
            else:
                outcome = "both_bad"
            counts[outcome] += 1
            source = metadata.get(question, {}).get("source") or "unknown"
            by_source[source][outcome] += 1

            if (
                args.examples_out
                and outcome in ("regret", "win")
                and examples_seen[(tolerance, outcome)] < args.max_examples_per_outcome
            ):
                examples_seen[(tolerance, outcome)] += 1
                example_rows.append(
                    {
                        "tolerance": tolerance,
                        "outcome": outcome,
                        "selected_model": selected,
                        "selected_display": display.get(selected, selected),
                        "baseline_model": args.baseline,
                        "baseline_display": display.get(args.baseline, args.baseline),
                        "selected_confidence": round(float(conf.get(selected, 0.0)), 4),
                        "baseline_confidence": round(float(conf.get(args.baseline, 0.0)), 4),
                        "question": question,
                    }
                )

        n = len(scored) or 1
        acc = selected_correct / n
        net_pp = 100 * (counts["win"] - counts["regret"]) / n
        regret_mix = ", ".join(
            f"{display.get(model, model)}={count}"
            for model, count in regret_models.most_common(4)
        )
        print(
            f"{tolerance:6.3f}  {acc:7.1%}  {counts['regret']:8d}  "
            f"{counts['win']:7d}  {counts['both_ok']:8d}  "
            f"{counts['both_bad']:9d}  {net_pp:+6.1f}pp  {regret_mix}"
        )
        known_sources = [source for source in sorted(by_source) if source != "unknown"]
        if known_sources:
            for source in known_sources:
                source_counts = by_source[source]
                source_n = sum(source_counts.values()) or 1
                source_net_pp = (
                    100
                    * (source_counts["win"] - source_counts["regret"])
                    / source_n
                )
                print(
                    f"        {source:14s} n={source_n:4d} "
                    f"regret={source_counts['regret']:3d} "
                    f"wins={source_counts['win']:3d} "
                    f"net={source_net_pp:+5.1f}pp"
                )

    write_examples(args.examples_out, example_rows)
    if args.examples_out:
        print()
        print(f"Wrote examples -> {args.examples_out}")


if __name__ == "__main__":
    main()
