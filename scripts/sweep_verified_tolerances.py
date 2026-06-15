#!/usr/bin/env python3
"""Exact tolerance sweep on a verified long-format label CSV.

This answers the production question the aggregate AUCCC curve does not:

  At tolerance X, what verified success rate do we get, what model mix is chosen,
  and what input/output rate proxy does that imply for the currently served pool?

It runs locally: no provider APIs are called. The CSV must be in the router's
long format: question,model,isCorrect,output_tokens.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict

from model_router_toolkit.config import load_config
from model_router_toolkit.prefill.router import PrefillRouter


def load_truth(path: str, model_names: list[str]) -> tuple[list[str], dict[str, dict[str, int]]]:
    truth: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            if model in model_names:
                truth[row["question"]][model] = int(row["isCorrect"])
    questions = [
        question
        for question, labels in truth.items()
        if all(model in labels for model in model_names)
    ]
    return questions, truth


def choose(
    *,
    model_names: list[str],
    confidences: list[float],
    input_rates: dict[str, float],
    output_rates: dict[str, float],
    routing_blends: dict[str, float],
    tolerance: float,
) -> str:
    conf = dict(zip(model_names, confidences))
    p_max = max(conf.values())
    threshold = p_max - tolerance
    selected = max(
        model_names,
        key=lambda m: (routing_blends[m], input_rates[m], output_rates[m], m),
    )
    for model in sorted(
        model_names,
        key=lambda m: (routing_blends[m], input_rates[m], output_rates[m], m),
    ):
        if conf[model] >= threshold:
            selected = model
            break
    return selected


def fmt_distribution(
    counts: Counter[str],
    *,
    display: dict[str, str],
    total: int,
    max_items: int,
) -> str:
    parts = []
    for model, count in counts.most_common(max_items):
        parts.append(f"{display.get(model, model)}={100 * count / total:.0f}%")
    return ", ".join(parts)


def parse_tolerances(values: str, ranges: list[str] | None) -> list[float]:
    tolerances = [float(x) for x in values.split(",") if x.strip()]
    for item in ranges or []:
        parts = item.split(":")
        if len(parts) != 3:
            raise SystemExit(
                f"Invalid --tolerance-range {item!r}; expected start:stop:step"
            )
        start, stop, step = (float(part) for part in parts)
        if step <= 0:
            raise SystemExit("--tolerance-range step must be positive")
        current = start
        # Include stop despite floating point representation drift.
        while current <= stop + (step / 10):
            tolerances.append(round(current, 6))
            current += step
    return sorted(set(tolerances))


def pareto_frontier(rows: list[dict]) -> list[dict]:
    frontier = []
    for row in rows:
        dominated = any(
            other is not row
            and other["accuracy"] >= row["accuracy"]
            and other["blended"] <= row["blended"]
            and (
                other["accuracy"] > row["accuracy"]
                or other["blended"] < row["blended"]
            )
            for other in rows
        )
        if not dominated:
            frontier.append(row)
    return sorted(
        frontier,
        key=lambda row: (row["delta_pp"], row["blended"]),
        reverse=True,
    )


def best_saving_under_loss(
    rows: list[dict],
    *,
    baseline_acc: float,
    allowed_loss_pp: float,
) -> dict | None:
    floor = baseline_acc - (allowed_loss_pp / 100)
    candidates = [row for row in rows if row["accuracy"] >= floor]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row["blended"], -row["accuracy"]))


def export_row(row: dict, *, allowed_loss_pp: float | None = None) -> dict:
    exported = {
        "tolerance": row["tolerance"],
        "accuracy_pct": 100 * row["accuracy"],
        "delta_pp": row["delta_pp"],
        "loss_pp": max(0.0, -row["delta_pp"]),
        "avg_input_per_m": row["avg_input"],
        "avg_output_per_m": row["avg_output"],
        "blend_per_m": row["blended"],
        "savings_pct": 100 * row["savings"],
        "distribution": row["distribution"],
        "model_counts": row["model_counts"],
    }
    if allowed_loss_pp is not None:
        exported["allowed_loss_pp"] = allowed_loss_pp
    return exported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/combined-pool.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data", default="data/full-test-split.csv")
    parser.add_argument("--baseline", default="claude-opus-4-6-high")
    parser.add_argument("--tolerances", default="0,0.02,0.05,0.10,0.15,0.20,0.25,0.35")
    parser.add_argument(
        "--tolerance-range",
        action="append",
        default=[],
        help="Add an inclusive start:stop:step tolerance grid, e.g. 0.04:0.15:0.005.",
    )
    parser.add_argument(
        "--output-token-weight",
        type=float,
        default=0.25,
        help="Blend proxy: input_rate + weight * output_rate. This is only a rate proxy.",
    )
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-dist", type=int, default=5)
    parser.add_argument(
        "--loss-bands-pp",
        default="0,0.5,1,2",
        help="Quality-loss bands, in percentage points below baseline, for best-saving picks.",
    )
    parser.add_argument(
        "--show-frontier",
        type=int,
        default=12,
        help="Print up to this many non-dominated accuracy/cost operating points.",
    )
    parser.add_argument(
        "--scores-cache",
        default="",
        help="Optional JSON cache for router confidences; avoids rescoring for tolerance-only sweeps.",
    )
    parser.add_argument(
        "--select-max-loss-pp",
        type=float,
        default=None,
        help=(
            "Select the cheapest tested operating point whose accuracy is within "
            "this many percentage points of the baseline."
        ),
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path for a structured calibration JSON artifact.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = args.checkpoint or cfg.routing.checkpoint
    model_names = cfg.model_names
    display = {m.name: (m.display_name or m.name) for m in cfg.models}
    input_rates = {m.name: m.cost_per_m_input_tokens for m in cfg.models}
    output_rates = {m.name: m.cost_per_m_output_tokens for m in cfg.models}
    route_output_weight = float(getattr(cfg.routing, "output_token_weight", 0.0) or 0.0)
    routing_blends = {
        m.name: (
            m.cost_per_m_input_tokens + route_output_weight * m.cost_per_m_output_tokens
        )
        * float(getattr(m, "routing_cost_multiplier", 1.0) or 1.0)
        for m in cfg.models
    }

    questions, truth = load_truth(args.data, model_names)
    if not questions:
        raise SystemExit(f"No fully covered questions in {args.data} for {len(model_names)} models")
    if args.sample and args.sample < len(questions):
        random.seed(args.seed)
        questions = random.sample(questions, args.sample)

    scored = []
    if args.scores_cache:
        try:
            with open(args.scores_cache) as f:
                cached = json.load(f)
            if (
                cached.get("checkpoint") == checkpoint
                and cached.get("data") == args.data
                and cached.get("model_names") == model_names
            ):
                scored = [
                    (row["question"], row["model_names"], row["confidences"])
                    for row in cached.get("scores", [])
                    if row.get("question") in truth
                ]
                print(f"Loaded {len(scored)} cached scores from {args.scores_cache}")
        except (OSError, json.JSONDecodeError):
            scored = []

    if not scored:
        print(f"Loading verified-trace router: {checkpoint}")
        router = PrefillRouter(config=cfg)
        router.load(checkpoint)

        for idx, question in enumerate(questions, 1):
            result = router.route(question, tolerance=0.0)
            scored.append((question, result.model_names, [float(c) for c in result.confidences]))
            if idx % 100 == 0:
                print(f"  scored {idx}/{len(questions)}")
        if args.scores_cache:
            with open(args.scores_cache, "w") as f:
                json.dump(
                    {
                        "checkpoint": checkpoint,
                        "config": args.config,
                        "data": args.data,
                        "model_names": model_names,
                        "scores": [
                            {
                                "question": question,
                                "model_names": names,
                                "confidences": confidences,
                            }
                            for question, names, confidences in scored
                        ],
                    },
                    f,
                )
            print(f"Wrote score cache -> {args.scores_cache}")

    baseline = args.baseline
    if baseline not in model_names:
        baseline = max(
            model_names,
            key=lambda m: (
                input_rates[m] + args.output_token_weight * output_rates[m],
                m,
            ),
        )
    best_single = max(
        model_names,
        key=lambda m: sum(truth[q][m] for q in questions) / len(questions),
    )
    baseline_acc = sum(truth[q][baseline] for q in questions) / len(questions)
    best_acc = sum(truth[q][best_single] for q in questions) / len(questions)
    baseline_blended = input_rates[baseline] + args.output_token_weight * output_rates[baseline]

    print()
    print(f"Data: {args.data} | questions: {len(questions)} | models: {len(model_names)}")
    print(
        f"Baseline {display.get(baseline, baseline)}: "
        f"acc={baseline_acc:.1%}, input=${input_rates[baseline]:.2f}/M, "
        f"output=${output_rates[baseline]:.2f}/M"
    )
    print(f"Best single {display.get(best_single, best_single)}: acc={best_acc:.1%}")
    print(f"Blended proxy = input + {args.output_token_weight:.2f} * output")
    print(f"Routing cost key = input + {route_output_weight:.2f} * output, then multiplier")
    print()
    print(
        f"{'tol':>5}  {'acc':>7}  {'d_base':>7}  {'in$/M':>7}  "
        f"{'out$/M':>7}  {'blend':>7}  {'save':>6}  distribution"
    )
    rows = []
    tolerances = parse_tolerances(args.tolerances, args.tolerance_range)
    for tolerance in tolerances:
        selected: list[str] = []
        correct = 0
        for question, names, confidences in scored:
            model = choose(
                model_names=names,
                confidences=confidences,
                input_rates=input_rates,
                output_rates=output_rates,
                routing_blends=routing_blends,
                tolerance=tolerance,
            )
            selected.append(model)
            correct += truth[question][model]

        n = len(selected)
        counts = Counter(selected)
        acc = correct / n
        avg_input = sum(input_rates[m] for m in selected) / n
        avg_output = sum(output_rates[m] for m in selected) / n
        blended = avg_input + args.output_token_weight * avg_output
        savings = 1 - blended / baseline_blended if baseline_blended else 0.0
        dist = fmt_distribution(
            counts,
            display=display,
            total=n,
            max_items=args.max_dist,
        )
        print(
            f"{tolerance:5.3f}  {acc:7.1%}  {100 * (acc - baseline_acc):+6.1f}pp  "
            f"{avg_input:7.2f}  {avg_output:7.2f}  {blended:7.2f}  "
            f"{100 * savings:5.0f}%  {dist}"
        )
        rows.append(
            {
                "tolerance": tolerance,
                "accuracy": acc,
                "delta_pp": 100 * (acc - baseline_acc),
                "avg_input": avg_input,
                "avg_output": avg_output,
                "blended": blended,
                "savings": savings,
                "distribution": dist,
                "model_counts": dict(counts),
            }
        )

    print()
    frontier = pareto_frontier(rows)
    if args.show_frontier > 0:
        print("Pareto frontier from tested tolerances:")
        for row in frontier[: args.show_frontier]:
            print(
                f"  tol={row['tolerance']:.3g}, "
                f"acc={row['accuracy']:.1%} ({row['delta_pp']:+.1f}pp), "
                f"blend=${row['blended']:.2f}/M, save={100 * row['savings']:.0f}%, "
                f"{row['distribution']}"
            )
        print()

    loss_bands = [float(x) for x in args.loss_bands_pp.split(",") if x.strip()]
    best_by_loss: list[dict] = []
    print("Best-saving operating points by allowed verified-label loss:")
    for band in loss_bands:
        best = best_saving_under_loss(
            rows,
            baseline_acc=baseline_acc,
            allowed_loss_pp=band,
        )
        if best is None:
            print(f"  <= {band:.1f}pp loss: no tested tolerance qualifies")
            continue
        best_by_loss.append(export_row(best, allowed_loss_pp=band))
        print(
            f"  <= {band:.2g}pp loss: tol={best['tolerance']:.3g}, "
            f"acc={best['accuracy']:.1%} ({best['delta_pp']:+.1f}pp), "
            f"blend=${best['blended']:.2f}/M, save={100 * best['savings']:.0f}%"
        )

    selected = None
    if args.select_max_loss_pp is not None:
        selected = best_saving_under_loss(
            rows,
            baseline_acc=baseline_acc,
            allowed_loss_pp=args.select_max_loss_pp,
        )
        print()
        if selected is None:
            print(
                "Selected bounded-loss point: none qualifies for "
                f"<={args.select_max_loss_pp:g}pp loss"
            )
        else:
            print(
                "Selected bounded-loss point: "
                f"tol={selected['tolerance']:.3g}, "
                f"loss={max(0.0, -selected['delta_pp']):.1f}pp, "
                f"blend=${selected['blended']:.2f}/M, "
                f"save={100 * selected['savings']:.0f}% "
                f"(budget <= {args.select_max_loss_pp:g}pp)"
            )

    if args.json_out:
        summary = {
            "schema_version": 2,
            "description": (
                "Offline verified-label operating points for the combined "
                "verified-trace router. This is calibration evidence only; live "
                "Goose route rows are still required for real savings and quality claims."
            ),
            "config": args.config,
            "checkpoint": checkpoint,
            "data": args.data,
            "questions": len(questions),
            "models": len(model_names),
            "baseline_model": baseline,
            "baseline_display": display.get(baseline, baseline),
            "baseline_accuracy_pct": 100 * baseline_acc,
            "best_single_model": best_single,
            "best_single_display": display.get(best_single, best_single),
            "best_single_accuracy_pct": 100 * best_acc,
            "output_token_weight": args.output_token_weight,
            "selection_policy": (
                {
                    "objective": "min_blended_rate_within_verified_label_loss_budget",
                    "max_loss_pp": args.select_max_loss_pp,
                }
                if args.select_max_loss_pp is not None
                else None
            ),
            "selected_trial": (
                export_row(selected, allowed_loss_pp=args.select_max_loss_pp)
                if selected is not None
                else None
            ),
            "operating_points": [export_row(row) for row in rows],
            "pareto_frontier": [export_row(row) for row in frontier],
            "best_by_allowed_loss": best_by_loss,
        }
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")
        print()
        print(f"Wrote calibration JSON -> {args.json_out}")


if __name__ == "__main__":
    main()
