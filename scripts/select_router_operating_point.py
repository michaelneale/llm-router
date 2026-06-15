#!/usr/bin/env python3
"""Select a router tolerance from public quality evidence and local Goose traffic.

This is local-only and does not call provider APIs.

Public verified labels answer the quality question:
  "How much verified-label loss does tolerance X imply?"

Local Goose replay answers the traffic-shape economics question:
  "On my actual agent sessions, what model mix and cache-aware savings would
   tolerance X produce?"

The selected point is the best Goose-savings point whose public verified-label
loss is inside the configured budget.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# This selector is local-only. Default to offline transformer loading so it does
# not stall or fail on Hugging Face metadata checks when the encoder is cached.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spotcheck_verified_router import (  # noqa: E402
    DB_DEFAULT,
    _connect_readonly,
    _cost_maps,
    _routing_cost_maps,
    _load_sessions,
    collect_events,
    compute_summaries,
)
from sweep_verified_tolerances import parse_tolerances  # noqa: E402

from model_router_toolkit.config import load_config  # noqa: E402
from model_router_toolkit.prefill.router import PrefillRouter  # noqa: E402


def load_calibration(path: str) -> dict:
    cal_path = Path(path).expanduser()
    if not cal_path.exists():
        raise SystemExit(f"Calibration file not found: {cal_path}")
    try:
        return json.loads(cal_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid calibration JSON {cal_path}: {exc}") from exc


def load_score_cache(path: str, *, config: str, checkpoint: str) -> dict[str, dict]:
    if not path:
        return {}
    cache_path = Path(path).expanduser()
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        return {}
    if data.get("config") != config or data.get("checkpoint") != checkpoint:
        return {}
    scores = data.get("scores")
    return scores if isinstance(scores, dict) else {}


def write_score_cache(
    path: str,
    *,
    config: str,
    checkpoint: str,
    scores: dict[str, dict],
) -> None:
    if not path:
        return
    cache_path = Path(path).expanduser()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config": config,
                "checkpoint": checkpoint,
                "scores": scores,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def loss_pp(point: dict) -> float | None:
    if point.get("loss_pp") is not None:
        return float(point["loss_pp"])
    if point.get("delta_pp") is not None:
        return max(0.0, -float(point["delta_pp"]))
    return None


def calibration_points(calibration: dict) -> dict[str, dict]:
    points = calibration.get("operating_points") or []
    selected = calibration.get("selected_trial")
    if selected:
        points = [*points, selected]
    out = {}
    for point in points:
        if point.get("tolerance") is None:
            continue
        out[f"{float(point['tolerance']):.6f}"] = point
    return out


def reprice_calibration_point(
    point: dict,
    *,
    rates: dict[str, tuple[float, float]],
    baseline: str,
    output_token_weight: float,
) -> dict:
    """Recompute calibration cost fields with the currently loaded pool rates."""

    counts = point.get("model_counts") or {}
    total = sum(int(count) for count in counts.values())
    if total <= 0 or baseline not in rates:
        return dict(point)

    missing = sorted(model for model in counts if model not in rates)
    if missing:
        out = dict(point)
        out["repricing_missing_models"] = missing
        return out

    avg_input = sum(int(count) * rates[model][0] for model, count in counts.items()) / total
    avg_output = sum(int(count) * rates[model][1] for model, count in counts.items()) / total
    blend = avg_input + output_token_weight * avg_output
    base_input, base_output = rates[baseline]
    baseline_blend = base_input + output_token_weight * base_output
    savings = 0.0
    if baseline_blend > 0:
        savings = 100.0 * (1.0 - (blend / baseline_blend))

    out = dict(point)
    out.update(
        {
            "avg_input_per_m": avg_input,
            "avg_output_per_m": avg_output,
            "blend_per_m": blend,
            "savings_pct": savings,
            "repriced_against_baseline": baseline,
        }
    )
    return out


def default_loss_budget(calibration: dict) -> float:
    policy = calibration.get("selection_policy") or {}
    if policy.get("max_loss_pp") is not None:
        return float(policy["max_loss_pp"])
    selected = calibration.get("selected_trial") or {}
    if selected.get("allowed_loss_pp") is not None:
        return float(selected["allowed_loss_pp"])
    return 2.0


def model_mix(model_counts: dict[str, int], display: dict[str, str], *, limit: int) -> str:
    total = sum(model_counts.values()) or 1
    counts = Counter(model_counts)
    parts = []
    for model, count in counts.most_common(limit):
        parts.append(f"{display.get(model, model)}={100 * count / total:.0f}%")
    return ", ".join(parts)


def objective_value(row: dict, objective: str) -> float:
    if objective == "goose_cache_savings":
        return float(row["goose_cache_savings_pct"])
    if objective == "goose_proxy_savings":
        return float(row["goose_proxy_savings_pct"])
    if objective == "public_blend":
        return -float(row["verified_blend_per_m"])
    raise AssertionError(objective)


def tie_break_value(row: dict, tie_breaker: str) -> tuple:
    loss = (
        float(row["verified_loss_pp"])
        if row.get("verified_loss_pp") is not None
        else float("inf")
    )
    verified_savings = (
        float(row["verified_savings_pct"])
        if row.get("verified_savings_pct") is not None
        else float("-inf")
    )
    tolerance = float(row["tolerance"])
    if tie_breaker == "lower_loss":
        return (-loss, verified_savings, float(row["goose_proxy_savings_pct"]), -tolerance)
    if tie_breaker == "verified_savings":
        return (verified_savings, float(row["goose_proxy_savings_pct"]), -loss, tolerance)
    if tie_breaker == "higher_tolerance":
        return (tolerance, verified_savings, -loss)
    raise AssertionError(tie_breaker)


def select_candidate(
    rows: list[dict],
    *,
    max_loss: float,
    objective: str,
    tie_breaker: str,
    objective_tie_slack: float,
) -> dict | None:
    candidates = [
        row
        for row in rows
        if row.get("verified_loss_pp") is not None
        and float(row["verified_loss_pp"]) <= max_loss
    ]
    if not candidates:
        return None
    best_value = max(objective_value(row, objective) for row in candidates)
    near_best = [
        row
        for row in candidates
        if objective_value(row, objective) >= best_value - objective_tie_slack
    ]
    selected = max(
        near_best,
        key=lambda row: tie_break_value(row, tie_breaker),
    )
    out = dict(selected)
    out["selection_objective_value"] = objective_value(selected, objective)
    out["selection_best_objective_value"] = best_value
    out["selection_objective_tie_slack"] = objective_tie_slack
    out["selection_near_best_candidates"] = len(near_best)
    out["selection_tie_breaker"] = tie_breaker
    return out


def strict_best_candidate(rows: list[dict], *, max_loss: float, objective: str) -> dict | None:
    candidates = [
        row
        for row in rows
        if row.get("verified_loss_pp") is not None
        and float(row["verified_loss_pp"]) <= max_loss
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            objective_value(row, objective),
            float(row["goose_proxy_savings_pct"]),
            -float(row["verified_loss_pp"]),
            -float(row["tolerance"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/combined-pool.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--calibration", default="configs/combined-calibration.json")
    parser.add_argument("--db", default=str(DB_DEFAULT))
    parser.add_argument("--sessions", type=int, default=24)
    parser.add_argument("--model-like", default="%nvidia-routed%")
    parser.add_argument("--max-user-turns", type=int, default=0)
    parser.add_argument("--baseline", default="claude-opus-4-6-high")
    parser.add_argument(
        "--tolerances",
        default="",
        help="Extra comma-separated tolerances to replay in addition to calibration points.",
    )
    parser.add_argument(
        "--tolerance-range",
        action="append",
        default=[],
        help="Extra inclusive start:stop:step tolerance grid.",
    )
    parser.add_argument("--max-loss-pp", type=float, default=None)
    parser.add_argument(
        "--loss-bands-pp",
        default="0,0.5,1,1.5,2",
        help=(
            "Comma-separated verified-label loss budgets to summarize. Include "
            "0 to show the equivalent-quality operating point."
        ),
    )
    parser.add_argument(
        "--objective",
        choices=("goose_cache_savings", "goose_proxy_savings", "public_blend"),
        default="goose_cache_savings",
    )
    parser.add_argument(
        "--tie-breaker",
        choices=("lower_loss", "verified_savings", "higher_tolerance"),
        default="lower_loss",
        help=(
            "How to choose when several tolerances are effectively tied on the "
            "primary objective. lower_loss is conservative; verified_savings is "
            "savings-first within the loss budget."
        ),
    )
    parser.add_argument(
        "--objective-tie-slack",
        type=float,
        default=0.0,
        help=(
            "Treat objective values within this amount of the best value as a tie. "
            "For savings objectives this is percentage points."
        ),
    )
    parser.add_argument("--cache-read-multiplier", type=float, default=0.10)
    parser.add_argument("--min-context-tokens", type=int, default=1000)
    parser.add_argument("--output-token-weight", type=float, default=0.25)
    parser.add_argument("--mix-items", type=int, default=4)
    parser.add_argument(
        "--scores-cache",
        default="/tmp/router-goose-route-scores.json",
        help=(
            "Cache local router confidence vectors by hashed task view. Stores no "
            "prompt text. Use an empty string to disable."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the recommendation and loss-budget ladder.",
    )
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = args.checkpoint or cfg.routing.checkpoint
    calibration = load_calibration(args.calibration)
    cal_points = calibration_points(calibration)
    if not cal_points:
        raise SystemExit(f"No operating points found in {args.calibration}")

    extra_tolerances = parse_tolerances(args.tolerances, args.tolerance_range)
    tolerances = sorted(
        {
            *(float(point["tolerance"]) for point in cal_points.values()),
            *extra_tolerances,
        }
    )

    db = _connect_readonly(Path(args.db).expanduser())
    sessions = _load_sessions(db, args.model_like, args.sessions)
    if not sessions:
        raise SystemExit(f"No Goose sessions matched model_config_json LIKE {args.model_like!r}")

    print(f"Loading verified-trace router: {checkpoint}", file=sys.stderr)
    router = PrefillRouter(config=cfg)
    router.load(checkpoint)

    score_cache = load_score_cache(
        args.scores_cache,
        config=args.config,
        checkpoint=checkpoint,
    )
    if args.scores_cache and score_cache:
        print(
            f"Loaded {len(score_cache)} cached Goose route scores from {args.scores_cache}",
            file=sys.stderr,
        )

    def flush_score_cache(scores: dict[str, dict]) -> None:
        write_score_cache(
            args.scores_cache,
            config=args.config,
            checkpoint=checkpoint,
            scores=scores,
        )

    events = collect_events(
        db=db,
        sessions=sessions,
        router=router,
        max_user_turns=args.max_user_turns or None,
        score_cache=score_cache if args.scores_cache else None,
        score_cache_flush=flush_score_cache if args.scores_cache else None,
        progress_every=args.progress_every,
    )
    if args.scores_cache:
        write_score_cache(
            args.scores_cache,
            config=args.config,
            checkpoint=checkpoint,
            scores=score_cache,
        )
    rates, display = _cost_maps(args.config)
    routing_rates = _routing_cost_maps(args.config)
    baseline = args.baseline if args.baseline in rates else max(
        rates,
        key=lambda model: (
            rates[model][0] + args.output_token_weight * rates[model][1],
            model,
        ),
    )
    summaries = compute_summaries(
        events=events,
        rates=rates,
        routing_rates=routing_rates,
        tolerances=tolerances,
        baseline=baseline,
        cache_read_multiplier=args.cache_read_multiplier,
        min_context_tokens=args.min_context_tokens,
        output_token_weight=args.output_token_weight,
    )

    rows: list[dict] = []
    for summary in summaries:
        key = f"{float(summary['tolerance']):.6f}"
        raw_point = cal_points.get(key)
        point = (
            reprice_calibration_point(
                raw_point,
                rates=rates,
                baseline=baseline,
                output_token_weight=args.output_token_weight,
            )
            if raw_point
            else None
        )
        row = {
            "tolerance": float(summary["tolerance"]),
            "has_verified_calibration": point is not None,
            "verified_loss_pp": loss_pp(point) if point else None,
            "verified_delta_pp": float(point["delta_pp"]) if point and point.get("delta_pp") is not None else None,
            "verified_savings_pct": float(point["savings_pct"]) if point and point.get("savings_pct") is not None else None,
            "verified_blend_per_m": float(point["blend_per_m"]) if point and point.get("blend_per_m") is not None else None,
            "goose_proxy_savings_pct": float(summary["proxy_savings_pct"]),
            "goose_cache_savings_pct": float(summary["cache_savings_pct"]),
            "goose_blend_per_m": float(summary["blend_per_m"]),
            "goose_baseline_share_pct": float(summary["baseline_share_pct"]),
            "counted_turns": int(summary["counted_turns"]),
            "skipped_cold_turns": int(summary["skipped_cold_turns"]),
            "model_counts": summary["model_counts"],
            "reason_counts": summary["reason_counts"],
        }
        rows.append(row)

    max_loss = args.max_loss_pp if args.max_loss_pp is not None else default_loss_budget(calibration)
    loss_bands = [float(x) for x in args.loss_bands_pp.split(",") if x.strip()]
    selected_by_loss: list[dict] = []
    for band in loss_bands:
        band_selected = select_candidate(
            rows,
            max_loss=band,
            objective=args.objective,
            tie_breaker=args.tie_breaker,
            objective_tie_slack=max(0.0, args.objective_tie_slack),
        )
        if band_selected is not None:
            band_row = dict(band_selected)
            band_row["allowed_loss_pp"] = band
            selected_by_loss.append(band_row)
    selected = select_candidate(
        rows,
        max_loss=max_loss,
        objective=args.objective,
        tie_breaker=args.tie_breaker,
        objective_tie_slack=max(0.0, args.objective_tie_slack),
    )
    strict_best = strict_best_candidate(rows, max_loss=max_loss, objective=args.objective)
    current_tolerance = float(cfg.routing.tolerance)
    current = min(rows, key=lambda row: abs(row["tolerance"] - current_tolerance))
    current_matches_selected = (
        selected is not None and abs(float(selected["tolerance"]) - current_tolerance) < 1e-9
    )

    print("Router Operating Point Selector")
    calibration_baseline = display.get(baseline) or calibration.get("baseline_display") or baseline
    print(
        f"Quality calibration: {args.calibration} "
        f"({calibration.get('questions', '?')} questions, "
        f"baseline={calibration_baseline})"
    )
    print(
        f"Goose replay: sessions={len(sessions)}, events={len(events)}, "
        f"route_decisions={sum(1 for event in events if event['kind'] == 'route')}, "
        f"model_like={args.model_like}"
    )
    print(
        f"Objective: {args.objective} under <= {max_loss:g}pp verified-label loss; "
        f"tie_breaker={args.tie_breaker}, tie_slack={max(0.0, args.objective_tie_slack):g}"
    )
    if not args.summary_only:
        print()
        print(
            f"{'tol':>6}  {'v_loss':>7}  {'v_save':>7}  {'g_proxy':>8}  "
            f"{'g_cache':>8}  {'opus%':>6}  mix"
        )
        for row in rows:
            marker = "* " if selected and row["tolerance"] == selected["tolerance"] else "  "
            v_loss = "-" if row["verified_loss_pp"] is None else f"{row['verified_loss_pp']:.1f}"
            v_save = "-" if row["verified_savings_pct"] is None else f"{row['verified_savings_pct']:.0f}%"
            print(
                f"{marker}{row['tolerance']:6.3f}  {v_loss:>7}  {v_save:>7}  "
                f"{row['goose_proxy_savings_pct']:7.0f}%  "
                f"{row['goose_cache_savings_pct']:7.0f}%  "
                f"{row['goose_baseline_share_pct']:5.0f}%  "
                f"{model_mix(row['model_counts'], display, limit=args.mix_items)}"
            )
        print()
    if selected is None:
        print("Recommendation: no calibrated tolerance satisfies the verified-label loss budget.")
    else:
        print(
            "Recommendation: "
            f"tol={selected['tolerance']:.3g}, "
            f"verified_loss={selected['verified_loss_pp']:.1f}pp, "
            f"goose_proxy_savings={selected['goose_proxy_savings_pct']:.0f}%, "
            f"goose_cache_savings={selected['goose_cache_savings_pct']:.0f}%"
        )
        if (
            strict_best is not None
            and abs(float(strict_best["tolerance"]) - float(selected["tolerance"])) > 1e-9
        ):
            print(
                "  note: strict best on the primary objective was "
                f"tol={strict_best['tolerance']:.3g}; selected used "
                f"{selected['selection_near_best_candidates']} near-best candidates "
                f"and {args.tie_breaker} tie-break."
            )
    if selected_by_loss:
        print()
        print("Quality-budget ladder from public labels + local Goose replay:")
        print(
            f"{'mode':>13}  {'budget':>8}  {'tol':>6}  {'v_loss':>7}  {'v_save':>7}  "
            f"{'g_proxy':>8}  {'g_cache':>8}  {'opus%':>6}"
        )
        for row in selected_by_loss:
            mode = "same-quality" if row["allowed_loss_pp"] == 0 else "bounded-loss"
            print(
                f"{mode:>13}  "
                f"{row['allowed_loss_pp']:7.1f}pp  "
                f"{row['tolerance']:6.3f}  "
                f"{row['verified_loss_pp']:6.1f}pp  "
                f"{row['verified_savings_pct']:6.0f}%  "
                f"{row['goose_proxy_savings_pct']:7.0f}%  "
                f"{row['goose_cache_savings_pct']:7.0f}%  "
                f"{row['goose_baseline_share_pct']:5.0f}%"
            )
    print(
        f"Current config tolerance: {current_tolerance:.3g} "
        f"({'matches recommendation' if current_matches_selected else 'differs from recommendation'})"
    )
    if current:
        print(
            f"Current replay row: verified_loss={current.get('verified_loss_pp', '-')}, "
            f"goose_proxy_savings={current['goose_proxy_savings_pct']:.0f}%, "
            f"goose_cache_savings={current['goose_cache_savings_pct']:.0f}%"
        )

    if args.json_out:
        out = Path(args.json_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "config": args.config,
                    "checkpoint": checkpoint,
                    "calibration": args.calibration,
                    "db": str(Path(args.db).expanduser()),
                    "sessions": [
                        {
                            "id": session["id"],
                            "name": session["name"],
                            "updated_at": session["updated_at"],
                        }
                        for session in sessions
                    ],
                    "event_count": len(events),
                    "kind_counts": dict(Counter(event["kind"] for event in events)),
                    "route_decisions": sum(1 for event in events if event["kind"] == "route"),
                    "max_loss_pp": max_loss,
                    "loss_bands_pp": loss_bands,
                    "objective": args.objective,
                    "tie_breaker": args.tie_breaker,
                    "objective_tie_slack": max(0.0, args.objective_tie_slack),
                    "calibration_costs_repriced": True,
                    "scores_cache": args.scores_cache,
                    "score_cache_entries": len(score_cache),
                    "current_tolerance": current_tolerance,
                    "current_matches_selected": current_matches_selected,
                    "selected": selected,
                    "selected_by_loss": selected_by_loss,
                    "strict_best": strict_best,
                    "rows": rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print()
        print(f"Wrote selector JSON -> {out}")


if __name__ == "__main__":
    main()
