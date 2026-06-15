#!/usr/bin/env python3
"""Aggregate a verified-trace router route log.

The live dashboard tracks token/cost usage. This script answers a different
question from the router's own decision log: what did the router *try* to do?

It summarizes route vs pin/utility decisions, model mix, rate-proxy savings
against a baseline model, session spread, and expensive decision previews. It is
local-only and reads JSONL emitted via ROUTER_ROUTE_LOG.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from model_router_toolkit.config import load_config


def load_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def fmt_ts(ts: float | int | None) -> str:
    if not ts:
        return "unknown"
    return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")


def display_counts(
    counts: Counter[str],
    *,
    display: dict[str, str],
    rows: int,
    max_items: int,
) -> list[str]:
    out = []
    for model, count in counts.most_common(max_items):
        out.append(f"{display.get(model, model):28s} {count:5d}  {count / rows:6.1%}")
    return out


def cache_adjusted_input_cost(
    rows: list[dict],
    *,
    selected_models: set[str],
    input_rate: dict[str, float],
    baseline: str,
    cache_read_multiplier: float,
    min_context_tokens: int,
) -> tuple[float, float, int]:
    """Return (actual_cost_units, baseline_cost_units, token_weight).

    Units are "token * $/M"; ratios are what matter. We model the selected
    model's prefix as cached when the previous known row in the same session used
    the same selected model. Baseline is modeled as one always-on model per
    session, so every row after the first per session is cache-read priced.
    """
    last_selected_by_session: dict[str, str] = {}
    seen_baseline_sessions: set[str] = set()
    actual = 0.0
    base = 0.0
    tokens_total = 0

    for idx, row in enumerate(rows):
        model = row.get("selected_model")
        if model not in selected_models:
            continue
        session_key = str(row.get("session_key") or "default")
        tokens = int(row.get("context_tokens_est") or 0)
        if tokens <= 0:
            tokens = min_context_tokens
        tokens_total += tokens

        cached_actual = last_selected_by_session.get(session_key) == model
        actual_multiplier = cache_read_multiplier if cached_actual else 1.0
        actual += tokens * input_rate[model] * actual_multiplier
        last_selected_by_session[session_key] = model

        cached_baseline = session_key in seen_baseline_sessions
        baseline_multiplier = cache_read_multiplier if cached_baseline else 1.0
        base += tokens * input_rate[baseline] * baseline_multiplier
        seen_baseline_sessions.add(session_key)

    return actual, base, tokens_total


def last_route_context(rows: list[dict]) -> dict[str, str]:
    context: dict[str, str] = {}
    out: dict[str, str] = {}
    for row in rows:
        session_key = str(row.get("session_key") or "default")
        task = " ".join((row.get("task_view") or "").split())
        if task:
            context[session_key] = task
        out[id(row)] = context.get(session_key, "")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="/tmp/router-routes-real.jsonl")
    parser.add_argument("--config", default="configs/combined-pool.yaml")
    parser.add_argument("--baseline", default="claude-opus-4-6-high")
    parser.add_argument(
        "--output-token-weight",
        type=float,
        default=0.25,
        help="Blend proxy: input_rate + weight * output_rate. This is only a rate proxy.",
    )
    parser.add_argument("--max-models", type=int, default=12)
    parser.add_argument("--show-expensive", type=int, default=8)
    parser.add_argument(
        "--cache-read-multiplier",
        type=float,
        default=0.10,
        help="Input-rate multiplier for a warm same-session/same-model context cache.",
    )
    parser.add_argument(
        "--min-context-tokens",
        type=int,
        default=1000,
        help="Minimum token weight for rows without context_tokens_est.",
    )
    parser.add_argument(
        "--include-probes",
        action="store_true",
        help="Include provider-free /router/route probe rows in the aggregate.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    display = {m.name: (m.display_name or m.name) for m in cfg.models}
    litellm_model = {m.name: (m.litellm_model or m.name) for m in cfg.models}
    input_rate = {m.name: m.cost_per_m_input_tokens for m in cfg.models}
    output_rate = {m.name: m.cost_per_m_output_tokens for m in cfg.models}
    model_names = set(display)

    rows = load_rows(Path(args.log))
    if not args.include_probes:
        rows = [
            row for row in rows
            if not str(row.get("request_id") or "").startswith("probe-")
        ]
    if not rows:
        print(f"No route rows found in {args.log}")
        return

    selected_rows = [row for row in rows if row.get("selected_model") in model_names]
    if not selected_rows:
        print(f"Read {len(rows)} rows from {args.log}, but none had known selected models.")
        return

    baseline = args.baseline if args.baseline in model_names else max(
        model_names,
        key=lambda m: input_rate[m] + args.output_token_weight * output_rate[m],
    )
    baseline_blend = input_rate[baseline] + args.output_token_weight * output_rate[baseline]

    selected = [row["selected_model"] for row in selected_rows]
    avg_input = sum(input_rate[m] for m in selected) / len(selected)
    avg_output = sum(output_rate[m] for m in selected) / len(selected)
    blended = avg_input + args.output_token_weight * avg_output
    savings = 1 - blended / baseline_blend if baseline_blend else 0.0
    cache_actual, cache_base, cache_tokens = cache_adjusted_input_cost(
        rows,
        selected_models=model_names,
        input_rate=input_rate,
        baseline=baseline,
        cache_read_multiplier=args.cache_read_multiplier,
        min_context_tokens=args.min_context_tokens,
    )
    cache_input_rate = cache_actual / cache_tokens if cache_tokens else 0.0
    cache_base_rate = cache_base / cache_tokens if cache_tokens else 0.0
    cache_savings = 1 - cache_actual / cache_base if cache_base else 0.0

    sessions = {row.get("session_key") for row in selected_rows if row.get("session_key")}
    decisions = Counter(row.get("decision", "unknown") for row in rows)
    models = Counter(selected)
    raw_models = Counter(
        row.get("raw_selected_model")
        for row in rows
        if row.get("raw_selected_model") in model_names
    )
    tolerances = Counter(
        row.get("tolerance", (row.get("metadata") or {}).get("tolerance"))
        for row in rows
        if row.get("tolerance") is not None or (row.get("metadata") or {}).get("tolerance") is not None
    )

    print(f"Route log: {args.log}")
    print(f"Rows: {len(rows)} ({len(selected_rows)} with known selected model)")
    print(f"Window: {fmt_ts(rows[0].get('ts'))} -> {fmt_ts(rows[-1].get('ts'))}")
    print(f"Sessions: {len(sessions)}")
    print()
    print("Decision mix:")
    for decision, count in decisions.most_common():
        print(f"  {decision:22s} {count:5d}  {count / len(rows):6.1%}")
    if tolerances:
        print("Tolerances: " + ", ".join(f"{tol}={count}" for tol, count in tolerances.most_common()))
    print()

    print(
        f"Rate proxy vs {display.get(baseline, baseline)} "
        f"(input + {args.output_token_weight:.2f} * output):"
    )
    print(f"  avg input rate  ${avg_input:.2f}/M")
    print(f"  avg output rate ${avg_output:.2f}/M")
    print(f"  blended rate    ${blended:.2f}/M")
    print(f"  proxy savings   {100 * savings:.0f}%")
    print()
    print(
        "Cache-aware input proxy "
        f"(same session+model cached at {args.cache_read_multiplier:.2f}x):"
    )
    print(f"  token weight           {cache_tokens:,}")
    print(f"  routed input rate      ${cache_input_rate:.2f}/M")
    print(f"  always-baseline rate   ${cache_base_rate:.2f}/M")
    print(f"  cache-aware savings    {100 * cache_savings:.0f}%")
    print()

    print("Selected model mix:")
    for line in display_counts(models, display=display, rows=len(selected_rows), max_items=args.max_models):
        print("  " + line)
    print()

    if raw_models:
        print("Raw route decisions before pin/switch gates:")
        for line in display_counts(raw_models, display=display, rows=sum(raw_models.values()), max_items=args.max_models):
            print("  " + line)
        print()

    print("Provider mapping for selected models:")
    for model in models:
        print(f"  {display.get(model, model):28s} -> {litellm_model.get(model, model)}")
    print()

    if args.show_expensive > 0:
        route_context = last_route_context(selected_rows)
        ranked = sorted(
            selected_rows,
            key=lambda row: (
                input_rate[row["selected_model"]]
                + args.output_token_weight * output_rate[row["selected_model"]],
                row.get("ts") or 0,
            ),
            reverse=True,
        )
        print(f"Most expensive selected turns ({min(args.show_expensive, len(ranked))}):")
        for row in ranked[: args.show_expensive]:
            model = row["selected_model"]
            text = " ".join((row.get("task_view") or "").split())[:180]
            if not text:
                text = route_context.get(id(row), "")[:180]
            print(
                f"  {fmt_ts(row.get('ts'))} {row.get('decision', 'unknown'):18s} "
                f"{display.get(model, model):20s} {row.get('session_key', '')}"
            )
            if text:
                print(f"    {text}")


if __name__ == "__main__":
    main()
