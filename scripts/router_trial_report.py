#!/usr/bin/env python3
"""One-command readout for a Goose routed trial.

This is local/provider-free. It combines:

  - live router route-log rows, excluding dry probes by default
  - rate-proxy and cache-aware savings estimates
  - mapping route-log session keys back to Goose sessions
  - weak quality/friction signals for matched routed sessions

The report is deliberately conservative: no live route rows means no real
savings claim, even if Goose has historical sessions requesting nvidia-routed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_goose_quality_signals import (
    DB_DEFAULT,
    analyze_session,
    connect_readonly,
    direct_text,
    group_key,
    load_messages,
    load_sessions,
    model_name,
    parse_content,
    summarize_group,
)
from analyze_route_log import cache_adjusted_input_cost, fmt_ts, load_rows
from model_router_toolkit.config import load_config


def is_probe(row: dict) -> bool:
    return str(row.get("request_id") or "").startswith("probe-")


def marker_timestamp(path: str) -> float | None:
    if not path:
        return None
    marker_path = Path(path).expanduser()
    if not marker_path.exists():
        raise SystemExit(f"Marker file not found: {marker_path}")
    text = marker_path.read_text().strip()
    if not text:
        raise SystemExit(f"Marker file is empty: {marker_path}")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = text
    if isinstance(value, dict):
        value = value.get("ts")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Marker file does not contain a numeric timestamp: {marker_path}") from exc


def write_marker(path: str) -> float:
    marker_path = Path(path).expanduser()
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.time()
    marker_path.write_text(json.dumps({"ts": ts}, sort_keys=True) + "\n")
    return ts


def filter_rows_since(rows: list[dict], since_ts: float | None) -> list[dict]:
    if since_ts is None:
        return rows
    return [row for row in rows if float(row.get("ts") or 0.0) >= since_ts]


def real_route_rows(path: Path, *, since_ts: float | None = None) -> list[dict]:
    rows = filter_rows_since(load_rows(path), since_ts)
    return [row for row in rows if not is_probe(row)]


def wait_for_real_rows(
    path: Path,
    *,
    min_rows: int,
    timeout_seconds: float,
    poll_seconds: float,
    since_ts: float | None,
) -> list[dict]:
    deadline = time.time() + timeout_seconds
    last_count = -1
    while True:
        rows = real_route_rows(path, since_ts=since_ts)
        if len(rows) >= min_rows:
            return rows
        if len(rows) != last_count:
            print(
                f"Waiting for real route rows: {len(rows)}/{min_rows} "
                f"in {path}"
            )
            last_count = len(rows)
        if time.time() >= deadline:
            print(
                f"Timed out waiting for {min_rows} real route rows "
                f"after {timeout_seconds:.1f}s."
            )
            return rows
        time.sleep(max(0.1, poll_seconds))


def first_user_text(db: sqlite3.Connection, session_id: str) -> str:
    for row in load_messages(db, session_id):
        if row["role"] != "user":
            continue
        text = direct_text(parse_content(row["content_json"])).strip()
        if text:
            return text
    return ""


def first_user_route_key(text: str) -> str:
    digest = hashlib.sha256(text[:2000].encode()).hexdigest()[:16]
    return f"first-user:{digest}"


def session_aliases(db: sqlite3.Connection, sessions: list[sqlite3.Row]) -> dict[str, sqlite3.Row]:
    aliases: dict[str, sqlite3.Row] = {}
    for session in sessions:
        aliases[session["id"]] = session
        text = first_user_text(db, session["id"])
        if text:
            aliases[first_user_route_key(text)] = session
    return aliases


def fetch_savings(base_url: str, timeout: float) -> dict | None:
    if not base_url:
        return None
    try:
        with urlopen(f"{base_url.rstrip('/')}/savings", timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (OSError, URLError, json.JSONDecodeError):
        return None


def row_tolerance(row: dict) -> float | None:
    value = row.get("tolerance")
    if value is None:
        value = (row.get("metadata") or {}).get("tolerance")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def observed_tolerances(rows: list[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        tolerance = row_tolerance(row)
        if tolerance is not None:
            counts[f"{tolerance:.6f}"] += 1
    return counts


def load_calibration(path: str) -> dict | None:
    if not path:
        return None
    cal_path = Path(path).expanduser()
    if not cal_path.exists():
        raise SystemExit(f"Calibration file not found: {cal_path}")
    try:
        return json.loads(cal_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid calibration JSON {cal_path}: {exc}") from exc


def calibration_point(calibration: dict | None, tolerance: float) -> dict | None:
    if not calibration:
        return None
    selected = calibration.get("selected_trial") or {}
    if selected and abs(float(selected.get("tolerance", -1)) - tolerance) < 1e-9:
        return selected
    points = calibration.get("operating_points") or []
    if not points:
        return None
    return min(points, key=lambda row: abs(float(row.get("tolerance", -1)) - tolerance))


def reprice_calibration_point(
    point: dict | None,
    *,
    cfg,
    baseline: str,
    output_token_weight: float,
) -> dict | None:
    """Recompute calibration economics from model counts and the active pool.

    The verified-label accuracy/loss comes from the calibration artifact; costs
    and display labels come from the pool actually being served.
    """
    if not point:
        return point
    counts = point.get("model_counts") or {}
    total = sum(int(v) for v in counts.values())
    models = {m.name: m for m in getattr(cfg, "models", [])}
    if total <= 0 or baseline not in models:
        return dict(point)
    if any(name not in models for name in counts):
        return dict(point)

    out = dict(point)
    avg_input = (
        sum(models[name].cost_per_m_input_tokens * int(count) for name, count in counts.items())
        / total
    )
    avg_output = (
        sum(models[name].cost_per_m_output_tokens * int(count) for name, count in counts.items())
        / total
    )
    blend = avg_input + output_token_weight * avg_output
    baseline_spec = models[baseline]
    baseline_blend = (
        baseline_spec.cost_per_m_input_tokens
        + output_token_weight * baseline_spec.cost_per_m_output_tokens
    )
    ranked = Counter({name: int(count) for name, count in counts.items()})
    parts = []
    for name, count in ranked.most_common(5):
        display = models[name].display_name or name
        parts.append(f"{display}={100 * count / total:.0f}%")
    out.update(
        {
            "avg_input_per_m": avg_input,
            "avg_output_per_m": avg_output,
            "blend_per_m": blend,
            "savings_pct": (
                100 * (1 - blend / baseline_blend) if baseline_blend else 0.0
            ),
            "distribution": ", ".join(parts),
            "costs_repriced": True,
        }
    )
    return out


def print_calibration(
    calibration: dict | None,
    tolerance: float,
    *,
    cfg,
    baseline: str,
    output_token_weight: float,
) -> dict:
    point = reprice_calibration_point(
        calibration_point(calibration, tolerance),
        cfg=cfg,
        baseline=baseline,
        output_token_weight=output_token_weight,
    )
    if not calibration:
        print("Offline verified-label calibration: unavailable")
        return {"available": False, "point": None}
    baseline_spec = cfg.get_model(baseline)
    baseline_display = (
        baseline_spec.display_name
        if baseline_spec is not None
        else calibration.get("baseline_display") or calibration.get("baseline_model", "-")
    )
    print("Offline verified-label calibration:")
    print(
        f"  source={calibration.get('data', '-')}, "
        f"questions={calibration.get('questions', '-')}, "
        f"baseline={baseline_display}"
    )
    policy = calibration.get("selection_policy") or {}
    selected = reprice_calibration_point(
        calibration.get("selected_trial") or {},
        cfg=cfg,
        baseline=baseline,
        output_token_weight=output_token_weight,
    )
    if policy:
        print(
            "  policy="
            f"{policy.get('objective', 'selected operating point')}, "
            f"max_loss={policy.get('max_loss_pp', 'n/a')}pp"
        )
    if selected:
        print(
            f"  selected tol={float(selected['tolerance']):.3g}: "
            f"acc={float(selected.get('accuracy_pct', 0)):.1f}%, "
            f"delta={float(selected.get('delta_pp', 0)):+.1f}pp, "
            f"savings={float(selected.get('savings_pct', 0)):.0f}%"
        )
    if point:
        approx = "approx " if point.get("approximate") else ""
        match = (
            "matches selected"
            if selected
            and abs(float(selected.get("tolerance", -1)) - tolerance) < 1e-9
            else "nearest calibration point"
        )
        print(
            f"  trial tol={float(point['tolerance']):.3g} ({match}): "
            f"acc={approx}{float(point.get('accuracy_pct', 0)):.1f}%, "
            f"delta={approx}{float(point.get('delta_pp', 0)):+.1f}pp, "
            f"savings={approx}{float(point.get('savings_pct', 0)):.0f}%"
        )
    else:
        print(f"  no operating point found near tol={tolerance:.3g}")
    return {
        "available": True,
        "point": point,
        "selected_trial": selected or None,
        "selection_policy": policy or None,
        "selected_matches_config": (
            bool(selected)
            and abs(float(selected.get("tolerance", -1)) - tolerance) < 1e-9
        ),
        "raw": calibration,
    }


def fmt_money(value) -> str:
    if value is None or value == "":
        return "-"
    return f"${float(value):.4f}"


def print_recent_routed_sessions(sessions: list[sqlite3.Row], show: int) -> None:
    routed = [s for s in sessions if model_name(s["model_config_json"]) == "nvidia-routed"]
    print(f"Recent Goose sessions requesting nvidia-routed: {len(routed)}")
    for session in routed[:show]:
        print(
            f"  {str(session['updated_at'])[:19]} {session['id']:12s} "
            f"tokens={session['total_tokens'] or '-':>8} "
            f"cost={fmt_money(session['accumulated_cost']):>9}  "
            f"{session['name']}"
        )
    missing_cost = sum(1 for s in routed if s["accumulated_cost"] is None)
    if missing_cost:
        print(f"  note: {missing_cost} routed sessions have no Goose accumulated_cost.")


def print_route_summary(
    *,
    rows: list[dict],
    config_path: str,
    baseline: str,
    output_token_weight: float,
    cache_read_multiplier: float,
    min_context_tokens: int,
    show: int,
) -> tuple[set[str], dict[str, Counter[str]], dict]:
    cfg = load_config(config_path)
    display = {m.name: (m.display_name or m.name) for m in cfg.models}
    input_rate = {m.name: m.cost_per_m_input_tokens for m in cfg.models}
    output_rate = {m.name: m.cost_per_m_output_tokens for m in cfg.models}
    model_names = set(display)
    baseline_model = baseline if baseline in model_names else max(
        model_names,
        key=lambda m: input_rate[m] + output_token_weight * output_rate[m],
    )
    selected_rows = [row for row in rows if row.get("selected_model") in model_names]
    selected = [row["selected_model"] for row in selected_rows]
    models = Counter(selected)
    decisions = Counter(row.get("decision", "unknown") for row in rows)
    raw = Counter(
        row.get("raw_selected_model")
        for row in rows
        if row.get("raw_selected_model") in model_names
    )
    metrics = {
        "rows": len(rows),
        "known_selected_rows": len(selected_rows),
        "decision_counts": dict(decisions),
        "model_counts": dict(models),
        "raw_model_counts": dict(raw),
        "baseline_model": baseline_model,
        "proxy_savings_pct": None,
        "cache_savings_pct": None,
        "blend_rate_per_m": None,
        "cache_token_weight": 0,
        "baseline_share_pct": (
            100 * models.get(baseline_model, 0) / len(selected)
            if selected
            else 0.0
        ),
    }

    print(f"Analyzed route rows: {len(rows)} ({len(selected_rows)} with known selected model)")
    if not rows:
        print("  no real route-log rows yet")
        return set(), {}, metrics
    print(f"Window: {fmt_ts(rows[0].get('ts'))} -> {fmt_ts(rows[-1].get('ts'))}")
    print("Decision mix: " + ", ".join(f"{k}={v}" for k, v in decisions.most_common()))

    if selected_rows:
        baseline_blend = (
            input_rate[baseline_model]
            + output_token_weight * output_rate[baseline_model]
        )
        avg_input = sum(input_rate[m] for m in selected) / len(selected)
        avg_output = sum(output_rate[m] for m in selected) / len(selected)
        blended = avg_input + output_token_weight * avg_output
        savings = 1 - blended / baseline_blend if baseline_blend else 0.0
        cache_actual, cache_base, cache_tokens = cache_adjusted_input_cost(
            rows,
            selected_models=model_names,
            input_rate=input_rate,
            baseline=baseline_model,
            cache_read_multiplier=cache_read_multiplier,
            min_context_tokens=min_context_tokens,
        )
        cache_savings = 1 - cache_actual / cache_base if cache_base else 0.0
        metrics.update(
            {
                "proxy_savings_pct": 100 * savings,
                "cache_savings_pct": 100 * cache_savings,
                "blend_rate_per_m": blended,
                "cache_token_weight": cache_tokens,
            }
        )
        print(
            f"Rate proxy vs {display.get(baseline_model, baseline_model)}: "
            f"blend=${blended:.2f}/M, savings={100 * savings:.0f}%"
        )
        print(
            f"Cache-aware input proxy: token_weight={cache_tokens:,}, "
            f"savings={100 * cache_savings:.0f}%"
        )
        print("Selected model mix:")
        for model, count in models.most_common(show):
            print(f"  {display.get(model, model):24s} {count:5d}  {count / len(selected):6.1%}")
        if raw:
            print("Raw route decisions before pin/switch gates:")
            for model, count in raw.most_common(show):
                print(f"  {display.get(model, model):24s} {count:5d}  {count / sum(raw.values()):6.1%}")

    session_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in selected_rows:
        session_key = str(row.get("session_key") or "")
        session_counts[session_key][row["selected_model"]] += 1
    return set(session_counts), session_counts, metrics


def signal_metrics(signals: list) -> dict:
    messages = sum(s.messages for s in signals)
    tool_errors = sum(s.tool_errors for s in signals)
    tool_responses = sum(s.tool_responses for s in signals)
    user_problem = sum(s.user_problem_turns for s in signals)
    user_text_turns = sum(s.user_text_turns for s in signals)
    assistant_problem = sum(s.assistant_problem_turns for s in signals)
    assistant_text_turns = sum(s.assistant_text_turns for s in signals)
    ended_user = sum(1 for s in signals if s.ends_on_user)
    return {
        "matched_sessions": len(signals),
        "messages": messages,
        "problem_score_total": sum(s.problem_score for s in signals),
        "avg_problem_score": (
            sum(s.problem_score for s in signals) / len(signals)
            if signals
            else None
        ),
        "avg_problem_score_per_100_messages": (
            sum(s.problem_score_per_100_messages for s in signals) / len(signals)
            if signals
            else None
        ),
        "user_problem_rate_pct": (
            100 * user_problem / user_text_turns if user_text_turns else 0.0
        ),
        "assistant_problem_rate_pct": (
            100 * assistant_problem / assistant_text_turns if assistant_text_turns else 0.0
        ),
        "tool_error_rate_pct": (
            100 * tool_errors / tool_responses if tool_responses else 0.0
        ),
        "ends_on_user_rate_pct": (
            100 * ended_user / len(signals) if signals else 0.0
        ),
    }


def print_matched_sessions(
    *,
    db: sqlite3.Connection,
    aliases: dict[str, sqlite3.Row],
    route_session_keys: set[str],
    session_counts: dict[str, Counter[str]],
    show: int,
) -> dict:
    matched: dict[str, sqlite3.Row] = {}
    for key in route_session_keys:
        session = aliases.get(key)
        if session is not None:
            matched[session["id"]] = session

    print(f"Matched route sessions to Goose DB: {len(matched)}")
    if not matched:
        if route_session_keys:
            preview = ", ".join(sorted(route_session_keys)[:show])
            print(f"  unmatched route session keys: {preview}")
        return signal_metrics([])

    for session in list(matched.values())[:show]:
        keys = [
            key
            for key, candidate in aliases.items()
            if candidate["id"] == session["id"] and key in session_counts
        ]
        model_mix = Counter()
        for key in keys:
            model_mix.update(session_counts[key])
        mix = ", ".join(f"{model}={count}" for model, count in model_mix.most_common(4))
        print(
            f"  {str(session['updated_at'])[:19]} {session['id']:12s} "
            f"{model_name(session['model_config_json']):14s} {session['name']} "
            f"route_mix={mix}"
        )

    print()
    print("Quality/friction signals for matched sessions:")
    signals = [analyze_session(db, session) for session in matched.values()]
    groups: dict[str, list] = defaultdict(list)
    for signal in signals:
        groups[group_key(signal)].append(signal)
    for name in sorted(groups):
        summarize_group(name, groups[name])
    return signal_metrics(signals)


def evaluate_gate(summary: dict, args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    route = summary["route"]
    quality = summary["quality"]
    if summary["real_rows"] < args.min_real_rows:
        failures.append(
            f"real route rows {summary['real_rows']} < required {args.min_real_rows}"
        )
    if route["known_selected_rows"] < args.min_real_rows:
        failures.append(
            "known selected route rows "
            f"{route['known_selected_rows']} < required {args.min_real_rows}"
        )
    proxy = route.get("proxy_savings_pct")
    if proxy is None or proxy < args.min_proxy_savings_pct:
        failures.append(
            "proxy savings "
            f"{proxy if proxy is not None else 'n/a'} < {args.min_proxy_savings_pct:.1f}%"
        )
    cache = route.get("cache_savings_pct")
    if cache is None or cache < args.min_cache_savings_pct:
        failures.append(
            "cache-aware savings "
            f"{cache if cache is not None else 'n/a'} < {args.min_cache_savings_pct:.1f}%"
        )
    if route["baseline_share_pct"] > args.max_baseline_share_pct:
        failures.append(
            f"baseline share {route['baseline_share_pct']:.1f}% > "
            f"{args.max_baseline_share_pct:.1f}%"
        )
    if not args.allow_unmatched and quality["matched_sessions"] == 0:
        failures.append("no route sessions matched back to Goose sessions")
    score100 = quality.get("avg_problem_score_per_100_messages")
    if score100 is not None and score100 > args.max_quality_score100:
        failures.append(
            f"matched-session score100 {score100:.1f} > {args.max_quality_score100:.1f}"
        )
    if quality["tool_error_rate_pct"] > args.max_tool_error_rate_pct:
        failures.append(
            f"matched-session tool error rate {quality['tool_error_rate_pct']:.1f}% > "
            f"{args.max_tool_error_rate_pct:.1f}%"
        )
    calibration = summary.get("calibration") or {}
    point = calibration.get("point")
    if args.require_calibration and not point:
        failures.append("required offline calibration point is unavailable")
    if len(summary.get("observed_tolerances") or {}) > 1:
        failures.append(
            "multiple observed route tolerances in one trial: "
            + ", ".join(
                f"{float(tol):.6g}={count}"
                for tol, count in sorted(summary["observed_tolerances"].items())
            )
        )
    selected = calibration.get("selected_trial")
    if args.require_selected_tolerance:
        if not selected:
            failures.append("required selected calibration tolerance is unavailable")
        elif (
            abs(float(selected.get("tolerance", -1)) - summary["trial_tolerance"])
            > args.tolerance_epsilon
        ):
            failures.append(
                "config tolerance "
                f"{summary['trial_tolerance']:.6g} does not match selected "
                f"bounded-loss tolerance {float(selected.get('tolerance')):.6g}"
            )
    if point:
        delta = point.get("delta_pp")
        if delta is not None and abs(float(delta)) > args.max_verified_loss_pp:
            failures.append(
                f"verified-label loss {abs(float(delta)):.1f}pp > "
                f"{args.max_verified_loss_pp:.1f}pp"
            )
        verified_savings = point.get("savings_pct")
        if (
            verified_savings is not None
            and float(verified_savings) < args.min_verified_savings_pct
        ):
            failures.append(
                f"verified-label savings {float(verified_savings):.1f}% < "
                f"{args.min_verified_savings_pct:.1f}%"
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DB_DEFAULT))
    parser.add_argument("--route-log", default="/tmp/router-routes-real.jsonl")
    parser.add_argument("--config", default="configs/combined-pool.yaml")
    parser.add_argument("--baseline", default="claude-opus-4-6-high")
    parser.add_argument("--sessions", type=int, default=80)
    parser.add_argument("--show", type=int, default=8)
    parser.add_argument("--include-probes", action="store_true")
    parser.add_argument("--base-url", default="http://localhost:4000")
    parser.add_argument("--savings-timeout", type=float, default=1.0)
    parser.add_argument("--output-token-weight", type=float, default=0.25)
    parser.add_argument("--cache-read-multiplier", type=float, default=0.10)
    parser.add_argument("--min-context-tokens", type=int, default=1000)
    parser.add_argument("--calibration", default="configs/combined-calibration.json")
    parser.add_argument(
        "--wait-real-rows",
        type=int,
        default=0,
        help="Wait until at least this many non-probe route rows exist before reporting.",
    )
    parser.add_argument("--wait-timeout", type=float, default=300.0)
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument(
        "--write-marker",
        default="",
        help="Write the current timestamp to this marker file before reporting.",
    )
    parser.add_argument(
        "--marker",
        default="",
        help="Only analyze route rows at or after the timestamp in this marker file.",
    )
    parser.add_argument(
        "--since-ts",
        type=float,
        default=0.0,
        help="Only analyze route rows at or after this epoch timestamp.",
    )
    parser.add_argument(
        "--since-minutes",
        type=float,
        default=0.0,
        help="Only analyze route rows from the last N minutes.",
    )
    parser.add_argument("--json-out", default="")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Exit non-zero if the trial fails conservative savings/quality gates.",
    )
    parser.add_argument("--min-real-rows", type=int, default=1)
    parser.add_argument("--min-proxy-savings-pct", type=float, default=30.0)
    parser.add_argument("--min-cache-savings-pct", type=float, default=0.0)
    parser.add_argument("--max-baseline-share-pct", type=float, default=35.0)
    parser.add_argument("--max-quality-score100", type=float, default=25.0)
    parser.add_argument("--max-tool-error-rate-pct", type=float, default=25.0)
    parser.add_argument("--require-calibration", action="store_true")
    parser.add_argument("--require-selected-tolerance", action="store_true")
    parser.add_argument("--tolerance-epsilon", type=float, default=1e-6)
    parser.add_argument("--max-verified-loss-pp", type=float, default=2.0)
    parser.add_argument("--min-verified-savings-pct", type=float, default=40.0)
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="Do not fail the gate when route rows cannot be matched to Goose sessions.",
    )
    args = parser.parse_args()

    db = connect_readonly(Path(args.db).expanduser())
    cfg = load_config(args.config)
    config_tolerance = float(cfg.routing.tolerance)
    calibration = load_calibration(args.calibration)
    sessions = load_sessions(db, args.sessions)
    aliases = session_aliases(db, sessions)

    route_log_path = Path(args.route_log).expanduser()
    marker_written_ts = write_marker(args.write_marker) if args.write_marker else None
    since_candidates = [
        ts for ts in (
            args.since_ts or None,
            marker_timestamp(args.marker),
            time.time() - (args.since_minutes * 60) if args.since_minutes else None,
        )
        if ts is not None
    ]
    since_ts = max(since_candidates) if since_candidates else None
    if args.wait_real_rows > 0:
        wait_for_real_rows(
            route_log_path,
            min_rows=args.wait_real_rows,
            timeout_seconds=args.wait_timeout,
            poll_seconds=args.poll_interval,
            since_ts=since_ts,
        )

    all_rows = filter_rows_since(load_rows(route_log_path), since_ts)
    probe_rows = [row for row in all_rows if is_probe(row)]
    real_rows = [row for row in all_rows if not is_probe(row)]
    if not args.include_probes:
        rows = real_rows
    else:
        rows = all_rows

    route_tolerances = observed_tolerances(rows)
    if route_tolerances:
        trial_tolerance = float(route_tolerances.most_common(1)[0][0])
    else:
        trial_tolerance = config_tolerance
    savings = fetch_savings(args.base_url, args.savings_timeout)

    print("Router Trial Report")
    print(f"Route log: {args.route_log}")
    if marker_written_ts is not None:
        print(f"Wrote marker: {args.write_marker} ({marker_written_ts:.3f})")
    if since_ts is not None:
        print(f"Since: {since_ts:.3f} ({fmt_ts(since_ts)})")
    print(
        f"Rows: analyzed={len(rows)} real={len(real_rows)} probes={len(probe_rows)}"
    )
    if savings:
        print(
            f"Savings endpoint: requests={savings.get('requests', 0)} "
            f"saved_pct={float(savings.get('saved_pct') or 0):.1f}% "
            f"saved={fmt_money(savings.get('saved_usd'))}"
        )
    else:
        print("Savings endpoint: unavailable")
    if route_tolerances:
        print(
            "Observed route tolerances: "
            + ", ".join(
                f"{float(tol):.6g}={count}"
                for tol, count in route_tolerances.most_common()
            )
        )
    else:
        print(f"Observed route tolerances: none; using config tolerance {config_tolerance:.6g}")
    print()

    calibration_summary = print_calibration(
        calibration,
        trial_tolerance,
        cfg=cfg,
        baseline=args.baseline,
        output_token_weight=args.output_token_weight,
    )
    print()

    print_recent_routed_sessions(sessions, args.show)
    print()
    route_session_keys, session_counts, route_metrics = print_route_summary(
        rows=rows,
        config_path=args.config,
        baseline=args.baseline,
        output_token_weight=args.output_token_weight,
        cache_read_multiplier=args.cache_read_multiplier,
        min_context_tokens=args.min_context_tokens,
        show=args.show,
    )
    print()
    quality_metrics = print_matched_sessions(
        db=db,
        aliases=aliases,
        route_session_keys=route_session_keys,
        session_counts=session_counts,
        show=args.show,
    )
    summary = {
        "route_log": args.route_log,
        "analyzed_rows": len(rows),
        "real_rows": len(real_rows),
        "probe_rows": len(probe_rows),
        "since_ts": since_ts,
        "marker_written_ts": marker_written_ts,
        "savings_endpoint": savings,
        "route": route_metrics,
        "quality": quality_metrics,
        "trial_tolerance": trial_tolerance,
        "config_tolerance": config_tolerance,
        "observed_tolerances": dict(route_tolerances),
        "calibration": calibration_summary,
    }
    print()
    if real_rows:
        print(
            "Verdict: real route-log evidence exists. Savings/model mix above are "
            "usable for the trial; quality remains heuristic unless outcomes were "
            "user-reviewed or head-to-head."
        )
    elif rows:
        print(
            "Verdict: only provider-free probe rows are present. Probe routing is "
            "useful for preflight, but a fresh Goose run through nvidia-routed is "
            "still required before claiming real savings or same quality."
        )
    else:
        print(
            "Verdict: no real live route rows yet. The router can be preflighted "
            "without provider calls, but a fresh Goose run through nvidia-routed "
            "is still required before claiming real savings or same quality."
        )

    gate_failures: list[str] = []
    if args.gate:
        gate_failures = evaluate_gate(summary, args)
        print()
        if gate_failures:
            print("Gate: FAIL")
            for failure in gate_failures:
                print(f"  - {failure}")
        else:
            print("Gate: PASS")

    if args.json_out:
        out_path = Path(args.json_out).expanduser()
        summary["gate"] = {
            "enabled": args.gate,
            "passed": not gate_failures if args.gate else None,
            "failures": gate_failures,
            "thresholds": {
                "wait_real_rows": args.wait_real_rows,
                "wait_timeout": args.wait_timeout,
                "since_ts": since_ts,
                "min_real_rows": args.min_real_rows,
                "min_proxy_savings_pct": args.min_proxy_savings_pct,
                "min_cache_savings_pct": args.min_cache_savings_pct,
                "max_baseline_share_pct": args.max_baseline_share_pct,
                "max_quality_score100": args.max_quality_score100,
                "max_tool_error_rate_pct": args.max_tool_error_rate_pct,
                "require_calibration": args.require_calibration,
                "require_selected_tolerance": args.require_selected_tolerance,
                "tolerance_epsilon": args.tolerance_epsilon,
                "max_verified_loss_pp": args.max_verified_loss_pp,
                "min_verified_savings_pct": args.min_verified_savings_pct,
                "allow_unmatched": args.allow_unmatched,
            },
        }
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"Wrote JSON summary -> {out_path}")

    if args.gate and gate_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
