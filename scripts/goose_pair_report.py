#!/usr/bin/env python3
"""Local report for one routed-vs-frontier Goose AB pair.

This does not call providers. It reads Goose's sessions.db, finds the most
recent sessions matching the names/models from a routed and direct frontier run,
and prints the same weak quality/friction counters used elsewhere in this repo.

The output is not an automatic correctness label. It is the minimum local
evidence needed before manual review: did the routed run obviously fail, stop
early, trigger more corrections, or hit more tool errors than the frontier run?
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "src"))

from analyze_goose_ab_pairs import first_user_text, fmt_signals, short  # noqa: E402
from analyze_goose_quality_signals import (  # noqa: E402
    DB_DEFAULT,
    analyze_session,
    connect_readonly,
    model_name,
)


def marker_timestamp(path: str) -> float | None:
    if not path:
        return None
    marker_path = Path(path).expanduser()
    if not marker_path.exists():
        raise SystemExit(f"Marker file not found: {marker_path}")
    try:
        value = json.loads(marker_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid marker JSON {marker_path}: {exc}") from exc
    try:
        return float(value.get("ts"))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Marker file does not contain numeric ts: {marker_path}") from exc


def find_session(
    db: sqlite3.Connection,
    *,
    name: str,
    model: str,
    since_ts: float | None,
) -> sqlite3.Row | None:
    rows = db.execute(
        """
        SELECT id, name, updated_at, provider_name, model_config_json,
               total_tokens, accumulated_cost
        FROM sessions
        WHERE session_type='user'
          AND name=?
          AND model_config_json LIKE ?
        ORDER BY updated_at DESC
        LIMIT 20
        """,
        (name, f"%{model}%"),
    ).fetchall()
    if since_ts is None:
        return rows[0] if rows else None
    for row in rows:
        # Goose timestamps are stored as ISO-ish local strings. For marker
        # filtering, use updated_at only as a best-effort guard; exact names are
        # the primary disambiguator.
        try:
            updated = time.mktime(time.strptime(str(row["updated_at"])[:19], "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            return row
        if updated >= since_ts:
            return row
    return None


def fmt_cost(value) -> str:
    if value is None or value == "":
        return "-"
    return f"${float(value):.4f}"


def session_summary(db: sqlite3.Connection, row: sqlite3.Row) -> dict:
    signals = analyze_session(db, row)
    first_user = first_user_text(db, row["id"])
    return {
        "id": row["id"],
        "name": row["name"],
        "updated_at": row["updated_at"],
        "provider": row["provider_name"],
        "model": model_name(row["model_config_json"]),
        "total_tokens": row["total_tokens"],
        "accumulated_cost": row["accumulated_cost"],
        "first_user": first_user,
        "signals": {
            "messages": signals.messages,
            "user_text_turns": signals.user_text_turns,
            "assistant_text_turns": signals.assistant_text_turns,
            "tool_responses": signals.tool_responses,
            "tool_errors": signals.tool_errors,
            "user_problem_turns": signals.user_problem_turns,
            "assistant_problem_turns": signals.assistant_problem_turns,
            "problem_score": signals.problem_score,
            "problem_score_per_100_messages": signals.problem_score_per_100_messages,
            "ends_on_user": signals.ends_on_user,
        },
        "_signals_obj": signals,
    }


def print_side(label: str, summary: dict) -> None:
    row = summary
    print(
        f"{label:8s} {row['id']:12s} {str(row['updated_at'])[:19]} "
        f"{row['model']:16s} tokens={row['total_tokens'] or '-':>8} "
        f"cost={fmt_cost(row['accumulated_cost']):>9} {row['name']}"
    )
    print(f"  prompt: {short(row['first_user'])}")
    print(f"  signals: {fmt_signals(row['_signals_obj'])}")


def evaluate_gate(delta: dict, args: argparse.Namespace) -> list[str]:
    failures = []
    if delta["score100"] > args.max_score100_delta:
        failures.append(
            f"score100 delta {delta['score100']:.1f} > {args.max_score100_delta:.1f}"
        )
    if delta["tool_errors"] > args.max_tool_error_delta:
        failures.append(
            f"tool error delta {delta['tool_errors']} > {args.max_tool_error_delta}"
        )
    if delta["user_problem_turns"] > args.max_user_problem_delta:
        failures.append(
            "user correction/friction delta "
            f"{delta['user_problem_turns']} > {args.max_user_problem_delta}"
        )
    if delta["assistant_problem_turns"] > args.max_assistant_problem_delta:
        failures.append(
            "assistant failure-language delta "
            f"{delta['assistant_problem_turns']} > {args.max_assistant_problem_delta}"
        )
    if args.fail_if_routed_ends_user and delta["ends_on_user"] > 0:
        failures.append("routed run ended on user while Opus did not")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DB_DEFAULT))
    parser.add_argument("--routed-name", required=True)
    parser.add_argument("--opus-name", required=True)
    parser.add_argument("--routed-model", default="nvidia-routed")
    parser.add_argument("--opus-model", default="claude-opus-4-8")
    parser.add_argument("--marker", default="")
    parser.add_argument("--since-ts", type=float, default=0.0)
    parser.add_argument("--json-out", default="")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Exit non-zero if routed run is materially worse on local friction signals.",
    )
    parser.add_argument("--max-score100-delta", type=float, default=15.0)
    parser.add_argument("--max-tool-error-delta", type=int, default=1)
    parser.add_argument("--max-user-problem-delta", type=int, default=0)
    parser.add_argument("--max-assistant-problem-delta", type=int, default=2)
    parser.add_argument("--fail-if-routed-ends-user", action="store_true")
    args = parser.parse_args()

    since_candidates = [
        ts
        for ts in (
            args.since_ts or None,
            marker_timestamp(args.marker),
        )
        if ts is not None
    ]
    since_ts = max(since_candidates) if since_candidates else None

    db = connect_readonly(Path(args.db).expanduser())
    routed = find_session(
        db,
        name=args.routed_name,
        model=args.routed_model,
        since_ts=since_ts,
    )
    opus = find_session(
        db,
        name=args.opus_name,
        model=args.opus_model,
        since_ts=since_ts,
    )
    if routed is None or opus is None:
        if routed is None:
            print(
                f"Missing routed session name={args.routed_name!r} "
                f"model_like={args.routed_model!r}"
            )
        if opus is None:
            print(
                f"Missing opus session name={args.opus_name!r} "
                f"model_like={args.opus_model!r}"
            )
        raise SystemExit(1)

    routed_summary = session_summary(db, routed)
    opus_summary = session_summary(db, opus)
    r_sig = routed_summary["signals"]
    o_sig = opus_summary["signals"]
    delta = {
        "score100": (
            r_sig["problem_score_per_100_messages"]
            - o_sig["problem_score_per_100_messages"]
        ),
        "tool_errors": r_sig["tool_errors"] - o_sig["tool_errors"],
        "user_problem_turns": (
            r_sig["user_problem_turns"] - o_sig["user_problem_turns"]
        ),
        "assistant_problem_turns": (
            r_sig["assistant_problem_turns"] - o_sig["assistant_problem_turns"]
        ),
        "ends_on_user": (
            int(r_sig["ends_on_user"]) - int(o_sig["ends_on_user"])
        ),
    }

    print("Goose AB Pair Report")
    if since_ts is not None:
        print(f"Since: {since_ts:.3f}")
    print_side("routed", routed_summary)
    print_side("opus", opus_summary)
    print()
    print("Delta routed - opus:")
    print(
        f"  score100={delta['score100']:+.1f}, "
        f"tool_errors={delta['tool_errors']:+d}, "
        f"user_flags={delta['user_problem_turns']:+d}, "
        f"assistant_flags={delta['assistant_problem_turns']:+d}, "
        f"ends_on_user={delta['ends_on_user']:+d}"
    )
    print()
    print(
        "Interpretation: this is friction evidence, not correctness proof. "
        "If the routed run has similar or lower friction, manually inspect the "
        "task outcome/diff before calling same-quality."
    )

    gate_failures = evaluate_gate(delta, args) if args.gate else []
    if args.gate:
        print()
        if gate_failures:
            print("Gate: FAIL")
            for failure in gate_failures:
                print(f"  - {failure}")
        else:
            print("Gate: PASS")

    if args.json_out:
        out = Path(args.json_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        for summary in (routed_summary, opus_summary):
            summary.pop("_signals_obj", None)
        out.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "since_ts": since_ts,
                    "routed": routed_summary,
                    "opus": opus_summary,
                    "delta": delta,
                    "gate": {
                        "enabled": args.gate,
                        "passed": not gate_failures if args.gate else None,
                        "failures": gate_failures,
                        "thresholds": {
                            "max_score100_delta": args.max_score100_delta,
                            "max_tool_error_delta": args.max_tool_error_delta,
                            "max_user_problem_delta": args.max_user_problem_delta,
                            "max_assistant_problem_delta": (
                                args.max_assistant_problem_delta
                            ),
                            "fail_if_routed_ends_user": (
                                args.fail_if_routed_ends_user
                            ),
                        },
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"Wrote JSON summary -> {out}")

    if args.gate and gate_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
