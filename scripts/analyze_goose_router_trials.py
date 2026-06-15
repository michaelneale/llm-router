#!/usr/bin/env python3
"""Summarize Goose usage of the verified router.

This is a local-only status check. It reads Goose's sessions DB and the router
route JSONL log to answer:

  - Did Goose sessions actually target nvidia-routed?
  - Did the live router log any selected backend models?
  - Do we have enough evidence to discuss real savings, or only offline/shadow
    evidence?

Goose stores the requested model/provider in the session config, but not the
router's final backend model per turn. The route log is therefore required for
real model-mix and savings analysis.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

from analyze_route_log import load_rows
from model_router_toolkit.config import load_config


DB_DEFAULT = Path.home() / ".local/share/goose/sessions/sessions.db"


def connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro&immutable=1"
    db = sqlite3.connect(uri, uri=True)
    db.row_factory = sqlite3.Row
    return db


def model_name(model_config_json: str | None) -> str:
    if not model_config_json:
        return ""
    try:
        config = json.loads(model_config_json)
    except json.JSONDecodeError:
        return ""
    return str(config.get("model_name") or "")


def session_rows(db: sqlite3.Connection, limit: int) -> list[dict]:
    rows = db.execute(
        """
        SELECT
          id,
          name,
          updated_at,
          provider_name,
          model_config_json,
          working_dir,
          total_tokens,
          input_tokens,
          output_tokens,
          accumulated_total_tokens,
          accumulated_input_tokens,
          accumulated_output_tokens,
          accumulated_cost
        FROM sessions
        WHERE session_type='user'
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def message_counts(db: sqlite3.Connection, session_ids: list[str]) -> dict[str, Counter[str]]:
    if not session_ids:
        return {}
    placeholders = ",".join("?" for _ in session_ids)
    rows = db.execute(
        f"""
        SELECT session_id, role, COUNT(*) AS n
        FROM messages
        WHERE session_id IN ({placeholders})
        GROUP BY session_id, role
        """,
        session_ids,
    ).fetchall()
    counts: dict[str, Counter[str]] = {sid: Counter() for sid in session_ids}
    for row in rows:
        counts[row["session_id"]][row["role"]] = int(row["n"])
    return counts


def fmt_money(value) -> str:
    if value is None or value == "":
        return "-"
    return f"${float(value):.4f}"


def fmt_int(value) -> str:
    if value is None or value == "":
        return "-"
    return f"{int(value):,}"


def fmt_time(value: float | None) -> str:
    if value is None:
        return "-"
    return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def route_log_info(path: Path, rows: list[dict]) -> dict:
    info = {
        "exists": path.exists(),
        "bytes": 0,
        "mtime": None,
        "first_row_ts": None,
        "last_row_ts": None,
    }
    if path.exists():
        stat = path.stat()
        info["bytes"] = stat.st_size
        info["mtime"] = stat.st_mtime
    ts_values = [float(row["ts"]) for row in rows if row.get("ts") is not None]
    if ts_values:
        info["first_row_ts"] = min(ts_values)
        info["last_row_ts"] = max(ts_values)
    return info


def print_recent_sessions(rows: list[dict], counts: dict[str, Counter[str]], max_rows: int) -> None:
    print("Recent Goose sessions:")
    print(
        "  "
        + f"{'updated_at':19s} {'id':11s} {'provider':13s} {'model':18s} "
        + f"{'msgs':>6s} {'tokens':>10s} {'cost':>10s}  name"
    )
    for row in rows[:max_rows]:
        model = model_name(row.get("model_config_json"))
        msg_count = sum(counts.get(row["id"], Counter()).values())
        print(
            "  "
            + f"{str(row.get('updated_at') or '')[:19]:19s} "
            + f"{row['id'][:11]:11s} "
            + f"{str(row.get('provider_name') or '')[:13]:13s} "
            + f"{model[:18]:18s} "
            + f"{msg_count:6d} "
            + f"{fmt_int(row.get('total_tokens')):>10s} "
            + f"{fmt_money(row.get('accumulated_cost')):>10s}  "
            + str(row.get("name") or "")[:50]
        )


def print_routed_sessions(rows: list[dict], counts: dict[str, Counter[str]], max_rows: int) -> None:
    routed = [row for row in rows if model_name(row.get("model_config_json")) == "nvidia-routed"]
    print()
    print(f"Goose sessions requesting nvidia-routed: {len(routed)} in inspected window")
    if not routed:
        return
    providers = Counter(str(row.get("provider_name") or "unknown") for row in routed)
    latest = max(str(row.get("updated_at") or "") for row in routed)
    print(
        "  provider mix: "
        + ", ".join(f"{provider}={count}" for provider, count in providers.most_common())
    )
    print(f"  latest routed session updated_at: {latest}")
    print(
        "  "
        + f"{'updated_at':19s} {'id':11s} {'provider':13s} "
        + f"{'user':>5s} {'assistant':>9s} {'tokens':>10s} {'cost':>10s}  name"
    )
    for row in routed[:max_rows]:
        c = counts.get(row["id"], Counter())
        print(
            "  "
            + f"{str(row.get('updated_at') or '')[:19]:19s} "
            + f"{row['id'][:11]:11s} "
            + f"{str(row.get('provider_name') or '')[:13]:13s} "
            + f"{c['user']:5d} "
            + f"{c['assistant']:9d} "
            + f"{fmt_int(row.get('total_tokens')):>10s} "
            + f"{fmt_money(row.get('accumulated_cost')):>10s}  "
            + str(row.get("name") or "")[:50]
        )
    missing_cost = sum(1 for row in routed if row.get("accumulated_cost") is None)
    if missing_cost:
        print(f"  note: {missing_cost} routed sessions have no Goose accumulated_cost.")


def print_route_log_status(rows: list[dict], config_path: str, max_models: int) -> None:
    cfg = load_config(config_path)
    display = {m.name: (m.display_name or m.name) for m in cfg.models}
    known = set(display)
    selected = [row.get("selected_model") for row in rows if row.get("selected_model") in known]
    decisions = Counter(row.get("decision", "unknown") for row in rows)
    models = Counter(selected)

    print()
    print(f"Route log rows: {len(rows)}")
    if not rows:
        print("  No backend selections are available from the live route log.")
        return
    print("  decisions: " + ", ".join(f"{k}={v}" for k, v in decisions.most_common()))
    print("  selected model mix:")
    for model, count in models.most_common(max_models):
        print(f"    {display.get(model, model):24s} {count:5d}  {count / len(selected):6.1%}")


def print_route_log_window(info: dict) -> None:
    print()
    if not info["exists"]:
        print("Route log file: missing")
        return
    print(
        "Route log file: "
        f"bytes={info['bytes']:,} mtime={fmt_time(info['mtime'])}"
    )
    if info["first_row_ts"] is None:
        print("Route row window: no rows")
    else:
        print(
            "Route row window: "
            f"{fmt_time(info['first_row_ts'])} -> {fmt_time(info['last_row_ts'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DB_DEFAULT))
    parser.add_argument("--route-log", default="/tmp/router-routes-real.jsonl")
    parser.add_argument("--config", default="configs/combined-pool.yaml")
    parser.add_argument("--sessions", type=int, default=40)
    parser.add_argument("--show", type=int, default=12)
    parser.add_argument(
        "--include-probes",
        action="store_true",
        help="Include provider-free /router/route probe rows in route-log status.",
    )
    args = parser.parse_args()

    db_path = Path(args.db).expanduser()
    route_log = Path(args.route_log)
    db = connect_readonly(db_path)
    sessions = session_rows(db, args.sessions)
    counts = message_counts(db, [row["id"] for row in sessions])
    route_rows = load_rows(route_log)
    if not args.include_probes:
        route_rows = [
            row for row in route_rows
            if not str(row.get("request_id") or "").startswith("probe-")
        ]
    log_info = route_log_info(route_log, route_rows)

    print(f"Goose DB: {db_path}")
    print(f"Route log: {route_log}")
    print_route_log_window(log_info)
    print()
    print_recent_sessions(sessions, counts, args.show)
    print_routed_sessions(sessions, counts, args.show)
    print_route_log_status(route_rows, args.config, args.show)

    print()
    if route_rows:
        print("Verdict: live route-log evidence exists; use analyze_route_log.py for savings/model mix.")
    else:
        print(
            "Verdict: Goose has requested nvidia-routed in recent sessions, but the current "
            "route log is empty. Existing sessions prove requested alias usage, not actual "
            "backend model mix or real savings."
        )
        print("Next: run one fresh task through the live proxy, then close it out:")
        print(
            "  scripts/goose-routed-task.sh --name router-smoke -- "
            "'what is going on in this repo'"
        )
        print("  scripts/router-service.sh summary")
        print(
            "If a new nvidia-routed Goose session appears after the route-log mtime but "
            "route rows stay at zero, the client is not reaching the local proxy; check "
            "GOOSE_PROVIDER/LITELLM_HOST or the equivalent OpenAI-compatible base URL."
        )


if __name__ == "__main__":
    main()
