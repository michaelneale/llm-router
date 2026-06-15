#!/usr/bin/env python3
"""Replay recent Goose sessions through a router config and print route distributions.

This is an offline routing check: it reads Goose's sessions.db, rebuilds the
message list turn-by-turn, and asks the local router which backend it would pick.
It does not call any LLM provider APIs.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from model_router_toolkit.adapters.litellm.strategy import ModelRoutingStrategy
from model_router_toolkit.config import load_config


DB_DEFAULT = Path.home() / ".local/share/goose/sessions/sessions.db"


def _load_sessions(db: sqlite3.Connection, model_like: str, limit: int) -> list[sqlite3.Row]:
    return db.execute(
        """
        SELECT id, name, updated_at
        FROM sessions
        WHERE session_type='user'
          AND model_config_json LIKE ?
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (model_like, limit),
    ).fetchall()


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") in ("toolResponse", "tool_result"):
                parts.append("[toolResponse]")
        return " ".join(parts).strip()
    return ""


def _messages_for_session(db: sqlite3.Connection, session_id: str) -> list[dict]:
    rows = db.execute(
        """
        SELECT role, content_json
        FROM messages
        WHERE session_id=?
        ORDER BY id
        """,
        (session_id,),
    ).fetchall()
    messages = []
    for row in rows:
        try:
            content = json.loads(row["content_json"])
        except (TypeError, ValueError):
            content = row["content_json"]
        messages.append({"role": row["role"], "content": content})
    return messages


def _fake_litellm_router(config_path: str):
    cfg = load_config(config_path)
    return type("LiteLLMRouter", (), {"model_list": [{"model_name": m.name} for m in cfg.models]})()


def _reset_strategy_state(strategy: ModelRoutingStrategy) -> None:
    with strategy._state_lock:
        strategy._last_result = None
        strategy._last_selected_model = None
        strategy._results_by_request.clear()
        strategy._session_selected.clear()


def _rates(config_path: str) -> dict[str, tuple[float, float]]:
    cfg = load_config(config_path)
    return {
        m.name: (m.cost_per_m_input_tokens, m.cost_per_m_output_tokens)
        for m in cfg.models
    }


def replay(
    strategy: ModelRoutingStrategy,
    sessions: list[sqlite3.Row],
    db: sqlite3.Connection,
    *,
    max_user_turns: int | None = None,
) -> dict:
    _reset_strategy_state(strategy)
    total = Counter()
    per_session: dict[str, Counter] = defaultdict(Counter)
    reasons = Counter()
    by_reason: dict[str, Counter] = defaultdict(Counter)
    decisions = 0

    for session in sessions:
        sid = session["id"]
        seen: list[dict] = []
        user_turns = 0
        for msg in _messages_for_session(db, sid):
            seen.append(msg)
            if msg.get("role") != "user":
                continue
            user_turns += 1
            if max_user_turns is not None and user_turns > max_user_turns:
                break
            dep = strategy.get_available_deployment(
                "nvidia-routed",
                messages=list(seen),
                request_kwargs={"metadata": {"router_session_id": sid}},
            )
            selected = dep.get("model_name", "<none>")
            result = strategy.last_result
            reason = "route"
            if result is not None:
                if result.metadata.get("utility"):
                    reason = "utility"
                elif result.metadata.get("pin_reason"):
                    reason = str(result.metadata.get("pin_reason"))
            total[selected] += 1
            per_session[sid][selected] += 1
            reasons[reason] += 1
            by_reason[reason][selected] += 1
            decisions += 1

    return {
        "total": total,
        "per_session": per_session,
        "reasons": reasons,
        "by_reason": by_reason,
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DB_DEFAULT))
    parser.add_argument("--config", default="configs/combined-pool.yaml")
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--model-like", default="%nvidia-routed%")
    parser.add_argument("--tolerances", default="0.0,0.02,0.05,0.10,0.20")
    parser.add_argument("--disable-switching", action="store_true")
    parser.add_argument("--max-user-turns", type=int, default=0)
    args = parser.parse_args()

    if args.disable_switching:
        os.environ["ROUTER_DISABLE_SWITCHING"] = "1"

    db_path = Path(args.db).expanduser()
    db = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    sessions = _load_sessions(db, args.model_like, args.sessions)
    if not sessions:
        raise SystemExit(f"No Goose sessions matched model_config_json LIKE {args.model_like!r}")

    print(f"Sessions: {len(sessions)}")
    for s in sessions:
        print(f"  {s['id']:12s} {s['updated_at']}  {s['name']}")
    print()

    base_cfg = load_config(args.config)
    display = {m.name: (m.display_name or m.name) for m in base_cfg.models}
    rates = _rates(args.config)
    tols = [float(x) for x in args.tolerances.split(",") if x.strip()]
    strategy = ModelRoutingStrategy.from_config(args.config)
    strategy.set_litellm_router(_fake_litellm_router(args.config))

    for tol in tols:
        strategy.tolerance = tol

        result = replay(
            strategy,
            sessions,
            db,
            max_user_turns=args.max_user_turns or None,
        )
        total: Counter = result["total"]
        n = result["decisions"] or 1
        avg_in_cost = sum(total[m] * rates.get(m, (0.0, 0.0))[0] for m in total) / n
        avg_out_cost = sum(total[m] * rates.get(m, (0.0, 0.0))[1] for m in total) / n
        print(f"Tolerance {tol:.2f} ({result['decisions']} user/tool turns routed)")
        print(f"  avg rates: input ${avg_in_cost:.2f}/M, output ${avg_out_cost:.2f}/M")
        print("  reasons: " + ", ".join(
            f"{k}={v}" for k, v in result["reasons"].most_common()
        ))
        for reason, counts in sorted(result["by_reason"].items()):
            breakdown = ", ".join(
                f"{display.get(model, model)}={count}"
                for model, count in counts.most_common()
            )
            print(f"  {reason}: {breakdown}")
        for model, count in total.most_common():
            print(f"  {display.get(model, model):28s} {count:4d}  {count / n:6.1%}")
        print()


if __name__ == "__main__":
    main()
