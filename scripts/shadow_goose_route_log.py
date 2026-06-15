#!/usr/bin/env python3
"""Replay Goose sessions through the verified router and write a shadow route log.

This is local-only: it does not call provider APIs. It reads Goose's sessions.db,
reconstructs user turns, runs the local prefill router for genuine decision
turns, applies the same simple pinning behavior used by the live strategy for
non-decision turns, and writes JSONL rows compatible with
``scripts/analyze_route_log.py``.

Use this when the live route log is missing or was reset after a Goose trial.
It estimates what the current router would do on those historical transcripts;
it is not proof of answer quality for the already-completed sessions.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

# This script is explicitly provider-free. Default to offline transformer
# loading so a local replay never stalls on Hugging Face metadata checks.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from model_router_toolkit.config import load_config
from model_router_toolkit.prefill.router import PrefillRouter
from model_router_toolkit.router import RoutingResult
from model_router_toolkit.task_view import (
    build_task_view,
    is_goose_title_request,
    is_info_only_request,
)
from spotcheck_verified_router import (
    DB_DEFAULT,
    _cheapest_model,
    _content_text,
    _context_tokens,
    _cost_maps,
    _routing_cost_maps,
    _load_sessions,
    _messages_for_session,
)


def _choose(
    *,
    result: RoutingResult,
    rates: dict[str, tuple[float, float]],
    tolerance: float,
) -> str:
    conf = dict(zip(result.model_names, result.confidences))
    threshold = max(conf.values()) - tolerance
    selected = max(result.model_names, key=lambda m: (rates[m][0], rates[m][1], m))
    for model in sorted(result.model_names, key=lambda m: (rates[m][0], rates[m][1], m)):
        if conf[model] >= threshold:
            selected = model
            break
    return selected


def _resolve(router: PrefillRouter, model: str) -> RoutingResult:
    result = router.resolve(model)
    if result is None:
        raise RuntimeError(f"Router cannot resolve model slot {model!r}")
    return result


def _with_selected(result: RoutingResult, selected: str, metadata: dict | None = None) -> RoutingResult:
    meta = dict(result.metadata)
    if metadata:
        meta.update(metadata)
    return RoutingResult(
        model_names=result.model_names,
        confidences=result.confidences,
        costs=result.costs,
        selected_model=selected,
        metadata=meta,
    )


def _row(
    *,
    session: sqlite3.Row,
    turn: int,
    decision: str,
    result: RoutingResult | None,
    task_view: str,
    messages_seen: list[dict],
    raw_selected_model: str | None = None,
) -> dict:
    row = {
        "ts": time.time(),
        "request_id": f"shadow:{session['id']}:{turn}",
        "session_key": session["id"],
        "session_name": session["name"],
        "session_updated_at": session["updated_at"],
        "decision": decision,
        "selected_model": result.selected_model if result else None,
        "session_depth": sum(1 for msg in messages_seen if msg.get("role") == "user"),
        "context_tokens_est": _context_tokens(messages_seen),
        "task_view": task_view[:1200],
        "metadata": result.metadata if result else {},
        "confidences": (
            {
                name: round(float(conf), 4)
                for name, conf in zip(result.model_names, result.confidences)
            }
            if result
            else {}
        ),
    }
    if raw_selected_model:
        row["raw_selected_model"] = raw_selected_model
    return row


def replay_sessions(
    *,
    db: sqlite3.Connection,
    sessions: list[sqlite3.Row],
    router: PrefillRouter,
    rates: dict[str, tuple[float, float]],
    tolerance: float,
    max_user_turns: int | None,
    progress_every: int,
) -> list[dict]:
    rows: list[dict] = []
    cheapest = _cheapest_model(rates)
    for idx, session in enumerate(sessions, 1):
        before_rows = len(rows)
        last_selected: str | None = None
        seen: list[dict] = []
        user_turns = 0
        for msg in _messages_for_session(db, session["id"]):
            seen.append(msg)
            if msg.get("role") != "user":
                continue
            user_turns += 1
            if max_user_turns is not None and user_turns > max_user_turns:
                break

            raw_text = _content_text(msg.get("content")).strip()
            if is_goose_title_request(raw_text):
                result = _resolve(router, cheapest)
                result.metadata["utility_reason"] = "goose_title_generation"
                last_selected = result.selected_model
                rows.append(
                    _row(
                        session=session,
                        turn=user_turns,
                        decision="cheap_utility",
                        result=result,
                        task_view=raw_text,
                        messages_seen=seen,
                    )
                )
                continue

            task_view = build_task_view(list(seen))
            if task_view is None:
                if last_selected:
                    result = _resolve(router, last_selected)
                    result.metadata["pin_reason"] = "non_decision_turn"
                    rows.append(
                        _row(
                            session=session,
                            turn=user_turns,
                            decision="pin_non_decision",
                            result=result,
                            task_view="",
                            messages_seen=seen,
                        )
                    )
                    continue
                if is_info_only_request(raw_text):
                    result = _resolve(router, cheapest)
                    result.metadata["utility_reason"] = "goose_info_only"
                    last_selected = result.selected_model
                    rows.append(
                        _row(
                            session=session,
                            turn=user_turns,
                            decision="cheap_utility",
                            result=result,
                            task_view=raw_text,
                            messages_seen=seen,
                        )
                    )
                    continue
                task_view = raw_text

            raw = router.route(task_view, tolerance=0.0)
            selected = _choose(result=raw, rates=rates, tolerance=tolerance)
            result = _with_selected(
                raw,
                selected,
                {"shadow": True, "tolerance": tolerance},
            )
            last_selected = selected
            rows.append(
                _row(
                    session=session,
                    turn=user_turns,
                    decision="route",
                    result=result,
                    task_view=task_view,
                    messages_seen=seen,
                    raw_selected_model=selected,
                )
            )
        if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == len(sessions)):
            print(
                "  shadowed "
                f"{idx}/{len(sessions)} sessions "
                f"({session['id']}, +{len(rows) - before_rows} rows, total={len(rows)})",
                flush=True,
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DB_DEFAULT))
    parser.add_argument("--config", default="configs/combined-pool.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-like", default="%nvidia-routed%")
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--max-user-turns", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument("--output", default="/tmp/router-routes-shadow.jsonl")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Print progress every N sessions while scoring. Use 0 to disable.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = args.checkpoint or cfg.routing.checkpoint
    tolerance = cfg.routing.tolerance if args.tolerance is None else args.tolerance
    rates, _display = _cost_maps(args.config)
    routing_rates = _routing_cost_maps(args.config)

    db_path = Path(args.db).expanduser()
    db = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    sessions = _load_sessions(db, args.model_like, args.sessions)
    if not sessions:
        raise SystemExit(f"No Goose sessions matched model_config_json LIKE {args.model_like!r}")

    print(f"Sessions: {len(sessions)}")
    for session in sessions:
        print(f"  {session['id']:12s} {session['updated_at']}  {session['name']}")
    print()
    print(f"Loading verified-trace router: {checkpoint}")
    router = PrefillRouter(config=cfg)
    router.load(checkpoint)

    rows = replay_sessions(
        db=db,
        sessions=sessions,
        router=router,
        rates=routing_rates,
        tolerance=tolerance,
        max_user_turns=args.max_user_turns or None,
        progress_every=args.progress_every,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} shadow route rows -> {out}")


if __name__ == "__main__":
    main()
