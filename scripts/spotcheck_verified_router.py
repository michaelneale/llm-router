#!/usr/bin/env python3
"""Fast offline spot-check of the verified-trace router on recent Goose traffic.

This does not call provider APIs. It reads Goose's sessions.db, reconstructs
user turns, scores only genuine routing decisions once with the verified-trace
checkpoint, then sweeps tolerance values offline. Non-decision turns are counted
as pins to the last routed model, matching the live proxy's cache-preserving
behavior closely enough for aggregate checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from model_router_toolkit.config import load_config
from model_router_toolkit.prefill.router import PrefillRouter
from model_router_toolkit.task_view import (
    build_task_view,
    is_goose_title_request,
    is_info_only_request,
)


DB_DEFAULT = Path.home() / ".local/share/goose/sessions/sessions.db"


def _connect_readonly(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    return db


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


def _context_tokens(messages: list[dict]) -> int:
    chars = 0
    for msg in messages:
        chars += len(_content_text(msg.get("content")))
    return max(1, chars // 4)


def _cost_maps(config_path: str) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    cfg = load_config(config_path)
    rates = {
        m.name: (m.cost_per_m_input_tokens, m.cost_per_m_output_tokens)
        for m in cfg.models
    }
    display = {m.name: (m.display_name or m.name) for m in cfg.models}
    return rates, display


def _routing_cost_maps(config_path: str) -> dict[str, tuple[float, float]]:
    cfg = load_config(config_path)
    output_weight = float(getattr(cfg.routing, "output_token_weight", 0.0) or 0.0)
    return {
        m.name: (
            (
                m.cost_per_m_input_tokens
                + output_weight * m.cost_per_m_output_tokens
            )
            * float(getattr(m, "routing_cost_multiplier", 1.0) or 1.0),
            0.0,
        )
        for m in cfg.models
    }


def _cheapest_model(rates: dict[str, tuple[float, float]]) -> str:
    return min(rates, key=lambda m: (rates[m][0], rates[m][1], m))


def _pick(
    model_names: list[str],
    confidences: list[float],
    rates: dict[str, tuple[float, float]],
    tolerance: float,
) -> str:
    conf = dict(zip(model_names, confidences))
    p_max = max(conf.values())
    threshold = p_max - tolerance
    selected = max(model_names, key=lambda m: (rates[m][0], rates[m][1], m))
    for model in sorted(model_names, key=lambda m: (rates[m][0], rates[m][1], m)):
        if conf[model] >= threshold:
            selected = model
            break
    return selected


def _top_confidences(event: dict, display: dict[str, str], n: int = 3) -> str:
    pairs = sorted(
        zip(event["model_names"], event["confidences"]),
        key=lambda item: item[1],
        reverse=True,
    )
    return ", ".join(f"{display.get(m, m)}={c:.3f}" for m, c in pairs[:n])


def _task_cache_key(task_view: str) -> str:
    return hashlib.sha256(task_view.encode()).hexdigest()


def _counter_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.items()}


def collect_events(
    *,
    db: sqlite3.Connection,
    sessions: list[sqlite3.Row],
    router: PrefillRouter,
    max_user_turns: int | None,
    score_cache: dict[str, dict] | None = None,
    score_cache_flush=None,
    progress_every: int = 0,
) -> list[dict]:
    events: list[dict] = []
    route_scores = 0
    cache_hits = 0
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

            raw_text = _content_text(msg.get("content")).strip()
            base = {
                "session_id": sid,
                "session_name": session["name"],
                "updated_at": session["updated_at"],
                "turn": user_turns,
                "raw_text": raw_text[:500],
                "context_tokens_est": _context_tokens(seen),
            }
            if is_goose_title_request(raw_text):
                events.append({**base, "kind": "utility", "reason": "goose_title_generation"})
                continue

            task_view = build_task_view(list(seen))
            if task_view is None:
                reason = "goose_info_only" if is_info_only_request(raw_text) else "non_decision_turn"
                events.append({**base, "kind": "non_decision", "reason": reason})
                continue

            cache_key = _task_cache_key(task_view)
            cached = score_cache.get(cache_key) if score_cache is not None else None
            if cached:
                model_names = list(cached["model_names"])
                confidences = [float(c) for c in cached["confidences"]]
                cache_hits += 1
            else:
                result = router.route(task_view, tolerance=0.0)
                model_names = list(result.model_names)
                confidences = [float(c) for c in result.confidences]
                if score_cache is not None:
                    score_cache[cache_key] = {
                        "task_sha256": cache_key,
                        "model_names": model_names,
                        "confidences": confidences,
                    }
                    if score_cache_flush is not None:
                        score_cache_flush(score_cache)
            route_scores += 1
            if progress_every > 0 and route_scores % progress_every == 0:
                print(
                    f"  scored route decisions: {route_scores} "
                    f"(cache hits {cache_hits})",
                    file=sys.stderr,
                )
            events.append(
                {
                    **base,
                    "kind": "route",
                    "reason": "route",
                    "task_view": task_view,
                    "model_names": model_names,
                    "confidences": confidences,
                }
            )
    return events


def compute_summaries(
    *,
    events: list[dict],
    rates: dict[str, tuple[float, float]],
    routing_rates: dict[str, tuple[float, float]] | None = None,
    tolerances: list[float],
    baseline: str,
    cache_read_multiplier: float,
    min_context_tokens: int,
    output_token_weight: float,
) -> list[dict]:
    cheapest = _cheapest_model(rates)
    baseline_blend = rates[baseline][0] + output_token_weight * rates[baseline][1]
    rows = []
    routing_rates = routing_rates or rates

    for tol in tolerances:
        total = Counter()
        reasons = Counter()
        last_by_session: dict[str, str] = {}
        cache_last_by_session: dict[str, str] = {}
        baseline_seen_sessions: set[str] = set()
        cache_actual = 0.0
        cache_base = 0.0
        cache_tokens = 0
        skipped = 0
        for event in events:
            sid = event["session_id"]
            selected: str | None = None
            reason = event["reason"]
            if event["kind"] == "route":
                selected = _pick(
                    event["model_names"],
                    event["confidences"],
                    routing_rates,
                    tol,
                )
                last_by_session[sid] = selected
            elif event["kind"] == "utility":
                selected = cheapest
            elif sid in last_by_session:
                selected = last_by_session[sid]
                reason = "pin_non_decision"
            elif event["reason"] == "goose_info_only":
                selected = cheapest
                last_by_session[sid] = selected
                reason = "cheap_info_cold"
            else:
                skipped += 1
                reason = "cold_non_decision_skipped"
            reasons[reason] += 1
            if selected:
                total[selected] += 1
                tokens = int(event.get("context_tokens_est") or 0)
                if tokens <= 0:
                    tokens = min_context_tokens
                cache_tokens += tokens
                cached_actual = cache_last_by_session.get(sid) == selected
                actual_mult = cache_read_multiplier if cached_actual else 1.0
                cache_actual += tokens * rates[selected][0] * actual_mult
                cache_last_by_session[sid] = selected
                cached_base = sid in baseline_seen_sessions
                base_mult = cache_read_multiplier if cached_base else 1.0
                cache_base += tokens * rates[baseline][0] * base_mult
                baseline_seen_sessions.add(sid)

        n = sum(total.values()) or 1
        avg_in = sum(total[m] * rates[m][0] for m in total) / n
        avg_out = sum(total[m] * rates[m][1] for m in total) / n
        cache_rate = cache_actual / cache_tokens if cache_tokens else 0.0
        base_cache_rate = cache_base / cache_tokens if cache_tokens else 0.0
        cache_savings = 1 - cache_actual / cache_base if cache_base else 0.0
        blended = avg_in + output_token_weight * avg_out
        proxy_savings = 1 - blended / baseline_blend if baseline_blend else 0.0
        rows.append(
            {
                "tolerance": tol,
                "counted_turns": n,
                "skipped_cold_turns": skipped,
                "avg_input_per_m": avg_in,
                "avg_output_per_m": avg_out,
                "blend_per_m": blended,
                "proxy_savings_pct": 100 * proxy_savings,
                "cache_input_rate_per_m": cache_rate,
                "cache_baseline_rate_per_m": base_cache_rate,
                "cache_savings_pct": 100 * cache_savings,
                "cache_token_weight": cache_tokens,
                "baseline_share_pct": 100 * total.get(baseline, 0) / n,
                "model_counts": _counter_dict(total),
                "reason_counts": _counter_dict(reasons),
            }
        )
    return rows


def summarize(
    *,
    events: list[dict],
    rates: dict[str, tuple[float, float]],
    display: dict[str, str],
    summaries: list[dict],
    preview_tolerance: float,
    show_decisions: int,
    baseline: str,
) -> None:
    print(f"Events: {len(events)}")
    print(
        "Kinds: "
        + ", ".join(f"{k}={v}" for k, v in Counter(e["kind"] for e in events).most_common())
    )
    route_events = [e for e in events if e["kind"] == "route"]
    print(f"Route decisions scored once: {len(route_events)}")
    print()

    for row in summaries:
        tol = float(row["tolerance"])
        total = Counter(row["model_counts"])
        reasons = Counter(row["reason_counts"])
        n = int(row["counted_turns"]) or 1
        print(
            f"Tolerance {tol:.3f} "
            f"({n} counted turns, {row['skipped_cold_turns']} cold skipped)"
        )
        print(
            f"  avg rates: input ${row['avg_input_per_m']:.2f}/M, "
            f"output ${row['avg_output_per_m']:.2f}/M, "
            f"blend ${row['blend_per_m']:.2f}/M "
            f"({row['proxy_savings_pct']:.0f}% proxy savings)"
        )
        print(
            f"  cache input proxy: routed ${row['cache_input_rate_per_m']:.2f}/M vs "
            f"baseline ${row['cache_baseline_rate_per_m']:.2f}/M "
            f"({row['cache_savings_pct']:.0f}% savings)"
        )
        print("  reasons: " + ", ".join(f"{k}={v}" for k, v in reasons.most_common()))
        for model, count in total.most_common():
            print(f"  {display.get(model, model):28s} {count:4d}  {count / n:6.1%}")
        print()

    if show_decisions <= 0:
        return

    print(f"Decision preview at tolerance {preview_tolerance:.2f}:")
    previews = []
    for event in route_events:
        selected = _pick(event["model_names"], event["confidences"], rates, preview_tolerance)
        previews.append(
            (
                rates[selected][0],
                str(event["session_id"]),
                int(event["turn"]),
                selected,
                event,
            )
        )
    previews.sort(reverse=True)
    for _, _, _, selected, event in previews[:show_decisions]:
        text = " ".join((event.get("task_view") or "").split())[:220]
        print(
            f"  {event['session_id']} turn {event['turn']:02d} -> "
            f"{display.get(selected, selected)} | {_top_confidences(event, display)}"
        )
        print(f"    {text}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DB_DEFAULT))
    parser.add_argument("--config", default="configs/combined-pool.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--model-like", default="%nvidia-routed%")
    parser.add_argument("--max-user-turns", type=int, default=8)
    parser.add_argument("--tolerances", default="0.0,0.02,0.05,0.10,0.20,0.35")
    parser.add_argument("--show-decisions", type=int, default=8)
    parser.add_argument("--preview-tolerance", type=float, default=None)
    parser.add_argument("--baseline", default="claude-opus-4-6-high")
    parser.add_argument(
        "--output-token-weight",
        type=float,
        default=0.25,
        help="Blend proxy: input_rate + weight * output_rate. This is only a rate proxy.",
    )
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
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = args.checkpoint or cfg.routing.checkpoint
    tolerances = [float(x) for x in args.tolerances.split(",") if x.strip()]
    preview_tolerance = (
        args.preview_tolerance if args.preview_tolerance is not None else cfg.routing.tolerance
    )

    db = _connect_readonly(Path(args.db).expanduser())
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

    max_turns = args.max_user_turns or None
    events = collect_events(
        db=db,
        sessions=sessions,
        router=router,
        max_user_turns=max_turns,
    )
    rates, display = _cost_maps(args.config)
    baseline = args.baseline if args.baseline in rates else max(
        rates,
        key=lambda m: (rates[m][0], rates[m][1], m),
    )
    summaries = compute_summaries(
        events=events,
        rates=rates,
        tolerances=tolerances,
        baseline=baseline,
        cache_read_multiplier=args.cache_read_multiplier,
        min_context_tokens=args.min_context_tokens,
        output_token_weight=args.output_token_weight,
    )
    summarize(
        events=events,
        rates=rates,
        display=display,
        summaries=summaries,
        preview_tolerance=preview_tolerance,
        show_decisions=args.show_decisions,
        baseline=baseline,
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
                    "db": str(Path(args.db).expanduser()),
                    "model_like": args.model_like,
                    "sessions": [
                        {
                            "id": session["id"],
                            "name": session["name"],
                            "updated_at": session["updated_at"],
                        }
                        for session in sessions
                    ],
                    "max_user_turns": max_turns,
                    "event_count": len(events),
                    "kind_counts": _counter_dict(Counter(e["kind"] for e in events)),
                    "route_decisions": sum(1 for e in events if e["kind"] == "route"),
                    "baseline": baseline,
                    "cache_read_multiplier": args.cache_read_multiplier,
                    "output_token_weight": args.output_token_weight,
                    "tolerances": summaries,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print()
        print(f"Wrote Goose replay JSON -> {out}")


if __name__ == "__main__":
    main()
