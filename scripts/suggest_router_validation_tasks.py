#!/usr/bin/env python3
"""Suggest representative Goose tasks for routed-vs-frontier validation.

This is local/provider-free. It mines recent Goose sessions, extracts the first
real user prompt, and proposes a small balanced set of tasks to rerun through
``scripts/goose-ab-task.sh``.

By default it prints privacy-safer previews. Use ``--emit-commands`` when you
want exact rerun commands containing the original prompts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_goose_quality_signals import (  # noqa: E402
    DB_DEFAULT,
    analyze_session,
    connect_readonly,
    direct_text,
    is_synthetic_user_summary,
    model_name,
    parse_content,
)


GENERIC_PROMPTS = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "ok",
    "yes",
    "no",
}

READ_ONLY_RE = re.compile(
    r"\b("
    r"analy[sz]e|check|explain|find|inspect|list|look|report|review|show|"
    r"status|summari[sz]e|tell me|what|where|why"
    r")\b",
    re.IGNORECASE,
)

MUTATING_RE = re.compile(
    r"\b("
    r"add|build|change|create|debug|delete|edit|fix|implement|install|kill|"
    r"make|patch|refactor|remove|restart|run|start|test|update|wire|write"
    r")\b",
    re.IGNORECASE,
)


def load_sessions(db: sqlite3.Connection, limit: int, model_like: str) -> list[sqlite3.Row]:
    return db.execute(
        """
        SELECT id, name, updated_at, provider_name, model_config_json, working_dir,
               total_tokens, accumulated_cost
        FROM sessions
        WHERE session_type='user'
          AND model_config_json LIKE ?
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (model_like, limit),
    ).fetchall()


def load_messages(db: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    return db.execute(
        """
        SELECT role, content_json
        FROM messages
        WHERE session_id=?
        ORDER BY id
        """,
        (session_id,),
    ).fetchall()


def first_user_prompt(db: sqlite3.Connection, session_id: str) -> str:
    for row in load_messages(db, session_id):
        if row["role"] != "user":
            continue
        text = direct_text(parse_content(row["content_json"]))
        text = " ".join(text.split())
        if not text or is_synthetic_user_summary(text):
            continue
        if text.startswith("/") or text.lower() in GENERIC_PROMPTS:
            continue
        return text
    return ""


def short(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value[:36] or "task"


def task_kind(prompt: str) -> str:
    mutating = bool(MUTATING_RE.search(prompt))
    read_only = bool(READ_ONLY_RE.search(prompt))
    if mutating and not read_only:
        return "mutating"
    if read_only and not mutating:
        return "read-only"
    if mutating and read_only:
        return "mixed"
    return "unknown"


def depth_bucket(tokens: int | None, tool_responses: int, messages: int) -> str:
    tokens = int(tokens or 0)
    if tokens >= 20_000 or tool_responses >= 25 or messages >= 60:
        return "deep"
    if tokens >= 8_000 or tool_responses >= 8 or messages >= 25:
        return "medium"
    return "small"


def score_candidate(prompt: str, tokens: int | None, tool_responses: int, messages: int) -> float:
    tokens = int(tokens or 0)
    return (
        min(len(prompt), 900) / 120
        + math.log1p(tokens) / 2
        + min(tool_responses, 60) / 4
        + min(messages, 100) / 20
    )


def candidate_rows(db: sqlite3.Connection, sessions: list[sqlite3.Row]) -> list[dict]:
    candidates = []
    seen_prompts: set[str] = set()
    for session in sessions:
        prompt = first_user_prompt(db, session["id"])
        norm = re.sub(r"\s+", " ", prompt.lower()).strip()
        if not prompt or norm in seen_prompts:
            continue
        seen_prompts.add(norm)
        signals = analyze_session(db, session)
        kind = task_kind(prompt)
        bucket = depth_bucket(
            session["total_tokens"],
            signals.tool_responses,
            signals.messages,
        )
        candidates.append(
            {
                "session_id": session["id"],
                "name": session["name"],
                "updated_at": session["updated_at"],
                "model": model_name(session["model_config_json"]),
                "working_dir": session["working_dir"],
                "tokens": session["total_tokens"],
                "messages": signals.messages,
                "tool_responses": signals.tool_responses,
                "tool_errors": signals.tool_errors,
                "kind": kind,
                "bucket": bucket,
                "prompt": prompt,
                "score": score_candidate(
                    prompt,
                    session["total_tokens"],
                    signals.tool_responses,
                    signals.messages,
                ),
            }
        )
    return candidates


def select_balanced(candidates: list[dict], *, count: int) -> list[dict]:
    preferred_order = [
        ("mutating", "medium"),
        ("read-only", "medium"),
        ("mutating", "deep"),
        ("read-only", "deep"),
        ("mixed", "medium"),
        ("mixed", "deep"),
        ("mutating", "small"),
        ("read-only", "small"),
        ("unknown", "medium"),
        ("unknown", "deep"),
    ]
    selected: list[dict] = []
    used: set[str] = set()
    by_group: dict[tuple[str, str], list[dict]] = {}
    for row in candidates:
        by_group.setdefault((row["kind"], row["bucket"]), []).append(row)
    for rows in by_group.values():
        rows.sort(key=lambda row: row["score"], reverse=True)

    for group in preferred_order:
        if len(selected) >= count:
            break
        for row in by_group.get(group, [])[:1]:
            if row["session_id"] not in used:
                selected.append(row)
                used.add(row["session_id"])
                break

    if len(selected) < count:
        for row in sorted(candidates, key=lambda item: item["score"], reverse=True):
            if row["session_id"] in used:
                continue
            selected.append(row)
            used.add(row["session_id"])
            if len(selected) >= count:
                break
    return selected


def filter_candidates(
    candidates: list[dict],
    *,
    max_tokens: int,
    max_tool_errors: int,
) -> list[dict]:
    out = []
    for row in candidates:
        if max_tokens > 0 and int(row["tokens"] or 0) > max_tokens:
            continue
        if max_tool_errors >= 0 and int(row["tool_errors"] or 0) > max_tool_errors:
            continue
        out.append(row)
    return out


def command_for(row: dict) -> str:
    name = f"router-ab-{slug(row['name'] or row['prompt'])}"
    args = ["scripts/goose-ab-task.sh", "--name", name]
    if row["kind"] in {"mutating", "mixed"}:
        args.append("--isolate-worktrees")
    args.extend(["--", row["prompt"]])
    return " ".join(shlex.quote(arg) for arg in args)


def print_table(rows: list[dict], *, preview_chars: int, emit_commands: bool) -> None:
    print(
        f"{'#':>2}  {'kind':10s} {'depth':7s} {'tokens':>8s} {'tools':>5s} "
        f"{'model':14s} {'updated':19s} name / prompt"
    )
    for idx, row in enumerate(rows, 1):
        print(
            f"{idx:2d}  {row['kind']:10s} {row['bucket']:7s} "
            f"{int(row['tokens'] or 0):8,d} {row['tool_responses']:5d} "
            f"{row['model'][:14]:14s} {str(row['updated_at'])[:19]:19s} "
            f"{row['name']}"
        )
        print(f"    {short(row['prompt'], preview_chars)}")
        if row["tool_errors"]:
            print(f"    note: source session had {row['tool_errors']} tool errors")
        if emit_commands:
            print(f"    command: {command_for(row)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DB_DEFAULT))
    parser.add_argument("--sessions", type=int, default=120)
    parser.add_argument("--model-like", default="%")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=60_000,
        help="Skip source sessions above this token count. Use 0 for no cap.",
    )
    parser.add_argument(
        "--max-tool-errors",
        type=int,
        default=3,
        help="Skip source sessions with more tool errors than this. Use -1 for no cap.",
    )
    parser.add_argument("--preview-chars", type=int, default=160)
    parser.add_argument("--emit-commands", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    db = connect_readonly(Path(args.db).expanduser())
    sessions = load_sessions(db, args.sessions, args.model_like)
    candidates = candidate_rows(db, sessions)
    filtered = filter_candidates(
        candidates,
        max_tokens=args.max_tokens,
        max_tool_errors=args.max_tool_errors,
    )
    selected = select_balanced(filtered, count=args.count)
    kind_counts = Counter(row["kind"] for row in filtered)
    bucket_counts = Counter(row["bucket"] for row in filtered)

    print("Router Validation Task Suggestions")
    print(
        f"Inspected sessions={len(sessions)}, candidates={len(candidates)}, "
        f"after_filters={len(filtered)}, selected={len(selected)}"
    )
    print(
        f"Filters: max_tokens={args.max_tokens or 'none'}, "
        f"max_tool_errors={args.max_tool_errors if args.max_tool_errors >= 0 else 'none'}"
    )
    if kind_counts:
        print("Kinds: " + ", ".join(f"{k}={v}" for k, v in sorted(kind_counts.items())))
    if bucket_counts:
        print("Depth: " + ", ".join(f"{k}={v}" for k, v in sorted(bucket_counts.items())))
    print()
    print_table(selected, preview_chars=args.preview_chars, emit_commands=args.emit_commands)
    print()
    print(
        "Interpretation: use 2-3 medium/deep candidates first. For mutating/mixed "
        "tasks, --isolate-worktrees gives the cleanest routed-vs-Opus comparison "
        "but requires the source git worktree to be clean."
    )

    if args.json_out:
        out = Path(args.json_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sessions_inspected": len(sessions),
                    "candidate_count": len(candidates),
                    "filtered_count": len(filtered),
                    "filters": {
                        "max_tokens": args.max_tokens,
                        "max_tool_errors": args.max_tool_errors,
                    },
                    "kind_counts": dict(kind_counts),
                    "bucket_counts": dict(bucket_counts),
                    "selected": selected,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"Wrote JSON suggestions -> {out}")


if __name__ == "__main__":
    main()
