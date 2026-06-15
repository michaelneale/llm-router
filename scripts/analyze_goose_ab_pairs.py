#!/usr/bin/env python3
"""Find local Goose routed-vs-Opus near-pairs.

This is local-only and uses existing session transcripts. It helps answer:
"Do we already have comparable tasks where one run used nvidia-routed and
another used direct Opus?"

Pairing is heuristic: first-user prompt similarity and session-name similarity.
It does not prove quality parity, but it identifies the best existing candidates
for manual review and shows the same quality-signal counters used by
``analyze_goose_quality_signals.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from analyze_goose_quality_signals import (
    DB_DEFAULT,
    analyze_session,
    connect_readonly,
    direct_text,
    group_key,
    load_messages,
    load_shadow_route_mix,
    model_name,
    parse_content,
)

GENERIC_NAMES = {
    "cli session",
    "casual greeting",
    "greeting and check in",
    "greetings and wellbeing",
    "user confirmation",
}
GENERIC_PROMPTS = {
    "hi",
    "hello",
    "hey",
    "how are you",
    "thanks",
    "thank you",
    "ok",
    "yes",
    "no",
}


@dataclass
class SessionSummary:
    row: sqlite3.Row
    model: str
    first_user: str


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(opus|routed|litellm|anthropic|openai)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity(a: str, b: str) -> float:
    a_norm = normalize(a)
    b_norm = normalize(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def generic_prompt(text: str) -> bool:
    norm = normalize(text)
    return norm in GENERIC_PROMPTS or len(norm.split()) < 4


def first_user_text(db: sqlite3.Connection, session_id: str) -> str:
    for row in load_messages(db, session_id):
        if row["role"] != "user":
            continue
        text = direct_text(parse_content(row["content_json"]))
        if text:
            return " ".join(text.split())
    return ""


def load_candidate_sessions(db: sqlite3.Connection, limit: int) -> list[SessionSummary]:
    rows = db.execute(
        """
        SELECT id, name, updated_at, provider_name, model_config_json,
               total_tokens, accumulated_cost
        FROM sessions
        WHERE session_type='user'
          AND (
            model_config_json LIKE '%nvidia-routed%'
            OR model_config_json LIKE '%claude-opus-4-8%'
          )
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    out = []
    for row in rows:
        out.append(
            SessionSummary(
                row=row,
                model=model_name(row["model_config_json"]),
                first_user=first_user_text(db, row["id"]),
            )
        )
    return out


def pair_score(routed: SessionSummary, opus: SessionSummary) -> tuple[float, float, float]:
    if not routed.first_user or not opus.first_user:
        return 0.0, 0.0, 0.0
    if generic_prompt(routed.first_user) or generic_prompt(opus.first_user):
        return 0.0, 0.0, 0.0
    prompt = similarity(routed.first_user, opus.first_user)
    routed_name = normalize(routed.row["name"])
    opus_name = normalize(opus.row["name"])
    if routed_name in GENERIC_NAMES or opus_name in GENERIC_NAMES:
        name = 0.0
    else:
        name = similarity(routed.row["name"], opus.row["name"])
    score = max(prompt, name * 0.92)
    return score, prompt, name


def short(text: str, limit: int = 130) -> str:
    text = " ".join((text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def fmt_cost(value) -> str:
    if value is None or value == "":
        return "-"
    return f"${float(value):.4f}"


def fmt_signals(signals) -> str:
    tool_rate = (
        100 * signals.tool_errors / signals.tool_responses
        if signals.tool_responses
        else 0.0
    )
    user_rate = (
        100 * signals.user_problem_turns / signals.user_text_turns
        if signals.user_text_turns
        else 0.0
    )
    return (
        f"msgs={signals.messages}, user_flags={signals.user_problem_turns}/"
        f"{signals.user_text_turns} ({user_rate:.1f}%), "
        f"tool_errors={signals.tool_errors}/{signals.tool_responses} ({tool_rate:.1f}%), "
        f"score100={signals.problem_score_per_100_messages:.1f}, "
        f"end_user={int(signals.ends_on_user)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DB_DEFAULT))
    parser.add_argument("--sessions", type=int, default=120)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--min-prompt-sim", type=float, default=0.55)
    parser.add_argument("--min-name-sim", type=float, default=0.72)
    parser.add_argument("--show", type=int, default=12)
    parser.add_argument("--route-log", default="/tmp/router-routes-shadow.jsonl")
    parser.add_argument("--config", default="configs/combined-pool.yaml")
    args = parser.parse_args()

    db = connect_readonly(Path(args.db).expanduser())
    summaries = load_candidate_sessions(db, args.sessions)
    routed = [s for s in summaries if s.model == "nvidia-routed"]
    opus = [s for s in summaries if s.model == "claude-opus-4-8"]
    route_mix = load_shadow_route_mix(args.route_log, args.config)

    pairs = []
    for r in routed:
        best = None
        for o in opus:
            score, prompt, name = pair_score(r, o)
            if prompt < args.min_prompt_sim and name < args.min_name_sim:
                continue
            if best is None or score > best[0]:
                best = (score, prompt, name, o)
        if best and best[0] >= args.min_score:
            pairs.append((best[0], best[1], best[2], r, best[3]))
    pairs.sort(key=lambda item: item[0], reverse=True)

    print(
        f"Candidates: routed={len(routed)}, opus={len(opus)}, "
        f"pairs>={args.min_score:.2f}={len(pairs)}"
    )
    print()

    for score, prompt_sim, name_sim, r, o in pairs[: args.show]:
        r_sig = analyze_session(db, r.row)
        o_sig = analyze_session(db, o.row)
        print(
            f"score={score:.2f} prompt={prompt_sim:.2f} name={name_sim:.2f}"
        )
        print(
            f"  routed {r.row['id']:12s} {r.row['updated_at']} "
            f"{r.row['name']} cost={fmt_cost(r.row['accumulated_cost'])}"
        )
        if r.row["id"] in route_mix:
            print(f"    shadow_mix: {route_mix[r.row['id']]}")
        print(f"    prompt: {short(r.first_user)}")
        print(f"    signals: {fmt_signals(r_sig)}")
        print(
            f"  opus   {o.row['id']:12s} {o.row['updated_at']} "
            f"{o.row['name']} cost={fmt_cost(o.row['accumulated_cost'])}"
        )
        print(f"    prompt: {short(o.first_user)}")
        print(f"    signals: {fmt_signals(o_sig)}")
        print()

    print(
        "Interpretation: pairs are heuristic candidates for manual review. "
        "Use them to choose head-to-head tasks or inspect existing outcomes; "
        "they are not automatic quality labels."
    )


if __name__ == "__main__":
    main()
