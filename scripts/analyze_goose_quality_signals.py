#!/usr/bin/env python3
"""Compare local Goose quality signals across model/provider groups.

This is a heuristic, local-only analyzer. It does not judge answer correctness
and does not call providers. It looks for weak but useful operational signals:

  - user correction/frustration follow-ups in direct user text
  - assistant apology/failure/error language
  - tool responses marked error or with non-zero exit codes
  - sessions ending on a user turn
  - sessions with no assistant text

Use it to spot obvious regressions in routed Goose sessions before spending on a
proper head-to-head quality run. Treat the output as triage, not proof.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from model_router_toolkit.config import load_config


DB_DEFAULT = Path.home() / ".local/share/goose/sessions/sessions.db"

USER_PROBLEM_RE = re.compile(
    r"\b("
    r"wrong|incorrect|no[, ]|not (?:right|correct|working|what i asked)|"
    r"does(?:n['’]?t| not) work|failed|failing|error|bad request|"
    r"shitting|sucks|broken|why (?:is|did)|come on|"
    r"you (?:missed|forgot|broke)|still (?:not|broken)"
    r")\b",
    re.IGNORECASE,
)

ASSISTANT_PROBLEM_RE = re.compile(
    r"\b("
    r"sorry|apologize|apologies|i can(?:not|'t)|i'?m unable|"
    r"failed|error|exception|traceback|bad request|not found|permission denied|"
    r"i don(?:'|’)t have|i'?m not sure"
    r")\b",
    re.IGNORECASE,
)

SYNTHETIC_USER_SUMMARY_RE = re.compile(
    r"^((a|an|the)\s+"
    r"(shell command|gh cli call|grep search|file(?: system)? search|command|"
    r"diff|test command|query|tool call)\b"
    r"|(a|an|the)\s+([\w()./-]+\s+){1,10}"
    r"(was|were|is|are|has|have|had|attempt\w*|execut\w*|ran|run|display\w*|"
    r"perform\w*|retriev\w*|creat\w*|found|confirm\w*|complet\w*|fail\w*|"
    r"return\w*|succeed\w*|check\w*|search\w*|inspect\w*|review\w*|map\w*|"
    r"show\w*|reveal\w*|report\w*|identif\w*|determin\w*|read|added|updated|"
    r"fixed|removed|verified)\b"
    r"|(checked|ran|searched|inspected|reviewed|confirmed|found|created|added|"
    r"updated|fixed|removed|verified|identified|determined|examined|attempted|"
    r"executed|performed|retrieved|displayed|completed|tested)\b)",
    re.IGNORECASE,
)


@dataclass
class SessionSignals:
    session_id: str
    name: str
    provider: str
    model: str
    updated_at: str
    messages: int = 0
    user_text_turns: int = 0
    assistant_text_turns: int = 0
    assistant_tool_requests: int = 0
    tool_responses: int = 0
    tool_errors: int = 0
    user_problem_turns: int = 0
    synthetic_user_summaries: int = 0
    assistant_problem_turns: int = 0
    ends_on_user: bool = False
    total_tokens: int | None = None
    accumulated_cost: float | None = None
    examples: list[str] = field(default_factory=list)

    @property
    def problem_score(self) -> int:
        score = self.user_problem_turns * 3 + self.assistant_problem_turns
        score += min(self.tool_errors, 5)
        if self.ends_on_user:
            score += 1
        if self.assistant_text_turns == 0 and self.messages > 0:
            score += 2
        return score

    @property
    def problem_score_per_100_messages(self) -> float:
        return 100 * self.problem_score / self.messages if self.messages else 0.0


def connect_readonly(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    return db


def model_name(model_config_json: str | None) -> str:
    if not model_config_json:
        return ""
    try:
        return str((json.loads(model_config_json) or {}).get("model_name") or "")
    except json.JSONDecodeError:
        return ""


def parse_content(content_json: str):
    try:
        content = json.loads(content_json)
    except json.JSONDecodeError:
        return content_json
    return content


def direct_text(content) -> str:
    """Text written by the user/assistant, excluding tool response payloads."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    return "\n".join(parts).strip()


def is_synthetic_user_summary(text: str) -> bool:
    return bool(SYNTHETIC_USER_SUMMARY_RE.match(" ".join(text.split())))


def count_tool_requests(content) -> int:
    if not isinstance(content, list):
        return 0
    return sum(1 for item in content if isinstance(item, dict) and item.get("type") == "toolRequest")


def tool_response_stats(content) -> tuple[int, int]:
    if not isinstance(content, list):
        return 0, 0
    responses = 0
    errors = 0
    for item in content:
        if not isinstance(item, dict) or item.get("type") not in {"toolResponse", "tool_result"}:
            continue
        responses += 1
        result = item.get("toolResult") or {}
        if result.get("status") == "error":
            errors += 1
            continue
        value = result.get("value") or {}
        if value.get("isError") is True:
            errors += 1
            continue
        structured = value.get("structuredContent") or {}
        exit_code = structured.get("exit_code")
        try:
            if exit_code is not None and int(exit_code) != 0:
                errors += 1
        except (TypeError, ValueError):
            pass
    return responses, errors


def load_sessions(db: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    return db.execute(
        """
        SELECT id, name, updated_at, provider_name, model_config_json,
               total_tokens, accumulated_cost
        FROM sessions
        WHERE session_type='user'
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (limit,),
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


def analyze_session(db: sqlite3.Connection, session: sqlite3.Row) -> SessionSignals:
    signals = SessionSignals(
        session_id=session["id"],
        name=session["name"],
        provider=session["provider_name"] or "",
        model=model_name(session["model_config_json"]),
        updated_at=session["updated_at"],
        total_tokens=session["total_tokens"],
        accumulated_cost=session["accumulated_cost"],
    )
    messages = load_messages(db, session["id"])
    signals.messages = len(messages)
    last_role = None
    for row in messages:
        role = row["role"]
        last_role = role
        content = parse_content(row["content_json"])
        text = direct_text(content)
        tool_responses, tool_errors = tool_response_stats(content)
        signals.tool_responses += tool_responses
        signals.tool_errors += tool_errors

        if role == "user" and text:
            if is_synthetic_user_summary(text):
                signals.synthetic_user_summaries += 1
                continue
            signals.user_text_turns += 1
            if signals.user_text_turns > 1 and USER_PROBLEM_RE.search(text):
                signals.user_problem_turns += 1
                if len(signals.examples) < 3:
                    signals.examples.append("user: " + " ".join(text.split())[:180])
        elif role == "assistant":
            signals.assistant_tool_requests += count_tool_requests(content)
            if text:
                signals.assistant_text_turns += 1
                if ASSISTANT_PROBLEM_RE.search(text):
                    signals.assistant_problem_turns += 1
                    if len(signals.examples) < 3:
                        signals.examples.append("assistant: " + " ".join(text.split())[:180])
    signals.ends_on_user = last_role == "user"
    return signals


def group_key(signals: SessionSignals) -> str:
    if signals.model == "nvidia-routed":
        return "nvidia-routed"
    if signals.model == "claude-opus-4-8":
        return "claude-opus-4-8"
    return signals.model or signals.provider or "unknown"


def load_shadow_route_mix(path: str, config_path: str) -> dict[str, str]:
    if not path:
        return {}
    route_path = Path(path).expanduser()
    if not route_path.exists():
        return {}
    cfg = load_config(config_path)
    display = {m.name: (m.display_name or m.name) for m in cfg.models}
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    with route_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            session = str(row.get("session_key") or "")
            selected = row.get("selected_model")
            if session and selected:
                counts[session][str(selected)] += 1
    out = {}
    for session, counter in counts.items():
        total = sum(counter.values()) or 1
        parts = []
        for model, count in counter.most_common(4):
            parts.append(f"{display.get(model, model)}={100 * count / total:.0f}%")
        out[session] = ", ".join(parts)
    return out


def summarize_group(name: str, sessions: list[SessionSignals]) -> None:
    n = len(sessions)
    if not n:
        return
    messages = sum(s.messages for s in sessions)
    tool_errors = sum(s.tool_errors for s in sessions)
    tool_responses = sum(s.tool_responses for s in sessions)
    user_problem = sum(s.user_problem_turns for s in sessions)
    user_text_turns = sum(s.user_text_turns for s in sessions)
    assistant_problem = sum(s.assistant_problem_turns for s in sessions)
    assistant_text_turns = sum(s.assistant_text_turns for s in sessions)
    synthetic = sum(s.synthetic_user_summaries for s in sessions)
    ended_user = sum(1 for s in sessions if s.ends_on_user)
    no_text = sum(1 for s in sessions if s.assistant_text_turns == 0 and s.messages > 0)
    avg_score = sum(s.problem_score for s in sessions) / n
    score_per_100 = sum(s.problem_score_per_100_messages for s in sessions) / n
    user_flag_rate = 100 * user_problem / user_text_turns if user_text_turns else 0.0
    assistant_flag_rate = (
        100 * assistant_problem / assistant_text_turns if assistant_text_turns else 0.0
    )
    tool_error_rate = 100 * tool_errors / tool_responses if tool_responses else 0.0
    print(
        f"{name:22s} sessions={n:3d} messages={messages:5d} "
        f"user_flags={user_problem:3d}/{user_text_turns:<3d} ({user_flag_rate:4.1f}%) "
        f"assistant_flags={assistant_problem:3d}/{assistant_text_turns:<3d} ({assistant_flag_rate:4.1f}%) "
        f"tool_errors={tool_errors:3d}/{tool_responses:<3d} ({tool_error_rate:4.1f}%) "
        f"ends_user={ended_user:2d} no_text={no_text:2d} "
        f"synthetic_user={synthetic:3d} avg_score={avg_score:.2f} "
        f"score100={score_per_100:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DB_DEFAULT))
    parser.add_argument("--sessions", type=int, default=80)
    parser.add_argument("--show", type=int, default=12)
    parser.add_argument(
        "--min-rank-messages",
        type=int,
        default=8,
        help="Minimum session length for the ranked flagged-session list.",
    )
    parser.add_argument(
        "--models",
        default="nvidia-routed,claude-opus-4-8",
        help="Comma-separated model_name filter. Empty means all models.",
    )
    parser.add_argument("--route-log", default="")
    parser.add_argument("--config", default="configs/combined-pool.yaml")
    args = parser.parse_args()

    db = connect_readonly(Path(args.db).expanduser())
    route_mix = load_shadow_route_mix(args.route_log, args.config)
    model_filter = {m.strip() for m in args.models.split(",") if m.strip()}
    analyzed = [
        analyze_session(db, session)
        for session in load_sessions(db, args.sessions)
    ]
    if model_filter:
        analyzed = [s for s in analyzed if s.model in model_filter]

    groups: dict[str, list[SessionSignals]] = defaultdict(list)
    for signals in analyzed:
        groups[group_key(signals)].append(signals)

    print(f"Analyzed sessions: {len(analyzed)}")
    print("Heuristic quality/problem signals by model:")
    for name in sorted(groups):
        summarize_group(name, groups[name])

    print()
    print(f"Most flagged sessions ({min(args.show, len(analyzed))}):")
    rankable = [s for s in analyzed if s.messages >= args.min_rank_messages]
    ranked = sorted(
        rankable,
        key=lambda s: (
            s.problem_score_per_100_messages,
            s.problem_score,
            s.user_problem_turns,
            s.tool_errors,
        ),
        reverse=True,
    )
    for s in ranked[: args.show]:
        print(
            f"  score={s.problem_score:2d} score100={s.problem_score_per_100_messages:5.1f} "
            f"{s.model:16s} {s.session_id:12s} "
            f"msgs={s.messages:4d} user_flags={s.user_problem_turns} "
            f"assistant_flags={s.assistant_problem_turns} "
            f"tool_errors={s.tool_errors}/{s.tool_responses} "
            f"synthetic_user={s.synthetic_user_summaries} "
            f"end_user={int(s.ends_on_user)}  {s.name}"
        )
        if s.session_id in route_mix:
            print(f"    route_mix: {route_mix[s.session_id]}")
        for example in s.examples:
            print(f"    {example}")

    print()
    print(
        "Interpretation: these are failure/friction signals, not quality labels. "
        "A low routed score is reassuring only for obvious regressions; same-quality "
        "still needs live route logs plus head-to-head or user-reviewed outcomes."
    )


if __name__ == "__main__":
    main()
