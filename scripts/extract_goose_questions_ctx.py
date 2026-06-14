"""Context-aware prompt extraction from goose's sessions.db (experiment A).

The plain extractor (extract_goose_questions.py) emits each user prompt as a
bare, decontextualized line. But in real agent sessions the *difficulty* of a
turn lives in the accumulated session state, not the sentence — "will it really
leak?" is a hard systems question only because of the 800 messages before it.
A prefill router that scores difficulty from text alone is therefore blind to
deep work and routes almost everything cheap.

This variant attaches a compact, rolling **session-context preamble** to each
user prompt, built from the conversation that precedes it:

  [ctx: <session name> | turn N/T | files:F tools:K | did: git-diff, grep, edit
        | re: <last assistant reasoning, trimmed>] <the user prompt>

The preamble is intentionally short and structural (model/tool activity, not
raw transcripts) so it (a) fits the encoder, (b) leaks no file contents, and
(c) gives the encoder a difficulty signal: a one-line prompt buried in a long,
tool-heavy session reads very differently from a cold-start one-liner.

Output is one example per line, drop-in for `model-router collect --questions`,
so it can be A/B'd against the plain set with the same pipeline.

Usage:
  python scripts/extract_goose_questions_ctx.py --output data/goose-questions-ctx.txt
  python scripts/extract_goose_questions_ctx.py --no-preamble   # ablation = plain
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

DB_DEFAULT = Path.home() / ".local/share/goose/sessions/sessions.db"

CONTINUATION = re.compile(
    r"^(yes|yeah|yep|ok|okay|no|nope|sure|go|go on|keep going|continue|do it|"
    r"try again|and |also |then |next |what about|how about|why|hrm|hmm|wait)\b",
    re.IGNORECASE,
)

# Auto-generated "summary" user turns goose injects (not real prompts). They
# read like prose narration of a tool result. We use them as *context* but
# never as the prompt being scored.
SUMMARY_HINT = re.compile(
    r"^(A |The |An )?\w+ (command|search|file|git|diff|directory|shell|script|"
    r"query|edit|call|read|listing) (was|were|executed|run|displayed|performed|"
    r"retrieved|created)\b",
    re.IGNORECASE,
)


def _msg_text(content) -> str:
    return " ".join(
        c.get("text", "") for c in content if c.get("type") == "text"
    ).strip()


def _tool_name(content) -> str | None:
    for c in content:
        if c.get("type") == "toolRequest":
            try:
                return c["toolCall"]["value"]["name"]
            except (KeyError, TypeError):
                return None
    return None


def _shell_verb(content) -> str | None:
    """Cheap label for what a shell/dev tool actually did (git diff, grep, ...)."""
    for c in content:
        if c.get("type") != "toolRequest":
            continue
        try:
            args = c["toolCall"]["value"].get("arguments", {})
            name = c["toolCall"]["value"].get("name", "")
        except (KeyError, TypeError):
            continue
        if name in ("shell", "developer__shell"):
            cmd = str(args.get("command", "")).strip().lower()
            for verb in ("git diff", "git merge", "git rebase", "cargo build",
                         "cargo test", "pytest", "grep", "rg ", "rsync", "ssh",
                         "git log", "git status", "git commit"):
                if verb in cmd:
                    return verb.strip()
            return (cmd.split() or ["shell"])[0]
        if name in ("edit", "write", "developer__text_editor", "str_replace"):
            return "edit"
        if "read" in name or "view" in name:
            return "read"
        return name
    return None


def _trim(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "\u2026"


def extract(db_path: Path, include_continuations: bool, add_preamble: bool):
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    sessions = [
        r["id"]
        for r in db.execute(
            "SELECT id FROM sessions WHERE session_type IN "
            "('user','acp','terminal') ORDER BY created_at"
        ).fetchall()
    ]

    stats = Counter()
    examples: list[str] = []

    for sid in sessions:
        rows = db.execute(
            "SELECT m.role, m.content_json, m.metadata_json, s.name AS sname "
            "FROM messages m JOIN sessions s ON s.id=m.session_id "
            "WHERE m.session_id=? ORDER BY m.created_timestamp",
            (sid,),
        ).fetchall()
        if not rows:
            continue
        sname = _trim(rows[0]["sname"] or "session", 40)

        # First pass: count total real user turns for "turn N/T".
        # Rolling context accumulators reset per session.
        files_touched: set[str] = set()
        tool_verbs: list[str] = []
        last_reasoning = ""
        n_tools = 0
        user_turn = 0

        for r in rows:
            try:
                content = json.loads(r["content_json"])
                meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
            except (ValueError, TypeError):
                continue

            if r["role"] == "assistant":
                txt = _msg_text(content)
                if txt:
                    last_reasoning = txt
                verb = _shell_verb(content)
                if verb:
                    tool_verbs.append(verb)
                    n_tools += 1
                continue

            # role == user
            txt = _msg_text(content)
            if not txt:
                continue

            # Auto-summary turns: fold into context, don't emit as a prompt.
            is_summary = SUMMARY_HINT.match(txt) or not (
                meta.get("userVisible") and meta.get("agentVisible")
            )
            if is_summary:
                if "git diff" in txt.lower() or "files modified" in txt.lower():
                    m = re.search(r"(\d+)\s+files?\s+modified", txt.lower())
                    if m:
                        files_touched |= {f"~{m.group(1)}f"}
                stats["context_summary"] += 1
                continue

            stats["user_prompts"] += 1
            user_turn += 1

            # --- filters mirror the plain extractor ---
            if txt.startswith("/") or txt.startswith("<info-msg>"):
                stats["skip_synthetic"] += 1
                continue
            flat = re.sub(r"\s+", " ", txt)
            if not (15 <= len(flat) <= 4000):
                stats["skip_len"] += 1
                continue
            if not include_continuations and CONTINUATION.match(flat) and len(flat) < 80:
                stats["skip_continuation"] += 1
                continue

            if add_preamble:
                recent = ", ".join(
                    f"{v}\u00d7{c}" if c > 1 else v
                    for v, c in Counter(tool_verbs[-12:]).most_common(4)
                ) or "none"
                fpart = f" files:{len(files_touched)}" if files_touched else ""
                pre = (
                    f"[ctx: {sname} | turn {user_turn} | tools:{n_tools}{fpart}"
                    f" | did: {recent}"
                )
                if last_reasoning:
                    pre += f" | re: {_trim(last_reasoning, 120)}"
                pre += "] "
                examples.append(_trim(pre + flat, 4000))
            else:
                examples.append(flat)

    # exact + near-dupe collapse on the *prompt* portion (strip preamble first)
    seen_exact, seen_prefix, out = set(), set(), []
    for ex in examples:
        core = ex.split("] ", 1)[-1] if ex.startswith("[ctx:") else ex
        if core in seen_exact:
            continue
        seen_exact.add(core)
        p = re.sub(r"[^a-z0-9]", "", core.lower())[:80]
        if p in seen_prefix:
            continue
        seen_prefix.add(p)
        out.append(ex)

    stats["kept"] = len(out)
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_DEFAULT))
    ap.add_argument("--output", default="data/goose-questions-ctx.txt")
    ap.add_argument("--include-continuations", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--no-preamble",
        action="store_true",
        help="ablation: emit bare prompts (should ~match the plain extractor)",
    )
    args = ap.parse_args()

    out, stats = extract(
        Path(args.db), args.include_continuations, add_preamble=not args.no_preamble
    )
    if args.limit:
        out = out[: args.limit]

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out) + "\n")

    print(f"real user prompts seen   : {stats['user_prompts']}")
    print(f"  context-summary folded : {stats['context_summary']}")
    print(f"  skipped (synthetic)    : {stats['skip_synthetic']}")
    print(f"  skipped (length)       : {stats['skip_len']}")
    print(f"  skipped (continuation) : {stats['skip_continuation']}")
    print(f"  kept (unique)          : {stats['kept']}")
    print(f"preamble                 : {'OFF (ablation)' if args.no_preamble else 'ON'}")
    print(f"wrote {len(out)} examples -> {dst}")


if __name__ == "__main__":
    main()
