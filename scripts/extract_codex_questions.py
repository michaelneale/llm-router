"""Extract user prompts from Codex rollout transcripts for router training.

Codex stores full session transcripts as JSONL rollouts under
~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl. Each line is an event; user turns
are response_item events with payload.role == "user" and payload.content a list
of {type: input_text|text, text: ...}.

These sessions skew much *deeper* than typical goose sessions (long debugging /
architecture / "prove this works" loops), so they're a valuable complement for
training a router to recognize genuinely hard work.

Context-aware: like extract_goose_questions_ctx.py, attaches a compact preamble
([ctx: <thread> | turn N | tools:K | ...]) built from the surrounding events so
prompt difficulty is visible to a prefill encoder.

Usage:
  python scripts/extract_codex_questions.py --output data/codex-questions-ctx.txt
  python scripts/extract_codex_questions.py --no-preamble    # bare prompts
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path.home() / ".codex/sessions"

CONTINUATION = re.compile(
    r"^(yes|yeah|yep|ok|okay|no|nope|sure|go|go on|keep going|continue|do it|"
    r"and fix|fix|try again|and |also |then |next |what about|how about|why|"
    r"hrm|hmm|wait)\b",
    re.IGNORECASE,
)
# codex injects control markers we never want as prompts
NOISE = re.compile(r"^(<turn_aborted>|<environment|<system|<info-msg>)", re.IGNORECASE)


def _user_text(payload) -> str:
    c = payload.get("content", [])
    if isinstance(c, str):
        return c
    return " ".join(
        x.get("text", "")
        for x in c
        if isinstance(x, dict) and x.get("type") in ("input_text", "text")
    ).strip()


def _tool_label(payload) -> str | None:
    t = payload.get("type")
    if t in ("function_call", "local_shell_call", "custom_tool_call"):
        name = payload.get("name", "")
        if "shell" in str(t) or name in ("shell", "bash"):
            try:
                args = json.loads(payload.get("arguments", "{}"))
                cmd = " ".join(args.get("command", [])) if isinstance(
                    args.get("command"), list
                ) else str(args.get("command", ""))
            except (ValueError, TypeError):
                cmd = ""
            cmd = cmd.lower()
            for v in ("git diff", "git merge", "cargo build", "cargo test",
                      "pytest", "grep", "rg ", "rsync", "ssh", "git log"):
                if v in cmd:
                    return v.strip()
            return (cmd.split() or ["shell"])[0] if cmd else "shell"
        return name or "tool"
    return None


def _trim(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "\u2026"


def extract(include_continuations: bool, add_preamble: bool):
    files = sorted(glob.glob(str(ROOT / "**/rollout-*.jsonl"), recursive=True))
    stats = Counter()
    out_examples: list[str] = []

    for fp in files:
        thread = "session"
        tools = 0
        tool_verbs: list[str] = []
        last_assistant = ""
        turn = 0
        try:
            lines = open(fp, errors="ignore").read().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if o.get("type") == "session_meta":
                thread = _trim(
                    (o.get("payload", {}) or {}).get("thread_name") or "session", 40
                )
                continue
            if o.get("type") != "response_item":
                continue
            p = o.get("payload", {}) or {}
            role = p.get("role")

            if role == "assistant":
                txt = _user_text(p)
                if txt:
                    last_assistant = txt
                continue
            verb = _tool_label(p)
            if verb:
                tools += 1
                tool_verbs.append(verb)
                continue
            if role != "user":
                continue

            txt = _user_text(p)
            if not txt:
                continue
            stats["user_prompts"] += 1
            turn += 1
            flat = re.sub(r"\s+", " ", txt)
            if NOISE.match(flat):
                stats["skip_noise"] += 1
                continue
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
                pre = f"[ctx: {thread} | turn {turn} | tools:{tools} | did: {recent}"
                if last_assistant:
                    pre += f" | re: {_trim(last_assistant, 120)}"
                pre += "] "
                out_examples.append(_trim(pre + flat, 4000))
            else:
                out_examples.append(flat)

    # dedupe on prompt core
    seen_exact, seen_prefix, out = set(), set(), []
    for ex in out_examples:
        core = ex.split("] ", 1)[-1] if ex.startswith("[ctx:") else ex
        if core in seen_exact:
            continue
        seen_exact.add(core)
        pre = re.sub(r"[^a-z0-9]", "", core.lower())[:80]
        if pre in seen_prefix:
            continue
        seen_prefix.add(pre)
        out.append(ex)
    stats["kept"] = len(out)
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/codex-questions-ctx.txt")
    ap.add_argument("--include-continuations", action="store_true")
    ap.add_argument("--no-preamble", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out, stats = extract(args.include_continuations, add_preamble=not args.no_preamble)
    if args.limit:
        out = out[: args.limit]
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out) + "\n")
    print(f"user prompts seen      : {stats['user_prompts']}")
    print(f"  skipped (noise)      : {stats['skip_noise']}")
    print(f"  skipped (length)     : {stats['skip_len']}")
    print(f"  skipped (continuation): {stats['skip_continuation']}")
    print(f"  kept (unique)        : {stats['kept']}")
    print(f"wrote {len(out)} -> {dst}")


if __name__ == "__main__":
    main()
