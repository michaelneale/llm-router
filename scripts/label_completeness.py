#!/usr/bin/env python3
"""Label prompts by AGENTIC COMPLETENESS, not single-turn correctness.

Why: the single-turn pass/fail labeling used by `model-router collect` is blind
to whether a model *sustains a task to done*. The real weakness we measured is
that cheap models bail on long/agentic work (write a script and quit, "I
recommend you run...", stop early) while a frontier model follows through.

Signal: real session traces contain human-labeled completeness failures — the
user telling the agent it bailed ("you didn't do anything", "don't stop", "you
failed to continue", "keep going you lazy prick"). The user prompt *immediately
before* such a correction is a task the model FAILED to complete (label 0). User
prompts that were accepted and the session moved on are completed (label 1).

Output CSV: question,completeness,session_id,signal
This is a real outcome label from human feedback, not a synthetic judge.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path

DEFAULT_DB = Path.home() / ".local/share/goose/sessions/sessions.db"

# Human telling the agent it did not finish / bailed / faked it.
FAILURE = re.compile(
    r"\b(you didn'?t (do|actually|even|check|run|sleep|continue|finish)"
    r"|you failed to continue|failed to continue|why did you stop|don'?t stop"
    r"|you are lazy|lazy prick|you just made (something|it) up|you forgot"
    r"|keep going|you were supposed|you didn'?t do anything|you useless"
    r"|you didn'?t check)\b",
    re.I,
)
# Neutral resumes ("sorry continue") are NOT failures — exclude.
NEUTRAL = re.compile(r"^\s*(sorry,?\s*(please\s*)?continue|continue on|carry on|please continue)\b", re.I)
# Compaction artifacts to skip.
ARTIFACT = re.compile(r"^\s*(<analysis>|<info-msg>|##\s)")


def clean(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--output", default="data/completeness-labels.csv")
    ap.add_argument("--min-len", type=int, default=15)
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    # ordered user turns per session
    rows = con.execute(
        """
        SELECT m.session_id, m.created_timestamp,
               json_extract(value,'$.text') AS txt,
               json_extract(value,'$.metadata.userVisible') AS uv
        FROM messages m, json_each(m.content_json)
        WHERE m.role='user' AND json_extract(value,'$.type')='text'
        ORDER BY m.session_id, m.created_timestamp
        """
    ).fetchall()

    # group user prompts by session in order
    by_sess: dict[str, list[str]] = {}
    for sid, ts, txt, uv in rows:
        if not txt:
            continue
        by_sess.setdefault(sid, []).append(txt)

    examples: list[tuple[str, int, str, str]] = []
    for sid, prompts in by_sess.items():
        for i, p in enumerate(prompts):
            if FAILURE.search(p) and not NEUTRAL.match(p):
                # the PREVIOUS real user prompt is the task that was not completed
                for j in range(i - 1, -1, -1):
                    prev = prompts[j]
                    if ARTIFACT.match(prev) or NEUTRAL.match(prev):
                        continue
                    if len(clean(prev)) >= args.min_len:
                        examples.append((clean(prev), 0, sid, "user_said_bailed"))
                        break

    # positive examples: user prompts that did NOT trigger a failure correction
    # and are followed by continued normal work (proxy for "completed").
    failed_prompts = {e[0] for e in examples}
    for sid, prompts in by_sess.items():
        for i, p in enumerate(prompts[:-1]):  # has a following turn
            cp = clean(p)
            nxt = prompts[i + 1]
            if (
                len(cp) >= args.min_len
                and not ARTIFACT.match(p)
                and cp not in failed_prompts
                and not FAILURE.search(nxt)  # next turn was not a complaint
                and not NEUTRAL.match(nxt)
            ):
                examples.append((cp, 1, sid, "session_progressed"))

    # dedupe on prompt, prefer the failure label (0) when conflicting
    best: dict[str, tuple[str, int, str, str]] = {}
    for ex in examples:
        q = ex[0]
        if q not in best or ex[1] < best[q][1]:
            best[q] = ex
    out = list(best.values())

    n0 = sum(1 for e in out if e[1] == 0)
    n1 = sum(1 for e in out if e[1] == 1)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "completeness", "session_id", "signal"])
        w.writerows(out)

    print(f"wrote {len(out)} labeled examples -> {args.output}")
    print(f"  completeness=0 (model bailed): {n0}")
    print(f"  completeness=1 (completed)    : {n1}")
    print(f"  from {len(by_sess)} sessions")


if __name__ == "__main__":
    main()
