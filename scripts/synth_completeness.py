#!/usr/bin/env python3
"""Synthesize a balanced completeness-labeled dataset.

The real completeness failures in session traces are strong but rare (~26 vs
5000). Too imbalanced to train on. This generates task prompts across the task
classes that the trace analysis showed drive bails (monitor/poll/track at 13x,
CI/build/test, check/verify, infra ops) plus easy control tasks, so a router can
learn "this class of task needs a model that follows through."

Two phases:
  generate  -> LLM expands seed templates into varied, realistic task prompts.
  (labeling happens via `model-router collect` on the output, scored for
   COMPLETION using --judge llm with the completeness rubric in JUDGE_RUBRIC.)

Output: a questions .txt (one prompt/line) ready for collect, tagged by class in
a sidecar .meta.csv so you can analyze per-class completion later.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

# Task classes, with the bail-lift measured from real traces.
SEED_TEMPLATES = {
    # 13.3x bail lift — the dominant completeness failure class.
    "monitor_poll": [
        "Monitor the CI run for PR #{n} on {repo}, poll every 60s until all checks finish, then report pass/fail.",
        "Watch the deploy of {svc} and keep checking until it's healthy or 10 minutes pass; report the outcome.",
        "Keep polling the GitHub Actions for branch {branch} until they complete, then summarize failures.",
        "Tail the logs of {svc} until you see 'ready', then confirm the endpoint responds.",
        "Run the test suite and keep re-running the flaky ones until they pass or you've tried 5 times.",
    ],
    # 1.7x — verification work that's easy to declare done prematurely.
    "ci_verify": [
        "Check whether the build of {repo} actually passes — run it, don't assume.",
        "Verify the {svc} endpoint returns the expected models list; actually call it.",
        "Confirm the migration applied cleanly by querying the table, not by reading the code.",
        "Make sure the new flag works end to end — exercise it, then report.",
    ],
    "infra_ops": [
        "Kill {svc} on the {node}, restart it as a client, and confirm it joined the mesh.",
        "Bring up a 2-node setup and verify peers=1 on both before declaring it working.",
        "Restart {svc}, then validate it's serving by hitting /v1/models.",
    ],
    # easy controls — should NOT need a follow-through model.
    "easy_control": [
        "What does `git rebase --onto` do?",
        "Rename the variable `foo` to `bar` in this file.",
        "What's the difference between a tokio and std Mutex?",
        "Add a docstring to this function.",
        "Format this JSON.",
    ],
}

FILL = {
    "n": ["610", "696", "712", "843"],
    "repo": ["mesh-llm", "aaif-goose/goose", "the workspace"],
    "svc": ["mesh-llm", "the relay", "the host runtime", "the proxy"],
    "branch": ["micn/launch-fix", "main", "micn/new-chat-shortcut"],
    "node": ["mini", "studio", "M4"],
}

JUDGE_RUBRIC = (
    "Score COMPLETENESS, not correctness. Did the assistant actually FOLLOW "
    "THROUGH on the task to a verifiable end (ran the loop, polled until done, "
    "actually executed and checked), or did it stop early / write a script and "
    "bail / say 'you can run this yourself' / declare success without verifying? "
    "Answer 1 only if it followed through, else 0."
)


def expand(template: str) -> str:
    def repl(m: str) -> str:
        import random

        return random.choice(FILL[m]) if m in FILL else m

    return re.sub(r"\{(\w+)\}", lambda mm: repl(mm.group(1)), template)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-template", type=int, default=4)
    ap.add_argument("--output", default="data/synth-completeness.txt")
    ap.add_argument("--meta", default="data/synth-completeness.meta.csv")
    ap.add_argument("--print-rubric", action="store_true")
    args = ap.parse_args()

    if args.print_rubric:
        print(JUDGE_RUBRIC)
        return

    seen: set[str] = set()
    rows: list[tuple[str, str]] = []
    for cls, templates in SEED_TEMPLATES.items():
        for t in templates:
            for _ in range(args.per_template):
                q = expand(t)
                if q not in seen:
                    seen.add(q)
                    rows.append((q, cls))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(q for q, _ in rows) + "\n")
    with open(args.meta, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "class"])
        w.writerows(rows)

    from collections import Counter

    by = Counter(c for _, c in rows)
    print(f"wrote {len(rows)} synthetic task prompts -> {args.output}")
    for c, n in by.items():
        print(f"  {c:14s}: {n}")
    print(f"\nNext: label by completion (NOT correctness):")
    print(f"  model-router collect --config configs/combined-pool.yaml \\")
    print(f"    --questions {args.output} --output data/synth-completeness-labels.csv \\")
    print(f"    --judge llm   # then score per class via the meta file")
    print(f"\nJudge rubric to use (see --print-rubric):\n  {JUDGE_RUBRIC[:90]}...")


if __name__ == "__main__":
    main()
