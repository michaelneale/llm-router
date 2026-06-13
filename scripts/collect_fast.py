"""Concurrent, resumable data collection for router training.

Improvements over the built-in serial `model-router collect`:
- async with bounded concurrency (default 8 in-flight calls)
- incremental append to CSV after every question (crash-safe / resumable)
- skips (question, model) pairs already present in the output CSV
- OpenAI judge by default (no OpenRouter dependency)
- per-call timeout + capped max_tokens so reasoning models can't run away
- live cost + progress estimate

Usage:
  .venv/bin/python scripts/collect_fast.py \
    --config configs/goose-mix.yaml \
    --questions data/goose-collect.txt \
    --output data/goose-collected.csv \
    --judge-model openai/gpt-5-mini \
    --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import re
import time
from pathlib import Path

import litellm
import yaml

litellm.drop_params = True  # tolerate unsupported kwargs across providers
litellm.suppress_debug_info = True

class _NullSem:
    """No-op async context manager (concurrency bounded by worker count)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


JUDGE_SYS = (
    "You are an expert answer evaluator. Given a question and a candidate answer, "
    "determine whether the answer is substantively correct and helpful.\n"
    'Respond with ONLY JSON: {"correct": true} or {"correct": false}'
)


def parse_verdict(text: str) -> int:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).lower()
    if re.search(r'correct["\':\s]*true', t):
        return 1
    if re.search(r'correct["\':\s]*false', t):
        return 0
    return 1 if '"correct": true' in t or "true" in t[:40] else 0


async def call_model(sem, model, q, system_prompt, kwargs, max_tokens, timeout):
    async with sem:
        try:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": q})
            r = await asyncio.wait_for(
                litellm.acompletion(
                    model=model, messages=msgs, max_tokens=max_tokens, **kwargs
                ),
                timeout=timeout,
            )
            content = r.choices[0].message.content or ""
            toks = getattr(r.usage, "completion_tokens", 0) or 0
            return content, toks, None
        except Exception as e:  # noqa: BLE001
            return "", 0, str(e)[:120]


async def judge(sem, judge_model, q, answer, timeout):
    if not answer.strip():
        return 0
    async with sem:
        try:
            r = await asyncio.wait_for(
                litellm.acompletion(
                    model=judge_model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYS},
                        {"role": "user", "content": f"Question: {q}\n\nAnswer: {answer}"},
                    ],
                    temperature=0.0,
                    max_tokens=2000,
                ),
                timeout=timeout,
            )
            return parse_verdict(r.choices[0].message.content or "")
        except Exception:  # noqa: BLE001
            return 0


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--judge-model", default="openai/gpt-5-mini")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    pool = yaml.safe_load(open(args.config))
    models = pool["models"]
    cost = {m["name"]: (m["cost_per_m_input_tokens"], m["cost_per_m_output_tokens"]) for m in models}

    questions = [l.strip() for l in open(args.questions) if l.strip()]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str]] = set()
    if out.exists():
        with open(out) as f:
            for row in csv.DictReader(f):
                done.add((row["question"], row["model"]))
        print(f"Resuming: {len(done)} (question,model) pairs already collected")
    else:
        with open(out, "w", newline="") as f:
            csv.writer(f).writerow(["question", "model", "isCorrect", "output_tokens"])

    # No outer semaphore: concurrency is bounded by the fixed number of workers,
    # each of which fully processes one pair (call -> judge -> write) before
    # pulling the next. This avoids the mass-gather connection-pool deadlock.
    nosem = _NullSem()
    total_pairs = len(questions) * len(models)
    fp = open(out, "a", newline="")
    writer = csv.writer(fp)
    write_lock = asyncio.Lock()
    t0 = time.time()
    state = {"completed": len(done), "cost": 0.0}

    queue: asyncio.Queue = asyncio.Queue()
    for q in questions:
        for m in models:
            if (q, m["name"]) not in done:
                queue.put_nowait((q, m))
    outstanding = queue.qsize()

    async def worker(wid: int):
        while True:
            try:
                q, m = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            content, toks, err = await call_model(
                nosem, m["litellm_model"], q, m.get("system_prompt", ""),
                m.get("chat_template_kwargs", {}) or {}, args.max_tokens, args.timeout,
            )
            v = await judge(nosem, args.judge_model, q, content, args.timeout)
            cin, cout = cost[m["name"]]
            async with write_lock:
                writer.writerow([q, m["name"], v, toks])
                fp.flush()
                state["completed"] += 1
                state["cost"] += (len(q) / 4 / 1e6 * cin) + (toks / 1e6 * cout)
                c = state["completed"]
                if err:
                    print(f"  ! {m['name'][:18]}: {err}", flush=True)
                if c % 25 == 0 or c == total_pairs:
                    rate = (c - len(done)) / max(time.time() - t0, 1)
                    eta = (total_pairs - c) / max(rate, 0.01) / 60
                    print(
                        f"pairs {c}/{total_pairs}  ~${state['cost']:.2f}  "
                        f"{rate:.1f} pair/s  ETA {eta:.0f}m",
                        flush=True,
                    )
            queue.task_done()

    print(f"Dispatching {outstanding} pairs across {args.concurrency} workers", flush=True)
    await asyncio.gather(*(worker(i) for i in range(args.concurrency)))

    fp.close()
    print(f"\nDONE. {state['completed']}/{total_pairs} pairs. Model cost ~${state['cost']:.2f}.")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY and ANTHROPIC_API_KEY")
    asyncio.run(main())
