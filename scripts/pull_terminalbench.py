#!/usr/bin/env python3
"""Pull terminal-bench trajectories: command-line/ops tasks across many models,
each with a verified `reward` (0/1). Closer to real goose/agent work than code-fix.

Output: data_public/terminalbench_matrix.jsonl -> (task_name, model, reward, task_text).
We need task TEXT for the encoder. steps may hold the prompt; fall back to task_name.
"""
import io, json, os, re, urllib.request
import pyarrow.parquet as pq
from collections import defaultdict

FILES = [
    "data/train-00000-of-00002.parquet",
    "data/train-00001-of-00002.parquet",
]
REPO = "yoonholee/terminalbench-trajectories"

def norm_model(m):
    # 'claude-opus-4-6@anthropic' -> 'claude-opus-4-6'
    return (m or "").split("@")[0].strip().lower()

def first_text(steps, task_name):
    """Pull the initial task/instruction text from the steps trajectory."""
    if isinstance(steps, str) and len(steps) > 20:
        return steps[:6000]
    if isinstance(steps, (list, tuple)):
        for s in steps[:4]:
            if isinstance(s, dict):
                for k in ("content", "instruction", "prompt", "task", "text"):
                    v = s.get(k)
                    if isinstance(v, str) and len(v) > 20:
                        return v[:6000]
            elif isinstance(s, str) and len(s) > 20:
                return s[:6000]
    return f"Terminal task: {task_name}"

os.makedirs("data_public", exist_ok=True)
out = open("data_public/terminalbench_matrix.jsonl", "w")
total = 0
task_text = {}
for f in FILES:
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{f}"
    raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "r/1"}), timeout=180).read()
    tbl = pq.read_table(io.BytesIO(raw))
    for r in tbl.to_pylist():
        task = r.get("task_name")
        model = norm_model(r.get("model"))
        reward = r.get("reward")
        if task is None or not model or reward is None:
            continue
        txt = first_text(r.get("steps"), task)
        if task not in task_text:
            task_text[task] = txt
        out.write(json.dumps({
            "task_name": task, "model": model,
            "reward": int(reward) if isinstance(reward, (int, float)) else int(str(reward) in ("1", "True", "true")),
            "task": txt,
            "cost_cents": r.get("cost_cents"),
        }) + "\n")
        total += 1
out.close()

# stats
rows = [json.loads(l) for l in open("data_public/terminalbench_matrix.jsonl")]
by_model = defaultdict(lambda: [0, 0])
tasks = set()
for r in rows:
    by_model[r["model"]][0] += r["reward"]; by_model[r["model"]][1] += 1
    tasks.add(r["task_name"])
print(f"rows: {total} | unique tasks: {len(tasks)} | models: {len(by_model)}")
for m, (c, t) in sorted(by_model.items(), key=lambda x: -x[1][0]/max(x[1][1],1)):
    if t >= 5:
        print(f"  {m:36s} {100*c/t:4.0f}%  ({t})")
