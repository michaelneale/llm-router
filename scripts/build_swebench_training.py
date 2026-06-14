#!/usr/bin/env python3
"""Build training CSV from the SWE-bench verified multi-model matrix.

Output: data/swebench-train.csv with columns question,model,isCorrect,output_tokens
where model = our pool slot names, isCorrect = verified `resolved`.
Optionally mixes in local goose traces (plausibility-judged) for personalization.
"""
import json, csv, os
from collections import defaultdict

# map tarsur385 model -> our pool checkpoint slot name (must match config `name`)
MAP = {
    "gpt-5-mini":            "nemotron-3-nano-reasoning",   # cheap slot
    "claude-4.5-haiku-high": "gpt-oss-20b-high",            # cheap slot
    "gemini-3-flash-high":   "gpt-4-1-nano-high",           # cheap slot
    "glm-5-high":            "gpt-oss-120b-high",           # mid slot
    "minimax-m2.5-high":     "gpt-4-1-mini-high",           # mid slot
    "claude-4.5-opus-high":  "claude-sonnet-4-6-high",      # strong slot
    "gpt-5.2-high":          "claude-haiku-4-5-high",       # strong slot
    "claude-opus-4.6":       "claude-opus-4-6-high",        # top slot
}

rows = [json.loads(l) for l in open("data_public/swebench_matrix.jsonl")]
# task text per instance
task_text = {}
for r in rows:
    if r["instance_id"] not in task_text and r.get("task"):
        task_text[r["instance_id"]] = r["task"]

# build per (task, model) isCorrect
out_rows = []
seen = set()
for r in rows:
    slot = MAP.get(r["model"])
    if not slot:
        continue
    q = task_text.get(r["instance_id"], "")
    if not q:
        continue
    key = (r["instance_id"], slot)
    if key in seen:
        continue
    seen.add(key)
    out_rows.append({
        "question": q[:4000],
        "model": slot,
        "isCorrect": int(bool(r["resolved"])),
        "output_tokens": r.get("n_msgs", 0) * 100,
    })

os.makedirs("llm-router/data", exist_ok=True)
path = "llm-router/data/swebench-train.csv"
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["question","model","isCorrect","output_tokens"])
    w.writeheader(); w.writerows(out_rows)

# stats
by_model = defaultdict(lambda: [0,0])
for r in out_rows:
    by_model[r["model"]][0] += r["isCorrect"]; by_model[r["model"]][1] += 1
print(f"wrote {len(out_rows)} rows -> {path}")
print(f"unique tasks: {len(task_text)} | models(slots): {len(by_model)}")
for m,(c,t) in sorted(by_model.items()):
    print(f"  {m:28s} {100*c/t:4.0f}% correct  ({t} rows)")
