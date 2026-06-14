#!/usr/bin/env python3
"""Join terminal-bench trajectory rewards (yoonholee) with real task instructions
(harborframework/terminal-bench-2.0), producing a (task, model, reward, text) matrix.
"""
import json, os, urllib.request, concurrent.futures as cf
from collections import defaultdict

INSTR = "https://huggingface.co/datasets/harborframework/terminal-bench-2.0/resolve/main/{task}/instruction.md"
os.makedirs("data_public/tb_instr", exist_ok=True)

rows = [json.loads(l) for l in open("data_public/terminalbench_matrix.jsonl")]
tasks = sorted({r["task_name"] for r in rows})
print(f"tasks needing instructions: {len(tasks)}")

def get_instr(task):
    cp = f"data_public/tb_instr/{task}.md"
    if os.path.exists(cp) and os.path.getsize(cp) > 20:
        return task, open(cp).read()
    try:
        url = INSTR.format(task=task)
        raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "r/1"}), timeout=40).read().decode("utf-8", "ignore")
        if len(raw) > 20:
            open(cp, "w").write(raw)
            return task, raw
    except Exception:
        pass
    return task, None

instr = {}
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for task, txt in ex.map(get_instr, tasks):
        if txt: instr[task] = txt
print(f"got instructions for {len(instr)}/{len(tasks)} tasks")

# rewrite matrix with real text
out = open("data_public/terminalbench_matrix_text.jsonl", "w")
n = 0
for r in rows:
    txt = instr.get(r["task_name"])
    if not txt: continue
    r["task"] = txt[:6000]
    out.write(json.dumps(r) + "\n"); n += 1
out.close()
print(f"wrote {n} rows with real task text -> data_public/terminalbench_matrix_text.jsonl")
