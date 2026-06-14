#!/usr/bin/env python3
"""Build the full-spectrum training set:
  RouterBench (easy->hard general Q&A, per-model correctness)
  + SWE-bench (coding) + terminal-bench (ops).
Maps each source's models onto our 8 pool slots by capability tier.
"""
import json, csv, os, random
import pandas as pd
from collections import defaultdict

SLOTS = ["nemotron-3-nano-reasoning","gpt-oss-20b-high","gpt-4-1-nano-high",
         "gpt-oss-120b-high","gpt-4-1-mini-high","claude-sonnet-4-6-high",
         "claude-haiku-4-5-high","claude-opus-4-6-high"]

out_rows=[]; seen=set()
def emit(tid, slot, correct, text):
    k=(tid,slot)
    if k in seen or not text or len(text)<10: return
    seen.add(k); out_rows.append({"question":text[:4000],"model":slot,"isCorrect":int(bool(correct)),"output_tokens":0})

# --- RouterBench: pick 8 models spanning weak->strong, map to slots ---
df=pd.read_pickle('/tmp/rb.pkl')
rb_models=['mistralai/mistral-7b-chat','WizardLM/WizardLM-13B-V1.2','mistralai/mixtral-8x7b-chat',
           'gpt-3.5-turbo-1106','claude-instant-v1','claude-v2','claude-v1','gpt-4-1106-preview']
rb_models=[m for m in rb_models if m in df.columns]
# rank by accuracy, evenly assign to slots
rb_models.sort(key=lambda m: df[m].mean())
idx=[round(k*(len(rb_models)-1)/7) for k in range(8)]
RB_MAP={rb_models[i]:SLOTS[k] for k,i in enumerate(idx)}
print("RouterBench model -> slot:")
for m,s in RB_MAP.items(): print(f"  {m:30s} {df[m].mean()*100:4.0f}% -> {s}")
# subsample (36k is a lot) - keep a balanced slice across eval categories
df_s=df.groupby('eval_name', group_keys=False).apply(lambda g: g.sample(min(len(g),120), random_state=1))
print(f"routerbench sampled: {len(df_s)} prompts")
for _,row in df_s.iterrows():
    p=row['prompt']
    if not isinstance(p,str) or len(p)<10: continue
    for m,slot in RB_MAP.items():
        sc=row[m]
        if pd.notna(sc): emit("rb::"+str(row['sample_id']), slot, sc>0.5, p)

# --- SWE-bench ---
swe=[json.loads(l) for l in open("data_public/swebench_matrix.jsonl")]
SWE_MAP={"gpt-5-mini":"nemotron-3-nano-reasoning","claude-4.5-haiku-high":"gpt-oss-20b-high",
 "gemini-3-flash-high":"gpt-4-1-nano-high","glm-5-high":"gpt-oss-120b-high",
 "minimax-m2.5-high":"gpt-4-1-mini-high","claude-4.5-opus-high":"claude-sonnet-4-6-high",
 "gpt-5.2-high":"claude-haiku-4-5-high","claude-opus-4.6":"claude-opus-4-6-high"}
swe_txt={r["instance_id"]:r.get("task","") for r in swe if r.get("task")}
for r in swe:
    s=SWE_MAP.get(r["model"])
    if s: emit("swe::"+r["instance_id"], s, r["resolved"], swe_txt.get(r["instance_id"],""))

# --- terminal-bench ---
tb=[json.loads(l) for l in open("data_public/terminalbench_matrix_text.jsonl")]
tb_rate=defaultdict(lambda:[0,0])
for r in tb: tb_rate[r["model"]][0]+=r["reward"]; tb_rate[r["model"]][1]+=1
elig=sorted([(m,c/t) for m,(c,t) in tb_rate.items() if t>=30], key=lambda x:x[1])
idx=[round(k*(len(elig)-1)/7) for k in range(8)]
TB_MAP={elig[i][0]:SLOTS[k] for k,i in enumerate(idx)}
for r in tb:
    s=TB_MAP.get(r["model"])
    if s: emit("tb::"+r["task_name"], s, r["reward"], r.get("task",""))

os.makedirs("llm-router/data",exist_ok=True)
path="llm-router/data/full-train.csv"
with open(path,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["question","model","isCorrect","output_tokens"]);w.writeheader();w.writerows(out_rows)
bm=defaultdict(lambda:[0,0])
for r in out_rows: bm[r["model"]][0]+=r["isCorrect"]; bm[r["model"]][1]+=1
print(f"\nwrote {len(out_rows)} rows ({len(set(r['question'] for r in out_rows))} unique tasks) -> {path}")
for m in SLOTS:
    c,t=bm[m]; print(f"  {m:28s} {100*c/max(t,1):4.0f}%  ({t})")
