#!/usr/bin/env python3
"""Robust pull with on-disk cache + retries/backoff (HF rate-limits bursts).
Resumable: skips tasks already cached. Builds data_public/swebench_matrix.jsonl.
"""
import json, os, time, urllib.request, urllib.error, concurrent.futures as cf

REPO = "tarsur385/swebench-verified-trajectories"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/swebench_verified_raw"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/swebench_verified_raw"
MODELS = ["gpt-5-mini","claude-4.5-haiku-high","gemini-3-flash-high","glm-5-high",
          "minimax-m2.5-high","claude-4.5-opus-high","gpt-5.2-high","claude-opus-4.6"]
N = int(os.environ.get("N_TASKS","300"))
CACHE = "data_public/cache"; os.makedirs(CACHE, exist_ok=True)

def get(url, timeout=90, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"router/1.0"})
            return urllib.request.urlopen(req, timeout=timeout).read()
        except urllib.error.HTTPError as e:
            if e.code in (429,503,500): time.sleep(2*(i+1)); continue
            raise
        except Exception:
            time.sleep(1.5*(i+1))
    return None

def list_tasks(model):
    r = get(f"{API}/{model}", 40)
    if not r: return []
    return [x["path"].split("/")[-1] for x in json.loads(r) if x.get("type")=="directory"]

def fetch(model, task):
    cp = f"{CACHE}/{model}__{task}.json"
    if os.path.exists(cp) and os.path.getsize(cp) > 50:
        return json.load(open(cp))
    raw = get(f"{BASE}/{model}/{task}/{task}.traj.json", 90)
    if not raw: return None
    try: d = json.loads(raw)
    except: return None
    info = d.get("info",{})
    resolved = info.get("resolved")
    if resolved is None: resolved = (info.get("scores") or {}).get("resolved")
    msgs = d.get("messages",[])
    rec = {"instance_id":task,"model":model,"resolved":bool(resolved),
           "task":str(next((m.get("content") for m in msgs if m.get("role")=="user"),""))[:6000],
           "n_msgs":len(msgs)}
    json.dump(rec, open(cp,"w"))
    return rec

tasks0 = list_tasks(MODELS[0])[:N]
print(f"tasks: {len(tasks0)} x {len(MODELS)} models")
out = open("data_public/swebench_matrix.jsonl","w"); total=0
for model in MODELS:
    avail = set(list_tasks(model))
    todo = [t for t in tasks0 if t in avail]
    rows=[]
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for r in ex.map(lambda t: fetch(model,t), todo):
            if r: rows.append(r)
    for r in rows: out.write(json.dumps(r)+"\n"); total+=1
    res=sum(r["resolved"] for r in rows)
    print(f"  {model:24s} {len(rows):3d}  resolved={res} ({100*res//max(len(rows),1)}%)")
out.close()
print(f"TOTAL {total} -> data_public/swebench_matrix.jsonl")
