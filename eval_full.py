#!/usr/bin/env python3
"""Evaluate the combined router on held-out tasks: route each task, look up the
chosen model's verified outcome, compare to fixed-model and oracle baselines."""
import csv, sys, os
from collections import defaultdict
sys.path.insert(0, "src")
os.environ.setdefault("ROUTER_DEVICE", "mps")

from model_router_toolkit.config import load_config, build_router_from_config

cfg = load_config("configs/combined-pool.yaml")
# point at combined checkpoint
cfg.routing.checkpoint = "checkpoints/full/prefill_router.pt"
router = build_router_from_config(cfg)

# held-out: task -> {slot: isCorrect}
rows = list(csv.DictReader(open("data/full-test-split.csv")))
tasks = defaultdict(dict)
text = {}
for r in rows:
    tasks[r["question"]][r["model"]] = int(r["isCorrect"])
    text[r["question"]] = r["question"]

# cost per slot (output $/M) from config
cost = {m.name: m.cost_per_m_output_tokens for m in cfg.models}
slots = [m.name for m in cfg.models]

def eval_strategy(pick):
    soln = c = 0.0
    for q, outcomes in tasks.items():
        m = pick(q, outcomes)
        if m in outcomes:
            soln += outcomes[m]; c += cost.get(m, 0)
    n = len(tasks)
    return 100*soln/n, c/n

print(f"held-out tasks: {len(tasks)}\n")
print(f"{'strategy':28s} {'resolved%':>9s} {'$/M':>6s}")
for tol in [0.0, 0.02, 0.05, 0.10]:
    def pick(q, oc, tol=tol):
        res = router.route(text[q], tolerance=tol)
        return res.selected_model
    a, c = eval_strategy(pick)
    print(f"router @ tol={tol:<4}            {a:8.1f} {c:6.1f}")
# fixed baselines
for slot in [slots[0], slots[3], slots[5], slots[-1]]:
    a, c = eval_strategy(lambda q, oc, s=slot: s)
    print(f"always {slot:21s} {a:8.1f} {c:6.1f}")
# oracle
a, _ = eval_strategy(lambda q, oc: max(oc, key=lambda m: oc[m]) if oc else slots[0])
print(f"{'oracle (any model solves)':28s} {a:8.1f}")
