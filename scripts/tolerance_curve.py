import csv, numpy as np, torch
from collections import defaultdict, Counter
from model_router_toolkit.config import load_config
from model_router_toolkit.evaluate import _build_shared_features
from model_router_toolkit.prefill.extract import extract_from_checkpoint
from model_router_toolkit.prefill.trunk import reconstruct_trunk, predict_proba

cfg = load_config("configs/goose-mix.yaml")
cost_out = {m.name: m.cost_per_m_output_tokens for m in cfg.models}
disp = {m.name: m.display_name for m in cfg.models}
ckpt = torch.load("checkpoints/prefill_router_goose.pt", map_location="cpu", weights_only=False)
names = ckpt["model_names"]

truth = defaultdict(dict); med_tok = defaultdict(list)
for r in csv.DictReader(open("data/goose-test.csv")):
    truth[r["question"]][r["model"]] = int(r["isCorrect"]); med_tok[r["model"]].append(int(r["output_tokens"]))
qs = list(truth); mt = {m: float(np.median(v)) for m, v in med_tok.items()}

pr = extract_from_checkpoint(ckpt, qs)
feats = _build_shared_features(ckpt, pr, names)
trunks = reconstruct_trunk(ckpt, device="cpu")
probs = predict_proba(trunks, feats, device="cpu")

Y = np.array([[truth[q].get(n, 0) for n in names] for q in qs])
costs = np.array([mt[n] / 1e6 * cost_out[n] * 1000 for n in names])
cheap_idx = [i for i, n in enumerate(names) if n in
             {"nemotron-3-nano-reasoning", "gpt-oss-20b-high", "nemotron-3-super", "qwen-3-5-35b"}]
base = costs[names.index("gpt-5-2-high")]

lines = [f"{'tol':>5} {'acc':>6} {'$/1k-q':>8} {'sav':>5} {'top chosen':22} {'%cheap':>7}"]
for tol in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    sel = np.array([
        min([j for j in range(len(names)) if probs[i][j] >= probs[i].max() - tol],
            key=lambda j: costs[j])
        for i in range(len(qs))
    ])
    acc = Y[np.arange(len(qs)), sel].mean()
    c = costs[sel].mean()
    top = disp[names[Counter(sel).most_common(1)[0][0]]]
    cheap = np.isin(sel, cheap_idx).mean()
    lines.append(f"{tol:5.2f} {acc:6.1%} ${c:7.3f} {(1 - c / base) * 100:4.0f}% {top[:22]:22} {cheap:6.0%}")
lines.append("")
lines.append(f"baseline gpt-5-2:  acc={Y[:, names.index('gpt-5-2-high')].mean():.1%} ${base:.3f}/1k-q")
oc = costs[names.index('claude-opus-4-6-high')]
lines.append(f"baseline opus-4.8: acc={Y[:, names.index('claude-opus-4-6-high')].mean():.1%} ${oc:.3f}/1k-q")
open("/tmp/tolcurve.txt", "w").write("\n".join(lines) + "\n")
print("WROTE /tmp/tolcurve.txt")
