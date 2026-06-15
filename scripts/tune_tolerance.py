#!/usr/bin/env python3
"""Personalize the router's tolerance to YOUR real prompt mix.

This does not retrain and does not judge answer quality. It reads your actual
session prompts (goose / codex), scores each one ONCE through the loaded router
(prefill -> per-model P(correct)), then sweeps tolerance offline to show the
cost / escalation trade-off on traffic that looks like yours.

The output is a recommended tolerance: the cheapest setting that does not start
sacrificing routing-up on your genuinely hard prompts (the knee of the curve).

Safe by construction: tolerance only slides the selection threshold over the
model's existing predictions. It cannot distort what the model thinks is hard.

Usage:
  python scripts/tune_tolerance.py \
      --checkpoint checkpoints/prefill_router_combined.pt \
      --config configs/combined-pool.yaml \
      [--prompts data/goose-questions-ctx.txt] \
      [--sample 400]

If --prompts is omitted it auto-discovers goose sessions.db and/or
data/*-questions-ctx.txt.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from model_router_toolkit.config import load_config
from model_router_toolkit.prefill.router import PrefillRouter

CTX_RE = re.compile(r"^\[ctx:[^\]]*\]\s*")


def discover_prompts() -> list[str]:
    """Find personal prompts from common local sources."""
    here = Path("data")
    for name in ("goose-questions-ctx.txt", "codex-questions-ctx.txt",
                 "personal-all-ctx.txt"):
        p = here / name
        if p.exists():
            return load_prompt_file(p)
    return []


def load_prompt_file(path: Path) -> list[str]:
    out = []
    for line in open(path):
        line = line.strip()
        if len(line) >= 10:
            out.append(line)
    return out


def select_for_pool(selected: str, cost_by_model: dict[str, float]) -> float:
    return cost_by_model.get(selected, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/prefill_router_combined.pt")
    ap.add_argument("--config", default="configs/combined-pool.yaml")
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--tolerances", default="0.0,0.01,0.02,0.03,0.05,0.08,0.12,0.2")
    args = ap.parse_args()

    cfg = load_config(args.config)
    prompts = load_prompt_file(Path(args.prompts)) if args.prompts else discover_prompts()
    if not prompts:
        print("no personal prompts found (looked in data/*-questions-ctx.txt). "
              "run scripts/extract_goose_questions_ctx.py first.", file=sys.stderr)
        sys.exit(1)

    # dedup on the user text (after any [ctx:] preamble), then sample
    seen, uniq = set(), []
    for p in prompts:
        k = CTX_RE.sub("", p)[:80].lower()
        if k not in seen:
            seen.add(k); uniq.append(p)
    import random
    random.seed(7); random.shuffle(uniq)
    uniq = uniq[: args.sample]

    print(f"loading router: {args.checkpoint}", file=sys.stderr)
    r = PrefillRouter(config=cfg)
    r.load(args.checkpoint)

    # cost per model slot (input cost as the ordering/headline price)
    cost_by_model = {}
    for name in r._model_names:
        spec = cfg.get_model(name)
        cost_by_model[name] = spec.cost_per_m_input_tokens if spec else 0.0
    # tier order = cheapest..priciest for "escalated?" accounting
    tiers = sorted(cost_by_model, key=lambda m: cost_by_model[m])
    top_tier = tiers[-1]
    median_cost = sorted(cost_by_model.values())[len(cost_by_model) // 2]

    print(f"scoring {len(uniq)} of your prompts once...", file=sys.stderr)
    # score each prompt ONCE; cache (model_names, confidences, costs)
    scored = []
    for i, p in enumerate(uniq):
        res = r.route(p, tolerance=0.0)  # any tol; we only want raw scores
        scored.append((res.model_names, res.confidences,
                       {m: c.cost_per_m_input_tokens for m, c in zip(res.model_names, res.costs)}))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(uniq)}", file=sys.stderr)

    def pick(model_names, confs, costs, tol):
        p_max = max(confs)
        thr = p_max - tol
        order = sorted(model_names, key=lambda m: costs[m])
        sel = order[-1]
        for m in order:
            idx = model_names.index(m)
            if confs[idx] >= thr:
                sel = m; break
        return sel

    tols = [float(x) for x in args.tolerances.split(",")]
    print(f"\n{'tol':>5}  {'avg $/M':>8}  {'vs always-top':>13}  {'%->top':>7}  {'%->cheapest':>11}")
    rows = []
    for tol in tols:
        total = 0.0; n_top = 0; n_cheap = 0
        for mn, cf, co in scored:
            sel = pick(mn, cf, co, tol)
            total += co[sel]
            if sel == top_tier: n_top += 1
            if sel == tiers[0]: n_cheap += 1
        avg = total / len(scored)
        always_top = cost_by_model[top_tier]
        saved_pct = (1 - avg / always_top) * 100 if always_top else 0
        rows.append((tol, avg, saved_pct, n_top / len(scored), n_cheap / len(scored)))
        print(f"{tol:>5.2f}  {avg:>8.2f}  {saved_pct:>11.0f}%  "
              f"{100*n_top/len(scored):>6.0f}%  {100*n_cheap/len(scored):>10.0f}%")

    # recommend: the knee — largest tolerance before %->top drops below 60% of
    # its tol=0 value (i.e. before we start starving genuinely-hard prompts).
    top0 = rows[0][3] or 1e-9
    rec = rows[0][0]
    for tol, avg, sp, ptop, pch in rows:
        if ptop >= 0.6 * top0:
            rec = tol
    print(f"\nrecommended tolerance for your traffic: {rec:.2f}", file=sys.stderr)
    print(f"  (keeps escalation on your hard prompts; "
          f"{[r[2] for r in rows if r[0]==rec][0]:.0f}% cheaper than always-top)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
