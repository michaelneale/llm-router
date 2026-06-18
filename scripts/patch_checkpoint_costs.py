"""Re-price an existing checkpoint's pool_config from a pool YAML.

The router ranks models by costs stored *inside* the checkpoint, not the
YAML. This script copies the bundled qwen08b checkpoint and overwrites its
pool_config costs with the values from a given config.

Usage: .venv/bin/python scripts/patch_checkpoint_costs.py \
           --src checkpoints/prefill_router_qwen08b.pt \
           --config configs/combined-pool.yaml \
           --out checkpoints/prefill_router_repriced.pt
"""

from __future__ import annotations

import argparse

import torch
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        pool = yaml.safe_load(f)
    prices = {
        m["name"]: (
            float(m["cost_per_m_input_tokens"]),
            float(m["cost_per_m_output_tokens"]),
        )
        for m in pool["models"]
    }

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    assert set(ckpt["model_names"]) == set(prices), (
        f"slot mismatch: checkpoint={ckpt['model_names']} config={list(prices)}"
    )

    pool_cfg = ckpt.get("pool_config", {})
    targets = pool_cfg.get("targets", pool_cfg) if isinstance(pool_cfg, dict) else pool_cfg
    n = 0
    for entry in targets:
        if isinstance(entry, dict) and entry.get("name") in prices:
            cin, cout = prices[entry["name"]]
            old = entry.get("cost_per_m_input_tokens")
            entry["cost_per_m_input_tokens"] = cin
            entry["cost_per_m_output_tokens"] = cout
            print(f"  {entry['name']:28s} ${old} -> ${cin}/M in, ${cout}/M out")
            n += 1

    assert n == len(prices), f"only patched {n}/{len(prices)} models"
    torch.save(ckpt, args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
