"""Head-to-head: router proxy vs. raw frontier model on real goose-repo tasks.

For each task we send the SAME prompt to:
  A) the router proxy  (http://localhost:4000, model alias 'nvidia-routed')
  B) raw Opus directly (anthropic/claude-opus-4-8) via litellm

and record, per side: which model answered (router picks; Opus is fixed),
token usage, estimated cost, latency, and the full answer text (for human
side-by-side quality judging). Writes a JSON report + a readable markdown diff.

Cost is computed from the pool's per-model rates in combined-pool.yaml so the
router's actual-model cost is comparable to Opus's.

Usage:
  python scripts/headtohead.py                 # built-in goose-repo tasks
  python scripts/headtohead.py --repo ~/Development/goose
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml

ROUTER_BASE = os.environ.get("ROUTER_BASE", "http://localhost:4000/v1")
ROUTER_ALIAS = os.environ.get("ROUTER_ALIAS", "nvidia-routed")
OPUS_MODEL = "anthropic/claude-opus-4-8"


def load_rates(pool_yaml: str) -> dict[str, tuple[float, float]]:
    cfg = yaml.safe_load(open(pool_yaml))
    rates = {}
    for m in cfg["models"]:
        rates[m["litellm_model"]] = (
            m["cost_per_m_input_tokens"],
            m["cost_per_m_output_tokens"],
        )
        rates[m["name"]] = rates[m["litellm_model"]]
        rates[m.get("display_name", m["name"])] = rates[m["litellm_model"]]
    return rates


def cost_of(model: str, pin: int, pout: int, rates) -> float:
    ci, co = rates.get(model, (5.0, 25.0))  # default to opus-ish if unknown
    return pin / 1e6 * ci + pout / 1e6 * co


def call_router(prompt: str) -> dict:
    import litellm

    t0 = time.time()
    r = litellm.completion(
        model=f"openai/{ROUTER_ALIAS}",
        api_base=ROUTER_BASE,
        api_key="sk-not-needed",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        timeout=180,
    )
    dt = time.time() - t0
    # the proxy patches response.model to the actually-routed deployment
    routed = r.model
    u = r.usage
    return {
        "answer": r.choices[0].message.content or "",
        "routed_model": routed,
        "prompt_tokens": getattr(u, "prompt_tokens", 0),
        "completion_tokens": getattr(u, "completion_tokens", 0),
        "latency_s": round(dt, 1),
    }


def call_opus(prompt: str) -> dict:
    import litellm

    t0 = time.time()
    r = litellm.completion(
        model=OPUS_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        timeout=180,
    )
    dt = time.time() - t0
    u = r.usage
    return {
        "answer": r.choices[0].message.content or "",
        "routed_model": OPUS_MODEL,
        "prompt_tokens": getattr(u, "prompt_tokens", 0),
        "completion_tokens": getattr(u, "completion_tokens", 0),
        "latency_s": round(dt, 1),
    }


def build_tasks(repo: Path) -> list[dict]:
    """Real, verifiable goose-repo tasks. Each embeds a code excerpt so the
    answer can be checked against ground truth rather than judged on vibes."""
    server = repo / "crates/goose/src/acp/server.rs"
    excerpt = ""
    if server.exists():
        lines = server.read_text(errors="ignore").splitlines()
        # grab the steering region for an anchored question
        start = next((i for i, l in enumerate(lines) if "discard_pending_steers" in l), 0)
        excerpt = "\n".join(lines[max(0, start - 5): start + 25])

    return [
        {
            "id": "explain-steering",
            "verifiable": "Mentions pending_steers queue, session_id keying, discard on some lifecycle event.",
            "prompt": (
                "This is from goose's ACP server (Rust). Explain precisely what "
                "the `pending_steers` mechanism does, when steers get discarded, "
                "and why it's keyed by session_id. Be concise and specific.\n\n"
                f"```rust\n{excerpt}\n```"
            ),
        },
        {
            "id": "rust-concurrency",
            "verifiable": "Should identify Arc<Mutex>/RwLock or channel patterns; correct Rust async reasoning.",
            "prompt": (
                "In an async Rust service (tokio) you have a shared "
                "`HashMap<SessionId, Vec<Steer>>` mutated from multiple request "
                "handlers and drained by a background task. Describe the safest "
                "concurrency primitive to use, the deadlock risk to avoid, and "
                "write a minimal correct skeleton. Be specific to tokio."
            ),
        },
        {
            "id": "trivial-git",
            "verifiable": "Trivial — any model should nail it. Tests router routes this cheap.",
            "prompt": "What does `git rebase --onto main feature~3 feature` do? One paragraph.",
        },
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path.home() / "Development/goose"))
    ap.add_argument("--pool", default="configs/combined-pool.yaml")
    ap.add_argument("--out", default="data/headtohead.json")
    args = ap.parse_args()

    rates = load_rates(args.pool)
    tasks = build_tasks(Path(args.repo))
    results = []

    for t in tasks:
        print(f"\n=== {t['id']} ===")
        print("  -> router ...", flush=True)
        rt = call_router(t["prompt"])
        print(f"     routed to {rt['routed_model']} "
              f"({rt['completion_tokens']} tok, {rt['latency_s']}s)")
        print("  -> opus   ...", flush=True)
        op = call_opus(t["prompt"])
        print(f"     opus ({op['completion_tokens']} tok, {op['latency_s']}s)")

        rt["cost_usd"] = round(cost_of(rt["routed_model"], rt["prompt_tokens"],
                                       rt["completion_tokens"], rates), 6)
        op["cost_usd"] = round(cost_of(OPUS_MODEL, op["prompt_tokens"],
                                       op["completion_tokens"], rates), 6)
        results.append({"task": t, "router": rt, "opus": op})

    # report
    tot_r = sum(x["router"]["cost_usd"] for x in results)
    tot_o = sum(x["opus"]["cost_usd"] for x in results)
    report = {
        "tasks": results,
        "totals": {
            "router_cost_usd": round(tot_r, 6),
            "opus_cost_usd": round(tot_o, 6),
            "saved_pct": round((1 - tot_r / tot_o) * 100, 1) if tot_o else 0,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    # readable markdown
    md = ["# Head-to-head: Router vs. raw Opus\n"]
    for x in results:
        t, rt, op = x["task"], x["router"], x["opus"]
        md.append(f"## {t['id']}\n")
        md.append(f"**Prompt:** {t['prompt'][:200]}...\n")
        md.append(f"**Check:** {t['verifiable']}\n")
        md.append(f"| | model | tokens | cost | latency |")
        md.append(f"|---|---|---|---|---|")
        md.append(f"| router | {rt['routed_model']} | {rt['completion_tokens']} | "
                  f"${rt['cost_usd']:.5f} | {rt['latency_s']}s |")
        md.append(f"| opus | {op['routed_model']} | {op['completion_tokens']} | "
                  f"${op['cost_usd']:.5f} | {op['latency_s']}s |\n")
        md.append(f"### Router answer\n\n{rt['answer']}\n")
        md.append(f"### Opus answer\n\n{op['answer']}\n")
        md.append("---\n")
    md.append(f"## Totals\n")
    md.append(f"- Router: **${tot_r:.5f}**  ·  Opus: **${tot_o:.5f}**  ·  "
              f"**Saved {report['totals']['saved_pct']}%**\n")
    Path(args.out).with_suffix(".md").write_text("\n".join(md))

    print(f"\n{'='*50}")
    print(f"Router total: ${tot_r:.5f}   Opus total: ${tot_o:.5f}   "
          f"Saved {report['totals']['saved_pct']}%")
    print(f"Wrote {args.out} and {Path(args.out).with_suffix('.md')}")


if __name__ == "__main__":
    main()
