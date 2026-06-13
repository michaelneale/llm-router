"""Offline routing benchmark: latency + selection distribution, no API calls."""

from __future__ import annotations

import json
import statistics
import time

from model_router_toolkit.config import build_router_from_config, load_config

QUERIES = {
    "trivial": [
        "What is 2+2?",
        "What does the ls command do?",
        "Capital of France?",
        "Convert 100F to Celsius",
    ],
    "simple-code": [
        "Write a Python one-liner to reverse a string",
        "Fix the typo in this sentence: 'teh quick brown fox'",
        "What does git stash do?",
        "Rename variable x to count in: for x in range(10): print(x)",
    ],
    "mid-agentic": [
        "Read the README, find all broken links, and summarize what the project does",
        "Write a bash script that watches a directory and rsyncs changes to a remote host",
        "Explain the difference between TCP and UDP and when to use each",
        "Refactor this function to use async/await and add error handling for timeouts",
    ],
    "hard": [
        "Prove that the square root of 2 is irrational",
        "Design a distributed rate limiter that works across 50 API gateway nodes with sub-millisecond overhead, handling clock skew",
        "Debug a race condition in a Rust tokio service where requests intermittently hang only under load above 10k rps",
        "Derive the Euler-Lagrange equation from the principle of least action",
    ],
}

TOLERANCES = [0.05, 0.20]


def main() -> None:
    config = load_config("configs/goose-mix.yaml")
    display = {m.name: m.display_name for m in config.models}
    cost_out = {m.name: m.cost_per_m_output_tokens for m in config.models}
    router = build_router_from_config(config)

    router.route("warmup", tolerance=0.2)  # load encoder

    results = []
    latencies = []
    for category, qs in QUERIES.items():
        for q in qs:
            t0 = time.perf_counter()
            r = router.route(q, tolerance=0.05)
            ms = (time.perf_counter() - t0) * 1000
            latencies.append(ms)
            r20 = router.route(q, tolerance=0.20)
            results.append(
                {
                    "category": category,
                    "q": q[:60],
                    "t05": r.selected_model,
                    "t05_conf": round(r.selected_confidence, 3),
                    "t20": r20.selected_model,
                    "ms": round(ms),
                }
            )

    print("\n=== Routing latency (CPU) ===")
    print(f"  mean={statistics.mean(latencies):.0f}ms  median={statistics.median(latencies):.0f}ms  "
          f"min={min(latencies):.0f}ms  max={max(latencies):.0f}ms  n={len(latencies)}")

    print("\n=== Selections (tol=0.05 vs tol=0.20) ===")
    for row in results:
        print(f"  [{row['category']:10s}] {row['q']:60s} -> {display[row['t05']][:22]:22s} "
              f"(c={row['t05_conf']}) | t20: {display[row['t20']][:22]}")

    for tol_key in ("t05", "t20"):
        from collections import Counter

        dist = Counter(r[tol_key] for r in results)
        total = len(results)
        blended = sum(cost_out[m] * n for m, n in dist.items()) / total
        print(f"\n=== Distribution tol={tol_key[1:]} ===   blended out-cost ${blended:.2f}/M "
              f"(vs ${cost_out['claude-opus-4-6-high']:.2f}/M all-Opus -> "
              f"{(1 - blended / cost_out['claude-opus-4-6-high']) * 100:.0f}% saved)")
        for m, n in dist.most_common():
            print(f"  {display[m]:28s} {n:2d} ({n / total * 100:.0f}%)")

    json.dump(results, open("/tmp/bench_routing.json", "w"), indent=1)


if __name__ == "__main__":
    main()
