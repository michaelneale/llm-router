# Session-Health Escalation Router

This is a second learned gate, separate from the prefill cost router.

The prefill router predicts which model is likely good enough for the current
task. The session-health gate predicts whether the recent agent trajectory is
going badly. If the score crosses the configured threshold, routing forces the
top tier.

## Data

The training data is public-HF-only. It does not use Goose sessions.

Sources:

- `tarsur385/swebench-verified-trajectories`: raw `.traj.json` agent traces with
  messages, tool outputs, error fields, and final `resolved`.
- `yoonholee/terminalbench-trajectories`: terminal/ops trajectories with
  `steps`, model metadata, duration/tokens/cost, and final `reward`.

Builder:

```bash
.venv/bin/python scripts/build_hf_session_health_windows.py \
  --output data_public/session-health-windows-full.csv \
  --swe-max-traces 1500 \
  --terminal-max-rows 3500 \
  --max-windows-per-trace 10 \
  --swe-workers 16
```

Observed build:

- 33,451 trajectory windows
- 14,980 SWE-bench windows, 19.6% positive
- 18,471 Terminal-Bench windows, 46.3% positive
- 34.3% positive overall

The label is `bad_next`: the trace prefix has reached a deterioration point.
Failed traces are not labeled positive from turn one; they become positive only
after local badness signals such as repeated errors, retry loops, or late-stage
non-recovery appear.

## Training

Trainer:

```bash
.venv/bin/python scripts/train_session_health.py \
  --data data_public/session-health-windows-full.csv \
  --output checkpoints/session_health_public.pkl \
  --report data_public/session-health-report.json \
  --min-precision 0.835
```

Held-out public-task split:

- threshold: 0.90
- precision: 0.837
- recall: 0.157
- F1: 0.265
- average precision: 0.718
- ROC-AUC: 0.806

The threshold is deliberately conservative. It is meant to limit false positives
and only force the frontier tier on clear bad-trajectory states.

Generated artifacts under `data_public/` and `checkpoints/*.pkl` are ignored and
should be regenerated locally.

## Runtime

Configured in `configs/combined-pool.yaml`:

```yaml
session_health:
  enabled: true
  checkpoint: checkpoints/session_health_public.pkl
  threshold: 0.90
  top_tier_model: claude-opus-4-6-high
```

If the score crosses threshold, the route decision is logged as
`session_health_escalation` and selected model is forced to the top tier.

Provider-free live probes on the local proxy:

- repeated failed test/tool trajectory: score `0.9027`, threshold `0.90`,
  selected `claude-opus-4-6-high` -> `anthropic/claude-opus-4-8`.
- clean simple prompt: score `0.0346`, threshold `0.90`, selected
  `gpt-4-1-nano-high` -> `openai/gpt-5-mini`.

