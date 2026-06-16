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

Observed v2 build:

- 35,766 trajectory windows
- 14,989 SWE-bench windows, 17.3% positive
- 20,777 Terminal-Bench windows, 46.0% positive
- 34.0% positive overall

The label is `bad_next`: the trace prefix has reached a deterioration point.
Failed traces are not labeled positive from turn one; they become positive only
after local badness signals such as repeated errors, retry loops, or late-stage
non-recovery appear.

## V2 Feature Pass

The v2 pass keeps the same public-HF-only setup but exposes more prefix-visible
trajectory structure to the classifier:

- Tool-call command extraction from `command` and `cmd` fields, plus bash code
  fences.
- Repeated command and repeated test-command counts.
- Repeated failing command attribution from assistant tool calls to following
  tool errors.
- Patch/edit recency, same-file edit repetition, same error after patch, and
  test failure after patch.
- Timeout, missing dependency, and tool parse-error counters.
- Recovery-after-error counters, so recovered failures can act as hard
  negatives.
- Initial user task descriptions can contain traceback/error text without
  counting as an operational trajectory failure.

The useful learned signal is not raw frustration wording. Public traces mostly
do not contain genuine human frustration. The useful public signal is machine
observable churn: late/deep unrecovered trajectories, repeated commands,
missing dependencies, timeouts, and patch/test/error loops.

## Training

Trainer:

```bash
.venv/bin/python scripts/train_session_health.py \
  --data data_public/session-health-windows-full.csv \
  --output checkpoints/session_health_public.pkl \
  --report data_public/session-health-report.json \
  --min-precision 0.835
```

Held-out public-task split, v2:

- threshold: 0.88
- precision: 0.844
- recall: 0.169
- F1: 0.281
- average precision: 0.722
- ROC-AUC: 0.813

Same v2 dataset with old numeric features only:

- threshold: 0.88
- precision: 0.841
- recall: 0.156
- F1: 0.263
- average precision: 0.718
- ROC-AUC: 0.811

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
  threshold: 0.88
  top_tier_model: claude-opus-4-6-high
```

If the score crosses threshold, the route decision is logged as
`session_health_escalation` and selected model is forced to the top tier.

Provider-free live probes on the local proxy after the v2 checkpoint:

- clean simple prompt: score `0.0219`, threshold `0.88`, selected
  `gpt-4-1-nano-high` -> `openai/gpt-5-mini`.
- short patch/test churn trajectory: score `0.8417`, threshold `0.88`,
  below the high-precision escalation bar.
- obvious unrecovered dependency/error loop: score `0.9963`, threshold `0.88`,
  selected `claude-opus-4-6-high` -> `anthropic/claude-opus-4-8`.
