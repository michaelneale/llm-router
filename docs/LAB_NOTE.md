# Lab note: what actually makes this router work

A consolidated record of the investigation into personalizing the prefill
complexity router, why most approaches failed, and the one that worked.

## The goal

Route each request to the cheapest model that won't hurt quality, and know when
a stronger model is genuinely warranted — without changing the core routing
engine too deeply.

## What we tried and why it failed

Every approach hit the same wall: **the training label was uninformative.**

1. **Train on goose/codex session prompts, label by single-shot correctness.**
   Labels came from `collect` (ask each model, judge the answer plausible or
   not). On open-ended agentic chatter almost every model's answer is
   "plausible", so every model scored ~95%+. The accuracy/cost front was flat:
   0/300 tasks where only a frontier model succeeded. The router correctly
   routed everything cheap — because by that label, cheap *was* always fine.

2. **Escalation keyword overlay** (force top tier on "poll/monitor/!hard").
   A hand-written rule, not learned. Caught the rare sustained-loop case but was
   a hack and didn't generalize.

3. **Depth-scaled tolerance** (tighten tolerance as a session deepens, because
   user-correction "you bailed" complaints cluster deep, not early). Mechanically
   works, but: safe only with a tolerance floor (>0), and within the safe band it
   barely moves routing — the checkpoint's per-tier confidence was too flat for
   small tolerance nudges to matter. At tol=0 it ratchets cost wastefully.

4. **Mine a completeness label from traces** (did the model follow through, or
   did the user have to push back). Real signal but irreducibly rare (~26–85
   positives vs thousands) and noisy (Q&A turns mislabeled as bails). Too
   imbalanced to train on.

The unifying lesson: single-shot prompt labeling — all the NVIDIA pipeline can
produce locally — cannot capture the property worth paying a frontier model for.
The original blueprint assumes **benchmark-style data with verifiable ground
truth** (MMLU, math, code with right/wrong answers), where the label genuinely
separates models. We fed it open-ended work where it doesn't.

## What worked: verified labels from multi-model benchmark traces

`tarsur385/swebench-verified-trajectories` runs the **same 500 SWE-bench Verified
tasks across ~10 models** (gpt-5-mini, claude-haiku, gemini-flash, glm-5,
minimax, claude-opus, gpt-5.2, …), each with a **machine-verified `resolved`
label** (did the generated patch make the repo's tests pass).

This gives, for free, the thing we never had: a per-(task, model) success matrix
with a *real* outcome, not a plausibility opinion. On 270 full-coverage tasks:

- 82% solvable by a cheap model
- 4% needed a mid model
- 2% frontier-only (cheap + mid failed)
- 10% nobody solved

A small but real escalation signal — vs 0/300 in our local traces.

### Result

Trained the existing prefill MLP on this matrix (encoder Qwen3.5-0.8B, shared
trunk ensemble). Per-model trunk AUC 0.89–0.95 — the model genuinely predicts,
from task text, which models will solve which task.

Routing front on held-out tasks (verified outcomes):

| strategy            | resolved % | avg $/M |
|---------------------|-----------:|--------:|
| router @ tol=0.05   |   ~86%     |   ~3.1  |
| always gpt-5-mini   |    60%     |    2.0  |
| always Opus (top)   |    79%     |   25.0  |

The routed ensemble of cheaper models **beats the single frontier model on
success rate at ~1/8th the cost**, and escalates only the genuinely hard tasks
(18%→3% to top tier as tolerance tightens). That is the whole thesis,
demonstrated — and it only became possible with a verifiable label.

## Takeaways

- The label is everything. Architecture and tolerance tuning are secondary.
- "Warrants a stronger model" is only learnable when you can *verify* that
  cheaper models fail and the stronger one succeeds on the same task.
- For a personalized router on real agent work, the open problem is getting
  verifiable success labels from your own traffic (test/CI/command outcomes),
  not more prompts or cleverer single-shot judging.
- Capability does not track price on these tasks (gemini-flash > gpt-5-mini), so
  the router routes on predicted success, not tier order — which is correct.

## Reproduce

```bash
python scripts/pull_swebench_traces.py        # pull verified multi-model matrix
python scripts/build_swebench_training.py      # -> data/swebench-train.csv
model-router train --config configs/swebench-pool.yaml \
  --data data/swebench-train.csv --output-dir checkpoints --device mps
```
