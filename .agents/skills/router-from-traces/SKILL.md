---
name: router-from-traces
description: >
  Build or retrain the llm-router prefill cost router from public Hugging Face /
  benchmark traces using the actual real provider model pool. Use when someone
  asks to train from HF/public traces, relabel a new model pool, avoid confusing
  fixed-slot remapping, create correctness labels, sweep cost-vs-quality loss,
  or produce a checkpoint/config for an OpenAI-compatible client. Agent sessions
  are for runtime spot checks, not the main training source.
---

# Router From Public Traces

Use this skill for the router workflow in this repository: train a cost router
from public HF/benchmark task traces, using the **actual models intended for
serving**.

## Non-Negotiable Rule

Do **not** map source-model labels onto confusing old slot names and then remap
those slots again to served models.

The training target names, config names, dashboard names, and served models
should mean the same thing. Prefer literal provider model IDs or clear stable
aliases that point one-to-one at the real model:

```text
openai/gpt-5-nano
openai/gpt-5-mini
openai/gpt-5.4-mini
anthropic/claude-sonnet-4-6
anthropic/claude-opus-4-8
```

If a previous checkpoint contains names like `gpt-oss-20b-high` or
`claude-haiku-4-5-high`, treat them as legacy learned slots. Do not use them as
evidence about the current real models unless those exact models were directly
labeled.

## What Labels Mean

The router trains on a correctness matrix:

```text
question,model,isCorrect,output_tokens
task A,openai/gpt-5-mini,1,842
task A,anthropic/claude-opus-4-8,1,911
task B,openai/gpt-5-nano,0,213
task B,anthropic/claude-sonnet-4-6,1,695
```

`isCorrect` means the model solved or acceptably answered the task. It is not
"which model appeared in the public trace" and not "which model we think is in
the same tier." For a new real model pool, run those real models and label them.

## Simple Workflow

1. Pick the real model pool.

   Use the models we actually want the router to choose from. Record current
   `litellm_model`, input cost, output cost, context limit, and any special max
   token settings.

2. Select public prompts/tasks.

   Use public HF/benchmark task sources already pulled into the repo where
   available. Existing flattened files are useful prompt sources:

   ```text
   data/swebench-train.csv
   data/combined-train.csv
   data/full-train.csv
   ```

   Their existing labels are only valid for the source/legacy slot setup that
   produced them. For a new real model pool, use the prompts/tasks and create a
   new label matrix.

3. Label the selected real models.

   Do this in the environment that has access to the required provider APIs.
   Hugging Face compute is useful after labels exist; it must not be assumed to
   have provider access unless the required keys/network path are configured
   there.

   For each prompt/task and each selected model:

   - call the actual provider model;
   - keep enough output-token headroom, especially for reasoning models;
   - judge or verify whether it solved the task;
   - emit `question,model,isCorrect,output_tokens`.

   Prefer verified task outcomes when available. For open-ended public prompts,
   use an LLM judge carefully and pilot-check for obvious judge/pathology issues.

4. Split train/eval.

   Keep a held-out split by task, not by row, so the same prompt does not appear
   in both train and eval under different model rows.

5. Train the prefill checkpoint.

   The prefill router learns:

   ```text
   P(real model is correct | task text/context features)
   ```

   It does not directly learn "pick cheap." Cost enters after prediction.

   Typical command shape:

   ```bash
   model-router train \
     --config configs/<real-pool>.yaml \
     --data data/<real-pool>-train.csv \
     --output-dir checkpoints/<real-pool> \
     --device mps
   ```

   This step can run locally or be offloaded to Hugging Face Jobs after the
   label CSV exists. Training/evaluation does not need provider API keys; it
   only needs the config, labels, encoder download/cache, and HF token if it
   will upload artifacts.

6. Sweep cost/loss.

   Routing selects:

   ```text
   p_max = max predicted correctness
   threshold = p_max - tolerance
   choose cheapest model with p >= threshold
   ```

   Sweep tolerance and output-token cost weighting on the held-out labels. Pick
   the Pareto/knee point: large savings with negligible or explicitly accepted
   quality loss.

7. Serve and validate.

   Update the runtime config so names are literal and dashboard-visible. Then:

   ```bash
   ./scripts/restart-router.sh
   ```

   Validate with public held-out metrics first, then with real client tasks as
   qualitative spot checks. The dashboard shows cost behavior, not quality.

## Recommended Provider/HF Split

If provider access is only available in one environment and Hugging Face is only
being used for compute/artifact storage, split the retrain like this:

| Phase | Where | Why |
| --- | --- | --- |
| Select public tasks | credentialed environment | cheap, reproducible, can inspect sample |
| Call provider models | credentialed environment | uses required provider API access |
| Judge/verify and write labels | credentialed environment | keeps raw outputs and retries resumable |
| Upload labels/raw outputs | credentialed environment -> HF dataset/model repo | parks expensive work |
| Split/train/evaluate/sweep | credentialed environment or HF Job | no provider keys required after labels exist |
| Upload checkpoint/config/calibration | HF | makes `just run-router` style downloads possible |
| Client spot checks | deployment/test environment | validates real client behavior |

The desired scripts are:

```text
scripts/label_public_ladder.py       # provider calls, resumable labels
scripts/hf_train_public_ladder.py    # no provider calls; train/eval/upload
```

Do not offload the labeling job to Hugging Face unless the required provider
keys and network access are explicitly available there. Prefer uploading a
completed `data/<pool>-labels.csv` plus raw-output JSONL shards, then running a
GPU HF Job only for hidden-state extraction, torch training, tolerance sweep,
and artifact upload.

## Time Expectations

Training is not usually the bottleneck. Labeling is.

For a pilot:

```text
1k-2k prompts x 5 models
Labeling:        same day, often 2-8 hours depending latency/rate limits
Feature extract: 10-60 min on MPS/GPU
Training:        5-30 min
Evaluation:      minutes
```

For a serious run:

```text
~10k prompts x 5 models
Labeling:        overnight to 1-2 days
Feature extract: 1-4 hours
Training:        30-90 min
Evaluation:      minutes
```

If prefill features for exactly the same prompt set are already cached, retrain
time drops sharply. If the model pool changes but the prompt set is identical,
the expensive part is still relabeling the real models.

With the provider/HF split, expect provider labeling to dominate wall-clock and
cost. HF GPU time is usually small because it only handles the router encoder
feature extraction, MLP training, evaluation, and upload.

## Current Repo Context

The active historical checkpoint was useful for getting the system working, but
it was trained from public labels mapped onto fixed legacy slots and then those
slots were mapped onto current callable models. That is confusing and weakens
claims about current model strengths.

For the next proper checkpoint:

- use real model names as `model` labels;
- produce a new CSV for the selected real model pool;
- train a checkpoint whose `model_names` match the served model names;
- regenerate the LiteLLM config from that same pool;
- avoid slot aliases unless they are purely display names and cannot be mistaken
  for learned legacy models.

## Useful Existing Files

Prompt/data builders:

```text
scripts/build_full_training.py
scripts/build_swebench_training.py
scripts/pull_swebench_traces.py
scripts/pull_terminalbench.py
```

Training/evaluation:

```text
src/model_router_toolkit/prefill/train.py
src/model_router_toolkit/prefill/router.py
scripts/select_router_operating_point.py
scripts/sweep_verified_tolerances.py
```

Serving:

```text
configs/combined-pool.yaml
scripts/restart-router.sh
scripts/router_artifacts.py
```

## Avoid

- Do not present legacy slot remapping as direct evidence about current real
  provider models.
- Keep checkpoint training grounded in public trace labels.
- Do not optimize for savings alone; report loss on held-out labels.
- Do not trust the dashboard as a quality signal.
- Do not add keyword escalation as a substitute for direct labels unless it is
  clearly documented as a policy overlay.
