---
name: router-with-agents
description: >
  Wire this repo's running LLM router proxy into Goose or another
  OpenAI-compatible agent, verify which real model was selected, inspect live
  routing scores, and troubleshoot dashboard/model-choice behavior.
---

# Router With Agents

Use this skill for this repository's runtime proxy work: starting the local
LiteLLM-compatible router, pointing Goose at it, checking which real provider
model was selected, and reading the dashboard or route log.

## Start Or Restart The Proxy

From the repo root:

```bash
PORT=4000 ./scripts/restart-router.sh
```

The script uses:

```text
configs/combined-pool.yaml
configs/litellm-combined.yaml
```

The dashboard is:

```text
http://localhost:4000/dashboard
```

The OpenAI-compatible base URL is:

```text
http://localhost:4000/v1
```

## Goose Config

Use the checked-in config files from this repo where possible:

```text
goose-prefill-config
goose-embedding-config
```

For direct environment wiring:

```bash
export LITELLM_HOST=http://localhost:4000
export LITELLM_API_KEY=sk-local
export GOOSE_PROVIDER=litellm
export GOOSE_MODEL=nvidia-routed
```

Use `embedding-routed` only when explicitly comparing the embedding ladder.

## Verify Routing

Check exposed model aliases:

```bash
curl -s http://localhost:4000/v1/models | python3 -m json.tool
```

Check the live counters and recent route-score rows:

```bash
curl -s http://localhost:4000/savings | python3 -m json.tool
```

Check route decisions directly:

```bash
tail -n 40 /tmp/router-routes.jsonl
```

Route rows include:

```text
decision
selected_model
confidences
metadata.complexity
metadata.pin_reason
task_view
```

The dashboard's recent routing section is backed by these rows.

## Dashboard Interpretation

The dashboard routing distribution is total provider traffic since the current
proxy process started or counters were reset. It is not the same as embedding
mode calibration.

When Nano dominates, check the route-log decision type:

```text
cheap_utility -> intentionally cheap utility/title/info requests
embedding_route -> embedding complexity ladder
route -> prefill router decision
pin_non_decision -> cache/session pin
session_health_escalation -> forced top tier after bad-session signal
```

For embedding mode, the moving score is `metadata.complexity`.
For prefill mode, use the selected model's value in `confidences`.

## Runtime Knobs

Set quality/cost tolerance:

```bash
curl -s -X POST http://localhost:4000/router/tuning \
  -H 'Content-Type: application/json' \
  -d '{"tolerance":0.04}'
```

Set cache pinning:

```bash
curl -s -X POST http://localhost:4000/router/tuning \
  -H 'Content-Type: application/json' \
  -d '{"cache_pin_mode":"dear_only"}'
```

Force top tier for 30 minutes:

```bash
curl -s -X POST http://localhost:4000/router/tuning \
  -H 'Content-Type: application/json' \
  -d '{"turbo":{"enabled":true,"duration_seconds":1800}}'
```

## Goose Session Data

The active Goose database is:

```text
~/.local/share/goose/sessions/sessions.db
```

Do not use the old zero-byte path:

```text
~/.local/share/goose/sessions.db
```

Recent Goose request logs are under:

```text
~/.local/state/goose/logs/llm_request*.jsonl
```

Use those logs to match a dashboard spike to the actual Goose session/model.
