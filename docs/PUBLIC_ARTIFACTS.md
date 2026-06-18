# Public Router Artifacts

The active router needs two generated checkpoints that are intentionally not
stored in git:

- `checkpoints/prefill_router_combined.pt`
- `checkpoints/session_health_public.pkl`

They are uploaded to Hugging Face:

```text
micdn/llm-router-goose-public
```

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[prefill,proxy,verified-data]'

just run-router
```

`just run-router` downloads missing checkpoints, prints Goose setup commands,
and starts the router on `http://localhost:4000`.

To use a different artifact repo:

```bash
ROUTER_ARTIFACT_REPO=namespace/repo just run-router
```

## Goose

Recommended persistent provider:

```text
~/.config/goose/custom_providers/router.json
```

```json
{
  "name": "router",
  "engine": "openai",
  "display_name": "llm-router",
  "description": "NVIDIA LLM Router v3 public-trace router",
  "base_url": "http://localhost:4000",
  "api_key_env": "",
  "requires_auth": false,
  "supports_streaming": true,
  "timeout_seconds": 600,
  "models": [
    {
      "name": "nvidia-routed",
      "context_limit": 200000
    }
  ]
}
```

Then use:

```bash
GOOSE_PROVIDER=router GOOSE_MODEL=nvidia-routed goose
```

Use `200000` as the current safe context limit because it is the smallest real
context window among the models this router may select. If the pool changes,
set it to the new minimum.

One-shot:

```bash
cd <repo-you-want-goose-to-work-on>
LITELLM_HOST=http://localhost:4000 LITELLM_API_KEY=sk-local GOOSE_CONTEXT_LIMIT=200000 \
GOOSE_PROVIDER=litellm GOOSE_MODEL=nvidia-routed \
goose run --name routed-task -t "<your task>"
```

Interactive:

```bash
LITELLM_HOST=http://localhost:4000 LITELLM_API_KEY=sk-local GOOSE_CONTEXT_LIMIT=200000 \
GOOSE_PROVIDER=litellm GOOSE_MODEL=nvidia-routed \
goose
```

For a hard task, prefix the task with `!hard` to force the top tier.
