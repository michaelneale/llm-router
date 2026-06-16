# Public Router Artifacts

The active router needs two generated checkpoints that are intentionally not
stored in git:

- `checkpoints/prefill_router_combined.pt`
- `checkpoints/session_health_public.pkl`

They are uploaded to Hugging Face:

```text
micdn/llm-router-goose-public
```

The bundle deliberately excludes `checkpoints/prefill_router_goose.pt`; that was
a private Goose-session experiment and is not used by `configs/combined-pool.yaml`.

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[proxy]'

just run-router
```

`just run-router` downloads missing checkpoints, prints Goose setup commands,
and starts the router on `http://localhost:4000`.

To use a different artifact repo:

```bash
ROUTER_ARTIFACT_REPO=namespace/repo just run-router
```

## Goose

One-shot:

```bash
cd <repo-you-want-goose-to-work-on>
LITELLM_HOST=http://localhost:4000 LITELLM_API_KEY=sk-local \
GOOSE_PROVIDER=litellm GOOSE_MODEL=nvidia-routed \
goose run --name routed-task -t "<your task>"
```

Interactive:

```bash
LITELLM_HOST=http://localhost:4000 LITELLM_API_KEY=sk-local \
GOOSE_PROVIDER=litellm GOOSE_MODEL=nvidia-routed \
goose
```

For a hard task, prefix the task with `!hard` to force the top tier.
