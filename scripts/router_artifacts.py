#!/usr/bin/env python3
"""Upload/download the public router artifact bundle.

The repository keeps checkpoints out of git. This script makes the active
public-derived checkpoints reproducible to fetch while deliberately excluding
private/personal experiment artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_REPO = "micdn/llm-router-goose-public"


@dataclass(frozen=True)
class Artifact:
    path: str
    description: str
    required_for_runtime: bool = True


ARTIFACTS = [
    Artifact(
        "checkpoints/prefill_router_combined.pt",
        "Active prefill cost-router checkpoint trained from public verified trace labels.",
    ),
    Artifact(
        "checkpoints/session_health_public.pkl",
        "Active v2 session-health checkpoint trained from public HF trajectory windows.",
    ),
    Artifact(
        "configs/combined-pool.yaml",
        "Active router pool config.",
        required_for_runtime=False,
    ),
    Artifact(
        "configs/litellm-combined.yaml",
        "Active LiteLLM model config.",
        required_for_runtime=False,
    ),
    Artifact(
        "docs/SESSION_HEALTH_ROUTER.md",
        "Session-health training and runtime notes.",
        required_for_runtime=False,
    ),
]

RUNTIME_ARTIFACTS = [artifact for artifact in ARTIFACTS if artifact.required_for_runtime]


def repo_from_args(value: str | None) -> str:
    return value or os.environ.get("ROUTER_ARTIFACT_REPO") or DEFAULT_REPO


def token() -> str:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or ""
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path, description: str, *, required_for_runtime: bool) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "description": description,
        "required_for_runtime": required_for_runtime,
    }


def build_manifest(repo_id: str) -> dict[str, Any]:
    artifacts = []
    for artifact in ARTIFACTS:
        path = Path(artifact.path)
        if not path.exists():
            raise SystemExit(f"missing artifact: {artifact.path}")
        artifacts.append(
            file_info(
                path,
                artifact.description,
                required_for_runtime=artifact.required_for_runtime,
            )
        )
    return {
        "repo_id": repo_id,
        "kind": "llm-router-public-artifacts",
        "version": 1,
        "active_runtime_config": "configs/combined-pool.yaml",
        "excluded": [
            {
                "path": "checkpoints/prefill_router_goose.pt",
                "reason": "private Goose-session experiment; not used by configs/combined-pool.yaml",
            }
        ],
        "artifacts": artifacts,
    }


def model_card(manifest: dict[str, Any]) -> str:
    files = "\n".join(
        f"- `{item['path']}` ({item['size_bytes'] / 1024 / 1024:.2f} MiB): "
        f"{item['description']}"
        for item in manifest["artifacts"]
    )
    return f"""---
license: apache-2.0
tags:
- llm-router
- model-routing
- agent-routing
- goose
---

# LLM Router Public Artifacts

Public artifact bundle for the `goose-personalized-router` branch of
`llm-router`.

## Files

{files}

## Provenance

- `prefill_router_combined.pt` is the active prefill router checkpoint used by
  `configs/combined-pool.yaml`. It maps public verified trace labels onto the
  callable OpenAI/Anthropic model pool in that config.
- `session_health_public.pkl` is the active v2 session-health checkpoint. Its
  embedded report points at `data_public/session-health-windows-v2.csv`, built
  from public SWE-bench and TerminalBench Hugging Face traces.
- `checkpoints/prefill_router_goose.pt` is deliberately not included; it was a
  private Goose-session experiment and is not used by the active config.

## Use

From the source repo:

```bash
just run-router
just goose-instructions
```

Or directly:

```bash
ROUTER_ARTIFACT_REPO={manifest['repo_id']} ./scripts/run-public-router.sh
```

The session-health checkpoint is a trusted Python pickle. Do not load arbitrary
pickle files from untrusted publishers.
"""


def hf_url(repo_id: str, path: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{path}"


def request(url: str) -> urllib.request.Request:
    headers = {"User-Agent": "llm-router-artifact-fetch/1"}
    auth = token()
    if auth:
        headers["Authorization"] = f"Bearer {auth}"
    return urllib.request.Request(url, headers=headers)


def fetch_json(repo_id: str, path: str) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(request(hf_url(repo_id, path)), timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def download_file(repo_id: str, path: str, expected_sha256: str | None = None) -> None:
    out = Path(path)
    if out.exists() and expected_sha256 and sha256(out) == expected_sha256:
        print(f"ok      {path}")
        return
    if out.exists() and expected_sha256 is None:
        print(f"exists  {path}")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    print(f"fetch   {path}")
    with urllib.request.urlopen(request(hf_url(repo_id, path)), timeout=300) as response:
        with tmp.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    if expected_sha256 and sha256(tmp) != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"checksum mismatch for {path}")
    os.replace(tmp, out)
    print(f"wrote   {path}")


def download(repo_id: str) -> None:
    manifest = fetch_json(repo_id, "artifact-manifest.json")
    expected = {
        item["path"]: item
        for item in (manifest or {}).get("artifacts", [])
        if item.get("required_for_runtime")
    }
    for artifact in RUNTIME_ARTIFACTS:
        info = expected.get(artifact.path, {})
        download_file(repo_id, artifact.path, info.get("sha256"))


def upload(repo_id: str, *, private: bool) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("upload requires `huggingface_hub` installed") from exc

    manifest = build_manifest(repo_id)
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        manifest_path = tmp / "artifact-manifest.json"
        card_path = tmp / "README.md"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        card_path.write_text(model_card(manifest))

        upload_items = [(manifest_path, "artifact-manifest.json"), (card_path, "README.md")]
        upload_items.extend((Path(item["path"]), item["path"]) for item in manifest["artifacts"])

        for local_path, remote_path in upload_items:
            print(f"upload {remote_path}")
            api.upload_file(
                repo_id=repo_id,
                repo_type="model",
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
            )
    print(f"uploaded https://huggingface.co/{repo_id}")


def goose_instructions(port: int, repo_id: str) -> None:
    print(
        textwrap.dedent(
            f"""\
            Router artifact repo: {repo_id}
            Router URL: http://localhost:{port}
            Dashboard:  http://localhost:{port}/dashboard

            Start the router:

              just run-router

            Recommended Goose provider:

              write ~/.config/goose/custom_providers/router.json with:
              {{
                "name": "router",
                "engine": "openai",
                "display_name": "llm-router",
                "base_url": "http://localhost:{port}",
                "api_key_env": "",
                "requires_auth": false,
                "supports_streaming": true,
                "timeout_seconds": 600,
                "models": [{{"name": "nvidia-routed", "context_limit": 200000}}]
              }}

              GOOSE_PROVIDER=router GOOSE_MODEL=nvidia-routed goose

            Env-only Goose fallback, one-shot:

              cd <repo-you-want-goose-to-work-on>
              LITELLM_HOST=http://localhost:{port} LITELLM_API_KEY=sk-local GOOSE_CONTEXT_LIMIT=200000 \\
              GOOSE_PROVIDER=litellm GOOSE_MODEL=nvidia-routed \\
              goose run --name routed-task -t "<your task>"

            Env-only Goose fallback, interactive:

              LITELLM_HOST=http://localhost:{port} LITELLM_API_KEY=sk-local GOOSE_CONTEXT_LIMIT=200000 \\
              GOOSE_PROVIDER=litellm GOOSE_MODEL=nvidia-routed \\
              goose

            Force top tier for a run:

              goose run --name hard-task -t "!hard <your task>"

            Check routing/savings:

              curl -s http://localhost:{port}/savings | python3 -m json.tool
            """
        ).strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    download_parser = sub.add_parser("download", help="download active runtime checkpoints")
    download_parser.add_argument("--repo", default=None, help=f"HF repo id; default {DEFAULT_REPO}")

    upload_parser = sub.add_parser("upload", help="upload active public artifacts")
    upload_parser.add_argument("--private", action="store_true", help="create/update a private repo")
    upload_parser.add_argument("--repo", default=None, help=f"HF repo id; default {DEFAULT_REPO}")

    instructions_parser = sub.add_parser("goose-instructions", help="print Goose setup commands")
    instructions_parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "4000")))
    instructions_parser.add_argument("--repo", default=None, help=f"HF repo id; default {DEFAULT_REPO}")

    args = parser.parse_args()
    repo_id = repo_from_args(getattr(args, "repo", None))
    if args.command == "download":
        download(repo_id)
    elif args.command == "upload":
        upload(repo_id, private=args.private)
    elif args.command == "goose-instructions":
        goose_instructions(args.port, repo_id)


if __name__ == "__main__":
    main()
