#!/usr/bin/env python3
"""Upload/download the public router artifact bundle.

The repository keeps checkpoints out of git. This script makes the active
public-derived checkpoints reproducible to fetch.
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
    runtime_mode: str = "default"


EMBEDDING_BUNDLE_PREFIX = "embedding/complexity_model"
EMBEDDING_BUNDLE_FILES = [
    "config.json",
    "embedder.onnx",
    "embedder_hf_config.json",
    "eval_report.md",
    "parity_fixture.jsonl",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "weights.safetensors",
]


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
    *[
        Artifact(
            f"{EMBEDDING_BUNDLE_PREFIX}/{name}",
            "Optional embedding-router bundle trained from public WildChat traces."
            if name == "config.json"
            else "Optional embedding-router bundle file.",
            required_for_runtime=False,
            runtime_mode="embedding",
        )
        for name in EMBEDDING_BUNDLE_FILES
    ],
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


def embedding_bundle_dir() -> Path:
    return Path(os.environ.get("ROUTER_EMBEDDING_BUNDLE", "~/.goose/complexity_model")).expanduser()


def local_path_for_artifact(artifact: Artifact) -> Path:
    prefix = f"{EMBEDDING_BUNDLE_PREFIX}/"
    if artifact.path.startswith(prefix):
        return embedding_bundle_dir() / artifact.path[len(prefix) :]
    return Path(artifact.path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(
    local_path: Path,
    remote_path: str,
    description: str,
    *,
    required_for_runtime: bool,
    runtime_mode: str,
) -> dict[str, Any]:
    return {
        "path": remote_path,
        "size_bytes": local_path.stat().st_size,
        "sha256": sha256(local_path),
        "description": description,
        "required_for_runtime": required_for_runtime,
        "runtime_mode": runtime_mode,
    }


def build_manifest(repo_id: str) -> dict[str, Any]:
    artifacts = []
    for artifact in ARTIFACTS:
        path = local_path_for_artifact(artifact)
        if not path.exists():
            raise SystemExit(f"missing artifact: {artifact.path} (local {path})")
        artifacts.append(
            file_info(
                path,
                artifact.path,
                artifact.description,
                required_for_runtime=artifact.required_for_runtime,
                runtime_mode=artifact.runtime_mode,
            )
        )
    return {
        "repo_id": repo_id,
        "kind": "llm-router-public-artifacts",
        "version": 1,
        "active_runtime_config": "configs/combined-pool.yaml",
        "artifacts": artifacts,
    }


def model_card(manifest: dict[str, Any]) -> str:
    files = "\n".join(
        f"- `{item['path']}` ({item.get('runtime_mode', 'default')}, "
        f"{item['size_bytes'] / 1024 / 1024:.2f} MiB): "
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

Public artifact bundle for the public-trace LLM router branch.

## Files

{files}

## Provenance

- `prefill_router_combined.pt` is the active prefill router checkpoint used by
  `configs/combined-pool.yaml`. It was built from public verified trace labels
  and the callable model pool in that config.
- `session_health_public.pkl` is the active v2 session-health checkpoint. Its
  embedded report points at `data_public/session-health-windows-v2.csv`, built
  from public SWE-bench and TerminalBench Hugging Face traces.
- `embedding/complexity_model/` is the optional embedding-router classifier bundle
  trained from public WildChat conversations labeled by an OpenAI judge. It is
  used only when routing through the `embedding-routed` alias.

## Use

From the source repo:

```bash
just run-router
just goose-instructions
```

Or directly:

```bash
ROUTER_ARTIFACT_REPO={manifest["repo_id"]} ./scripts/run-public-router.sh
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


def download_file(
    repo_id: str,
    path: str,
    expected_sha256: str | None = None,
    *,
    out_path: Path | None = None,
) -> None:
    out = out_path or Path(path)
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
    if out.as_posix() == path:
        print(f"wrote   {path}")
    else:
        print(f"wrote   {path} -> {out}")


def artifacts_for_mode(mode: str) -> list[Artifact]:
    if mode == "all":
        return ARTIFACTS
    if mode == "embedding":
        return [artifact for artifact in ARTIFACTS if artifact.runtime_mode == "embedding"]
    return RUNTIME_ARTIFACTS


def remote_info_for_artifact(
    expected: dict[str, dict[str, Any]], artifact: Artifact
) -> tuple[str, dict[str, Any]]:
    info = expected.get(artifact.path)
    if info:
        return artifact.path, info
    prefix = f"{EMBEDDING_BUNDLE_PREFIX}/"
    if artifact.path.startswith(prefix):
        legacy_path = f"{LEGACY_EMBEDDING_BUNDLE_PREFIX}/" + artifact.path[len(prefix) :]
        info = expected.get(legacy_path)
        if info:
            return legacy_path, info
    return artifact.path, {}


def download(repo_id: str, *, mode: str = "default") -> None:
    manifest = fetch_json(repo_id, "artifact-manifest.json")
    expected = {item["path"]: item for item in (manifest or {}).get("artifacts", [])}
    for artifact in artifacts_for_mode(mode):
        remote_path, info = remote_info_for_artifact(expected, artifact)
        download_file(
            repo_id,
            remote_path,
            info.get("sha256"),
            out_path=local_path_for_artifact(artifact),
        )


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
        upload_items.extend(
            (local_path_for_artifact(Artifact(item["path"], item["description"])), item["path"])
            for item in manifest["artifacts"]
        )

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

            Run the router:

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
                "models": [
                  {{"name": "nvidia-routed", "context_limit": 200000}},
                  {{"name": "embedding-routed", "context_limit": 200000}}
                ]
              }}

              GOOSE_PROVIDER=router GOOSE_MODEL=nvidia-routed goose
              GOOSE_PROVIDER=router GOOSE_MODEL=embedding-routed goose

            Quick one-shot comparison:

              curl -s -X POST http://localhost:{port}/savings/reset
              source ./goose-prefill-config
              goose run --name prefill-router-smoke -t "Reply with exactly: prefill ready"
              source ./goose-embedding-config
              goose run --name embedding-router-smoke -t "Reply with exactly: embedding ready"
              curl -s http://localhost:{port}/savings | python3 -m json.tool

            Interactive Goose:

              source ./goose-prefill-config
              goose

              source ./goose-embedding-config
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
    download_parser.add_argument(
        "--mode",
        choices=["default", "embedding", "all"],
        default="default",
        help="artifact set to download; default skips optional embedding bundle",
    )

    download_embedding_parser = sub.add_parser(
        "download-embedding",
        help="download optional embedding-router bundle",
    )
    download_embedding_parser.add_argument(
        "--repo",
        default=None,
        help=f"HF repo id; default {DEFAULT_REPO}",
    )

    upload_parser = sub.add_parser("upload", help="upload active public artifacts")
    upload_parser.add_argument(
        "--private",
        action="store_true",
        help="create/update a private repo",
    )
    upload_parser.add_argument("--repo", default=None, help=f"HF repo id; default {DEFAULT_REPO}")

    instructions_parser = sub.add_parser("goose-instructions", help="print Goose setup commands")
    instructions_parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "4000")),
    )
    instructions_parser.add_argument(
        "--repo",
        default=None,
        help=f"HF repo id; default {DEFAULT_REPO}",
    )

    args = parser.parse_args()
    repo_id = repo_from_args(getattr(args, "repo", None))
    if args.command == "download":
        download(repo_id, mode=args.mode)
    elif args.command == "download-embedding":
        download(repo_id, mode="embedding")
    elif args.command == "upload":
        upload(repo_id, private=args.private)
    elif args.command == "goose-instructions":
        goose_instructions(args.port, repo_id)


if __name__ == "__main__":
    main()
