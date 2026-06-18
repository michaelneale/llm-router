"""Embedding-router bundle scoring for the LiteLLM proxy experiment.

It loads the portable public WildChat-trained bundle: tokenizer.json,
embedder.onnx, weights.safetensors, and config.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ANCHOR_MARKER = ">>>"
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_ARTIFACT_REPO = "micdn/llm-router-goose-public"
DEFAULT_ARTIFACT_PREFIX = "embedding/complexity_model"
MINIMUM_BUNDLE_FILES = (
    "config.json",
    "embedder.onnx",
    "tokenizer.json",
    "weights.safetensors",
)


@dataclass(frozen=True)
class EmbeddingScore:
    complexity: float
    tool_calls_norm: float
    elapsed_ms: int
    rendered: str


def default_bundle_dir() -> Path:
    return Path(os.environ.get("ROUTER_EMBEDDING_BUNDLE", "~/.goose/complexity_model")).expanduser()


def default_artifact_repo() -> str:
    return (
        os.environ.get("ROUTER_EMBEDDING_ARTIFACT_REPO")
        or os.environ.get("ROUTER_ARTIFACT_REPO")
        or DEFAULT_ARTIFACT_REPO
    )


def default_artifact_prefixes() -> list[str]:
    configured = os.environ.get("ROUTER_EMBEDDING_ARTIFACT_PREFIX")
    if configured:
        return [configured.strip("/")]
    return [DEFAULT_ARTIFACT_PREFIX]


def default_artifact_prefix() -> str:
    return os.environ.get("ROUTER_EMBEDDING_ARTIFACT_PREFIX", DEFAULT_ARTIFACT_PREFIX).strip("/")


def render_messages_for_embedding(messages: list[dict] | None) -> str:
    """Render OpenAI-format messages like the embedding training/runtime contract.

    We keep user/assistant turns through the last user turn and mark that final
    user turn with ``>>>``. Tool/system messages are omitted for this first proxy
    spike, matching the minimal Goose-side runtime.
    """
    if not messages:
        return ""

    anchor_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user" and _content_text(messages[i].get("content")):
            anchor_idx = i
            break
    if anchor_idx is None:
        return ""

    lines: list[str] = []
    for i, msg in enumerate(messages[: anchor_idx + 1]):
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = _content_text(msg.get("content")).strip()
        if not text:
            continue
        prefix = f"{ANCHOR_MARKER} " if i == anchor_idx else ""
        lines.append(f"{prefix}{role}: {text}")
    return "\n".join(lines)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return " ".join(parts)
    return ""


class EmbeddingComplexityScorer:
    """Lazy ONNX + NumPy scorer for a portable embedding-router bundle."""

    def __init__(self, bundle_dir: str | Path | None = None):
        self.bundle_dir = Path(bundle_dir or default_bundle_dir()).expanduser()
        self._loaded = False
        self._session = None
        self._tokenizer = None
        self._input_names: set[str] = set()
        self._layers: list[tuple[np.ndarray, np.ndarray]] = []
        self._output_dim = 2
        self._embedder_dim = 0

    def score_messages(self, messages: list[dict] | None) -> EmbeddingScore:
        rendered = render_messages_for_embedding(messages)
        if not rendered:
            raise ValueError("Embedding router could not find a user turn to score")
        return self.score(rendered)

    def score(self, rendered: str) -> EmbeddingScore:
        self._ensure_loaded()
        started = time.perf_counter()
        emb = self._embed(rendered)
        x = emb
        last = len(self._layers) - 1
        for i, (weight, bias) in enumerate(self._layers):
            x = weight @ x + bias
            if i != last:
                x = np.maximum(x, 0.0)
        x = 1.0 / (1.0 + np.exp(-x))
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return EmbeddingScore(
            complexity=float(x[0]),
            tool_calls_norm=float(x[1]) if x.shape[0] > 1 else 0.0,
            elapsed_ms=elapsed_ms,
            rendered=rendered,
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        self._ensure_bundle_present()

        try:
            import onnxruntime as ort
            from safetensors.numpy import load_file as safe_load
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Embedding routing requires onnxruntime, tokenizers, and safetensors. "
                "Install the proxy extras or run: .venv/bin/pip install onnxruntime"
            ) from exc

        cfg_path = self.bundle_dir / "config.json"
        with cfg_path.open() as f:
            cfg = json.load(f)

        if int(cfg.get("format_version", 0)) != 1:
            raise ValueError(f"unsupported embedding bundle format: {cfg.get('format_version')}")

        embedder = cfg["embedder"]
        head = cfg["head"]
        self._embedder_dim = int(embedder["output_dim"])
        self._output_dim = int(head.get("output_dim", 2))

        tokenizer = Tokenizer.from_file(str(self.bundle_dir / embedder["tokenizer_file"]))
        tokenizer.enable_truncation(max_length=DEFAULT_MAX_SEQ_LEN)
        self._tokenizer = tokenizer

        self._session = ort.InferenceSession(
            str(self.bundle_dir / embedder["onnx_file"]),
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {inp.name for inp in self._session.get_inputs()}

        weights = safe_load(str(self.bundle_dir / "weights.safetensors"))
        layers: list[tuple[np.ndarray, np.ndarray]] = []
        prev = int(head["input_dim"])
        sequential_idx = 0
        for hidden in head.get("hidden_dims", []):
            hidden = int(hidden)
            layers.append(
                (
                    _as_f32(weights[f"trunk.{sequential_idx}.weight"], (hidden, prev)),
                    _as_f32(weights[f"trunk.{sequential_idx}.bias"], (hidden,)),
                )
            )
            prev = hidden
            sequential_idx += 3
        layers.append(
            (
                _as_f32(weights["out.weight"], (self._output_dim, prev)),
                _as_f32(weights["out.bias"], (self._output_dim,)),
            )
        )
        self._layers = layers
        self._loaded = True

    def _ensure_bundle_present(self) -> None:
        if _has_minimum_bundle_files(self.bundle_dir):
            return
        if _download_disabled():
            return
        _download_public_bundle(self.bundle_dir)

    def _embed(self, text: str) -> np.ndarray:
        assert self._tokenizer is not None
        assert self._session is not None

        encoding = self._tokenizer.encode(text)
        ids = np.asarray(encoding.ids, dtype=np.int64)[None, :]
        mask = np.asarray(encoding.attention_mask, dtype=np.int64)[None, :]
        feed = {
            "input_ids": ids,
            "attention_mask": mask,
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(ids, dtype=np.int64)

        outputs = self._session.run(None, feed)
        output_names = [out.name for out in self._session.get_outputs()]
        try:
            last_hidden = outputs[output_names.index("last_hidden_state")]
        except ValueError:
            last_hidden = outputs[0]

        if last_hidden.ndim != 3:
            raise ValueError(f"expected 3D embedder output, got {last_hidden.shape}")
        if last_hidden.shape[2] != self._embedder_dim:
            raise ValueError(
                f"embedder hidden dim {last_hidden.shape[2]} != config {self._embedder_dim}"
            )

        weights = mask.astype(np.float32)
        denom = float(weights.sum())
        if denom <= 0.0:
            raise ValueError("attention mask is all zeros")
        pooled = (last_hidden[0] * weights[0, :, None]).sum(axis=0) / denom
        return pooled.astype(np.float32, copy=False)


def _as_f32(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"expected tensor shape {shape}, got {arr.shape}")
    return arr


def _download_disabled() -> bool:
    value = os.environ.get("ROUTER_EMBEDDING_DISABLE_DOWNLOAD", "")
    return value.lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _has_minimum_bundle_files(bundle_dir: Path) -> bool:
    return all((bundle_dir / name).exists() for name in MINIMUM_BUNDLE_FILES)


def _hf_token() -> str:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or ""
    )


def _hf_url(repo_id: str, path: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{path}"


def _request(url: str) -> urllib.request.Request:
    headers = {"User-Agent": "llm-router-embedding-fetch/1"}
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_json(repo_id: str, path: str) -> dict[str, Any]:
    with urllib.request.urlopen(_request(_hf_url(repo_id, path)), timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_public_bundle(bundle_dir: Path) -> None:
    repo_id = default_artifact_repo()
    try:
        manifest = _download_json(repo_id, "artifact-manifest.json")
    except urllib.error.HTTPError as exc:
        raise FileNotFoundError(
            f"Embedding bundle missing at {bundle_dir}; could not fetch artifact manifest "
            f"from {repo_id}: HTTP {exc.code}"
        ) from exc

    artifacts = []
    selected_prefix = ""
    for prefix in default_artifact_prefixes():
        prefix_with_slash = f"{prefix}/"
        artifacts = [
            item
            for item in manifest.get("artifacts", [])
            if item.get("runtime_mode") == "embedding"
            and str(item.get("path", "")).startswith(prefix_with_slash)
        ]
        if artifacts:
            selected_prefix = prefix
            break
    if not artifacts:
        raise FileNotFoundError(
            f"Embedding bundle missing at {bundle_dir}; manifest {repo_id} has no "
            "embedding bundle files"
        )

    bundle_dir.mkdir(parents=True, exist_ok=True)
    prefix_with_slash = f"{selected_prefix}/"
    for item in artifacts:
        remote_path = str(item["path"])
        rel = remote_path[len(prefix_with_slash) :]
        out = bundle_dir / rel
        expected_sha = item.get("sha256")
        if out.exists() and expected_sha and _sha256(out) == expected_sha:
            continue

        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=out.name + ".",
            suffix=".tmp",
            dir=str(out.parent),
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            try:
                with urllib.request.urlopen(
                    _request(_hf_url(repo_id, remote_path)),
                    timeout=300,
                ) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise

        if expected_sha and _sha256(tmp_path) != expected_sha:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch for downloaded embedding artifact {remote_path}")
        os.replace(tmp_path, out)

    if not _has_minimum_bundle_files(bundle_dir):
        raise FileNotFoundError(f"downloaded embedding bundle is incomplete: {bundle_dir}")
