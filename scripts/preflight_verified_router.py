#!/usr/bin/env python3
"""Local preflight checks for the verified Goose router.

No provider calls are made. The checks cover the failure modes that can make a
real Goose trial unusable before quality/savings can even be measured:

  - pool slots mapped to provider model IDs absent from Goose inventory
  - generated LiteLLM config drifting away from the pool config
  - current OpenAI-compatible base URL pointing back at the local router
  - live proxy missing the nvidia-routed alias or baseline metadata
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

from model_router_toolkit.config import load_config

DB_DEFAULT = Path.home() / ".local/share/goose/sessions/sessions.db"


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"FAIL: {message}")


def warn(message: str) -> None:
    print(f"WARN: {message}")


def ok(message: str) -> None:
    print(f"OK:   {message}")


def inventory_models(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        warn(f"Goose inventory DB not found: {path}")
        return {}
    uri = f"file:{path}?mode=ro&immutable=1"
    db = sqlite3.connect(uri, uri=True)
    rows = db.execute(
        """
        SELECT e.provider_family, m.model_id
        FROM provider_inventory_entries e
        JOIN provider_inventory_models m ON e.inventory_key=m.inventory_key
        """
    ).fetchall()
    out: dict[str, set[str]] = {}
    for provider, model in rows:
        out.setdefault(str(provider), set()).add(str(model))
    return out


def split_litellm_model(value: str) -> tuple[str, str]:
    if "/" not in value:
        return "", value
    provider, model = value.split("/", 1)
    return provider, model


def check_pool_inventory(
    config_path: str,
    inventory: dict[str, set[str]],
    failures: list[str],
) -> None:
    cfg = load_config(config_path)
    if not inventory:
        warn("Skipping provider inventory checks because no inventory was loaded.")
        return

    provider_alias = {
        "anthropic": "anthropic",
        "openai": "openai",
    }
    missing = []
    for model in cfg.models:
        litellm_model = model.litellm_model or ""
        provider, model_id = split_litellm_model(litellm_model)
        family = provider_alias.get(provider)
        if not family:
            warn(f"{model.name}: cannot inventory-check provider mapping {litellm_model!r}")
            continue
        if model_id not in inventory.get(family, set()):
            missing.append(f"{model.name} -> {litellm_model}")

    if missing:
        fail(
            "pool maps slots to model IDs absent from Goose inventory: " + "; ".join(missing),
            failures,
        )
    else:
        ok("all OpenAI/Anthropic pool mappings appear in Goose provider inventory")


def check_litellm_config(pool_path: str, litellm_path: str, failures: list[str]) -> None:
    pool = load_config(pool_path)
    try:
        with open(litellm_path) as f:
            data = yaml.safe_load(f) or {}
    except OSError as exc:
        fail(f"could not read LiteLLM config {litellm_path}: {exc}", failures)
        return

    routed_aliases = {"nvidia-routed", "embedding-routed"}
    actual = {
        row.get("model_name"): (row.get("litellm_params") or {}).get("model")
        for row in data.get("model_list", [])
        if row.get("model_name") not in routed_aliases
    }
    expected = {m.name: m.litellm_model for m in pool.models}
    drift = [
        f"{slot}: expected {expected_model}, got {actual.get(slot)}"
        for slot, expected_model in expected.items()
        if actual.get(slot) != expected_model
    ]
    missing = [slot for slot in expected if slot not in actual]
    if missing:
        drift.extend(f"{slot}: missing from LiteLLM config" for slot in missing)
    if drift:
        fail("LiteLLM config differs from pool config: " + "; ".join(drift), failures)
    else:
        ok("LiteLLM config model mappings match pool config")

    exposed = {row.get("model_name") for row in data.get("model_list", [])}
    if "nvidia-routed" not in exposed:
        warn("LiteLLM config does not expose nvidia-routed")
    else:
        ok("LiteLLM config exposes nvidia-routed")
    if "embedding-routed" not in exposed:
        warn("LiteLLM config does not expose embedding-routed")
    else:
        ok("LiteLLM config exposes embedding-routed")


def check_local_base_url(port: int, failures: list[str]) -> None:
    bad = []
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        value = os.environ.get(name, "")
        if f"localhost:{port}" in value or f"127.0.0.1:{port}" in value:
            bad.append(f"{name}={value}")
    if bad:
        message = (
            "current environment points OpenAI upstream back at the router: "
            + "; ".join(bad)
            + ". run.sh unsets this for the proxy unless ROUTER_PRESERVE_OPENAI_BASE_URL=1."
        )
        if os.environ.get("ROUTER_PRESERVE_OPENAI_BASE_URL") == "1":
            fail(message, failures)
        else:
            warn(message)
    else:
        ok("current shell does not point OpenAI upstream at the local router")


def fetch_json(url: str, timeout: float) -> dict | None:
    try:
        with urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (OSError, URLError, json.JSONDecodeError) as exc:
        warn(f"could not fetch {url}: {exc}")
        return None


def post_json(url: str, payload: dict, timeout: float) -> dict | None:
    body = json.dumps(payload).encode()
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (OSError, HTTPError, URLError, json.JSONDecodeError) as exc:
        warn(f"could not POST {url}: {exc}")
        return None


def check_live_proxy(
    base_url: str,
    timeout: float,
    failures: list[str],
    config_path: str,
) -> None:
    cfg = load_config(config_path)
    trial_tolerance = float(cfg.routing.tolerance)
    cheapest = (
        min(
            cfg.models,
            key=lambda model: (
                model.cost_per_m_input_tokens,
                model.cost_per_m_output_tokens,
                model.name,
            ),
        ).name
        if cfg.models
        else ""
    )

    models = fetch_json(f"{base_url.rstrip('/')}/v1/models", timeout)
    if models is None:
        fail(f"live proxy is not reachable at {base_url.rstrip('/')}/v1/models", failures)
        return
    ids = {row.get("id") for row in models.get("data", []) if isinstance(row, dict)}
    if "nvidia-routed" not in ids:
        fail("live proxy /v1/models does not expose nvidia-routed", failures)
    else:
        ok("live proxy exposes nvidia-routed")

    savings = fetch_json(f"{base_url.rstrip('/')}/savings", timeout)
    if savings is None:
        fail(
            f"live proxy savings endpoint is not reachable at {base_url.rstrip('/')}/savings",
            failures,
        )
        return
    if not savings.get("baseline_model"):
        fail("live proxy /savings has no baseline_model", failures)
    else:
        ok(f"live proxy baseline is {savings['baseline_model']}")
    if "cached_input_tokens" in savings:
        ok("live proxy exposes cache-aware savings fields")
    else:
        warn("live proxy does not expose cached_input_tokens; restart on current code")

    utility_probe = post_json(
        f"{base_url.rstrip('/')}/router/route",
        {
            "model": "nvidia-routed",
            "messages": [{"role": "user", "content": "list files in this directory"}],
            "metadata": {"router_session_id": f"preflight-utility-{uuid.uuid4().hex}"},
            "tolerance": trial_tolerance,
        },
        max(timeout, 30.0),
    )
    if utility_probe is None:
        fail("live proxy provider-free /router/route utility probe failed", failures)
        return
    if not utility_probe.get("selected_model"):
        fail("live proxy /router/route utility probe returned no selected_model", failures)
    elif cheapest and utility_probe.get("selected_model") != cheapest:
        fail(
            "live proxy utility probe did not select cheapest model: "
            f"expected {cheapest}, got {utility_probe.get('selected_model')}",
            failures,
        )
    else:
        ok(f"live proxy utility probe selected {utility_probe['selected_model']}")

    learned_probe = post_json(
        f"{base_url.rstrip('/')}/router/route",
        {
            "model": "nvidia-routed",
            "messages": [
                {
                    "role": "user",
                    "content": "explain how this repo works with mesh-llm",
                }
            ],
            "metadata": {"router_session_id": f"preflight-learned-{uuid.uuid4().hex}"},
            "tolerance": trial_tolerance,
        },
        max(timeout, 30.0),
    )
    if learned_probe is None:
        fail("live proxy provider-free /router/route learned probe failed", failures)
        return
    metadata = (learned_probe.get("routing") or {}).get("metadata") or {}
    if not learned_probe.get("selected_model"):
        fail("live proxy /router/route learned probe returned no selected_model", failures)
    elif metadata.get("pin_reason") == "cheap_utility_pattern":
        fail("live proxy learned probe incorrectly matched cheap utility overlay", failures)
    else:
        ok(f"live proxy learned probe selected {learned_probe['selected_model']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/combined-pool.yaml")
    parser.add_argument("--litellm-config", default="configs/litellm-combined.yaml")
    parser.add_argument("--goose-db", default=str(DB_DEFAULT))
    parser.add_argument("--base-url", default="http://localhost:4000")
    parser.add_argument("--port", type=int, default=4000)
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()

    failures: list[str] = []
    inventory = inventory_models(Path(args.goose_db).expanduser())
    check_pool_inventory(args.config, inventory, failures)
    check_litellm_config(args.config, args.litellm_config, failures)
    check_local_base_url(args.port, failures)
    if not args.skip_live:
        check_live_proxy(args.base_url, args.timeout, failures, args.config)

    if failures:
        print()
        print(f"Preflight failed with {len(failures)} issue(s).")
        sys.exit(1)
    print()
    print("Preflight passed.")


if __name__ == "__main__":
    main()
