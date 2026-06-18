"""Strategy injection hook for the LiteLLM Proxy.

Imports the litellm proxy's FastAPI app, registers a startup event that
patches the proxy's internal Router with our ModelRoutingStrategy, then
runs the server via uvicorn.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from starlette.requests import Request

logger = logging.getLogger(__name__)

_MIN_LITELLM_VERSION = "1.50.0"


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _env_falsey(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"0", "false", "no", "off"}


def _routing_cost_blend(model: Any, output_token_weight: float) -> float:
    multiplier = float(getattr(model, "routing_cost_multiplier", 1.0) or 1.0)
    return round(
        (
            float(getattr(model, "cost_per_m_input_tokens", 0.0) or 0.0)
            + output_token_weight * float(getattr(model, "cost_per_m_output_tokens", 0.0) or 0.0)
        )
        * multiplier,
        6,
    )


def _model_knob(model: Any, output_token_weight: float) -> dict[str, Any]:
    internal_name = getattr(model, "name", "")
    configured_display = getattr(model, "display_name", "")
    provider_model = getattr(model, "litellm_model", "")
    display_name = (
        configured_display
        if configured_display and configured_display != internal_name
        else provider_model or configured_display or internal_name
    )
    return {
        "display_name": display_name,
        "model_id": provider_model or display_name,
        "provider_model": provider_model,
        "litellm_model": provider_model,
        "input_cost": float(getattr(model, "cost_per_m_input_tokens", 0.0) or 0.0),
        "output_cost": float(getattr(model, "cost_per_m_output_tokens", 0.0) or 0.0),
        "routing_cost_multiplier": float(getattr(model, "routing_cost_multiplier", 1.0) or 1.0),
        "routing_blend": _routing_cost_blend(model, output_token_weight),
    }


def _public_model_ladder(
    models_with_slots: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Collapse trained-slot duplicates into the real provider models users see."""
    ladder: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, knob in models_with_slots:
        key = str(knob.get("provider_model") or knob.get("model_id") or "")
        if key in seen:
            continue
        seen.add(key)
        ladder.append(knob)
    return ladder


def _recommended_tolerances() -> list[dict[str, Any]]:
    return [
        {
            "label": "Quality",
            "value": 0.040,
            "note": "equivalent-quality check",
        },
        {
            "label": "Small loss",
            "value": 0.110,
            "note": "conservative savings",
        },
        {
            "label": "Aggressive",
            "value": 0.145,
            "note": "current high-savings trial",
        },
        {
            "label": "Cheaper",
            "value": 0.180,
            "note": "explore lower cost",
        },
    ]


def _cache_pin_modes() -> list[dict[str, Any]]:
    return [
        {
            "label": "Off",
            "mode": "off",
            "note": "route non-decision turns",
        },
        {
            "label": "Dear only",
            "mode": "dear_only",
            "note": "pin only costly incumbents",
        },
        {
            "label": "All",
            "mode": "all",
            "note": "old sticky-cache behavior",
        },
    ]


def _build_routing_knobs(
    pool: Any,
    *,
    router_config: str,
    litellm_config: str,
) -> dict[str, Any]:
    routing = getattr(pool, "routing", None)
    switching = getattr(pool, "switching", None)
    escalation = getattr(pool, "escalation", None)
    session_health = getattr(pool, "session_health", None)
    utility = getattr(pool, "utility", None)

    configured_tolerance = float(getattr(routing, "tolerance", 0.0) or 0.0)
    env_tolerance = os.environ.get("ROUTER_TOLERANCE")
    try:
        effective_tolerance = (
            float(env_tolerance) if env_tolerance not in (None, "") else configured_tolerance
        )
    except ValueError:
        effective_tolerance = configured_tolerance

    output_weight = float(getattr(routing, "output_token_weight", 0.0) or 0.0)
    tolerance_source = "env" if env_tolerance not in (None, "") else "config"
    models_with_slots = sorted(
        (
            (getattr(model, "name", ""), _model_knob(model, output_weight))
            for model in getattr(pool, "models", [])
        ),
        key=lambda item: (
            item[1]["routing_blend"],
            item[1]["input_cost"],
            item[1]["output_cost"],
            item[0],
        ),
    )
    by_slot = {slot: knob for slot, knob in models_with_slots}
    top_slot = getattr(escalation, "top_tier_model", "") if escalation else ""

    switching_disabled = _env_truthy("ROUTER_DISABLE_SWITCHING") or _env_falsey("ROUTER_SWITCHING")
    switching_configured = bool(getattr(switching, "enabled", False))
    cache_pin_mode = os.environ.get(
        "ROUTER_CACHE_PINNING",
        getattr(switching, "cache_pin_mode", "off") if switching else "off",
    )
    cache_pin_min_blend = float(
        os.environ.get(
            "ROUTER_CACHE_PIN_MIN_BLEND",
            getattr(switching, "cache_pin_min_blend", 3.0) if switching else 3.0,
        )
        or 0.0
    )

    patterns = (
        list(getattr(escalation, "force_top_tier_when_prompt_matches", []) or [])
        if escalation
        else []
    )
    return {
        "pool_config": router_config,
        "litellm_config": litellm_config,
        "route_log": os.environ.get("ROUTER_ROUTE_LOG", ""),
        "configured_tolerance": configured_tolerance,
        "env_tolerance": env_tolerance or "",
        "effective_tolerance": max(0.0, min(1.0, effective_tolerance)),
        "startup_tolerance": max(0.0, min(1.0, effective_tolerance)),
        "tolerance_source": tolerance_source,
        "recommended_tolerances": _recommended_tolerances(),
        "output_token_weight": output_weight,
        "switching": {
            "configured": switching_configured,
            "effective": switching_configured and not switching_disabled,
            "disabled_by_env": switching_disabled,
            "up_margin": float(getattr(switching, "up_margin", 0.0) or 0.0),
            "down_margin": float(getattr(switching, "down_margin", 0.0) or 0.0),
            "down_margin_per_100k": float(getattr(switching, "down_margin_per_100k", 0.0) or 0.0),
            "max_down_margin": float(getattr(switching, "max_down_margin", 0.0) or 0.0),
        },
        "cache_pinning": {
            "mode": str(cache_pin_mode or "off"),
            "configured_mode": str(cache_pin_mode or "off"),
            "min_blend": cache_pin_min_blend,
            "recommended_modes": _cache_pin_modes(),
        },
        "top_tier": by_slot.get(top_slot),
        "manual_override": "!hard" if any("!hard" in p for p in patterns) else "",
        "session_health": {
            "configured": bool(session_health and getattr(session_health, "enabled", False)),
            "checkpoint": (
                str(getattr(session_health, "checkpoint", "")) if session_health else ""
            ),
            "threshold": float(
                os.environ.get(
                    "ROUTER_SESSION_HEALTH_THRESHOLD",
                    getattr(session_health, "threshold", 0.0) if session_health else 0.0,
                )
                or 0.0
            ),
            "top_tier_model": (
                getattr(session_health, "top_tier_model", "") if session_health else ""
            ),
            "disabled_by_env": os.environ.get("ROUTER_SESSION_HEALTH", "").lower()
            in {"0", "false", "no", "off"},
        },
        "escalation_patterns": patterns,
        "cheap_patterns": (
            list(getattr(utility, "cheap_when_prompt_matches", []) or []) if utility else []
        ),
        "models": _public_model_ladder(models_with_slots),
    }


def _check_proxy_available() -> None:
    """Verify litellm[proxy] is installed with a compatible version."""
    try:
        import litellm
    except ImportError:
        raise ImportError(
            "litellm is not installed. Run: pip install 'model-router-toolkit[proxy]'"
        ) from None

    from packaging.version import Version

    try:
        if Version(litellm.__version__) < Version(_MIN_LITELLM_VERSION):
            logger.warning(
                "litellm %s is older than the tested minimum %s — "
                "proxy integration may not work correctly",
                litellm.__version__,
                _MIN_LITELLM_VERSION,
            )
    except Exception:
        pass

    try:
        from litellm.proxy import proxy_server  # noqa: F401
    except ImportError:
        raise ImportError(
            "litellm proxy extras are not installed. Run: pip install 'litellm[proxy]'"
        ) from None


def _inject_strategy(router_config: str):
    """Patch the litellm proxy's internal Router with our strategy.

    Returns the ModelRoutingStrategy instance so callers can access
    ``last_result`` for response patching.
    """
    import litellm.proxy.proxy_server as proxy_module

    from model_router_toolkit.adapters.litellm.strategy import ModelRoutingStrategy

    llm_router = proxy_module.llm_router
    if llm_router is None:
        raise RuntimeError(
            "litellm proxy did not initialize a Router. "
            "Ensure your litellm config.yaml contains a model_list."
        )

    print("Model Router Toolkit: building strategy from", router_config)
    t0 = time.time()
    strategy = ModelRoutingStrategy.from_config(router_config)
    strategy.set_litellm_router(llm_router)
    llm_router.set_custom_routing_strategy(strategy)
    elapsed = time.time() - t0

    print(f"Model Router Toolkit: strategy registered in {elapsed:.1f}s")
    print(f"  Routing method : {strategy.router.__class__.__name__}")
    print(f"  Tolerance      : {strategy.tolerance}")
    print(f"  Models in proxy: {[d.get('model_name') for d in llm_router.model_list]}")

    try:
        result = strategy.router.route("warmup", tolerance=0.5)
        print(f"  Warmup route   : {result.selected_model}")
    except Exception as e:
        logger.warning("Warmup route failed: %s", e)

    return strategy


def start_proxy(
    litellm_config: str,
    router_config: str,
    *,
    host: str = "0.0.0.0",
    port: int = 4000,
) -> None:
    """Start the litellm proxy with our custom routing strategy injected."""
    _check_proxy_available()

    litellm_config = str(Path(litellm_config).resolve())
    router_config_abs = str(Path(router_config).resolve())

    os.environ["CONFIG_FILE_PATH"] = litellm_config

    from litellm.proxy.proxy_server import app as litellm_app
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response

    from model_router_toolkit.adapters.litellm.savings import (
        extract_usage,
        extract_usage_from_sse,
        get_tracker,
    )

    tracker = get_tracker()
    model_display_names: dict[str, str] = {}
    routing_knobs: dict[str, Any] = {
        "pool_config": router_config_abs,
        "litellm_config": litellm_config,
        "error": "router config was not loaded",
    }
    # Map internal checkpoint slot names -> the real model they call, so the
    # dashboard shows "openai/gpt-5-mini" instead of the cosmetic NVIDIA slot
    # name "gpt-oss-120b-high".
    try:
        from model_router_toolkit.config import load_config

        _pool = load_config(router_config_abs)
        model_display_names = {
            m.name: (m.display_name or m.litellm_model or m.name) for m in _pool.models
        }
        tracker.set_display_names(
            {m.name: (m.litellm_model or m.display_name or m.name) for m in _pool.models}
        )
        tracker.set_model_rates(
            {m.name: (m.cost_per_m_input_tokens, m.cost_per_m_output_tokens) for m in _pool.models}
        )
        routing_knobs = _build_routing_knobs(
            _pool,
            router_config=router_config_abs,
            litellm_config=litellm_config,
        )
    except Exception as e:  # pragma: no cover - display sugar only
        logger.warning("Could not load model rates for dashboard: %s", e)

    _strategy_ref = None

    # litellm >= ~1.60 uses a lifespan context; legacy @app.on_event("startup")
    # handlers are silently ignored when a custom lifespan is set. Wrap the
    # existing lifespan so injection runs after litellm initializes its Router.
    from contextlib import asynccontextmanager

    _original_lifespan = litellm_app.router.lifespan_context

    @asynccontextmanager
    async def _routing_lifespan(app):
        nonlocal _strategy_ref
        async with _original_lifespan(app) as state:
            try:
                _strategy_ref = _inject_strategy(router_config_abs)
            except Exception:
                logger.exception("Failed to inject routing strategy at startup")
            yield state

    litellm_app.router.lifespan_context = _routing_lifespan

    import json as _json

    _COMPLETION_PATHS = (
        "/chat/completions",
        "/completions",
        "/v1/chat/completions",
        "/v1/completions",
    )

    def _content_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ).strip()
        return ""

    def _router_session_id(messages) -> str:
        import hashlib

        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    text = _content_text(msg.get("content")).strip()
                    if text:
                        digest = hashlib.sha256(text[:2000].encode()).hexdigest()[:16]
                        return f"first-user:{digest}"
        return f"request:{uuid.uuid4().hex}"

    class _RouterProxyMiddleware(BaseHTTPMiddleware):
        """Patch the response ``model`` field to reflect the actual routed
        model (litellm echoes the request model name, not the deployment) and
        record real token usage for the savings tracker.

        Handles both non-streaming JSON responses and streaming SSE responses.
        For streaming, we force ``stream_options.include_usage`` on the inbound
        request so the upstream emits a final usage chunk, then tap the SSE
        stream as it passes through to read it — without buffering the whole
        response, so interactive streaming stays live.
        """

        async def dispatch(self, request: Request, call_next) -> Response:
            is_completion = any(request.url.path.endswith(p) for p in _COMPLETION_PATHS)
            request_id: str | None = None

            # 1. For completion requests, force include_usage on streaming so we
            #    can count tokens. We must read + replace the request body.
            if is_completion:
                raw = await request.body()
                new_raw, request_id = _prepare_completion_body(raw)
                if new_raw != raw:
                    # Re-serve the (possibly modified) body to downstream handlers.
                    async def _receive():
                        return {"type": "http.request", "body": new_raw, "more_body": False}

                    request = Request(request.scope, _receive)

            response = await call_next(request)

            if not is_completion:
                return response

            strategy = _strategy_ref
            result = None
            if strategy:
                pop_result = getattr(strategy, "pop_result", None)
                if callable(pop_result):
                    result = pop_result(request_id)
                if result is None:
                    result = getattr(strategy, "last_result", None)
            if result is None:
                return response

            selected = result.selected_model
            response.headers["X-Model-Router-Selected"] = selected
            content_type = response.headers.get("content-type", "")

            # --- Non-streaming JSON ---
            if content_type.startswith("application/json"):
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk if isinstance(chunk, bytes) else chunk.encode()

                try:
                    data = _json.loads(body)
                    data["model"] = selected
                    body = _json.dumps(data).encode()
                except (ValueError, KeyError):
                    pass

                usage = extract_usage(body)
                if usage is not None:
                    tracker.record(result, usage)

                headers = dict(response.headers)
                headers["content-length"] = str(len(body))

                from starlette.responses import Response as StarletteResponse

                return StarletteResponse(
                    content=body,
                    status_code=response.status_code,
                    headers=headers,
                    media_type=response.media_type,
                )

            # --- Streaming SSE: pass chunks through live, tap for usage ---
            if "text/event-stream" in content_type:
                from starlette.responses import StreamingResponse

                upstream = response.body_iterator

                async def _tap():
                    buf_parts: list[str] = []
                    async for chunk in upstream:
                        text = chunk.decode() if isinstance(chunk, bytes) else chunk
                        buf_parts.append(text)
                        yield chunk
                    # Stream finished — extract usage from the accumulated SSE.
                    usage = extract_usage_from_sse("".join(buf_parts))
                    if usage is not None:
                        tracker.record(result, usage)

                headers = dict(response.headers)
                headers.pop("content-length", None)
                return StreamingResponse(
                    _tap(),
                    status_code=response.status_code,
                    headers=headers,
                    media_type=response.media_type,
                )

            return response

    def _prepare_completion_body(raw: bytes) -> tuple[bytes, str | None]:
        """Attach router request/session metadata and include streaming usage.

        Returns ``(body, request_id)``.
        """
        if not raw:
            return raw, None
        try:
            data = _json.loads(raw)
        except (ValueError, TypeError):
            return raw, None
        if not isinstance(data, dict):
            return raw, None

        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        request_id = str(metadata.get("router_request_id") or uuid.uuid4().hex)
        metadata["router_request_id"] = request_id
        metadata.setdefault("router_session_id", _router_session_id(data.get("messages")))
        if str(data.get("model") or "") == "embedding-routed":
            metadata["router_mode"] = "embedding"
        data["metadata"] = metadata
        data["litellm_metadata"] = dict(metadata)

        if data.get("stream"):
            opts = data.get("stream_options")
            if not isinstance(opts, dict):
                opts = {}
            opts["include_usage"] = True
            data["stream_options"] = opts

        return _json.dumps(data).encode(), request_id

    litellm_app.add_middleware(_RouterProxyMiddleware)

    from starlette.responses import HTMLResponse, JSONResponse

    from model_router_toolkit.adapters.litellm.dashboard import DASHBOARD_HTML

    def _current_routing_knobs() -> dict[str, Any]:
        knobs = dict(routing_knobs)
        strategy = _strategy_ref
        if strategy is None or "error" in knobs:
            return knobs
        current = float(strategy.tolerance)
        startup = float(knobs.get("startup_tolerance", current) or current)
        knobs["effective_tolerance"] = current
        knobs["runtime_tolerance"] = current
        if abs(current - startup) > 1e-9:
            knobs["tolerance_source"] = "runtime"
        cache_pinning = dict(knobs.get("cache_pinning") or {})
        cache_pinning["mode"] = getattr(strategy, "cache_pin_mode", "off")
        cache_pinning["min_blend"] = float(
            getattr(strategy, "cache_pin_min_blend", cache_pinning.get("min_blend", 3.0)) or 0.0
        )
        cache_pinning.setdefault("recommended_modes", _cache_pin_modes())
        knobs["cache_pinning"] = cache_pinning
        return knobs

    @litellm_app.get("/savings")
    async def _savings():  # noqa: ANN202
        snapshot = tracker.snapshot()
        snapshot["routing_knobs"] = _current_routing_knobs()
        return JSONResponse(snapshot)

    @litellm_app.post("/savings/reset")
    async def _savings_reset():  # noqa: ANN202
        tracker.reset()
        return JSONResponse({"status": "reset"})

    @litellm_app.get("/router/tuning")
    async def _router_tuning():  # noqa: ANN202
        return JSONResponse({"routing_knobs": _current_routing_knobs()})

    @litellm_app.post("/router/tuning")
    async def _router_tuning_update(request: Request):  # noqa: ANN202
        strategy = _strategy_ref
        if strategy is None:
            return JSONResponse(
                {"error": "routing strategy is not initialized"},
                status_code=503,
            )
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}

        if "tolerance" in body:
            try:
                tolerance = float(body["tolerance"])
            except (TypeError, ValueError):
                return JSONResponse(
                    {"error": "tolerance must be a number"},
                    status_code=400,
                )
            if not 0.0 <= tolerance <= 1.0:
                return JSONResponse(
                    {"error": "tolerance must be between 0 and 1"},
                    status_code=400,
                )
            strategy.tolerance = tolerance

        cache_pin_mode = body.get("cache_pin_mode")
        if cache_pin_mode is None and isinstance(body.get("cache_pinning"), dict):
            cache_pin_mode = body["cache_pinning"].get("mode")
        if cache_pin_mode is not None:
            try:
                strategy.cache_pin_mode = str(cache_pin_mode)
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)

        cache_pin_min_blend = body.get("cache_pin_min_blend")
        if cache_pin_min_blend is None and isinstance(body.get("cache_pinning"), dict):
            cache_pin_min_blend = body["cache_pinning"].get("min_blend")
        if cache_pin_min_blend is not None:
            try:
                strategy.cache_pin_min_blend = float(cache_pin_min_blend)
            except (TypeError, ValueError):
                return JSONResponse(
                    {"error": "cache_pin_min_blend must be a number"},
                    status_code=400,
                )

        return JSONResponse({"routing_knobs": _current_routing_knobs()})

    @litellm_app.post("/router/route")
    async def _route_probe(request: Request):  # noqa: ANN202
        """Route without calling a provider.

        This exercises the same LiteLLM routing strategy as chat completions,
        including task-view shaping, pins, policy escalation, switch gates, and
        ROUTER_ROUTE_LOG writes. Request ids are prefixed with ``probe-`` so
        aggregate real-traffic analysis can ignore these rows by default.
        """

        strategy = _strategy_ref
        if strategy is None:
            return JSONResponse(
                {"error": "routing strategy is not initialized"},
                status_code=503,
            )

        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}

        messages = body.get("messages") or []
        if "tolerance" in body:
            strategy.set_request_tolerance(float(body.get("tolerance", 0.20)))

        metadata = body.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        else:
            metadata = dict(metadata)
        if "models" in body:
            metadata["models"] = body["models"]
        request_id = f"probe-{uuid.uuid4().hex}"
        metadata["router_request_id"] = request_id
        metadata.setdefault("router_session_id", _router_session_id(messages))

        dep = strategy.get_available_deployment(
            model=body.get("model") or "nvidia-routed",
            messages=messages,
            request_kwargs={"metadata": metadata},
        )
        result = strategy.pop_result(request_id) or strategy.last_result
        selected = result.selected_model if result is not None else dep.get("model_name")
        litellm_params = dep.get("litellm_params") or {}
        response = {
            "id": request_id,
            "object": "router.route",
            "selected_model": selected,
            "selected_display": model_display_names.get(selected, selected),
            "deployment": dep.get("model_name"),
            "litellm_model": litellm_params.get("model"),
            "routing": None,
        }
        if result is not None:
            response["routing"] = {
                "selected_model": result.selected_model,
                "confidences": dict(zip(result.model_names, result.confidences)),
                "metadata": result.metadata,
            }
        return JSONResponse(response)

    @litellm_app.get("/dashboard")
    async def _dashboard():  # noqa: ANN202
        return HTMLResponse(DASHBOARD_HTML)

    import uvicorn

    print("\nStarting LiteLLM Proxy with Model Router Toolkit")
    print(f"  LiteLLM config : {litellm_config}")
    print(f"  Router config  : {router_config_abs}")
    print(f"  Listening on   : http://{host}:{port}\n")

    uvicorn.run(litellm_app, host=host, port=port)
