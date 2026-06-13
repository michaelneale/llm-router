"""Strategy injection hook for the LiteLLM Proxy.

Imports the litellm proxy's FastAPI app, registers a startup event that
patches the proxy's internal Router with our ModelRoutingStrategy, then
runs the server via uvicorn.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_MIN_LITELLM_VERSION = "1.50.0"


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

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response

    from litellm.proxy.proxy_server import app as litellm_app

    from model_router_toolkit.adapters.litellm.savings import (
        extract_usage,
        extract_usage_from_sse,
        get_tracker,
    )

    tracker = get_tracker()
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

    _COMPLETION_PATHS = ("/chat/completions", "/completions", "/v1/chat/completions",
                         "/v1/completions")

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
            # 1. For completion requests, force include_usage on streaming so we
            #    can count tokens. We must read + replace the request body.
            if any(request.url.path.endswith(p) for p in _COMPLETION_PATHS):
                raw = await request.body()
                new_raw = _force_include_usage(raw)
                if new_raw is not raw:
                    # Re-serve the (possibly modified) body to downstream handlers.
                    async def _receive():
                        return {"type": "http.request", "body": new_raw,
                                "more_body": False}

                    request = Request(request.scope, _receive)

            response = await call_next(request)

            strategy = _strategy_ref
            if not (strategy and getattr(strategy, "last_result", None)):
                return response

            result = strategy.last_result
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
                    tracker.record(result, usage[0], usage[1])

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
                        tracker.record(result, usage[0], usage[1])

                headers = dict(response.headers)
                headers.pop("content-length", None)
                return StreamingResponse(
                    _tap(),
                    status_code=response.status_code,
                    headers=headers,
                    media_type=response.media_type,
                )

            return response

    def _force_include_usage(raw: bytes):
        """If the request is a streaming completion, set
        ``stream_options.include_usage=true`` so the upstream emits usage.

        Returns the original ``raw`` unchanged if no edit is needed, else new bytes.
        """
        if not raw:
            return raw
        try:
            data = _json.loads(raw)
        except (ValueError, TypeError):
            return raw
        if not isinstance(data, dict) or not data.get("stream"):
            return raw
        opts = data.get("stream_options")
        if not isinstance(opts, dict):
            opts = {}
        if opts.get("include_usage") is True:
            return raw
        opts["include_usage"] = True
        data["stream_options"] = opts
        return _json.dumps(data).encode()

    litellm_app.add_middleware(_RouterProxyMiddleware)

    from starlette.responses import HTMLResponse, JSONResponse

    from model_router_toolkit.adapters.litellm.dashboard import DASHBOARD_HTML

    @litellm_app.get("/savings")
    async def _savings():  # noqa: ANN202
        return JSONResponse(tracker.snapshot())

    @litellm_app.post("/savings/reset")
    async def _savings_reset():  # noqa: ANN202
        tracker.reset()
        return JSONResponse({"status": "reset"})

    @litellm_app.get("/dashboard")
    async def _dashboard():  # noqa: ANN202
        return HTMLResponse(DASHBOARD_HTML)

    import uvicorn

    print("\nStarting LiteLLM Proxy with Model Router Toolkit")
    print(f"  LiteLLM config : {litellm_config}")
    print(f"  Router config  : {router_config_abs}")
    print(f"  Listening on   : http://{host}:{port}\n")

    uvicorn.run(litellm_app, host=host, port=port)
