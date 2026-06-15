"""OpenAI-compatible chat completions endpoint."""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from model_router_toolkit.router import extract_user_text

router = APIRouter()


async def _handle_completion(request: Request, body: dict) -> JSONResponse | StreamingResponse:
    litellm_router = request.app.state.litellm_router
    strategy = request.app.state.strategy
    config = request.app.state.config

    messages = body.get("messages", [])
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 4096)

    if "tolerance" in body:
        strategy.set_request_tolerance(float(body.get("tolerance", 0.20)))

    # Use the first model name from config as the LiteLLM model group.
    # The routing strategy intercepts the call and picks the actual deployment.
    model_group = config.models[0].name if config.models else "default"

    metadata: dict[str, Any] = {}
    if "models" in body:
        metadata["models"] = body["models"]
    request_id = uuid.uuid4().hex
    metadata["router_request_id"] = request_id

    kwargs: dict[str, Any] = {
        "model": model_group,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    kwargs["metadata"] = metadata

    if stream:

        async def sse_stream():
            response_stream = await litellm_router.acompletion(**kwargs)
            async for chunk in response_stream:
                data = chunk.model_dump(exclude_none=True)
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    response = await litellm_router.acompletion(**kwargs)

    result_data = response.model_dump(exclude_none=True)

    for choice in result_data.get("choices", []):
        msg = choice.get("message", {})
        if "content" not in msg:
            msg["content"] = ""

    result = None
    pop_result = getattr(strategy, "pop_result", None)
    if callable(pop_result):
        result = pop_result(request_id)
    if result is None:
        result = strategy.last_result

    if result:
        result_data["routing"] = {
            "selected_model": result.selected_model,
            "confidences": dict(
                zip(
                    result.model_names,
                    result.confidences,
                )
            ),
            "metadata": result.metadata,
        }

    from model_router_toolkit import telemetry

    if telemetry.enabled() and result:
        user_text = extract_user_text(messages)
        telemetry.log_chat(
            session_id=None,
            question=user_text,
            selected_model=result.selected_model,
        )

    return JSONResponse(content=result_data)


@router.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    return await _handle_completion(request, body)


@router.post("/route")
async def route_only(request: Request):
    """Return the LiteLLM strategy's routing decision without calling a provider.

    This intentionally exercises the same strategy path as /chat/completions
    (task-view shaping, pins, policy escalation, switching gates, and route
    logging) but stops before LiteLLM sends the request upstream.
    """

    body = await request.json()
    strategy = request.app.state.strategy
    config = request.app.state.config

    messages = body.get("messages", [])
    if "tolerance" in body:
        strategy.set_request_tolerance(float(body.get("tolerance", 0.20)))

    metadata: dict[str, Any] = {}
    body_metadata = body.get("metadata") or {}
    if isinstance(body_metadata, dict):
        metadata.update(body_metadata)
    if "models" in body:
        metadata["models"] = body["models"]
    request_id = f"probe-{uuid.uuid4().hex}"
    metadata["router_request_id"] = request_id

    model_group = body.get("model") or "nvidia-routed"
    dep = strategy.get_available_deployment(
        model=model_group,
        messages=messages,
        request_kwargs={"metadata": metadata},
    )
    result = strategy.pop_result(request_id) or strategy.last_result
    selected = result.selected_model if result else dep.get("model_name")

    litellm_params = dep.get("litellm_params") or {}
    display = {
        model.name: (model.display_name or model.name)
        for model in getattr(config, "models", [])
    }
    response = {
        "id": request_id,
        "object": "router.route",
        "model": model_group,
        "selected_model": selected,
        "selected_display": display.get(selected, selected),
        "litellm_model": litellm_params.get("model"),
        "deployment": dep.get("model_name"),
        "routing": None,
    }
    if result is not None:
        response["routing"] = {
            "selected_model": result.selected_model,
            "confidences": dict(zip(result.model_names, result.confidences)),
            "metadata": result.metadata,
        }
    return JSONResponse(content=response)
