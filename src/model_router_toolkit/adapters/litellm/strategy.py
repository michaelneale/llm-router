"""LiteLLM custom routing strategy wrapping any BaseRouter.

This is the primary integration point. Usage:

    from litellm import Router
    from model_router_toolkit import ModelRoutingStrategy

    router = Router(model_list=my_models)
    strategy = ModelRoutingStrategy.from_config("pool_config.yaml")
    router.set_custom_routing_strategy(strategy)
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import os
import threading
import time
from typing import Any

from model_router_toolkit.router import BaseRouter, RoutingResult, extract_user_text
from model_router_toolkit.task_view import (
    build_task_view,
    is_goose_title_request,
    is_info_only_request,
    strip_info_messages,
)

logger = logging.getLogger(__name__)

EMBEDDING_ROUTED_ALIAS = "embedding-routed"
EMBEDDING_COMPLEXITY_BANDS = (0.15, 0.35, 0.50, 0.54)
SESSION_HEALTH_MIN_EVENTS = 4
SESSION_HEALTH_MIN_AGENTIC_EVENTS = 2
ANTHROPIC_CACHE_CONTROL = {"type": "ephemeral"}
MAX_ANTHROPIC_CACHE_MARKERS = 4

_request_tolerance: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "request_tolerance",
    default=None,
)

_CACHE_PIN_ALIASES = {
    "0": "off",
    "false": "off",
    "no": "off",
    "none": "off",
    "off": "off",
    "1": "all",
    "true": "all",
    "yes": "all",
    "on": "all",
    "all": "all",
    "dear": "dear_only",
    "dear_only": "dear_only",
    "expensive": "dear_only",
    "expensive_only": "dear_only",
}


def _normalize_cache_pin_mode(value: Any) -> str:
    mode = _CACHE_PIN_ALIASES.get(str(value or "off").strip().lower())
    if mode is None:
        raise ValueError("cache_pin_mode must be one of: off, dear_only, all")
    return mode


def inject_anthropic_cache_markers(
    deployment: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
    request_kwargs: dict[str, Any] | None,
) -> int:
    """Mirror Goose's Anthropic prompt-cache markers after routing picks Claude."""
    if not isinstance(deployment, dict):
        return 0
    params = deployment.get("litellm_params") or {}
    if not str(params.get("model") or "").startswith("anthropic/"):
        return 0

    def count(obj: Any) -> int:
        if isinstance(obj, dict):
            return (1 if "cache_control" in obj else 0) + sum(count(v) for v in obj.values())
        if isinstance(obj, list):
            return sum(count(v) for v in obj)
        return 0

    existing = count(messages) + count(request_kwargs)
    added = 0

    def mark(block: Any) -> bool:
        nonlocal added
        if (
            existing + added >= MAX_ANTHROPIC_CACHE_MARKERS
            or not isinstance(block, dict)
            or "cache_control" in block
        ):
            return False
        block["cache_control"] = dict(ANTHROPIC_CACHE_CONTROL)
        added += 1
        return True

    def mark_message(message: dict[str, Any]) -> None:
        nonlocal added
        if existing + added >= MAX_ANTHROPIC_CACHE_MARKERS:
            return
        content = message.get("content")
        if isinstance(content, str):
            if content.strip():
                message["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": dict(ANTHROPIC_CACHE_CONTROL),
                    }
                ]
                added += 1
            return
        if isinstance(content, list):
            for block in reversed(content):
                if isinstance(block, dict) and (
                    block.get("text") or block.get("content") or block.get("type")
                ):
                    if mark(block):
                        return

    if request_kwargs:
        system = request_kwargs.get("system")
        if isinstance(system, list) and system:
            mark(system[0])
        elif (
            isinstance(system, str)
            and system.strip()
            and existing + added < MAX_ANTHROPIC_CACHE_MARKERS
        ):
            request_kwargs["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": dict(ANTHROPIC_CACHE_CONTROL),
                }
            ]
            added += 1

        tools = request_kwargs.get("tools")
        if isinstance(tools, list) and tools:
            mark(tools[-1])

    if messages:
        seen: set[int] = set()
        for idx in (max(0, len(messages) - 5), len(messages) - 1):
            if idx in seen:
                continue
            seen.add(idx)
            if existing + added >= MAX_ANTHROPIC_CACHE_MARKERS:
                break
            mark_message(messages[idx])
    return added


class ModelRoutingStrategy:
    """Wraps a BaseRouter and implements the LiteLLM custom routing interface.

    Implements async_get_available_deployment() and get_available_deployment()
    as required by litellm.router.CustomRoutingStrategyBase.

    Tolerance can be overridden per-request via set_request_tolerance() which
    uses contextvars for async-safe, per-request scoping.
    """

    def __init__(
        self,
        router: BaseRouter,
        *,
        tolerance: float = 0.20,
        models: list[str] | None = None,
        escalation_patterns: list[str] | None = None,
        escalation_model: str = "",
        cheap_patterns: list[str] | None = None,
        depth_tolerance: Any = None,
        switching: Any = None,
        session_health: Any = None,
    ):
        import re as _re

        self._router = router
        self._tolerance = tolerance
        self._models = models
        self._litellm_router: Any = None
        self._last_result: RoutingResult | None = None
        self._results_by_request: dict[str, RoutingResult] = {}
        self._session_selected: dict[str, str] = {}
        self._turbo_until = 0.0
        self._turbo_duration_seconds = 1800.0
        self._state_lock = threading.Lock()
        # Last model we actually routed to, kept as a process-wide debug mirror.
        # Pinning uses _session_selected instead.
        self._last_selected_model: str | None = None
        # Depth-scaled tolerance: tighten tolerance (-> escalate) as the session
        # deepens, because completeness failures cluster deep, not early.
        self._depth_tol = depth_tolerance
        # Cache-aware asymmetric switching gate (opt-in).
        self._switching = switching
        self._session_health_config = session_health
        self._session_health_scorer: Any = None
        self._session_health_threshold = float(
            os.environ.get(
                "ROUTER_SESSION_HEALTH_THRESHOLD",
                getattr(session_health, "threshold", 0.75) if session_health else 0.75,
            )
            or 0.75
        )
        if self._session_health_enabled():
            checkpoint = getattr(session_health, "checkpoint", "") if session_health else ""
            if checkpoint:
                from model_router_toolkit.session_health import SessionHealthScorer

                self._session_health_scorer = SessionHealthScorer.load(checkpoint)
        self._cache_pin_mode = _normalize_cache_pin_mode(
            os.environ.get(
                "ROUTER_CACHE_PINNING",
                getattr(switching, "cache_pin_mode", "off") if switching else "off",
            )
        )
        self._cache_pin_min_blend = float(
            os.environ.get(
                "ROUTER_CACHE_PIN_MIN_BLEND",
                getattr(switching, "cache_pin_min_blend", 3.0) if switching else 3.0,
            )
            or 0.0
        )
        # Policy escalation overlay: force the top tier when the prompt matches
        # (sustained agentic loops the prefill encoder can't see, or !hard).
        self._escalation_model = escalation_model
        self._escalation_res = [_re.compile(p, _re.IGNORECASE) for p in (escalation_patterns or [])]
        self._cheap_res = [_re.compile(p, _re.IGNORECASE) for p in (cheap_patterns or [])]
        self._route_log_path = os.environ.get("ROUTER_ROUTE_LOG", "")
        self._route_log_lock = threading.Lock()
        self._embedding_scorer: Any = None
        self._embedding_load_error: Exception | None = None

    @classmethod
    def from_config(cls, config_path: str, **kwargs: Any) -> ModelRoutingStrategy:
        """Build a strategy from a pool_config.yaml file."""
        from model_router_toolkit.config import build_router_from_config, load_config

        config = load_config(config_path)
        router = build_router_from_config(config)
        esc = config.escalation
        utility = config.utility
        if "tolerance" not in kwargs:
            tolerance = config.routing.tolerance
            if env_tolerance := os.environ.get("ROUTER_TOLERANCE"):
                try:
                    tolerance = float(env_tolerance)
                except ValueError as exc:
                    raise ValueError(
                        f"ROUTER_TOLERANCE must be a float, got {env_tolerance!r}"
                    ) from exc
            kwargs["tolerance"] = tolerance
        kwargs.setdefault("escalation_patterns", esc.force_top_tier_when_prompt_matches)
        kwargs.setdefault("escalation_model", esc.top_tier_model)
        kwargs.setdefault("cheap_patterns", utility.cheap_when_prompt_matches)
        kwargs.setdefault("depth_tolerance", config.depth_tolerance)
        kwargs.setdefault("switching", getattr(config, "switching", None))
        kwargs.setdefault("session_health", getattr(config, "session_health", None))
        return cls(router, **kwargs)

    def _escalates(self, text: str) -> bool:
        return bool(self._escalation_model) and any(r.search(text) for r in self._escalation_res)

    def _cheap_utility(self, text: str) -> bool:
        return any(r.search(text) for r in self._cheap_res)

    def _session_depth(self, messages: list[dict] | None) -> int:
        """Estimate how deep the session is from the incoming message list.

        'turns' = number of assistant messages so far; 'tools' = number of
        tool/function messages (proxy for tool-call depth)."""
        if not messages:
            return 0
        metric = getattr(self._depth_tol, "depth_metric", "turns")
        if metric == "tools":
            return sum(
                1
                for m in messages
                if m.get("role") == "tool" or m.get("tool_calls") or m.get("tool_call_id")
            )
        return sum(1 for m in messages if m.get("role") == "assistant")

    def _depth_scaled_tolerance(self, base: float, messages: list[dict] | None) -> float:
        """Tighten tolerance toward min_tolerance as session depth grows."""
        dt = self._depth_tol
        if not dt or not getattr(dt, "enabled", False):
            return base
        depth = self._session_depth(messages)
        full = max(1, getattr(dt, "depth_full", 30))
        frac = min(1.0, depth / full)
        # Floor at 0.001 (never exactly 0): at tol=0 the cost term drops out and
        # routing becomes incoherent — trivial deep turns ratchet to the top tier.
        floor = max(0.001, getattr(dt, "min_tolerance", 0.0))
        scaled = base - (base - floor) * frac
        return max(floor, min(1.0, scaled))

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: float) -> None:
        self._tolerance = max(0.0, min(1.0, value))

    def set_request_tolerance(self, value: float) -> None:
        """Set tolerance for the current async request context only."""
        _request_tolerance.set(max(0.0, min(1.0, value)))

    @property
    def models(self) -> list[str] | None:
        return self._models

    @models.setter
    def models(self, value: list[str] | None) -> None:
        self._models = value

    @property
    def effective_tolerance(self) -> float:
        """Tolerance for the current request: per-request override or default."""
        val = _request_tolerance.get()
        return val if val is not None else self._tolerance

    @property
    def cache_pin_mode(self) -> str:
        return self._cache_pin_mode

    @cache_pin_mode.setter
    def cache_pin_mode(self, value: str) -> None:
        self._cache_pin_mode = _normalize_cache_pin_mode(value)

    @property
    def cache_pin_min_blend(self) -> float:
        return self._cache_pin_min_blend

    @cache_pin_min_blend.setter
    def cache_pin_min_blend(self, value: float) -> None:
        self._cache_pin_min_blend = max(0.0, float(value))

    def set_turbo(self, *, enabled: bool, duration_seconds: float = 1800.0) -> dict[str, Any]:
        duration = max(1.0, float(duration_seconds))
        with self._state_lock:
            if enabled:
                self._turbo_duration_seconds = duration
                self._turbo_until = time.time() + duration
            else:
                self._turbo_until = 0.0
        return self.turbo_state()

    def turbo_state(self) -> dict[str, Any]:
        now = time.time()
        with self._state_lock:
            until = float(self._turbo_until or 0.0)
            duration = float(self._turbo_duration_seconds or 1800.0)
            if until and until <= now:
                self._turbo_until = 0.0
                until = 0.0
        remaining = max(0.0, until - now)
        return {
            "active": remaining > 0,
            "remaining_seconds": int(round(remaining)),
            "until_ts": until if remaining > 0 else 0.0,
            "duration_seconds": int(round(duration)),
        }

    @property
    def last_result(self) -> RoutingResult | None:
        return self._last_result

    def pop_result(self, request_id: str | None) -> RoutingResult | None:
        """Return and remove the routing result for a specific request."""
        if not request_id:
            return None
        with self._state_lock:
            return self._results_by_request.pop(request_id, None)

    @property
    def router(self) -> BaseRouter:
        return self._router

    def _extract_user_text(self, messages: list[dict[str, str]] | None) -> str:
        return extract_user_text(messages)

    def _find_deployment(self, model_name: str) -> dict | None:
        if self._litellm_router is None:
            return None
        for dep in self._litellm_router.model_list:
            if isinstance(dep, dict) and dep.get("model_name") == model_name:
                return dep
        return None

    def _pool_model_names(self) -> list[str]:
        config = getattr(self._router, "_config", None)
        if config is not None and hasattr(config, "model_names"):
            return list(config.model_names)
        if self._models:
            return list(self._models)
        if self._litellm_router is not None:
            return [
                dep.get("model_name")
                for dep in self._litellm_router.model_list
                if isinstance(dep, dict) and dep.get("model_name")
            ]
        return []

    def _cheapest_model(self, allowed: list[str] | None = None) -> str | None:
        allowed_set = set(allowed) if allowed else None
        candidates: list[tuple[float, float, str]] = []
        for name in self._pool_model_names():
            if allowed_set is not None and name not in allowed_set:
                continue
            result = self._router.resolve(name)
            if result is None:
                continue
            cost = result.selected_cost
            candidates.append((cost.cost_per_m_input_tokens, cost.cost_per_m_output_tokens, name))
        if not candidates:
            return None
        return min(candidates)[2]

    def _dearest_model(self, allowed: list[str] | None = None) -> str | None:
        allowed_set = set(allowed) if allowed else None
        candidates: list[tuple[float, float, str]] = []
        for name in self._pool_model_names():
            if allowed_set is not None and name not in allowed_set:
                continue
            result = self._router.resolve(name)
            if result is None:
                continue
            cost = result.selected_cost
            candidates.append((cost.cost_per_m_input_tokens, cost.cost_per_m_output_tokens, name))
        if not candidates:
            return None
        return max(candidates)[2]

    def _session_health_enabled(self) -> bool:
        if os.environ.get("ROUTER_SESSION_HEALTH", "").lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return False
        if os.environ.get("ROUTER_SESSION_HEALTH", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return True
        return bool(
            self._session_health_config and getattr(self._session_health_config, "enabled", False)
        )

    def _session_health_top_model(self, allowed: list[str] | None = None) -> str | None:
        configured = (
            getattr(self._session_health_config, "top_tier_model", "")
            if self._session_health_config
            else ""
        )
        configured = configured or self._escalation_model
        if configured and (not allowed or configured in set(allowed)):
            return configured
        return self._dearest_model(allowed)

    def _score_session_health(
        self,
        messages: list[dict] | None,
        *,
        task: str,
    ) -> Any | None:
        scorer = self._session_health_scorer
        if scorer is None:
            return None
        try:
            score = scorer.score_messages(messages, task=task)
        except Exception:
            logger.exception("Session-health scoring failed")
            return None
        score.threshold = self._session_health_threshold
        score.should_escalate = (
            score.score >= self._session_health_threshold
            and self._session_health_has_actionable_history(score)
        )
        return score

    def _session_health_has_actionable_history(self, score: Any) -> bool:
        features = getattr(score, "features", None) or {}

        def value(name: str) -> float:
            try:
                return float(features.get(name, 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        event_count = value("event_count")
        assistant_count = value("assistant_count")
        tool_count = value("tool_count")
        command_count = value("command_count")
        error_count = value("error_count")
        retry_count = value("retry_count")
        repeated_error_recent = value("repeated_error_recent")
        test_failure_recent = value("test_failure_recent")

        if event_count < SESSION_HEALTH_MIN_EVENTS:
            return False
        if assistant_count + tool_count < SESSION_HEALTH_MIN_AGENTIC_EVENTS:
            return False
        return any(
            signal > 0
            for signal in (
                tool_count,
                command_count,
                error_count,
                retry_count,
                repeated_error_recent,
                test_failure_recent,
            )
        )

    def _select_utility_model(
        self,
        *,
        session_key: str,
        request_id: str | None,
        messages: list[dict] | None,
        task_view: str,
        reason: str,
        remember: bool = False,
        allowed: list[str] | None = None,
    ) -> dict | None:
        model_name = self._cheapest_model(allowed)
        if not model_name:
            return None
        dep = self._find_deployment(model_name)
        if dep is None:
            return None
        result = self._router.resolve(model_name)
        if result is not None:
            result.metadata["pin_reason"] = reason
            result.metadata["utility"] = True
            if remember:
                self._set_last_selected(session_key, model_name)
            self._store_result(request_id, result)
            self._log_route(
                request_id=request_id,
                session_key=session_key,
                decision="cheap_utility",
                result=result,
                task_view=task_view,
                messages=messages,
            )
        return dep

    def _metadata(self, request_kwargs: dict | None) -> dict:
        if not request_kwargs:
            return {}
        merged = {}
        for key in ("metadata", "litellm_metadata"):
            metadata = request_kwargs.get(key) or {}
            if isinstance(metadata, dict):
                merged.update(metadata)
        return merged

    def _request_id(self, request_kwargs: dict | None) -> str | None:
        metadata = self._metadata(request_kwargs)
        rid = metadata.get("router_request_id")
        return str(rid) if rid else None

    def _is_embedding_request(self, model: str, request_kwargs: dict | None) -> bool:
        metadata = self._metadata(request_kwargs)
        mode = str(metadata.get("router_mode") or "").lower()
        return model == EMBEDDING_ROUTED_ALIAS or mode == "embedding"

    def _session_key(
        self,
        messages: list[dict] | None,
        request_kwargs: dict | None,
    ) -> str:
        metadata = self._metadata(request_kwargs)
        for key in (
            "router_session_id",
            "session_id",
            "conversation_id",
            "thread_id",
            "chat_id",
        ):
            val = metadata.get(key)
            if val:
                return str(val)

        # Goose does not currently send a session id through the OpenAI API path.
        # Hash the first real user message so all later full-context requests in
        # the same session share routing/cache state without leaking the prompt in
        # process state.
        if messages:
            for msg in messages:
                if msg.get("role") == "user":
                    text = self._extract_user_text([msg]).strip()
                    if is_goose_title_request(text):
                        continue
                    text = strip_info_messages(text).strip()
                    if text:
                        digest = hashlib.sha256(text[:2000].encode()).hexdigest()[:16]
                        return f"first-user:{digest}"
        return "default"

    def _get_last_selected(self, session_key: str) -> str | None:
        with self._state_lock:
            return self._session_selected.get(session_key)

    def _set_last_selected(self, session_key: str, model_name: str) -> None:
        with self._state_lock:
            self._session_selected[session_key] = model_name
            self._last_selected_model = model_name

    def _store_result(self, request_id: str | None, result: RoutingResult) -> None:
        with self._state_lock:
            self._last_result = result
            if request_id:
                self._results_by_request[request_id] = result

    def _get_embedding_scorer(self):
        if self._embedding_scorer is not None:
            return self._embedding_scorer
        if self._embedding_load_error is not None:
            raise RuntimeError(
                "Embedding-router scorer failed to load earlier"
            ) from self._embedding_load_error
        try:
            from model_router_toolkit.adapters.litellm.embedding import (
                EmbeddingComplexityScorer,
            )

            bundle = os.environ.get("ROUTER_EMBEDDING_BUNDLE", "")
            self._embedding_scorer = EmbeddingComplexityScorer(bundle or None)
            return self._embedding_scorer
        except Exception as exc:
            self._embedding_load_error = exc
            raise

    def _embedding_ladder(self, allowed: list[str] | None) -> list[str]:
        allowed_set = set(allowed) if allowed else None
        candidates: list[tuple[tuple[float, float], str]] = []
        seen_costs: set[tuple[float, float]] = set()
        for name in self._pool_model_names():
            if allowed_set is not None and name not in allowed_set:
                continue
            result = self._router.resolve(name)
            if result is None:
                continue
            cost = result.selected_cost
            cost_key = (cost.cost_per_m_input_tokens, cost.cost_per_m_output_tokens)
            if cost_key in seen_costs:
                continue
            seen_costs.add(cost_key)
            candidates.append((cost_key, name))
        candidates.sort(key=lambda item: item[0])
        return [name for _, name in candidates]

    def _select_embedding_ladder_model(
        self, complexity: float, allowed: list[str] | None
    ) -> str | None:
        ladder = self._embedding_ladder(allowed)
        if not ladder:
            return None
        if len(ladder) == len(EMBEDDING_COMPLEXITY_BANDS) + 1:
            idx = 0
            for threshold in EMBEDDING_COMPLEXITY_BANDS:
                if complexity < threshold:
                    break
                idx += 1
        else:
            idx = int(max(0.0, min(0.999999, complexity)) * len(ladder))
        return ladder[min(idx, len(ladder) - 1)]

    def _route_embedding(
        self,
        *,
        request_id: str | None,
        session_key: str,
        messages: list[dict] | None,
        request_kwargs: dict | None,
        allowed: list[str] | None,
    ) -> dict | None:
        router_mode = "embedding"
        decision = "embedding_route"
        error: str | None = None
        try:
            score = self._get_embedding_scorer().score_messages(messages)
            selected = self._select_embedding_ladder_model(score.complexity, allowed)
            if not selected:
                return None
            task_view = score.rendered
            ladder = self._embedding_ladder(allowed)
            rung_index = ladder.index(selected) + 1 if selected in ladder else None
            metadata = {
                "router_mode": router_mode,
                "complexity": round(float(score.complexity), 4),
                "tool_calls_norm": round(float(score.tool_calls_norm), 4),
                "elapsed_ms": int(score.elapsed_ms),
                "ladder": ladder,
                "rung_index": rung_index,
                "rung_count": len(ladder),
            }
        except Exception as exc:
            logger.exception("Embedding routing failed; falling back to dearest model")
            selected = self._dearest_model(allowed)
            if not selected:
                return None
            task_view = self._extract_user_text(messages)
            decision = "embedding_fallback_main"
            error = str(exc)
            metadata = {
                "router_mode": router_mode,
                "ladder": self._embedding_ladder(allowed),
                "error": error[:300],
            }

        result = self._router.resolve(selected)
        if result is None:
            return None
        result.metadata.update(metadata)
        self._store_result(request_id, result)
        self._set_last_selected(session_key, selected)
        self._log_route(
            request_id=request_id,
            session_key=session_key,
            decision=decision,
            result=result,
            task_view=task_view,
            messages=messages,
            extra={"router_mode": router_mode, "embedding_error": error},
        )
        logger.info("Embedding route -> %s (%s)", selected, result.metadata)
        return self._find_deployment(selected)

    def _switching_enabled(self) -> bool:
        sw = self._switching
        if os.environ.get("ROUTER_DISABLE_SWITCHING", "").lower() in {"1", "true", "yes", "on"}:
            return False
        if os.environ.get("ROUTER_SWITCHING", "").lower() in {"0", "false", "no", "off"}:
            return False
        return bool(sw and getattr(sw, "enabled", False))

    def _routing_blend(self, model_name: str) -> float:
        config = getattr(self._router, "_config", None)
        if config is not None:
            spec = config.get_model(model_name) if hasattr(config, "get_model") else None
            if spec is not None:
                routing = getattr(config, "routing", None)
                output_weight = float(getattr(routing, "output_token_weight", 0.0) or 0.0)
                multiplier = float(getattr(spec, "routing_cost_multiplier", 1.0) or 1.0)
                return (
                    spec.cost_per_m_input_tokens + output_weight * spec.cost_per_m_output_tokens
                ) * multiplier

        resolved = self._router.resolve(model_name)
        if resolved is None:
            return 0.0
        for name, cost in zip(resolved.model_names, resolved.costs):
            if name == model_name:
                return cost.cost_per_m_input_tokens + 0.25 * cost.cost_per_m_output_tokens
        return 0.0

    def _cache_pin_allowed(self, model_name: str) -> tuple[bool, float]:
        mode = self.cache_pin_mode
        if mode == "off":
            return False, self._routing_blend(model_name)
        if mode == "all":
            return True, self._routing_blend(model_name)
        blend = self._routing_blend(model_name)
        return blend >= self.cache_pin_min_blend, blend

    def _log_route(
        self,
        *,
        request_id: str | None,
        session_key: str,
        decision: str,
        result: RoutingResult | None,
        task_view: str,
        messages: list[dict] | None,
        extra: dict | None = None,
    ) -> None:
        if not self._route_log_path:
            return
        row = {
            "ts": time.time(),
            "request_id": request_id,
            "session_key": session_key,
            "decision": decision,
            "selected_model": result.selected_model if result else None,
            "session_depth": self._session_depth(messages),
            "context_tokens_est": self._context_tokens(messages),
            "task_view": task_view[:1200],
            "metadata": result.metadata if result else {},
            "confidences": (
                {
                    name: round(float(conf), 4)
                    for name, conf in zip(result.model_names, result.confidences)
                }
                if result
                else {}
            ),
        }
        if extra:
            row.update(extra)
        try:
            with self._route_log_lock:
                with open(self._route_log_path, "a") as f:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
        except OSError:
            logger.warning("Could not write route log to %s", self._route_log_path)

    def _context_tokens(self, messages: list[dict] | None) -> int:
        """Rough size of the cached prefix at risk, in tokens (~4 chars/token)."""
        if not messages:
            return 0
        chars = 0
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                chars += len(c)
            elif isinstance(c, list):
                for p in c:
                    if isinstance(p, dict):
                        chars += len(p.get("text", "") or "")
        return chars // 4

    def _apply_switch_gate(
        self, result: RoutingResult, messages: list[dict] | None, session_key: str
    ) -> RoutingResult:
        """Cache-aware asymmetric switching. The router has chosen the best model
        for this turn; decide whether the predicted gain justifies leaving the
        model we are already on (which holds a warm prompt cache). Stay put unless
        the candidate beats the incumbent's predicted P(success) by a margin that
        is small when escalating and large (and context-scaled) when de-escalating.
        """
        sw = self._switching
        if not self._switching_enabled():
            return result
        cur = self._get_last_selected(session_key)
        cand = result.selected_model
        if not cur or cur == cand or cur not in result.model_names:
            return result

        names = result.model_names
        confs = result.confidences
        costs = result.costs
        ci = names.index(cur)
        cj = names.index(cand)
        cur_cost = costs[ci].cost_per_m_input_tokens if costs else 0.0
        cand_cost = costs[cj].cost_per_m_input_tokens if costs else 0.0
        going_down = cand_cost < cur_cost

        gain = confs[cj] - confs[ci]  # predicted quality gain from switching
        if going_down:
            ctx = self._context_tokens(messages)
            margin = min(
                getattr(sw, "max_down_margin", 0.25),
                getattr(sw, "down_margin", 0.06)
                + getattr(sw, "down_margin_per_100k", 0.04) * (ctx / 100_000),
            )
        else:
            margin = getattr(sw, "up_margin", 0.0)

        if gain >= margin:
            return result  # worth switching

        # not worth leaving the cached incumbent — stay put
        logger.info(
            "Switch gate: stay on %s (cand %s gain %.3f < margin %.3f, %s)",
            cur,
            cand,
            gain,
            margin,
            "down" if going_down else "up",
        )
        meta = dict(result.metadata)
        meta["switch_gate"] = {
            "stayed_on": cur,
            "candidate": cand,
            "gain": round(float(gain), 4),
            "margin": round(float(margin), 4),
            "direction": "down" if going_down else "up",
        }
        return RoutingResult(
            model_names=result.model_names,
            confidences=result.confidences,
            costs=result.costs,
            selected_model=cur,
            metadata=meta,
        )

    def _pin_to_last(
        self,
        session_key: str,
        request_id: str | None,
        messages: list[dict] | None,
        task_view: str,
    ) -> dict | None:
        """Return the deployment for the last model we routed to, or None if we
        have not made a routing decision yet this session. Used for non-decision
        turns (tool results / narration) so we stay on the cached model.
        """
        last_selected = self._get_last_selected(session_key)
        if not last_selected:
            return None
        allowed, blend = self._cache_pin_allowed(last_selected)
        if not allowed:
            return None
        dep = self._find_deployment(last_selected)
        if dep is None:
            return None
        pinned = self._router.resolve(last_selected)
        if pinned is not None:
            pinned.metadata["pin_reason"] = "non_decision_turn"
            pinned.metadata["cache_pin_mode"] = self.cache_pin_mode
            pinned.metadata["routing_blend"] = round(float(blend), 6)
            self._store_result(request_id, pinned)
            self._log_route(
                request_id=request_id,
                session_key=session_key,
                decision="pin_non_decision",
                result=pinned,
                task_view=task_view,
                messages=messages,
            )
        return dep

    def _try_pin(
        self,
        request_kwargs: dict | None,
        *,
        session_key: str,
        request_id: str | None,
        messages: list[dict] | None,
    ) -> dict | None:
        """Check request_kwargs for an explicit pin_model directive.

        Returns the pinned deployment dict, or None to fall through to
        normal routing.  The pin_model value must match a model in the
        pool; unknown names are silently ignored.
        """
        if not request_kwargs:
            return None
        metadata = self._metadata(request_kwargs)
        pin = metadata.get("pin_model")
        if not pin or not self._router.has_model(pin):
            return None
        pinned = self._router.resolve(pin)
        if pinned is None:
            return None
        pinned.metadata["pin_reason"] = "explicit_pin_model"
        self._set_last_selected(session_key, pin)
        self._store_result(request_id, pinned)
        self._log_route(
            request_id=request_id,
            session_key=session_key,
            decision="explicit_pin",
            result=pinned,
            task_view="",
            messages=messages,
        )
        return self._find_deployment(pin)

    def _try_turbo(
        self,
        *,
        session_key: str,
        request_id: str | None,
        messages: list[dict] | None,
        task_view: str,
        allowed: list[str] | None,
    ) -> dict | None:
        turbo = self.turbo_state()
        if not turbo["active"]:
            return None
        model_name = self._session_health_top_model(allowed)
        dep = self._find_deployment(model_name) if model_name else None
        if dep is None:
            return None
        result = self._router.resolve(model_name)
        if result is None:
            return dep
        result.metadata["escalated"] = True
        result.metadata["turbo"] = turbo
        self._set_last_selected(session_key, model_name)
        self._store_result(request_id, result)
        self._log_route(
            request_id=request_id,
            session_key=session_key,
            decision="turbo_escalation",
            result=result,
            task_view=task_view,
            messages=messages,
            extra={
                "turbo_remaining_seconds": turbo["remaining_seconds"],
                "allowed_models": allowed,
            },
        )
        logger.info(
            "Turbo mode %.0fs remaining -> %s",
            turbo["remaining_seconds"],
            model_name,
        )
        return dep

    def _route_and_select(
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,
        input: str | list | None = None,
        request_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> dict:
        request_id = self._request_id(request_kwargs)
        session_key = self._session_key(messages, request_kwargs)
        req_models = self._metadata(request_kwargs).get("models")
        allowed = req_models or self._models
        raw_text = self._extract_user_text(messages).strip()
        embedding_request = self._is_embedding_request(model, request_kwargs)

        dep = self._try_turbo(
            session_key=session_key,
            request_id=request_id,
            messages=messages,
            task_view=raw_text,
            allowed=allowed,
        )
        if dep:
            return dep

        # Explicit pin via metadata — for router-per-subagent flows.
        dep = self._try_pin(
            request_kwargs,
            session_key=session_key,
            request_id=request_id,
            messages=messages,
        )
        if dep:
            return dep

        # Reconstruct a training-shaped task statement from the turn-in-context.
        # Returns None when this turn is NOT a routing decision (tool result or
        # agent narration). By default we still route those turns rather than
        # blindly pinning to the incumbent; optional cache-pin modes can preserve
        # the incumbent only for dear models, or for every model.
        if is_goose_title_request(raw_text):
            dep = self._select_utility_model(
                session_key=session_key,
                request_id=request_id,
                messages=messages,
                task_view=raw_text,
                reason="goose_title_generation",
                allowed=allowed,
            )
            if dep is not None:
                return dep

        if (
            not embedding_request
            and self._cheap_utility(raw_text)
            and self._get_last_selected(session_key) is None
        ):
            dep = self._select_utility_model(
                session_key=session_key,
                request_id=request_id,
                messages=messages,
                task_view=raw_text,
                reason="cheap_utility_pattern",
                remember=True,
                allowed=allowed,
            )
            if dep is not None:
                return dep

        if not embedding_request and is_info_only_request(raw_text):
            pinned = self._pin_to_last(session_key, request_id, messages, "")
            if pinned is not None:
                return pinned
            dep = self._select_utility_model(
                session_key=session_key,
                request_id=request_id,
                messages=messages,
                task_view=raw_text,
                reason="goose_info_only",
                remember=True,
                allowed=allowed,
            )
            if dep is not None:
                return dep

        text = build_task_view(messages)
        is_real_routing_turn = text is not None
        if text is None:
            pinned = self._pin_to_last(session_key, request_id, messages, "")
            if pinned is not None:
                return pinned
            # no prior decision yet (cold session opening on a tool turn) — fall
            # back to bare extraction so we still route something sensible.
            text = self._extract_user_text(messages)
        if not text and isinstance(input, str):
            text = input

        if not text:
            with self._state_lock:
                self._last_result = None
            if self._litellm_router:
                return self._litellm_router.model_list[0]
            return {}

        # Policy escalation: explicit hard overrides or configured high-stakes
        # patterns. Imposed by rule, not learned; the learned trajectory-health
        # gate is handled separately.
        if self._escalates(text):
            dep = self._find_deployment(self._escalation_model)
            if dep:
                result = self._router.resolve(self._escalation_model)
                if result is None:
                    return dep
                result.metadata["escalated"] = True
                self._set_last_selected(session_key, self._escalation_model)
                self._store_result(request_id, result)
                self._log_route(
                    request_id=request_id,
                    session_key=session_key,
                    decision="policy_escalation",
                    result=result,
                    task_view=text,
                    messages=messages,
                )
                logger.info("Escalation policy -> %s", self._escalation_model)
                return dep

        health = self._score_session_health(messages, task=text) if is_real_routing_turn else None
        if health is not None and health.should_escalate:
            model_name = self._session_health_top_model(allowed)
            dep = self._find_deployment(model_name) if model_name else None
            if dep:
                result = self._router.resolve(model_name)
                if result is None:
                    return dep
                result.metadata["escalated"] = True
                result.metadata["session_health"] = {
                    "score": round(float(health.score), 4),
                    "threshold": round(float(health.threshold), 4),
                    "features": {k: round(float(v), 4) for k, v in health.features.items()},
                }
                self._set_last_selected(session_key, model_name)
                self._store_result(request_id, result)
                self._log_route(
                    request_id=request_id,
                    session_key=session_key,
                    decision="session_health_escalation",
                    result=result,
                    task_view=text,
                    messages=messages,
                    extra={
                        "session_health_score": round(float(health.score), 4),
                        "session_health_threshold": round(float(health.threshold), 4),
                        "allowed_models": allowed,
                    },
                )
                logger.info(
                    "Session-health escalation %.3f >= %.3f -> %s",
                    health.score,
                    health.threshold,
                    model_name,
                )
                return dep

        health_extra = {}
        if health is not None:
            health_extra = {
                "session_health_score": round(float(health.score), 4),
                "session_health_threshold": round(float(health.threshold), 4),
            }

        if embedding_request:
            dep = self._route_embedding(
                request_id=request_id,
                session_key=session_key,
                messages=messages,
                request_kwargs=request_kwargs,
                allowed=allowed,
            )
            if dep is not None:
                return dep

        tol = self._depth_scaled_tolerance(self.effective_tolerance, messages)
        result = self._router.route(
            text,
            tolerance=tol,
            models=allowed,
        )
        raw_selected = result.selected_model
        result = self._apply_switch_gate(result, messages, session_key)
        self._store_result(request_id, result)
        self._set_last_selected(session_key, result.selected_model)
        self._log_route(
            request_id=request_id,
            session_key=session_key,
            decision="route",
            result=result,
            task_view=text,
            messages=messages,
            extra={
                "raw_selected_model": raw_selected,
                "tolerance": tol,
                "allowed_models": allowed,
                **health_extra,
            },
        )

        dep = self._find_deployment(result.selected_model)
        if dep:
            return dep

        logger.warning(
            "Routed model %r not found in litellm model_list. "
            "LiteLLM will fall back to default routing. "
            "Ensure all pool models are in model_list.",
            result.selected_model,
        )

        if self._litellm_router:
            return self._litellm_router.model_list[0]
        return {}

    async def async_get_available_deployment(
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,
        input: str | list | None = None,
        specific_deployment: bool | None = False,
        request_kwargs: dict | None = None,
    ) -> dict:
        dep = await asyncio.to_thread(
            self._route_and_select,
            model,
            messages,
            input,
            request_kwargs,
        )
        inject_anthropic_cache_markers(dep, messages, request_kwargs)
        return dep

    def get_available_deployment(
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,
        input: str | list | None = None,
        specific_deployment: bool | None = False,
        request_kwargs: dict | None = None,
    ) -> dict:
        dep = self._route_and_select(model, messages, input, request_kwargs)
        inject_anthropic_cache_markers(dep, messages, request_kwargs)
        return dep

    def set_litellm_router(self, litellm_router: Any) -> None:
        """Called internally when plugged into a litellm.Router."""
        self._litellm_router = litellm_router
