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
    ):
        import re as _re

        self._router = router
        self._tolerance = tolerance
        self._models = models
        self._litellm_router: Any = None
        self._last_result: RoutingResult | None = None
        self._results_by_request: dict[str, RoutingResult] = {}
        self._session_selected: dict[str, str] = {}
        self._state_lock = threading.Lock()
        # Last model we actually routed to, kept as a process-wide debug mirror.
        # Pinning uses _session_selected instead.
        self._last_selected_model: str | None = None
        # Depth-scaled tolerance: tighten tolerance (-> escalate) as the session
        # deepens, because completeness failures cluster deep, not early.
        self._depth_tol = depth_tolerance
        # Cache-aware asymmetric switching gate (opt-in).
        self._switching = switching
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
        self._escalation_res = [
            _re.compile(p, _re.IGNORECASE) for p in (escalation_patterns or [])
        ]
        self._cheap_res = [
            _re.compile(p, _re.IGNORECASE) for p in (cheap_patterns or [])
        ]
        self._route_log_path = os.environ.get("ROUTER_ROUTE_LOG", "")
        self._route_log_lock = threading.Lock()

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
        return cls(router, **kwargs)

    def _escalates(self, text: str) -> bool:
        return bool(self._escalation_model) and any(
            r.search(text) for r in self._escalation_res
        )

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
            candidates.append(
                (cost.cost_per_m_input_tokens, cost.cost_per_m_output_tokens, name)
            )
        if not candidates:
            return None
        return min(candidates)[2]

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
                output_weight = float(
                    getattr(routing, "output_token_weight", 0.0) or 0.0
                )
                multiplier = float(
                    getattr(spec, "routing_cost_multiplier", 1.0) or 1.0
                )
                return (
                    spec.cost_per_m_input_tokens
                    + output_weight * spec.cost_per_m_output_tokens
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
            cur, cand, gain, margin, "down" if going_down else "up",
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
        raw_text = self._extract_user_text(messages).strip()
        req_models = self._metadata(request_kwargs).get("models")
        allowed = req_models or self._models

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

        if self._cheap_utility(raw_text) and self._get_last_selected(session_key) is None:
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

        text = build_task_view(messages)
        if text is None:
            pinned = self._pin_to_last(session_key, request_id, messages, "")
            if pinned is not None:
                return pinned
            if is_info_only_request(raw_text):
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

        # Policy escalation: sustained-loop / high-stakes prompts that look easy
        # per-turn but need the top tier for the whole task. Imposed by rule, not
        # learned — see docs/ROUTING_FINDINGS.md (the "poll CI until done" case).
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
        return await asyncio.to_thread(
            self._route_and_select,
            model,
            messages,
            input,
            request_kwargs,
        )

    def get_available_deployment(
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,
        input: str | list | None = None,
        specific_deployment: bool | None = False,
        request_kwargs: dict | None = None,
    ) -> dict:
        return self._route_and_select(model, messages, input, request_kwargs)

    def set_litellm_router(self, litellm_router: Any) -> None:
        """Called internally when plugged into a litellm.Router."""
        self._litellm_router = litellm_router
