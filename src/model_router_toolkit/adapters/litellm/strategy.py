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
import logging
from typing import Any

from model_router_toolkit.router import BaseRouter, RoutingResult, extract_user_text
from model_router_toolkit.task_view import build_task_view

logger = logging.getLogger(__name__)

_request_tolerance: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "request_tolerance",
    default=None,
)


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
        depth_tolerance: Any = None,
        switching: Any = None,
    ):
        import re as _re

        self._router = router
        self._tolerance = tolerance
        self._models = models
        self._litellm_router: Any = None
        self._last_result: RoutingResult | None = None
        # Last model we actually routed to. When a turn is NOT a routing decision
        # (a tool result or agent narration, ~75% of agent turns), we pin to this
        # instead of scoring noise — which also preserves the provider's prompt
        # cache across the turn (switching busts the cache on big-context agents).
        self._last_selected_model: str | None = None
        # Depth-scaled tolerance: tighten tolerance (-> escalate) as the session
        # deepens, because completeness failures cluster deep, not early.
        self._depth_tol = depth_tolerance
        # Cache-aware asymmetric switching gate (opt-in).
        self._switching = switching
        # Policy escalation overlay: force the top tier when the prompt matches
        # (sustained agentic loops the prefill encoder can't see, or !hard).
        self._escalation_model = escalation_model
        self._escalation_res = [
            _re.compile(p, _re.IGNORECASE) for p in (escalation_patterns or [])
        ]

    @classmethod
    def from_config(cls, config_path: str, **kwargs: Any) -> ModelRoutingStrategy:
        """Build a strategy from a pool_config.yaml file."""
        from model_router_toolkit.config import build_router_from_config, load_config

        config = load_config(config_path)
        router = build_router_from_config(config)
        esc = config.escalation
        kwargs.setdefault("escalation_patterns", esc.force_top_tier_when_prompt_matches)
        kwargs.setdefault("escalation_model", esc.top_tier_model)
        kwargs.setdefault("depth_tolerance", config.depth_tolerance)
        kwargs.setdefault("switching", getattr(config, "switching", None))
        return cls(router, tolerance=config.routing.tolerance, **kwargs)

    def _escalates(self, text: str) -> bool:
        return bool(self._escalation_model) and any(
            r.search(text) for r in self._escalation_res
        )

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
    def last_result(self) -> RoutingResult | None:
        return self._last_result

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
        self, result: RoutingResult, messages: list[dict] | None
    ) -> RoutingResult:
        """Cache-aware asymmetric switching. The router has chosen the best model
        for this turn; decide whether the predicted gain justifies leaving the
        model we are already on (which holds a warm prompt cache). Stay put unless
        the candidate beats the incumbent's predicted P(success) by a margin that
        is small when escalating and large (and context-scaled) when de-escalating.
        """
        sw = self._switching
        if not sw or not getattr(sw, "enabled", False):
            return result
        cur = self._last_selected_model
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
        stay = self._router.resolve(cur)
        return stay if stay is not None else result

    def _pin_to_last(self) -> dict | None:
        """Return the deployment for the last model we routed to, or None if we
        have not made a routing decision yet this session. Used for non-decision
        turns (tool results / narration) so we stay on the cached model.
        """
        if not self._last_selected_model:
            return None
        dep = self._find_deployment(self._last_selected_model)
        if dep is None:
            return None
        pinned = self._router.resolve(self._last_selected_model)
        if pinned is not None:
            self._last_result = pinned
        return dep

    def _try_pin(self, request_kwargs: dict | None) -> dict | None:
        """Check request_kwargs for an explicit pin_model directive.

        Returns the pinned deployment dict, or None to fall through to
        normal routing.  The pin_model value must match a model in the
        pool; unknown names are silently ignored.
        """
        if not request_kwargs:
            return None
        metadata = request_kwargs.get("metadata") or {}
        pin = metadata.get("pin_model")
        if not pin or not self._router.has_model(pin):
            return None
        pinned = self._router.resolve(pin)
        if pinned is None:
            return None
        self._last_result = pinned
        return self._find_deployment(pin)

    def _route_and_select(
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,
        input: str | list | None = None,
        request_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> dict:
        # Explicit pin via metadata — for router-per-subagent flows.
        dep = self._try_pin(request_kwargs)
        if dep:
            return dep

        # Reconstruct a training-shaped task statement from the turn-in-context.
        # Returns None when this turn is NOT a routing decision (tool result or
        # agent narration). In that case pin to the model we last chose: it keeps
        # the request on a model that already has the conversation cached, and
        # avoids scoring machine-noise the classifier was never trained on.
        text = build_task_view(messages)
        if text is None:
            pinned = self._pin_to_last()
            if pinned is not None:
                return pinned
            # no prior decision yet (cold session opening on a tool turn) — fall
            # back to bare extraction so we still route something sensible.
            text = self._extract_user_text(messages)
        if not text and isinstance(input, str):
            text = input

        if not text:
            self._last_result = None
            if self._litellm_router:
                return self._litellm_router.model_list[0]
            return {}

        # Policy escalation: sustained-loop / high-stakes prompts that look easy
        # per-turn but need the top tier for the whole task. Imposed by rule, not
        # learned — see docs/ROUTING_FINDINGS.md (the "poll CI until done" case).
        if self._escalates(text):
            from model_router_toolkit.router import RoutingResult as _RR

            dep = self._find_deployment(self._escalation_model)
            if dep:
                self._last_result = _RR(
                    model_names=[self._escalation_model],
                    confidences=[1.0],
                    costs=[],
                    selected_model=self._escalation_model,
                    metadata={"escalated": True},
                )
                logger.info("Escalation policy -> %s", self._escalation_model)
                return dep

        req_models = ((request_kwargs or {}).get("metadata") or {}).get("models")
        allowed = req_models or self._models
        tol = self._depth_scaled_tolerance(self.effective_tolerance, messages)
        result = self._router.route(
            text,
            tolerance=tol,
            models=allowed,
        )
        result = self._apply_switch_gate(result, messages)
        self._last_result = result
        self._last_selected_model = result.selected_model

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
