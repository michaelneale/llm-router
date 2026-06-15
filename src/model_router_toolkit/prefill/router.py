"""Prefill-based routing: single forward pass -> hidden states -> MLP scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_router_toolkit.router import BaseRouter, CostEstimate, RoutingResult


class PrefillRouter(BaseRouter):
    """Routes queries by running prefill through encoder models and scoring with an MLP.

    This is a wrapper that adapts the complexity_router.ComplexityScorer to
    the BaseRouter interface. The actual extraction and scoring logic lives in
    the prefill/ subpackage modules (extract, trunk, transforms, sweep).
    """

    def __init__(self, *, config: Any = None):
        self._config = config
        self._scorer = None
        self._model_names: list[str] = []

    def load(self, checkpoint_path: str | Path) -> None:
        from model_router_toolkit.prefill.scorer import load_scorer

        self._scorer = load_scorer(checkpoint_path, config=self._config)
        self._scorer._ensure_loaded()
        self._model_names = self._scorer.model_names

    def _routing_cost_key(self, model_name: str, cost: CostEstimate) -> tuple[float, str]:
        output_weight = 0.0
        multiplier = 1.0
        if self._config is not None:
            routing = getattr(self._config, "routing", None)
            output_weight = float(getattr(routing, "output_token_weight", 0.0) or 0.0)
            spec = self._config.get_model(model_name) if hasattr(self._config, "get_model") else None
            if spec is not None:
                multiplier = float(getattr(spec, "routing_cost_multiplier", 1.0) or 1.0)
        weighted_cost = (
            cost.cost_per_m_input_tokens
            + output_weight * cost.cost_per_m_output_tokens
        ) * multiplier
        return (weighted_cost, model_name)

    def route(
        self,
        question: str,
        *,
        tolerance: float = 0.20,
        models: list[str] | None = None,
    ) -> RoutingResult:
        if self._scorer is None:
            raise RuntimeError("Router not loaded. Call load() first.")

        if models:
            unknown = set(models) - set(self._model_names)
            if unknown:
                raise ValueError(f"Models not in pool: {unknown}")

        raw = self._scorer.score(question)
        allowed = set(models) if models else set(raw.model_names)

        allowed_confs = [c for m, c in zip(raw.model_names, raw.confidences) if m in allowed]
        p_max = max(allowed_confs)
        threshold = p_max - tolerance

        cost_sorted = sorted(
            zip(raw.model_names, raw.confidences, raw.costs),
            key=lambda x: self._routing_cost_key(x[0], x[2]),
        )

        selected = [n for n, _, _ in cost_sorted if n in allowed][-1]
        for name, conf, _ in cost_sorted:
            if name in allowed and conf >= threshold:
                selected = name
                break

        return RoutingResult(
            model_names=raw.model_names,
            confidences=raw.confidences,
            costs=raw.costs,
            selected_model=selected,
            metadata={
                "p_max": p_max,
                "threshold": threshold,
                "tolerance": tolerance,
                "allowed_models": sorted(allowed),
            },
        )

    def has_model(self, model_name: str) -> bool:
        if self._config and hasattr(self._config, "model_names"):
            return model_name in self._config.model_names
        return model_name in self._model_names

    def resolve(self, model_name: str) -> RoutingResult | None:
        if not self.has_model(model_name):
            return None
        model_names = self._config.model_names if self._config else list(self._model_names)
        confidences = [1.0 if m == model_name else 0.0 for m in model_names]
        costs = []
        for m in model_names:
            spec = self._config.get_model(m) if self._config else None
            costs.append(
                CostEstimate(
                    median_output_tokens=500,
                    cost_per_m_input_tokens=spec.cost_per_m_input_tokens if spec else 0,
                    cost_per_m_output_tokens=spec.cost_per_m_output_tokens if spec else 0,
                )
            )
        return RoutingResult(
            model_names=model_names,
            confidences=confidences,
            costs=costs,
            selected_model=model_name,
            metadata={"pinned": True},
        )

    def unload(self) -> None:
        if self._scorer is not None:
            self._scorer.unload()
            self._scorer = None
