"""Configuration system for model-router-toolkit.

Loads pool_config.yaml and validates it with pydantic.
The config determines routing method, model pool, and provider settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    name: str
    display_name: str = ""
    litellm_model: str = ""
    cost_per_m_input_tokens: float = 0.0
    cost_per_m_output_tokens: float = 0.0
    routing_cost_multiplier: float = 1.0
    system_prompt: str = ""
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    api_base: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.display_name:
            self.display_name = self.name


class RoutingConfig(BaseModel):
    method: str = "prefill"
    checkpoint: str = ""
    tolerance: float = 0.20
    output_token_weight: float = 0.0

    encoder: str = ""
    encoder_server: str = ""
    training_mode: str = "auto"
    encoder_backend: str = "transformers"


class EscalationConfig(BaseModel):
    """Policy overlay: force the top tier for tasks the prefill encoder can't
    judge (sustained agentic loops, explicit high-stakes). Matched against the
    user prompt text before the learned router scores it.
    """

    force_top_tier_when_prompt_matches: list[str] = Field(default_factory=list)
    top_tier_model: str = ""


class SessionHealthConfig(BaseModel):
    """Learned trajectory-health escalation.

    This is intentionally separate from the normal model-correctness router. It
    predicts whether the recent session state is deteriorating; if it fires, the
    LiteLLM strategy escalates to the configured top tier.
    """

    enabled: bool = False
    checkpoint: str = ""
    threshold: float = 0.75
    top_tier_model: str = ""


class UtilityConfig(BaseModel):
    """Cheap-route overlay for obvious utility prompts.

    These are cold-session prompts where provider quality is not the scarce
    resource: greetings, title-like niceties, or simple local command/listing
    requests. Keep patterns narrow; substantive tasks should go through the
    learned router.
    """

    cheap_when_prompt_matches: list[str] = Field(default_factory=list)


class DepthToleranceConfig(BaseModel):
    """Scale routing tolerance by session depth. Trace analysis shows completeness
    failures cluster DEEP in sessions (21+ tool calls / turn 11+), not early — the
    model loses the thread as context fills. So tighten tolerance (demand a closer
    quality match -> escalate) as the conversation deepens. Depth is read from the
    incoming message list (assistant + tool turns); no retrain, no core change.

    effective_tolerance = base - (base - min_tolerance) * min(1, depth / depth_full)
    Disabled by default; opt in with enabled: true.
    """

    enabled: bool = False
    min_tolerance: float = 0.0  # tolerance at/after depth_full (most escalation)
    depth_full: int = 30  # tool/assistant turns at which min_tolerance is reached
    depth_metric: str = "turns"  # "turns" (assistant msgs) or "tools" (tool msgs)


class SwitchingConfig(BaseModel):
    """Cache-aware, asymmetric switching policy.

    The trained router proposes the best model per turn. But moving OFF the model
    we are currently on has a real cost: the provider's prompt cache is warm on
    the current model and cold on any other, so switching re-pays full input
    price on the (often huge) conversation prefix. That cost is asymmetric:

      - Bump UP (cheap -> dearer): the abandoned cache was cheap; if the turn
        looks harder, escalate readily. Low resistance.
      - Bump DOWN (dearer -> cheaper): the current model already has the prefix
        cached at ~0.1x input; the cheaper model would pay full price cold on the
        whole prefix, often costing MORE than staying. So resist going down,
        more so the larger the cached context.

    Implemented as a switch margin the *candidate* must beat the *incumbent* by,
    in predicted P(success), before we move. Up-margin is small; down-margin
    grows with cached context size. Heuristic (cache size modeled from token
    counts, not measured); disabled by default.
    """

    enabled: bool = False
    up_margin: float = 0.0      # extra P(success) gain needed to escalate
    down_margin: float = 0.06   # base extra gain needed to de-escalate
    # additional down-margin per 100k cached context tokens (stickier when big)
    down_margin_per_100k: float = 0.04
    max_down_margin: float = 0.25
    # Non-decision turns are Goose tool results / narration that do not produce
    # a fresh task view. Default to routing them normally rather than blindly
    # preserving the incumbent cache; optional modes can favor cache only for
    # expensive incumbents, or for every incumbent.
    cache_pin_mode: str = "off"  # "off", "dear_only", or "all"
    cache_pin_min_blend: float = 3.0


class PoolConfig(BaseModel):
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    session_health: SessionHealthConfig = Field(default_factory=SessionHealthConfig)
    utility: UtilityConfig = Field(default_factory=UtilityConfig)
    depth_tolerance: DepthToleranceConfig = Field(default_factory=DepthToleranceConfig)
    switching: SwitchingConfig = Field(default_factory=SwitchingConfig)
    models: list[ModelSpec] = Field(default_factory=list)

    @property
    def model_names(self) -> list[str]:
        return [m.name for m in self.models]

    def get_model(self, name: str) -> ModelSpec | None:
        for m in self.models:
            if m.name == name:
                return m
        return None


def load_config(path: str | Path) -> PoolConfig:
    """Load and validate a pool_config.yaml file."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PoolConfig.model_validate(raw)


def build_router_from_config(config: PoolConfig):
    """Construct the appropriate BaseRouter from config."""

    method = config.routing.method.lower()

    if method == "prefill":
        from model_router_toolkit.prefill.router import PrefillRouter

        router = PrefillRouter(config=config)
        if config.routing.checkpoint:
            router.load(config.routing.checkpoint)
        return router

    else:
        raise ValueError(f"Unknown routing method: {method!r}. Supported: 'prefill'.")
