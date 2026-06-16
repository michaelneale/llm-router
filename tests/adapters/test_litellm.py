import pytest

from model_router_toolkit.adapters.litellm.strategy import ModelRoutingStrategy
from model_router_toolkit.router import BaseRouter, CostEstimate, RoutingResult


class FakeRouter(BaseRouter):
    """Deterministic router for testing the strategy wrapper."""

    def __init__(self, selected: str = "model-a"):
        self._selected = selected
        self._pool = ["model-a", "model-b"]

    def load(self, checkpoint_path):
        pass

    def route(self, question, *, tolerance=0.10, models=None):
        allowed = set(models) if models else set(self._pool)
        selected = (
            self._selected
            if self._selected in allowed
            else next(m for m in self._pool if m in allowed)
        )
        return RoutingResult(
            model_names=["model-a", "model-b"],
            confidences=[0.9, 0.7],
            costs=[
                CostEstimate(
                    median_output_tokens=100,
                    cost_per_m_input_tokens=0.1,
                    cost_per_m_output_tokens=0.1,
                ),
                CostEstimate(
                    median_output_tokens=200,
                    cost_per_m_input_tokens=1.0,
                    cost_per_m_output_tokens=1.0,
                ),
            ],
            selected_model=selected,
            metadata={"test": True, "allowed_models": sorted(allowed)},
        )

    def has_model(self, model_name):
        return model_name in self._pool

    def resolve(self, model_name):
        if model_name not in self._pool:
            return None
        return RoutingResult(
            model_names=self._pool,
            confidences=[1.0 if m == model_name else 0.0 for m in self._pool],
            costs=[
                CostEstimate(
                    median_output_tokens=100,
                    cost_per_m_input_tokens=0.1,
                    cost_per_m_output_tokens=0.1,
                ),
                CostEstimate(
                    median_output_tokens=200,
                    cost_per_m_input_tokens=1.0,
                    cost_per_m_output_tokens=1.0,
                ),
            ],
            selected_model=model_name,
            metadata={"pinned": True},
        )


class TestModelRoutingStrategy:
    def test_sync_routing(self):
        strategy = ModelRoutingStrategy(FakeRouter("model-a"), tolerance=0.20)
        strategy._litellm_router = type(
            "R",
            (),
            {
                "model_list": [
                    {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                    {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
                ]
            },
        )()

        dep = strategy.get_available_deployment(
            model="test",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert dep["model_name"] == "model-a"
        assert strategy.last_result is not None
        assert strategy.last_result.selected_model == "model-a"

    def test_tolerance_bounds(self):
        strategy = ModelRoutingStrategy(FakeRouter(), tolerance=0.10)
        strategy.tolerance = -0.5
        assert strategy.tolerance == 0.0
        strategy.tolerance = 1.5
        assert strategy.tolerance == 1.0

    def test_from_config_tolerance_env_override(self, tmp_path, monkeypatch):
        cfg = tmp_path / "pool.yaml"
        cfg.write_text(
            """
routing:
  method: prefill
  tolerance: 0.02
models: []
"""
        )
        monkeypatch.setenv("ROUTER_TOLERANCE", "0.125")

        strategy = ModelRoutingStrategy.from_config(str(cfg))

        assert strategy.tolerance == 0.125

    def test_empty_messages(self):
        strategy = ModelRoutingStrategy(FakeRouter(), tolerance=0.20)
        strategy._litellm_router = type("R", (), {"model_list": [{"model_name": "x"}]})()
        strategy.get_available_deployment(model="test", messages=[])
        assert strategy.last_result is None

    def test_extract_user_text(self):
        strategy = ModelRoutingStrategy(FakeRouter())
        text = strategy._extract_user_text(
            [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "What is 2+2?"},
            ]
        )
        assert text == "What is 2+2?"

    def test_extract_multipart_content(self):
        strategy = ModelRoutingStrategy(FakeRouter())
        text = strategy._extract_user_text(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": "World"},
                    ],
                },
            ]
        )
        assert text == "Hello World"

    @pytest.mark.asyncio
    async def test_async_routing(self):
        strategy = ModelRoutingStrategy(FakeRouter("model-b"), tolerance=0.20)
        strategy._litellm_router = type(
            "R",
            (),
            {
                "model_list": [
                    {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                    {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
                ]
            },
        )()

        dep = await strategy.async_get_available_deployment(
            model="test",
            messages=[{"role": "user", "content": "Test"}],
        )
        assert dep["model_name"] == "model-b"

    def test_pin_model_metadata_bypasses_routing(self):
        """When request_kwargs has pin_model, skip ML and return pinned model."""
        strategy = ModelRoutingStrategy(FakeRouter("model-a"), tolerance=0.20)
        strategy._litellm_router = type(
            "R",
            (),
            {
                "model_list": [
                    {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                    {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
                ]
            },
        )()

        dep = strategy.get_available_deployment(
            model="model-a",
            messages=[{"role": "user", "content": "Hello"}],
            request_kwargs={"metadata": {"pin_model": "model-b"}},
        )
        assert dep["model_name"] == "model-b"
        assert strategy.last_result is not None
        assert strategy.last_result.selected_model == "model-b"
        assert strategy.last_result.metadata.get("pinned") is True

    def test_pin_model_unknown_falls_through(self):
        """When pin_model is not in the pool, route normally via ML."""
        strategy = ModelRoutingStrategy(FakeRouter("model-a"), tolerance=0.20)
        strategy._litellm_router = type(
            "R",
            (),
            {
                "model_list": [
                    {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                    {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
                ]
            },
        )()

        dep = strategy.get_available_deployment(
            model="model-a",
            messages=[{"role": "user", "content": "Hello"}],
            request_kwargs={"metadata": {"pin_model": "unknown-model"}},
        )
        assert dep["model_name"] == "model-a"
        assert strategy.last_result.metadata.get("pinned") is None

    def test_no_pin_model_routes_normally(self):
        """Without pin_model metadata, always route via ML even if model is a pool name."""
        strategy = ModelRoutingStrategy(FakeRouter("model-a"), tolerance=0.20)
        strategy._litellm_router = type(
            "R",
            (),
            {
                "model_list": [
                    {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                    {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
                ]
            },
        )()

        dep = strategy.get_available_deployment(
            model="model-b",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert dep["model_name"] == "model-a"
        assert strategy.last_result.metadata.get("test") is True

    @pytest.mark.asyncio
    async def test_pin_model_async(self):
        """Async path also respects pin_model metadata."""
        strategy = ModelRoutingStrategy(FakeRouter("model-a"), tolerance=0.20)
        strategy._litellm_router = type(
            "R",
            (),
            {
                "model_list": [
                    {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                    {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
                ]
            },
        )()

        dep = await strategy.async_get_available_deployment(
            model="model-a",
            messages=[{"role": "user", "content": "Test"}],
            request_kwargs={"metadata": {"pin_model": "model-b"}},
        )
        assert dep["model_name"] == "model-b"
        assert strategy.last_result.metadata.get("pinned") is True

    def test_models_via_request_metadata(self):
        """Models subset passed via request metadata restricts routing."""
        strategy = ModelRoutingStrategy(FakeRouter("model-a"), tolerance=0.20)
        strategy._litellm_router = type(
            "R",
            (),
            {
                "model_list": [
                    {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                    {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
                ]
            },
        )()

        dep = strategy.get_available_deployment(
            model="test",
            messages=[{"role": "user", "content": "Hello"}],
            request_kwargs={"metadata": {"models": ["model-b"]}},
        )
        assert dep["model_name"] == "model-b"
        assert "model-b" in strategy.last_result.metadata["allowed_models"]

    def test_models_server_default(self):
        """Server-wide models default restricts routing when no per-request override."""
        strategy = ModelRoutingStrategy(
            FakeRouter("model-a"),
            tolerance=0.20,
            models=["model-b"],
        )
        strategy._litellm_router = type(
            "R",
            (),
            {
                "model_list": [
                    {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                    {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
                ]
            },
        )()

        dep = strategy.get_available_deployment(
            model="test",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert dep["model_name"] == "model-b"

    def test_models_request_overrides_server_default(self):
        """Per-request models override the server-wide default."""
        strategy = ModelRoutingStrategy(
            FakeRouter("model-a"),
            tolerance=0.20,
            models=["model-b"],
        )
        strategy._litellm_router = type(
            "R",
            (),
            {
                "model_list": [
                    {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                    {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
                ]
            },
        )()

        dep = strategy.get_available_deployment(
            model="test",
            messages=[{"role": "user", "content": "Hello"}],
            request_kwargs={"metadata": {"models": ["model-a"]}},
        )
        assert dep["model_name"] == "model-a"

    def test_goose_title_generation_uses_cheapest_model(self):
        """Goose title prompts are utility work, not a routing decision."""
        strategy = ModelRoutingStrategy(FakeRouter("model-b"), tolerance=0.20)
        strategy._litellm_router = type(
            "R",
            (),
            {
                "model_list": [
                    {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                    {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
                ]
            },
        )()

        dep = strategy.get_available_deployment(
            model="test",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "---BEGIN USER MESSAGES---\n"
                        "how does this repo work?\n"
                        "---END USER MESSAGES---\n\n"
                        "Generate a short title for the above messages."
                    ),
                }
            ],
        )

        assert dep["model_name"] == "model-a"
        assert strategy.last_result is not None
        assert strategy.last_result.metadata["pin_reason"] == "goose_title_generation"

    def test_cheap_utility_pattern_uses_cheapest_model_for_cold_session(self):
        strategy = ModelRoutingStrategy(
            FakeRouter("model-b"),
            tolerance=0.20,
            cheap_patterns=[r"^\s*(list|show)\s+(the\s+)?files\b"],
        )
        strategy._litellm_router = type(
            "R",
            (),
            {
                "model_list": [
                    {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                    {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
                ]
            },
        )()

        dep = strategy.get_available_deployment(
            model="test",
            messages=[{"role": "user", "content": "list files in this directory"}],
            request_kwargs={"metadata": {"router_session_id": "cold"}},
        )

        assert dep["model_name"] == "model-a"
        assert strategy.last_result is not None
        assert strategy.last_result.metadata["pin_reason"] == "cheap_utility_pattern"

    def test_goose_info_only_cold_uses_cheapest_model(self):
        """Cold context-refresh turns should not independently escalate."""
        strategy = ModelRoutingStrategy(FakeRouter("model-b"), tolerance=0.20)
        strategy._litellm_router = type(
            "R",
            (),
            {
                "model_list": [
                    {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                    {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
                ]
            },
        )()

        dep = strategy.get_available_deployment(
            model="test",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "<info-msg>\n"
                        "Working directory: /tmp/repo\n"
                        "Context: ~7k/128k tokens used (6%)\n"
                        "</info-msg>"
                    ),
                }
            ],
        )

        assert dep["model_name"] == "model-a"
        assert strategy.last_result is not None
        assert strategy.last_result.metadata["pin_reason"] == "goose_info_only"

    def test_goose_info_only_default_does_not_pin_existing_session_model(self):
        """Default mode routes context-refresh turns instead of cache-pinning."""
        strategy = ModelRoutingStrategy(FakeRouter("model-b"), tolerance=0.20)
        strategy._litellm_router = type(
            "R",
            (),
            {
                "model_list": [
                    {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                    {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
                ]
            },
        )()

        strategy.get_available_deployment(
            model="test",
            messages=[{"role": "user", "content": "Fix the failing tests"}],
            request_kwargs={"metadata": {"router_session_id": "s1"}},
        )
        dep = strategy.get_available_deployment(
            model="test",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "<info-msg>\n"
                        "Working directory: /tmp/repo\n"
                        "Context: ~7k/128k tokens used (6%)\n"
                        "</info-msg>"
                    ),
                }
            ],
            request_kwargs={"metadata": {"router_session_id": "s1"}},
        )

        assert dep["model_name"] == "model-a"
        assert strategy.last_result is not None
        assert strategy.last_result.metadata["pin_reason"] == "goose_info_only"

    def test_goose_info_only_can_pin_existing_session_when_enabled(self):
        """The old sticky behavior is still available as an explicit mode."""
        strategy = ModelRoutingStrategy(FakeRouter("model-b"), tolerance=0.20)
        strategy.cache_pin_mode = "all"
        strategy._litellm_router = type(
            "R",
            (),
            {
                "model_list": [
                    {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                    {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
                ]
            },
        )()

        strategy.get_available_deployment(
            model="test",
            messages=[{"role": "user", "content": "Fix the failing tests"}],
            request_kwargs={"metadata": {"router_session_id": "s1"}},
        )
        dep = strategy.get_available_deployment(
            model="test",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "<info-msg>\n"
                        "Working directory: /tmp/repo\n"
                        "Context: ~7k/128k tokens used (6%)\n"
                        "</info-msg>"
                    ),
                }
            ],
            request_kwargs={"metadata": {"router_session_id": "s1"}},
        )

        assert dep["model_name"] == "model-b"
        assert strategy.last_result is not None
        assert strategy.last_result.metadata["pin_reason"] == "non_decision_turn"
        assert strategy.last_result.metadata["cache_pin_mode"] == "all"

    def test_dear_only_cache_pin_skips_cheap_incumbent(self):
        strategy = ModelRoutingStrategy(FakeRouter("model-a"), tolerance=0.20)
        strategy.cache_pin_mode = "dear_only"
        strategy.cache_pin_min_blend = 3.0
        strategy._litellm_router = type(
            "R",
            (),
            {
                "model_list": [
                    {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                    {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
                ]
            },
        )()

        strategy.get_available_deployment(
            model="test",
            messages=[{"role": "user", "content": "list files in this directory"}],
            request_kwargs={"metadata": {"router_session_id": "s1"}},
        )
        dep = strategy.get_available_deployment(
            model="test",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "<info-msg>\n"
                        "Working directory: /tmp/repo\n"
                        "</info-msg>"
                    ),
                }
            ],
            request_kwargs={"metadata": {"router_session_id": "s1"}},
        )

        assert dep["model_name"] == "model-a"
        assert strategy.last_result is not None
        assert strategy.last_result.metadata["pin_reason"] == "goose_info_only"
