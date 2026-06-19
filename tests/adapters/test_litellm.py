import pytest

from model_router_toolkit.adapters.litellm.strategy import (
    ModelRoutingStrategy,
    inject_anthropic_cache_markers,
)
from model_router_toolkit.adapters.litellm.embedding import render_messages_for_embedding
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


class FakeEmbeddingScorer:
    def __init__(self, complexity: float):
        self.complexity = complexity

    def score_messages(self, messages):
        return type(
            "Score",
            (),
            {
                "complexity": self.complexity,
                "tool_calls_norm": 0.0,
                "elapsed_ms": 1,
                "rendered": ">>> user: test",
            },
        )()


class FakeSessionHealthScorer:
    def __init__(self, score: float, features: dict[str, float]):
        self.score = score
        self.features = features

    def score_messages(self, messages, *, task: str = ""):
        return type(
            "HealthScore",
            (),
            {
                "score": self.score,
                "threshold": 0.0,
                "should_escalate": False,
                "features": self.features,
                "window_text": task,
            },
        )()


def _count_cache_markers(obj):
    if isinstance(obj, dict):
        return (1 if "cache_control" in obj else 0) + sum(
            _count_cache_markers(v) for v in obj.values()
        )
    if isinstance(obj, list):
        return sum(_count_cache_markers(v) for v in obj)
    return 0


class TierRouter(BaseRouter):
    """Router with one model per cost tier for embedding-ladder tests."""

    def __init__(self, n: int = 5):
        self._pool = [f"model-{i}" for i in range(n)]

    def load(self, checkpoint_path):
        pass

    def route(self, question, *, tolerance=0.10, models=None):
        return self.resolve(self._pool[0])

    def has_model(self, model_name):
        return model_name in self._pool

    def resolve(self, model_name):
        if model_name not in self._pool:
            return None
        costs = [
            CostEstimate(
                median_output_tokens=100,
                cost_per_m_input_tokens=float(i + 1),
                cost_per_m_output_tokens=float(i + 1),
            )
            for i, _ in enumerate(self._pool)
        ]
        return RoutingResult(
            model_names=self._pool,
            confidences=[1.0 if m == model_name else 0.0 for m in self._pool],
            costs=costs,
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

    def test_anthropic_selection_injects_prompt_cache_markers(self):
        dep = {
            "model_name": "model-b",
            "litellm_params": {"model": "anthropic/claude-opus-4-8"},
        }
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Fix this bug"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "I'll inspect it."}]},
            {"role": "user", "content": [{"type": "tool_result", "content": "first result"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "I found a lead."}]},
            {"role": "user", "content": [{"type": "tool_result", "content": "second result"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "I'll patch it."}]},
            {"role": "user", "content": [{"type": "tool_result", "content": "test output"}]},
        ]
        request_kwargs = {
            "system": [{"type": "text", "text": "You are goose."}],
            "tools": [
                {"name": "shell", "description": "run commands", "input_schema": {"type": "object"}}
            ],
        }

        added = inject_anthropic_cache_markers(dep, messages, request_kwargs)

        assert added == 4
        assert request_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert request_kwargs["tools"][0]["cache_control"] == {"type": "ephemeral"}
        assert messages[2]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert messages[-1]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert _count_cache_markers({"messages": messages, **request_kwargs}) == 4

    def test_openai_selection_does_not_inject_prompt_cache_markers(self):
        dep = {"model_name": "model-a", "litellm_params": {"model": "openai/a"}}
        messages = [{"role": "user", "content": "hello"}]
        request_kwargs = {
            "system": "You are goose.",
            "tools": [{"name": "shell", "description": "run commands"}],
        }

        added = inject_anthropic_cache_markers(dep, messages, request_kwargs)

        assert added == 0
        assert _count_cache_markers({"messages": messages, **request_kwargs}) == 0

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

    def test_embedding_routed_low_complexity_uses_cheapest_model(self):
        """The embedding alias bypasses prefill and chooses from the cost ladder."""
        strategy = ModelRoutingStrategy(FakeRouter("model-b"), tolerance=0.20)
        strategy._embedding_scorer = FakeEmbeddingScorer(0.10)
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
            model="embedding-routed",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert dep["model_name"] == "model-a"
        assert strategy.last_result is not None
        assert strategy.last_result.selected_model == "model-a"
        assert strategy.last_result.metadata["router_mode"] == "embedding"
        assert strategy.last_result.metadata["complexity"] == 0.1

    def test_embedding_routed_high_complexity_uses_dearest_model(self):
        """High complexity reaches the top of the cost ladder."""
        strategy = ModelRoutingStrategy(FakeRouter("model-a"), tolerance=0.20)
        strategy._embedding_scorer = FakeEmbeddingScorer(0.80)
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
            model="embedding-routed",
            messages=[{"role": "user", "content": "Debug this flaky test"}],
        )

        assert dep["model_name"] == "model-b"
        assert strategy.last_result is not None
        assert strategy.last_result.selected_model == "model-b"
        assert strategy.last_result.metadata["router_mode"] == "embedding"
        assert strategy.last_result.metadata["complexity"] == 0.8

    def test_embedding_routed_standard_complexity_uses_middle_ladder(self):
        """The public complexity rubric maps standard work to the middle tier."""
        strategy = ModelRoutingStrategy(TierRouter(5), tolerance=0.20)
        strategy._embedding_scorer = FakeEmbeddingScorer(0.44)
        strategy._litellm_router = type(
            "R",
            (),
            {
                "model_list": [
                    {"model_name": f"model-{i}", "litellm_params": {"model": f"openai/{i}"}}
                    for i in range(5)
                ]
            },
        )()

        dep = strategy.get_available_deployment(
            model="embedding-routed",
            messages=[{"role": "user", "content": "Fix this focused coding bug"}],
        )

        assert dep["model_name"] == "model-2"
        assert strategy.last_result is not None
        assert strategy.last_result.selected_model == "model-2"
        assert strategy.last_result.metadata["ladder"] == [
            "model-0",
            "model-1",
            "model-2",
            "model-3",
            "model-4",
        ]

    def test_embedding_render_anchors_last_real_user_not_goose_info(self):
        rendered = render_messages_for_embedding(
            [
                {"role": "user", "content": "Fix the failing CLI test"},
                {"role": "assistant", "content": "I will inspect the failure."},
                {"role": "tool", "content": "pytest failed with exit code: 1"},
                {
                    "role": "user",
                    "content": "<info-msg>\nWorking directory: /tmp/repo\n</info-msg>",
                },
            ]
        )

        assert ">>> user: Fix the failing CLI test" in rendered
        assert "Working directory" not in rendered
        assert "tool: pytest failed with exit code: 1" in rendered
        assert rendered.count(">>>") == 1

    def test_embedding_render_drops_stale_history_before_latest_user(self):
        rendered = render_messages_for_embedding(
            [
                {"role": "user", "content": "Fix the CORS issue"},
                {
                    "role": "assistant",
                    "content": "Old CORS reasoning " * 80,
                },
                {
                    "role": "user",
                    "content": "ok open the draft PR and monitor CI",
                },
                {"role": "tool", "content": "created draft PR https://example.test/pr/1"},
                {
                    "role": "tool",
                    "content": "Lint Rust Code pending\nBuild and Test pending",
                },
                {
                    "role": "user",
                    "content": "<info-msg>\nWorking directory: /tmp/repo\n</info-msg>",
                },
            ]
        )

        assert rendered.startswith(">>> user: ok open the draft PR and monitor CI")
        assert "Old CORS reasoning" not in rendered
        assert "Working directory" not in rendered
        assert "tool: Lint Rust Code pending" in rendered

    def test_embedding_render_budget_keeps_newest_state(self):
        messages = [{"role": "user", "content": "Fix the failing build"}]
        messages.extend(
            {
                "role": "tool",
                "content": f"old compile warning {i}\n" + ("noise " * 120),
            }
            for i in range(12)
        )
        messages.append(
            {
                "role": "tool",
                "content": "LATEST cargo test passed\nfinished target/debug/deps/acp_cors_test",
            }
        )

        rendered = render_messages_for_embedding(messages)

        assert len(rendered) <= 1400
        assert "LATEST cargo test passed" in rendered
        assert "old compile warning 0" not in rendered

    def test_embedding_routed_info_only_followup_still_scores_embedding(self):
        strategy = ModelRoutingStrategy(FakeRouter("model-b"), tolerance=0.20)
        strategy._embedding_scorer = FakeEmbeddingScorer(0.80)
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
            model="embedding-routed",
            messages=[
                {"role": "user", "content": "Fix the failing CLI test"},
                {"role": "assistant", "content": "The first attempt failed."},
                {
                    "role": "user",
                    "content": "<info-msg>\nWorking directory: /tmp/repo\n</info-msg>",
                },
            ],
        )

        assert dep["model_name"] == "model-b"
        assert strategy.last_result.metadata["router_mode"] == "embedding"
        assert strategy.last_result.metadata.get("utility") is None

    def test_embedding_routed_title_generation_stays_cheap(self):
        """Goose title-generation bookkeeping should not spend the embedding scorer."""
        strategy = ModelRoutingStrategy(FakeRouter("model-b"), tolerance=0.20)
        strategy._embedding_scorer = FakeEmbeddingScorer(0.99)
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
            model="embedding-routed",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "---BEGIN USER MESSAGES---\n"
                        "fix this bug\n"
                        "---END USER MESSAGES---\n\n"
                        "Generate a short title for the above messages."
                    ),
                }
            ],
        )

        assert dep["model_name"] == "model-a"
        assert strategy.last_result is not None
        assert strategy.last_result.selected_model == "model-a"
        assert strategy.last_result.metadata["pin_reason"] == "goose_title_generation"

    def test_embedding_routed_hard_override_uses_escalation_model(self):
        """The alternate scorer still shares the common hard-override policy."""
        strategy = ModelRoutingStrategy(
            FakeRouter("model-a"),
            tolerance=0.20,
            escalation_patterns=[r"(^|\s)!hard\b"],
            escalation_model="model-b",
        )
        strategy._embedding_scorer = FakeEmbeddingScorer(0.10)
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
            model="embedding-routed",
            messages=[{"role": "user", "content": "!hard fix this"}],
        )

        assert dep["model_name"] == "model-b"
        assert strategy.last_result is not None
        assert strategy.last_result.selected_model == "model-b"
        assert strategy.last_result.metadata["escalated"] is True

    def test_embedding_routed_ignores_shallow_session_health_spike(self):
        """A first-turn health false positive should not override the ladder."""
        strategy = ModelRoutingStrategy(
            FakeRouter("model-b"),
            tolerance=0.20,
            escalation_model="model-b",
        )
        strategy._embedding_scorer = FakeEmbeddingScorer(0.10)
        strategy._session_health_threshold = 0.88
        strategy._session_health_scorer = FakeSessionHealthScorer(
            0.96,
            {
                "event_count": 2.0,
                "assistant_count": 0.0,
                "tool_count": 0.0,
                "error_count": 2.0,
                "error_rate": 1.0,
            },
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
            model="embedding-routed",
            messages=[{"role": "user", "content": "Reply with exactly: ready"}],
        )

        assert dep["model_name"] == "model-a"
        assert strategy.last_result is not None
        assert strategy.last_result.selected_model == "model-a"
        assert strategy.last_result.metadata["router_mode"] == "embedding"
        assert strategy.last_result.metadata.get("escalated") is None

    def test_embedding_routed_escalates_bad_session_health_with_history(self):
        """A high health score with real trajectory evidence still forces top tier."""
        strategy = ModelRoutingStrategy(
            FakeRouter("model-a"),
            tolerance=0.20,
            escalation_model="model-b",
        )
        strategy._embedding_scorer = FakeEmbeddingScorer(0.10)
        strategy._session_health_threshold = 0.88
        strategy._session_health_scorer = FakeSessionHealthScorer(
            0.96,
            {
                "event_count": 5.0,
                "assistant_count": 2.0,
                "tool_count": 2.0,
                "command_count": 2.0,
                "error_count": 2.0,
                "repeated_error_recent": 1.0,
            },
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
            model="embedding-routed",
            messages=[{"role": "user", "content": "Fix this failing loop"}],
        )

        assert dep["model_name"] == "model-b"
        assert strategy.last_result is not None
        assert strategy.last_result.selected_model == "model-b"
        assert strategy.last_result.metadata["escalated"] is True
        assert "session_health" in strategy.last_result.metadata

    def test_embedding_routed_info_only_does_not_trip_session_health(self):
        """Goose bookkeeping continuations should not repeatedly force top tier."""
        strategy = ModelRoutingStrategy(
            FakeRouter("model-b"),
            tolerance=0.20,
            escalation_model="model-b",
        )
        strategy._embedding_scorer = FakeEmbeddingScorer(0.10)
        strategy._session_health_threshold = 0.88
        strategy._session_health_scorer = FakeSessionHealthScorer(
            0.99,
            {
                "event_count": 12.0,
                "assistant_count": 5.0,
                "tool_count": 5.0,
                "command_count": 5.0,
                "error_count": 12.0,
                "repeated_error_recent": 4.0,
            },
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
            model="embedding-routed",
            messages=[
                {"role": "user", "content": "Fix this failing loop"},
                {"role": "assistant", "content": "The test command failed with exit code: 1"},
                {
                    "role": "user",
                    "content": "<info-msg>\nWorking directory: /tmp/repo\n</info-msg>",
                },
            ],
        )

        assert dep["model_name"] == "model-a"
        assert strategy.last_result is not None
        assert strategy.last_result.selected_model == "model-a"
        assert strategy.last_result.metadata["router_mode"] == "embedding"
        assert strategy.last_result.metadata.get("session_health") is None
        assert strategy.last_result.metadata.get("escalated") is None

    def test_turbo_forces_top_model(self):
        """Dashboard turbo mode forces the configured top model."""
        strategy = ModelRoutingStrategy(
            FakeRouter("model-a"),
            tolerance=0.20,
            escalation_model="model-b",
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

        state = strategy.set_turbo(enabled=True, duration_seconds=1800)
        dep = strategy.get_available_deployment(
            model="nvidia-routed",
            messages=[{"role": "user", "content": "hello"}],
        )

        assert state["active"] is True
        assert 0 < state["remaining_seconds"] <= 1800
        assert dep["model_name"] == "model-b"
        assert strategy.last_result is not None
        assert strategy.last_result.selected_model == "model-b"
        assert strategy.last_result.metadata["escalated"] is True
        assert strategy.last_result.metadata["turbo"]["active"] is True

    def test_turbo_overrides_title_generation_cheap_path(self):
        """When turbo is on, even bookkeeping requests use the top tier."""
        strategy = ModelRoutingStrategy(
            FakeRouter("model-a"),
            tolerance=0.20,
            escalation_model="model-b",
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
        strategy.set_turbo(enabled=True, duration_seconds=1800)

        dep = strategy.get_available_deployment(
            model="embedding-routed",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "---BEGIN USER MESSAGES---\n"
                        "fix this bug\n"
                        "---END USER MESSAGES---\n\n"
                        "Generate a short title for the above messages."
                    ),
                }
            ],
        )

        assert dep["model_name"] == "model-b"
        assert strategy.last_result is not None
        assert strategy.last_result.selected_model == "model-b"
        assert strategy.last_result.metadata["turbo"]["active"] is True

    def test_turbo_expires(self):
        """Expired turbo mode falls back to normal routing."""
        strategy = ModelRoutingStrategy(
            FakeRouter("model-a"),
            tolerance=0.20,
            escalation_model="model-b",
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
        strategy.set_turbo(enabled=True, duration_seconds=1)
        strategy._turbo_until = 1.0

        state = strategy.turbo_state()
        dep = strategy.get_available_deployment(
            model="nvidia-routed",
            messages=[{"role": "user", "content": "hello"}],
        )

        assert state["active"] is False
        assert dep["model_name"] == "model-a"
        assert strategy.last_result is not None
        assert "turbo" not in strategy.last_result.metadata

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
                    "content": ("<info-msg>\nWorking directory: /tmp/repo\n</info-msg>"),
                }
            ],
            request_kwargs={"metadata": {"router_session_id": "s1"}},
        )

        assert dep["model_name"] == "model-a"
        assert strategy.last_result is not None
        assert strategy.last_result.metadata["pin_reason"] == "goose_info_only"
