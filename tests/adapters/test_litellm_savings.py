from model_router_toolkit.adapters.litellm.savings import (
    SavingsTracker,
    Usage,
    extract_usage,
    extract_usage_from_sse,
)
from model_router_toolkit.router import CostEstimate, RoutingResult


def _result(selected: str = "cheap") -> RoutingResult:
    return RoutingResult(
        model_names=["cheap", "frontier"],
        confidences=[0.8, 0.9],
        costs=[
            CostEstimate(
                median_output_tokens=100,
                cost_per_m_input_tokens=1.0,
                cost_per_m_output_tokens=10.0,
            ),
            CostEstimate(
                median_output_tokens=100,
                cost_per_m_input_tokens=5.0,
                cost_per_m_output_tokens=25.0,
            ),
        ],
        selected_model=selected,
    )


def test_snapshot_reports_configured_baseline_before_first_request():
    tracker = SavingsTracker()

    tracker.set_display_names(
        {
            "cheap": "openai/cheap",
            "frontier": "anthropic/frontier",
        }
    )
    tracker.set_model_rates(
        {
            "cheap": (0.1, 0.4),
            "frontier": (5.0, 25.0),
        }
    )

    snapshot = tracker.snapshot()

    assert snapshot["baseline_model"] == "anthropic/frontier"
    assert snapshot["requests"] == 0


def test_snapshot_respects_baseline_override_before_first_request():
    tracker = SavingsTracker(baseline_model="cheap")

    tracker.set_display_names({"cheap": "openai/cheap"})
    tracker.set_model_rates(
        {
            "cheap": (0.1, 0.4),
            "frontier": (5.0, 25.0),
        }
    )

    assert tracker.snapshot()["baseline_model"] == "openai/cheap"


def test_extract_usage_includes_openai_cached_prompt_tokens():
    usage = extract_usage(
        b"""
        {
          "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "prompt_tokens_details": {"cached_tokens": 750}
          }
        }
        """
    )

    assert usage == Usage(
        input_tokens=1000,
        output_tokens=200,
        cached_input_tokens=750,
    )


def test_extract_usage_from_sse_keeps_final_cached_usage():
    usage = extract_usage_from_sse(
        "\n".join(
            [
                'data: {"choices":[{"delta":{"content":"hi"}}]}',
                (
                    'data: {"usage":{"input_tokens":1000,"output_tokens":200,'
                    '"cache_read_input_tokens":500}}'
                ),
                "data: [DONE]",
            ]
        )
    )

    assert usage == Usage(
        input_tokens=1000,
        output_tokens=200,
        cached_input_tokens=500,
    )


def test_extract_usage_includes_anthropic_cache_creation_tokens():
    usage = extract_usage(
        b"""
        {
          "usage": {
            "input_tokens": 1000,
            "output_tokens": 200,
            "cache_creation_input_tokens": 700,
            "cache_read_input_tokens": 100
          }
        }
        """
    )

    assert usage == Usage(
        input_tokens=1000,
        output_tokens=200,
        cached_input_tokens=100,
        cache_creation_input_tokens=700,
    )


def test_extract_usage_does_not_double_count_litellm_cache_details():
    usage = extract_usage(
        b"""
        {
          "usage": {
            "input_tokens": 1000,
            "output_tokens": 200,
            "prompt_tokens_details": {
              "cached_tokens": 100,
              "cache_creation_tokens": 700
            },
            "cache_creation_input_tokens": 700,
            "cache_read_input_tokens": 100
          }
        }
        """
    )

    assert usage == Usage(
        input_tokens=1000,
        output_tokens=200,
        cached_input_tokens=100,
        cache_creation_input_tokens=700,
    )


def test_record_prices_cached_input_tokens_with_multiplier(monkeypatch):
    monkeypatch.setenv("ROUTER_CACHE_READ_MULTIPLIER", "0.10")
    monkeypatch.setenv("ROUTER_CACHE_WRITE_MULTIPLIER", "1.25")
    tracker = SavingsTracker(baseline_model="frontier")
    tracker.set_display_names({"cheap": "openai/cheap", "frontier": "anthropic/frontier"})

    tracker.record(_result(), Usage(input_tokens=1000, output_tokens=100, cached_input_tokens=600))

    snapshot = tracker.snapshot()

    assert snapshot["requests"] == 1
    assert snapshot["input_tokens"] == 1000
    assert snapshot["cached_input_tokens"] == 600
    assert snapshot["cache_creation_input_tokens"] == 0
    assert snapshot["output_tokens"] == 100
    assert snapshot["actual_cost_usd"] == 0.00146
    assert snapshot["baseline_cost_usd"] == 0.0048


def test_record_prices_cache_creation_tokens_with_multiplier(monkeypatch):
    monkeypatch.setenv("ROUTER_CACHE_READ_MULTIPLIER", "0.10")
    monkeypatch.setenv("ROUTER_CACHE_WRITE_MULTIPLIER", "1.25")
    tracker = SavingsTracker(baseline_model="frontier")
    tracker.set_display_names({"cheap": "openai/cheap", "frontier": "anthropic/frontier"})

    tracker.record(
        _result(),
        Usage(
            input_tokens=1000,
            output_tokens=100,
            cached_input_tokens=200,
            cache_creation_input_tokens=600,
        ),
    )

    snapshot = tracker.snapshot()

    assert snapshot["cached_input_tokens"] == 200
    assert snapshot["cache_creation_input_tokens"] == 600
    assert snapshot["actual_cost_usd"] == 0.00197
    assert snapshot["baseline_cost_usd"] == 0.00685
