import importlib.util
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "router_trial_report",
    ROOT / "scripts" / "router_trial_report.py",
)
assert SPEC is not None
router_trial_report = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(router_trial_report)


def _args(**overrides):
    values = {
        "min_real_rows": 1,
        "min_proxy_savings_pct": 30.0,
        "min_cache_savings_pct": 0.0,
        "max_baseline_share_pct": 35.0,
        "max_quality_score100": 25.0,
        "max_tool_error_rate_pct": 25.0,
        "require_calibration": True,
        "require_selected_tolerance": False,
        "tolerance_epsilon": 1e-6,
        "max_verified_loss_pp": 1.5,
        "min_verified_savings_pct": 50.0,
        "allow_unmatched": True,
    }
    values.update(overrides)
    return Namespace(**values)


def _summary(*, delta_pp: float, savings_pct: float, observed=None):
    return {
        "real_rows": 1,
        "trial_tolerance": 0.11,
        "observed_tolerances": observed or {"0.110000": 1},
        "route": {
            "known_selected_rows": 1,
            "proxy_savings_pct": 38.0,
            "cache_savings_pct": 59.0,
            "baseline_share_pct": 12.0,
        },
        "quality": {
            "matched_sessions": 0,
            "avg_problem_score_per_100_messages": 0.0,
            "tool_error_rate_pct": 0.0,
        },
        "calibration": {
            "point": {
                "delta_pp": delta_pp,
                "savings_pct": savings_pct,
            },
            "selected_trial": None,
        },
    }


def test_observed_tolerances_reads_top_level_and_metadata_values():
    rows = [
        {"tolerance": 0.11},
        {"metadata": {"tolerance": "0.11"}},
        {"metadata": {"tolerance": 0.125}},
        {"metadata": {"tolerance": "not-a-number"}},
    ]

    assert router_trial_report.observed_tolerances(rows) == {
        "0.110000": 2,
        "0.125000": 1,
    }


def test_gate_accepts_selected_small_loss_high_savings_point():
    failures = router_trial_report.evaluate_gate(
        _summary(delta_pp=-1.144, savings_pct=51.6),
        _args(),
    )

    assert failures == []


def test_gate_rejects_more_aggressive_loss_even_when_savings_are_high():
    failures = router_trial_report.evaluate_gate(
        _summary(delta_pp=-1.945, savings_pct=57.1, observed={"0.125000": 1}),
        _args(),
    )

    assert any("verified-label loss 1.9pp > 1.5pp" in failure for failure in failures)


def test_gate_rejects_mixed_tolerances_in_one_trial_window():
    failures = router_trial_report.evaluate_gate(
        _summary(
            delta_pp=-1.144,
            savings_pct=51.6,
            observed={"0.110000": 2, "0.125000": 1},
        ),
        _args(),
    )

    assert any("multiple observed route tolerances" in failure for failure in failures)
