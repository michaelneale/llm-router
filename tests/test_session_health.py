from model_router_toolkit.session_health import (
    TraceEvent,
    choose_window_labels,
    events_from_openai_messages,
    health_features,
)


def test_health_features_detect_recent_tool_errors():
    events = events_from_openai_messages(
        [
            {"role": "user", "content": "fix the tests"},
            {"role": "assistant", "content": "I will run pytest."},
            {"role": "tool", "content": "Exit code: 1\nFAILED tests/test_app.py"},
            {"role": "assistant", "content": "I will fix the import and try again."},
            {"role": "tool", "content": "Exit code: 1\nFAILED tests/test_app.py"},
        ]
    )

    feats = health_features(events)

    assert feats["error_recent"] >= 2
    assert feats["repeated_error_recent"] >= 1
    assert feats["last_tool_error"] == 1


def test_failed_trace_labels_onset_not_first_turn():
    events = [
        TraceEvent("user", "fix the bug"),
        TraceEvent("assistant", "I will inspect the files"),
        TraceEvent("tool", "Exit code: 0\nfiles listed", is_tool=True, has_success=True),
        TraceEvent("assistant", "I will run tests"),
        TraceEvent("tool", "Exit code: 1\nFAILED tests/test_app.py", is_tool=True, has_error=True),
        TraceEvent("assistant", "Trying another fix"),
        TraceEvent("tool", "Exit code: 1\nFAILED tests/test_app.py", is_tool=True, has_error=True),
    ]

    labels = choose_window_labels(events, final_success=False, max_windows=7)

    assert labels[0][1] == 0
    assert any(label == 1 for _, label, _ in labels)


def test_successful_recovered_errors_stay_negative():
    events = [
        TraceEvent("user", "fix the bug"),
        TraceEvent("assistant", "I will run tests"),
        TraceEvent("tool", "Exit code: 1\nFAILED tests/test_app.py", is_tool=True, has_error=True),
        TraceEvent("assistant", "I found the issue and fixed it"),
        TraceEvent("tool", "Exit code: 0\nall tests passed", is_tool=True, has_success=True),
    ]

    labels = choose_window_labels(events, final_success=True, max_windows=5)

    assert labels
    assert all(label == 0 for _, label, _ in labels)

