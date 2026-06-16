from model_router_toolkit.session_health import (
    TraceEvent,
    choose_window_labels,
    events_from_openai_messages,
    health_features,
    window_text,
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


def test_health_features_detect_command_patch_test_loop():
    events = events_from_openai_messages(
        [
            {"role": "user", "content": "fix the failing test"},
            {
                "role": "assistant",
                "content": "I will run the focused test.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": {"command": "cd /testbed && pytest tests/test_app.py"},
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": "Exit code: 1\nFAILED tests/test_app.py::test_widget",
            },
            {
                "role": "assistant",
                "content": "I will patch app.py.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": {"command": "cd /testbed && sed -i s/foo/bar/ app.py"},
                        }
                    }
                ],
            },
            {"role": "tool", "content": "Exit code: 0\npatched app.py"},
            {
                "role": "assistant",
                "content": "Rerunning the test.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": {"command": "cd /testbed && pytest tests/test_app.py"},
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": "Exit code: 1\nFAILED tests/test_app.py::test_widget",
            },
        ]
    )

    feats = health_features(events)

    assert feats["command_count"] >= 3
    assert feats["patch_recent"] >= 1
    assert feats["repeated_failing_command_recent"] >= 1
    assert feats["test_failure_after_patch_recent"] >= 1
    assert feats["same_error_after_patch_recent"] >= 1


def test_initial_user_error_log_is_not_operational_failure():
    events = events_from_openai_messages(
        [
            {
                "role": "user",
                "content": "Please fix this Traceback: AssertionError in tests/test_app.py",
            }
        ]
    )

    feats = health_features(events)

    assert feats["error_count"] == 0
    assert feats["last_event_error"] == 0
    assert "[error]" not in window_text(events)


def test_tool_call_parse_error_counts_as_operational_error():
    events = events_from_openai_messages(
        [
            {"role": "user", "content": "run the tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": {
                    "function": "bash",
                    "arguments": {"cmd": "pytest"},
                    "parse_error": "invalid tool call json",
                },
            },
        ]
    )

    feats = health_features(events)

    assert feats["error_count"] == 1
    assert feats["parse_error_count"] == 1


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
