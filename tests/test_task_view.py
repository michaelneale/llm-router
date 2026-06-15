from model_router_toolkit.task_view import build_task_view


def test_build_task_view_drops_goose_title_request():
    view = build_task_view(
        [
            {
                "role": "user",
                "content": (
                    "---BEGIN USER MESSAGES---\n"
                    "how does this repo work?\n"
                    "---END USER MESSAGES---\n\n"
                    "Generate a short title for the above messages."
                ),
            }
        ]
    )

    assert view is None


def test_build_task_view_drops_info_only_request():
    view = build_task_view(
        [
            {
                "role": "user",
                "content": (
                    "<info-msg>\n"
                    "Working directory: /tmp/repo\n"
                    "Context: ~7k/128k tokens used (6%)\n"
                    "</info-msg>"
                ),
            }
        ]
    )

    assert view is None


def test_build_task_view_strips_info_msg_from_real_turn():
    view = build_task_view(
        [
            {
                "role": "user",
                "content": (
                    "does this repo have a PR open by me?\n"
                    "<info-msg>\n"
                    "Working directory: /tmp/repo\n"
                    "</info-msg>"
                ),
            }
        ]
    )

    assert view == "does this repo have a PR open by me?"
