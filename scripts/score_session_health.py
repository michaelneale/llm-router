#!/usr/bin/env python3
"""Score a message list or text snippet with a session-health checkpoint."""

from __future__ import annotations

import argparse
import json
import sys

from model_router_toolkit.session_health import (
    SessionHealthScorer,
    TraceEvent,
    events_from_openai_messages,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/session_health_public.pkl")
    parser.add_argument("--messages-json", default="")
    parser.add_argument("--text", default="")
    args = parser.parse_args()

    scorer = SessionHealthScorer.load(args.checkpoint)
    if args.messages_json:
        messages = json.loads(args.messages_json)
        score = scorer.score_events(events_from_openai_messages(messages))
    else:
        text = args.text or sys.stdin.read()
        score = scorer.score_events([TraceEvent(role="user", text=text)])
    print(
        json.dumps(
            {
                "score": score.score,
                "threshold": score.threshold,
                "should_escalate": score.should_escalate,
                "features": score.features,
                "window_text": score.window_text,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

