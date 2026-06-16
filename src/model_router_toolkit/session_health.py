"""Learned session-health detector for agentic routing.

The normal router predicts which model can solve the current task. This module
predicts a different target: whether the recent trajectory already looks like it
is going badly. Runtime policy can then escalate when the learned score crosses
the configured threshold.
"""

from __future__ import annotations

import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ERROR_RE = re.compile(
    r"\b("
    r"exit code:\s*[1-9]\d*|traceback|exception|assertionerror|failed|failure|"
    r"error:|errno|no such file|not found|command not found|permission denied|"
    r"timeout|timed out|segmentation fault|syntaxerror|typeerror|valueerror|"
    r"pytest.*failed|tests?\s+failed|cannot|can't|unable"
    r")\b",
    re.I,
)
SUCCESS_RE = re.compile(
    r"\b(exit code:\s*0|all tests? passed|passed\b|success(?:ful|fully)?|done)\b",
    re.I,
)
RETRY_RE = re.compile(
    r"\b(try again|retry|rerun|re-run|fix|debug|investigate|still|again|"
    r"failed|failure|error|issue|problem)\b",
    re.I,
)
COMMAND_RE = re.compile(r"```(?:bash|sh|shell)?\s*(.*?)```", re.I | re.S)

NUMERIC_FEATURES = [
    "event_count",
    "assistant_count",
    "user_count",
    "tool_count",
    "error_count",
    "error_recent",
    "error_rate",
    "recent_error_rate",
    "success_count",
    "success_recent",
    "retry_count",
    "retry_recent",
    "repeated_error_recent",
    "consecutive_error_tail",
    "last_event_error",
    "last_tool_error",
    "chars_recent",
    "avg_chars_recent",
    "tool_fraction",
    "late_fraction",
]


@dataclass
class TraceEvent:
    role: str
    text: str
    is_tool: bool = False
    has_error: bool = False
    has_success: bool = False


def clean_text(text: Any, *, max_chars: int = 2000) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False, sort_keys=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "input_text"}:
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return clean_text(content)


def _event(role: str, text: Any, *, is_tool: bool = False, error: Any = None) -> TraceEvent:
    raw = content_text(text)
    error_text = clean_text(error) if error not in (None, "", False) else ""
    combined = f"{raw}\n{error_text}" if error_text else raw
    return TraceEvent(
        role=role or "unknown",
        text=clean_text(combined, max_chars=2500),
        is_tool=is_tool,
        has_error=bool(error_text) or bool(ERROR_RE.search(combined or "")),
        has_success=bool(SUCCESS_RE.search(combined or "")),
    )


def events_from_swe_messages(messages: list[dict[str, Any]]) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for msg in messages:
        role = str(msg.get("role") or "unknown")
        events.append(
            _event(
                role,
                msg.get("content", ""),
                is_tool=role == "tool" or bool(msg.get("tool_call_id")),
                error=msg.get("error"),
            )
        )
    return [e for e in events if e.text]


def events_from_terminal_steps(steps: Any) -> list[TraceEvent]:
    if isinstance(steps, str):
        if steps.strip().lower() in {"", "null", "none"}:
            return []
        try:
            steps = json.loads(steps)
        except json.JSONDecodeError:
            return [_event("trace", steps)]
    if not isinstance(steps, list):
        return []
    events: list[TraceEvent] = []
    for step in steps:
        if not isinstance(step, dict):
            events.append(_event("trace", step))
            continue
        role = str(step.get("src") or step.get("role") or "trace")
        text = step.get("msg")
        if text in (None, ""):
            text = step.get("content") or step.get("text") or step.get("obs")
        obs = step.get("obs")
        if obs not in (None, ""):
            text = f"{content_text(text)}\nObservation: {content_text(obs)}"
        tools = step.get("tools")
        if tools not in (None, "", []):
            text = f"{content_text(text)}\nTools: {clean_text(tools, max_chars=500)}"
        events.append(
            _event(
                role,
                text,
                is_tool=role in {"tool", "observation", "env"} or obs not in (None, ""),
                error=step.get("error"),
            )
        )
    return [e for e in events if e.text]


def events_from_openai_messages(messages: list[dict[str, Any]] | None) -> list[TraceEvent]:
    if not messages:
        return []
    events: list[TraceEvent] = []
    for msg in messages:
        role = str(msg.get("role") or "unknown")
        is_tool = role in {"tool", "function"} or bool(msg.get("tool_call_id"))
        text = content_text(msg.get("content", ""))
        if msg.get("tool_calls"):
            text = f"{text}\nTool calls: {clean_text(msg.get('tool_calls'), max_chars=700)}"
        events.append(_event(role, text, is_tool=is_tool))
    return [e for e in events if e.text]


def error_signature(text: str) -> str:
    text = text.lower()
    text = re.sub(r"0x[0-9a-f]+", "0xADDR", text)
    text = re.sub(r"\b\d+\b", "N", text)
    text = re.sub(r"[/\w.-]+(?:\.py|\.js|\.ts|\.rs|\.go|\.java)", "FILE", text)
    matches = ERROR_RE.findall(text)
    if matches:
        return "|".join(sorted(set(matches)))[:120]
    return text[:120]


def health_features(events: list[TraceEvent], idx: int | None = None) -> dict[str, float]:
    if not events:
        return {name: 0.0 for name in NUMERIC_FEATURES}
    if idx is None:
        idx = len(events) - 1
    idx = max(0, min(idx, len(events) - 1))
    prefix = events[: idx + 1]
    recent = prefix[-8:]
    errors = [e for e in prefix if e.has_error]
    recent_errors = [e for e in recent if e.has_error]
    successes = [e for e in prefix if e.has_success]
    recent_successes = [e for e in recent if e.has_success]
    retries = [e for e in prefix if RETRY_RE.search(e.text)]
    recent_retries = [e for e in recent if RETRY_RE.search(e.text)]
    sig_counts: dict[str, int] = {}
    for e in recent_errors:
        sig = error_signature(e.text)
        sig_counts[sig] = sig_counts.get(sig, 0) + 1
    repeated = sum(max(0, c - 1) for c in sig_counts.values())
    tail = 0
    for e in reversed(prefix):
        if not e.has_error:
            break
        tail += 1
    chars_recent = sum(len(e.text) for e in recent)
    event_count = len(prefix)
    tool_count = sum(1 for e in prefix if e.is_tool)
    return {
        "event_count": float(event_count),
        "assistant_count": float(sum(1 for e in prefix if e.role == "assistant" or e.role == "agent")),
        "user_count": float(sum(1 for e in prefix if e.role == "user")),
        "tool_count": float(tool_count),
        "error_count": float(len(errors)),
        "error_recent": float(len(recent_errors)),
        "error_rate": len(errors) / max(1.0, event_count),
        "recent_error_rate": len(recent_errors) / max(1.0, len(recent)),
        "success_count": float(len(successes)),
        "success_recent": float(len(recent_successes)),
        "retry_count": float(len(retries)),
        "retry_recent": float(len(recent_retries)),
        "repeated_error_recent": float(repeated),
        "consecutive_error_tail": float(tail),
        "last_event_error": float(prefix[-1].has_error),
        "last_tool_error": float(bool(prefix[-1].is_tool and prefix[-1].has_error)),
        "chars_recent": float(chars_recent),
        "avg_chars_recent": chars_recent / max(1.0, len(recent)),
        "tool_fraction": tool_count / max(1.0, event_count),
        "late_fraction": idx / max(1.0, len(events) - 1),
    }


def feature_vector(features: dict[str, float]) -> list[float]:
    return [float(features.get(name, 0.0) or 0.0) for name in NUMERIC_FEATURES]


def window_text(
    events: list[TraceEvent],
    idx: int | None = None,
    *,
    task: str = "",
    max_events: int = 10,
    max_chars: int = 6000,
) -> str:
    if idx is None:
        idx = len(events) - 1
    idx = max(0, min(idx, len(events) - 1)) if events else 0
    recent = events[max(0, idx + 1 - max_events) : idx + 1]
    parts: list[str] = []
    if task:
        parts.append("Task: " + clean_text(task, max_chars=1200))
    for e in recent:
        tag = e.role
        if e.is_tool:
            tag = f"{tag}/tool"
        flags = []
        if e.has_error:
            flags.append("error")
        if e.has_success:
            flags.append("success")
        suffix = f" [{' '.join(flags)}]" if flags else ""
        parts.append(f"{tag}{suffix}: {clean_text(e.text, max_chars=800)}")
    text = "\n".join(parts)
    return text[-max_chars:]


def heuristic_badness(features: dict[str, float]) -> float:
    score = 0.0
    score += 1.4 * min(3.0, features.get("error_recent", 0.0))
    score += 1.2 * min(2.0, features.get("repeated_error_recent", 0.0))
    score += 1.0 * min(2.0, features.get("consecutive_error_tail", 0.0))
    score += 0.8 * features.get("last_event_error", 0.0)
    score += 0.8 * features.get("last_tool_error", 0.0)
    score += 0.5 * min(4.0, features.get("retry_recent", 0.0))
    score += 0.03 * min(80.0, features.get("event_count", 0.0))
    score -= 0.9 * min(2.0, features.get("success_recent", 0.0))
    return score


def candidate_indices(events: list[TraceEvent], *, max_windows: int = 12) -> list[int]:
    if not events:
        return []
    interesting = [
        i
        for i, e in enumerate(events)
        if i >= 2 and (e.has_error or e.is_tool or e.role in {"user", "assistant", "agent"})
    ]
    if not interesting:
        interesting = list(range(len(events)))
    if len(interesting) <= max_windows:
        return interesting
    step = len(interesting) / max_windows
    picks = [interesting[min(len(interesting) - 1, int(math.floor(i * step)))] for i in range(max_windows)]
    return sorted(set(picks))


def choose_window_labels(
    events: list[TraceEvent],
    *,
    final_success: bool,
    max_windows: int = 12,
) -> list[tuple[int, int, dict[str, float]]]:
    """Return ``(event_idx, label, features)`` for training.

    Failed traces are not labeled positive from the first turn. A failed trace
    becomes positive only after the prefix has local deterioration signals, or
    near the end if the trace failed without obvious machine-detectable errors.
    Successful traces provide negative examples, including recovered-error hard
    negatives.
    """

    labels: list[tuple[int, int, dict[str, float]]] = []
    indices = candidate_indices(events, max_windows=max_windows)
    if not indices:
        return labels
    onset: int | None = None
    if not final_success:
        for i in indices:
            feats = health_features(events, i)
            if heuristic_badness(feats) >= 3.0:
                onset = i
                break
        if onset is None:
            onset = indices[max(0, int(len(indices) * 0.65) - 1)]
    for i in indices:
        feats = health_features(events, i)
        label = int((not final_success) and onset is not None and i >= onset)
        labels.append((i, label, feats))
    return labels


@dataclass
class SessionHealthScore:
    score: float
    threshold: float
    should_escalate: bool
    features: dict[str, float]
    window_text: str


class SessionHealthScorer:
    def __init__(self, checkpoint: dict[str, Any]):
        self.checkpoint = checkpoint
        self.vectorizer = checkpoint["vectorizer"]
        self.scaler = checkpoint["scaler"]
        self.classifier = checkpoint["classifier"]
        self.threshold = float(checkpoint.get("threshold", 0.75))

    @classmethod
    def load(cls, path: str | Path) -> "SessionHealthScorer":
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def _score_one(self, text: str, features: dict[str, float]) -> float:
        from scipy.sparse import hstack

        x_text = self.vectorizer.transform([text])
        x_num = self.scaler.transform(np.asarray([feature_vector(features)], dtype=float))
        x = hstack([x_text, x_num])
        if hasattr(self.classifier, "predict_proba"):
            return float(self.classifier.predict_proba(x)[0, 1])
        decision = float(self.classifier.decision_function(x)[0])
        return 1.0 / (1.0 + math.exp(-decision))

    def score_events(self, events: list[TraceEvent], *, task: str = "") -> SessionHealthScore:
        feats = health_features(events)
        text = window_text(events, task=task)
        score = self._score_one(text, feats)
        return SessionHealthScore(
            score=score,
            threshold=self.threshold,
            should_escalate=score >= self.threshold,
            features=feats,
            window_text=text,
        )

    def score_messages(self, messages: list[dict[str, Any]] | None, *, task: str = "") -> SessionHealthScore:
        return self.score_events(events_from_openai_messages(messages), task=task)

