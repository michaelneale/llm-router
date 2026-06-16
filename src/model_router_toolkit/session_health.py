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
TEST_COMMAND_RE = re.compile(
    r"\b(pytest|tox|npm\s+test|yarn\s+test|pnpm\s+test|cargo\s+test|go\s+test|"
    r"mvn\s+test|gradle\s+test|runtests?\.py|unittest|nosetests|rspec|jest|vitest)\b",
    re.I,
)
TEST_FAILURE_RE = re.compile(
    r"\b("
    r"failed\b|failures?\b|assertionerror|error collecting|tests?\s+failed|"
    r"\d+\s+failed|\d+\s+errors?|not ok\b"
    r")\b",
    re.I,
)
TIMEOUT_RE = re.compile(r"\b(timeout|timed out|time limit|deadline exceeded|killed)\b", re.I)
MISSING_DEP_RE = re.compile(
    r"\b("
    r"no module named|modulenotfounderror|importerror|cannot find module|"
    r"module not found|command not found|no such file|package not found"
    r")\b",
    re.I,
)
PARSE_ERROR_RE = re.compile(r"parse[_ -]?error|\"parse_error\"\s*:\s*(?!null)", re.I)
PATCH_RE = re.compile(
    r"\b(apply_patch|diff --git|git apply|edit_file|write_file|write_text|"
    r"sed\s+-i|perl\s+-pi|cat\s+>|tee\s+[^|\s>]+|patch\.txt)\b|"
    r"^@@|^\+\+\+ b/|^--- a/",
    re.I | re.M,
)
FILE_RE = re.compile(
    r"(?:^|[\s\"'`])(?:[ab]/)?([\w./-]+\."
    r"(?:py|js|ts|tsx|jsx|rs|go|java|rb|php|c|cc|cpp|h|hpp|swift|kt|scala|cs|"
    r"md|rst|toml|yaml|yml|json|ini|cfg|txt|sh|sql))\b",
    re.I,
)
DIFF_FILE_RE = re.compile(r"diff --git a/([^\s]+) b/([^\s]+)", re.I)
COMMAND_VALUE_RE = re.compile(
    r"[\"'](?:command|cmd)[\"']\s*:\s*([\"'])(.{1,1200}?)\1",
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
    "command_count",
    "command_recent",
    "test_command_recent",
    "repeated_command_recent",
    "repeated_failing_command_recent",
    "repeated_test_command_recent",
    "test_failure_count",
    "test_failure_recent",
    "patch_count",
    "patch_recent",
    "edit_after_error_recent",
    "same_error_after_patch_recent",
    "test_failure_after_patch_recent",
    "same_file_edit_recent",
    "timeout_count",
    "timeout_recent",
    "missing_dependency_count",
    "missing_dependency_recent",
    "parse_error_count",
    "parse_error_recent",
    "recovery_after_error_recent",
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


def iter_tool_calls(tool_calls: Any) -> list[Any]:
    if not tool_calls:
        return []
    return tool_calls if isinstance(tool_calls, list) else [tool_calls]


def tool_call_parse_errors(tool_calls: Any) -> list[Any]:
    return [
        call.get("parse_error")
        for call in iter_tool_calls(tool_calls)
        if isinstance(call, dict) and call.get("parse_error") not in (None, "", False)
    ]


def tool_calls_text(tool_calls: Any) -> str:
    calls = iter_tool_calls(tool_calls)
    if not calls:
        return ""
    parts: list[str] = []
    for call in calls:
        if not isinstance(call, dict):
            parts.append(f"tool_call: {clean_text(call, max_chars=800)}")
            continue
        function = call.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            arguments = function.get("arguments")
            if name:
                parts.append(f"tool_function: {clean_text(name, max_chars=100)}")
            if arguments:
                parts.append(f"tool_arguments: {clean_text(arguments, max_chars=1000)}")
        elif function:
            parts.append(f"tool_function: {clean_text(function, max_chars=100)}")
        arguments = call.get("arguments")
        if arguments:
            parts.append(f"tool_arguments: {clean_text(arguments, max_chars=1000)}")
        view = call.get("view")
        if isinstance(view, dict) and view.get("content"):
            parts.append(f"tool_view: {clean_text(view.get('content'), max_chars=1000)}")
        parse_error = call.get("parse_error")
        if parse_error not in (None, "", False):
            parts.append(f"tool_parse_error: {clean_text(parse_error, max_chars=500)}")
    return "\n".join(parts)


def events_from_swe_messages(messages: list[dict[str, Any]]) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for msg in messages:
        role = str(msg.get("role") or "unknown")
        text = content_text(msg.get("content", ""))
        calls_text = tool_calls_text(msg.get("tool_calls"))
        if calls_text:
            text = f"{text}\n{calls_text}" if text else calls_text
        parse_errors = tool_call_parse_errors(msg.get("tool_calls"))
        events.append(
            _event(
                role,
                text,
                is_tool=role == "tool" or bool(msg.get("tool_call_id")),
                error=parse_errors or msg.get("error"),
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
            text = f"{text}\n{tool_calls_text(msg.get('tool_calls'))}"
        events.append(
            _event(
                role,
                text,
                is_tool=is_tool,
                error=tool_call_parse_errors(msg.get("tool_calls")),
            )
        )
    return [e for e in events if e.text]


def _shell_lines(text: str) -> list[str]:
    text = text.replace("\\n", "\n")
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("$ "):
            line = line[2:].strip()
        lines.append(line)
    return lines


def normalize_command(command: str) -> str:
    command = clean_text(command.replace("\\n", " "), max_chars=500).lower()
    command = re.sub(r"\bcd\s+/\S+\s*(?:&&|;)\s*", "", command)
    command = re.sub(r"\b(?:python3?|/opt/miniconda3/bin/python)\b", "python", command)
    command = re.sub(r"\s+", " ", command).strip()
    return command[:220]


def extract_commands(text: str) -> list[str]:
    commands: list[str] = []
    for block in COMMAND_RE.findall(text or ""):
        commands.extend(normalize_command(line) for line in _shell_lines(block))
    for _, value in COMMAND_VALUE_RE.findall(text or ""):
        commands.extend(normalize_command(line) for line in _shell_lines(value))
    deduped: list[str] = []
    seen: set[str] = set()
    for command in commands:
        if command and command not in seen:
            seen.add(command)
            deduped.append(command)
    return deduped


def extract_files(text: str) -> list[str]:
    files: set[str] = set()
    for left, right in DIFF_FILE_RE.findall(text or ""):
        files.add(right or left)
    for file_name in FILE_RE.findall(text or ""):
        files.add(file_name.removeprefix("a/").removeprefix("b/"))
    return sorted(files)


def is_test_command(command: str) -> bool:
    return bool(TEST_COMMAND_RE.search(command))


def is_patch_event(text: str, commands: list[str], files: list[str]) -> bool:
    if PATCH_RE.search(text or ""):
        return True
    if any(PATCH_RE.search(command) for command in commands):
        return True
    return bool(files and any(command.startswith(("sed -i", "perl -pi")) for command in commands))


def _operational_error(index: int, event: TraceEvent) -> bool:
    return bool(event.has_error and not (index == 0 and event.role == "user"))


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
    errors = [e for i, e in enumerate(prefix) if _operational_error(i, e)]
    recent_start = len(prefix) - len(recent)
    recent_errors = [
        e for i, e in enumerate(recent, start=recent_start) if _operational_error(i, e)
    ]
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
    for i in range(len(prefix) - 1, -1, -1):
        if not _operational_error(i, prefix[i]):
            break
        tail += 1
    chars_recent = sum(len(e.text) for e in recent)
    event_count = len(prefix)
    tool_count = sum(1 for e in prefix if e.is_tool)
    event_infos: list[dict[str, Any]] = []
    last_commands: list[str] = []
    for i, event in enumerate(prefix):
        commands = extract_commands(event.text)
        active_commands = commands or last_commands
        files = extract_files(event.text)
        has_error = _operational_error(i, event)
        test_command = any(is_test_command(command) for command in commands)
        failed_commands = active_commands[-3:] if has_error and active_commands else []
        has_test_failure = bool(TEST_FAILURE_RE.search(event.text)) or bool(
            has_error and any(is_test_command(command) for command in active_commands)
        )
        info = {
            "commands": commands,
            "failed_commands": failed_commands,
            "files": files,
            "has_error": has_error,
            "has_success": event.has_success,
            "is_patch": is_patch_event(event.text, commands, files),
            "has_test_command": test_command,
            "has_test_failure": has_test_failure,
            "has_timeout": bool(TIMEOUT_RE.search(event.text)),
            "has_missing_dependency": bool(MISSING_DEP_RE.search(event.text)),
            "has_parse_error": bool(PARSE_ERROR_RE.search(event.text)),
            "error_sig": error_signature(event.text) if has_error else "",
        }
        event_infos.append(info)
        if commands:
            last_commands = commands
    recent_infos = event_infos[-8:]

    def repeated_count(values: list[str]) -> float:
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return float(sum(max(0, count - 1) for count in counts.values()))

    recent_commands = [
        command for info in recent_infos for command in info["commands"] if command
    ]
    recent_failed_commands = [
        command for info in recent_infos for command in info["failed_commands"] if command
    ]
    recent_test_commands = [command for command in recent_commands if is_test_command(command)]
    recent_patch_files = [
        file_name for info in recent_infos if info["is_patch"] for file_name in info["files"]
    ]
    same_error_after_patch = 0
    last_error_index: dict[str, int] = {}
    for i, info in enumerate(recent_infos):
        sig = info["error_sig"]
        if not sig:
            continue
        prev = last_error_index.get(sig)
        if prev is not None and any(item["is_patch"] for item in recent_infos[prev + 1 : i]):
            same_error_after_patch += 1
        last_error_index[sig] = i
    seen_error = False
    edit_after_error = 0
    test_failure_after_patch = 0
    seen_patch = False
    recovery_after_error = 0
    for info in recent_infos:
        if info["is_patch"] and seen_error:
            edit_after_error += 1
        if info["has_test_failure"] and seen_patch:
            test_failure_after_patch += 1
        if info["has_success"] and seen_error:
            recovery_after_error += 1
        if info["has_error"]:
            seen_error = True
        if info["is_patch"]:
            seen_patch = True
    last_idx = len(prefix) - 1
    last_event_error = _operational_error(last_idx, prefix[-1])
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
        "last_event_error": float(last_event_error),
        "last_tool_error": float(bool(prefix[-1].is_tool and last_event_error)),
        "chars_recent": float(chars_recent),
        "avg_chars_recent": chars_recent / max(1.0, len(recent)),
        "tool_fraction": tool_count / max(1.0, event_count),
        "late_fraction": idx / max(1.0, len(events) - 1),
        "command_count": float(sum(len(info["commands"]) for info in event_infos)),
        "command_recent": float(len(recent_commands)),
        "test_command_recent": float(sum(1 for info in recent_infos if info["has_test_command"])),
        "repeated_command_recent": repeated_count(recent_commands),
        "repeated_failing_command_recent": repeated_count(recent_failed_commands),
        "repeated_test_command_recent": repeated_count(recent_test_commands),
        "test_failure_count": float(sum(1 for info in event_infos if info["has_test_failure"])),
        "test_failure_recent": float(sum(1 for info in recent_infos if info["has_test_failure"])),
        "patch_count": float(sum(1 for info in event_infos if info["is_patch"])),
        "patch_recent": float(sum(1 for info in recent_infos if info["is_patch"])),
        "edit_after_error_recent": float(edit_after_error),
        "same_error_after_patch_recent": float(same_error_after_patch),
        "test_failure_after_patch_recent": float(test_failure_after_patch),
        "same_file_edit_recent": repeated_count(recent_patch_files),
        "timeout_count": float(sum(1 for info in event_infos if info["has_timeout"])),
        "timeout_recent": float(sum(1 for info in recent_infos if info["has_timeout"])),
        "missing_dependency_count": float(
            sum(1 for info in event_infos if info["has_missing_dependency"])
        ),
        "missing_dependency_recent": float(
            sum(1 for info in recent_infos if info["has_missing_dependency"])
        ),
        "parse_error_count": float(sum(1 for info in event_infos if info["has_parse_error"])),
        "parse_error_recent": float(sum(1 for info in recent_infos if info["has_parse_error"])),
        "recovery_after_error_recent": float(recovery_after_error),
    }


def feature_vector(
    features: dict[str, float],
    names: list[str] | tuple[str, ...] | None = None,
) -> list[float]:
    feature_names = names or NUMERIC_FEATURES
    return [float(features.get(name, 0.0) or 0.0) for name in feature_names]


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
    for absolute_idx, e in enumerate(recent, start=max(0, idx + 1 - max_events)):
        tag = e.role
        if e.is_tool:
            tag = f"{tag}/tool"
        flags = []
        if _operational_error(absolute_idx, e):
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
    score += 1.0 * min(2.0, features.get("repeated_failing_command_recent", 0.0))
    score += 0.9 * min(2.0, features.get("same_error_after_patch_recent", 0.0))
    score += 0.7 * min(3.0, features.get("test_failure_after_patch_recent", 0.0))
    score += 0.5 * min(3.0, features.get("edit_after_error_recent", 0.0))
    score += 0.6 * min(2.0, features.get("timeout_recent", 0.0))
    score += 0.03 * min(80.0, features.get("event_count", 0.0))
    score -= 0.9 * min(2.0, features.get("success_recent", 0.0))
    score -= 0.8 * min(2.0, features.get("recovery_after_error_recent", 0.0))
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
        self.numeric_features = list(checkpoint.get("numeric_features") or NUMERIC_FEATURES)

    @classmethod
    def load(cls, path: str | Path) -> "SessionHealthScorer":
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def _score_one(self, text: str, features: dict[str, float]) -> float:
        from scipy.sparse import hstack

        x_text = self.vectorizer.transform([text])
        x_num = self.scaler.transform(
            np.asarray([feature_vector(features, self.numeric_features)], dtype=float)
        )
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

    def score_messages(
        self,
        messages: list[dict[str, Any]] | None,
        *,
        task: str = "",
    ) -> SessionHealthScore:
        return self.score_events(events_from_openai_messages(messages), task=task)
