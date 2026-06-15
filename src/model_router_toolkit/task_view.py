"""Reconstruct a training-shaped task statement from an agent message list.

Problem this solves
-------------------
The prefill classifier was trained on *self-contained problem statements*
(SWE-bench issues, RouterBench questions, terminal-bench instructions): a few
hundred chars of text that fully describe a task. At runtime in an agent
session we were instead feeding it the bare last user message — often a
fragment ("why?", "run the tests") or a raw tool-result dump — whose difficulty
lives in conversation state the encoder never sees. That input is
out-of-distribution for the classifier, so its difficulty estimates are noise.

`build_task_view` maps the live message list onto something dimensionally and
semantically closer to the training distribution:

  - SKIP turns that are not a routing decision (tool results, empty turns):
    callers should pin to the previous model rather than score noise.
  - For a real user turn, build a compact, self-contained task statement:
        the user's intent + a short synopsis of what's being worked on +
        the active tools, trimmed toward the training length (~300-700 chars).

It deliberately carries NO file contents or large transcripts — just enough
structural context to make a fragment read like a stated task.
"""
from __future__ import annotations

import re

# training-row length stats (data/full-train.csv): median 433, p75 686, p90 1447
TARGET_CHARS = 700
USER_TEXT_CAP = 500
SYNOPSIS_CAP = 180

_TOOL_NOISE = re.compile(r"(toolResponse|tool_call_id|\"status\"\s*:\s*\"success\")")

# Agents (goose) inject user-role messages that *narrate* a tool result rather
# than express user intent — e.g. "A grep search of the two config files found
# that...", "A streaming test was attempted, but it failed...", "Checked the X
# mapping in both files." In goose's own store these are flagged
# userVisible:false, but that flag does not survive to the proxy over HTTP, so
# we detect them structurally: 3rd-person past-tense report openers that no
# human types as an instruction.
#
# Real user turns are imperative/interrogative ("fix the bug", "why?", "run the
# tests", "ok carry on"). Narration is declarative past-tense about an action
# already taken. We match the latter.
_NARRATION_VERB = (
    r"(was|were|is|are|has|have|had|"
    r"attempt\w*|execut\w*|ran|run|display\w*|perform\w*|retriev\w*|creat\w*|"
    r"found|confirm\w*|complet\w*|fail\w*|return\w*|succeed\w*|check\w*|"
    r"search\w*|inspect\w*|review\w*|map\w*|show\w*|reveal\w*|report\w*|"
    r"identif\w*|determin\w*|read|added|updated|fixed|removed|verified)"
)
_SUMMARY_NARRATION = re.compile(
    # opener: "A/An/The <words> <narration-verb>"  OR  bare past-tense verb start
    r"^((a|an|the)\s+([\w-]+\s+){1,8}" + _NARRATION_VERB + r"\b"
    r"|(checked|ran|searched|inspected|reviewed|confirmed|found|created|added|"
    r"updated|fixed|removed|verified|identified|determined|examined|attempted|"
    r"executed|performed|retrieved|displayed|completed|tested)\b)",
    re.IGNORECASE,
)


def _content_text(content) -> str:
    """Pull plain text out of an OpenAI-format content field; '' if none."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ).strip()
    return ""


def _is_tool_turn(msg: dict) -> bool:
    """True if this message is a tool result / tool output, not user intent."""
    if msg.get("role") == "tool":
        return True
    if msg.get("tool_call_id") or msg.get("tool_calls"):
        return True
    content = msg.get("content")
    if isinstance(content, list):
        if any(isinstance(p, dict) and p.get("type") in ("toolResponse", "tool_result")
               for p in content):
            return True
    if isinstance(content, str) and _TOOL_NOISE.search(content):
        return True
    return False


def _last_user_intent(messages: list[dict]) -> str | None:
    """The most recent genuine user message text, or None if the last
    user-role turn is actually a tool result / empty (= not a routing event)."""
    # The routing decision is about the CURRENT (last) message. If the latest
    # message is not a genuine user turn — a tool result, assistant turn, or an
    # agent-injected narration — this is not a routing event; pin instead.
    last = messages[-1]
    if last.get("role") != "user" or _is_tool_turn(last):
        return None
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        if _is_tool_turn(msg):
            return None  # last user turn is a tool dump -> not a decision
        text = _content_text(msg.get("content")).strip()
        if not text:
            return None
        if _SUMMARY_NARRATION.match(text):
            return None  # agent narrating a tool result, not user intent -> pin
        return text
    return None


def _synopsis(messages: list[dict]) -> str:
    """Short 'what's being worked on' line from recent assistant intent + tools."""
    last_asst = ""
    tools: list[str] = []
    for msg in reversed(messages):
        role = msg.get("role")
        if role == "assistant" and not last_asst:
            t = _content_text(msg.get("content"))
            if t:
                last_asst = t[:SYNOPSIS_CAP]
        for tc in msg.get("tool_calls") or []:
            fn = (tc.get("function") or {}).get("name")
            if fn and fn not in tools:
                tools.append(fn)
        if last_asst and len(tools) >= 4:
            break
    bits = []
    if tools:
        bits.append("tools in use: " + ", ".join(tools[:4]))
    if last_asst:
        bits.append("prior step: " + last_asst.replace("\n", " ").strip())
    return " | ".join(bits)


def build_task_view(messages: list[dict] | None) -> str | None:
    """Return a training-shaped task statement, or None if this turn is not a
    routing decision (tool result / empty) and the caller should pin instead.
    """
    if not messages:
        return None
    intent = _last_user_intent(messages)
    if not intent:
        return None  # tool result or empty -> pin, don't score noise

    intent = intent.strip()[:USER_TEXT_CAP]
    syn = _synopsis(messages)
    if not syn:
        return intent  # cold-start turn already looks task-like

    view = f"Task: {intent}\nContext: {syn}"
    return view[:TARGET_CHARS]
