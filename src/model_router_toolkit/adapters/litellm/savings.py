"""Live savings tracker for the router proxy.

Accumulates, per request, the actual cost (selected model x real token usage)
versus a baseline cost (a chosen reference model x the same tokens). Exposes
the running totals via GET /savings on the proxy.

Savings are always *relative to a baseline you pick*. Default baseline is the
most expensive model in the pool (the "if I'd sent everything to the top tier"
comparison). Override with ROUTER_SAVINGS_BASELINE=<slot name>.

State is in-memory (resets on proxy restart) and optionally persisted to
ROUTER_SAVINGS_LOG as JSONL for durable analysis across restarts.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class _Totals:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    actual_cost: float = 0.0
    baseline_cost: float = 0.0
    per_model_requests: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_model_actual: dict[str, float] = field(default_factory=lambda: defaultdict(float))


class SavingsTracker:
    """Thread-safe accumulator of actual-vs-baseline spend."""

    def __init__(self, baseline_model: str | None = None, log_path: str | None = None):
        self._lock = threading.Lock()
        self._t = _Totals()
        self._started = time.time()
        self._baseline_override = baseline_model or os.environ.get(
            "ROUTER_SAVINGS_BASELINE"
        )
        self._log_path = log_path or os.environ.get("ROUTER_SAVINGS_LOG")
        # rate cache: model_name -> (in_per_m, out_per_m), filled from RoutingResult
        self._rates: dict[str, tuple[float, float]] = {}
        self._baseline_name: str | None = None
        # slot name -> real model label (e.g. "gpt-oss-120b-high" -> "openai/gpt-5-mini")
        self._display: dict[str, str] = {}

    def set_display_names(self, mapping: dict[str, str]) -> None:
        """Map internal checkpoint slot names to the real model they call, so the
        dashboard shows e.g. 'openai/gpt-5-mini' not the cosmetic 'gpt-oss-120b-high'."""
        self._display.update({k: v for k, v in mapping.items() if v})

    def _label(self, slot: str | None) -> str | None:
        if slot is None:
            return None
        return self._display.get(slot, slot)

    def _ingest_rates(self, result) -> None:
        """Capture per-model token rates from a RoutingResult (once is enough,
        but cheap to refresh)."""
        for name, cost in zip(result.model_names, result.costs):
            self._rates[name] = (
                cost.cost_per_m_input_tokens,
                cost.cost_per_m_output_tokens,
            )
        # resolve baseline: explicit override, else most expensive by output rate
        if self._baseline_override and self._baseline_override in self._rates:
            self._baseline_name = self._baseline_override
        else:
            self._baseline_name = max(
                self._rates, key=lambda n: self._rates[n][1], default=None
            )

    def record(self, result, in_tokens: int, out_tokens: int) -> None:
        """Record one completed request.

        result: the RoutingResult for this request (has selected_model + rates)
        in_tokens / out_tokens: real usage from the response.
        """
        with self._lock:
            self._ingest_rates(result)
            selected = result.selected_model
            sel_in, sel_out = self._rates.get(selected, (0.0, 0.0))
            actual = in_tokens / 1e6 * sel_in + out_tokens / 1e6 * sel_out

            base_in, base_out = self._rates.get(
                self._baseline_name, (sel_in, sel_out)
            )
            baseline = in_tokens / 1e6 * base_in + out_tokens / 1e6 * base_out

            self._t.requests += 1
            self._t.input_tokens += in_tokens
            self._t.output_tokens += out_tokens
            self._t.actual_cost += actual
            self._t.baseline_cost += baseline
            self._t.per_model_requests[selected] += 1
            self._t.per_model_actual[selected] += actual

            if self._log_path:
                try:
                    with open(self._log_path, "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "ts": time.time(),
                                    "selected": selected,
                                    "baseline": self._baseline_name,
                                    "in_tokens": in_tokens,
                                    "out_tokens": out_tokens,
                                    "actual_cost": round(actual, 8),
                                    "baseline_cost": round(baseline, 8),
                                    "confidence": round(result.selected_confidence, 4),
                                }
                            )
                            + "\n"
                        )
                except OSError:
                    pass

    def snapshot(self) -> dict:
        with self._lock:
            t = self._t
            saved = t.baseline_cost - t.actual_cost
            pct = (saved / t.baseline_cost * 100) if t.baseline_cost else 0.0
            dist = {
                self._label(m): {
                    "requests": n,
                    "share_pct": round(n / t.requests * 100, 1) if t.requests else 0.0,
                    "actual_cost_usd": round(t.per_model_actual[m], 6),
                }
                for m, n in sorted(
                    t.per_model_requests.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
            }
            return {
                "uptime_seconds": round(time.time() - self._started, 1),
                "baseline_model": self._label(self._baseline_name),
                "requests": t.requests,
                "input_tokens": t.input_tokens,
                "output_tokens": t.output_tokens,
                "actual_cost_usd": round(t.actual_cost, 6),
                "baseline_cost_usd": round(t.baseline_cost, 6),
                "saved_usd": round(saved, 6),
                "saved_pct": round(pct, 1),
                "projected_monthly_saving_usd": (
                    round(saved / (time.time() - self._started) * 2_592_000, 2)
                    if (time.time() - self._started) > 0
                    else 0.0
                ),
                "routing_distribution": dist,
            }

    def reset(self) -> None:
        with self._lock:
            self._t = _Totals()
            self._started = time.time()


def extract_usage(body_bytes: bytes) -> tuple[int, int] | None:
    """Pull (prompt_tokens, completion_tokens) from a JSON completion body."""
    try:
        data = json.loads(body_bytes)
    except (ValueError, TypeError):
        return None
    return _usage_from_obj(data)


def _usage_from_obj(data) -> tuple[int, int] | None:
    """Pull (prompt_tokens, completion_tokens) from a parsed JSON object."""
    if not isinstance(data, dict):
        return None
    usage = data.get("usage") or {}
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    if pt is None and ct is None:
        return None
    return int(pt or 0), int(ct or 0)


def extract_usage_from_sse(sse_text: str) -> tuple[int, int] | None:
    """Pull token usage from a Server-Sent Events stream body.

    OpenAI-style streaming emits ``data: {json}`` lines; when
    ``stream_options.include_usage`` is set, one (typically final, pre-[DONE])
    chunk carries a top-level ``usage`` block. We scan all chunks and return the
    last usage we see.
    """
    found: tuple[int, int] | None = None
    for line in sse_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except (ValueError, TypeError):
            continue
        u = _usage_from_obj(obj)
        if u is not None:
            found = u
    return found


# module-level singleton the proxy middleware writes to and the route reads from
_TRACKER: SavingsTracker | None = None


def get_tracker() -> SavingsTracker:
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = SavingsTracker()
    return _TRACKER
