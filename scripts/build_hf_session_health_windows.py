#!/usr/bin/env python3
"""Build public-HF session-health training windows.

This intentionally ignores private Goose/Codex sessions. It reads public raw
agent traces from Hugging Face and emits rolling trajectory windows for a binary
target: whether the current trajectory prefix is going badly.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from model_router_toolkit.session_health import (
    NUMERIC_FEATURES,
    choose_window_labels,
    events_from_swe_messages,
    events_from_terminal_steps,
    window_text,
)

UA = {"User-Agent": "model-router-session-health/1"}
HF = "https://huggingface.co"
DATASET_API = "https://datasets-server.huggingface.co/rows"
SWE_REPO = "tarsur385/swebench-verified-trajectories"
TB_REPO = "yoonholee/terminalbench-trajectories"


def get_bytes(url: str, *, timeout: int = 60, tries: int = 4) -> bytes | None:
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url, headers=UA),
                timeout=timeout,
            ).read()
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504}:
                raise
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def get_json(url: str, *, timeout: int = 60) -> Any:
    raw = get_bytes(url, timeout=timeout)
    if raw is None:
        raise RuntimeError(f"could not fetch {url}")
    return json.loads(raw)


def swe_paths(*, max_traces: int, seed: int) -> list[str]:
    meta = get_json(f"{HF}/api/datasets/{SWE_REPO}", timeout=60)
    paths = [
        item["rfilename"]
        for item in meta.get("siblings", [])
        if item.get("rfilename", "").endswith(".traj.json")
    ]
    rng = random.Random(seed)
    rng.shuffle(paths)
    if max_traces > 0:
        paths = paths[:max_traces]
    return paths


def load_swe_trace(path: str, cache_dir: Path) -> dict[str, Any] | None:
    local = cache_dir / path
    if local.exists() and local.stat().st_size > 50:
        return json.loads(local.read_text())
    url = f"{HF}/datasets/{SWE_REPO}/resolve/main/{path}"
    raw = get_bytes(url, timeout=90)
    if raw is None:
        return None
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(raw)
    return json.loads(raw)


def terminal_rows(
    *,
    max_rows: int,
    page_size: int = 100,
    max_page_failures: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    failures = 0
    while max_rows <= 0 or len(rows) < max_rows:
        length = min(page_size, max_rows - len(rows)) if max_rows > 0 else page_size
        params = urllib.parse.urlencode(
            {
                "dataset": TB_REPO,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": length,
            }
        )
        try:
            data = get_json(f"{DATASET_API}?{params}", timeout=60)
            failures = 0
        except Exception as exc:
            failures += 1
            print(f"terminalbench offset {offset} fetch failed ({exc}); retry window {failures}")
            if failures >= max_page_failures:
                print(
                    f"terminalbench stopping after {failures} failed page fetches; "
                    f"kept {len(rows)} rows"
                )
                break
            offset += page_size
            continue
        page = data.get("rows") or []
        if not page:
            break
        for item in page:
            row = item.get("row", item)
            if isinstance(row, dict):
                rows.append(row)
                if max_rows > 0 and len(rows) >= max_rows:
                    break
        offset += len(page)
        total = data.get("num_rows_total")
        if total is not None and offset >= int(total):
            break
    return rows


def emit_trace_windows(
    writer: csv.DictWriter,
    *,
    source: str,
    task_id: str,
    model: str,
    trace_id: str,
    final_success: bool,
    task: str,
    events: list,
    max_windows_per_trace: int,
) -> tuple[int, int]:
    n = 0
    positives = 0
    for event_idx, label, feats in choose_window_labels(
        events,
        final_success=final_success,
        max_windows=max_windows_per_trace,
    ):
        row = {
            "source": source,
            "task_id": task_id,
            "model": model,
            "trace_id": trace_id,
            "turn_idx": event_idx,
            "final_success": int(final_success),
            "bad_next": label,
            "window_text": window_text(events, event_idx, task=task),
        }
        for name in NUMERIC_FEATURES:
            row[name] = round(float(feats.get(name, 0.0) or 0.0), 6)
        writer.writerow(row)
        n += 1
        positives += int(label)
    return n, positives


def trace_window_rows(
    *,
    source: str,
    task_id: str,
    model: str,
    trace_id: str,
    final_success: bool,
    task: str,
    events: list,
    max_windows_per_trace: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event_idx, label, feats in choose_window_labels(
        events,
        final_success=final_success,
        max_windows=max_windows_per_trace,
    ):
        row: dict[str, Any] = {
            "source": source,
            "task_id": task_id,
            "model": model,
            "trace_id": trace_id,
            "turn_idx": event_idx,
            "final_success": int(final_success),
            "bad_next": label,
            "window_text": window_text(events, event_idx, task=task),
        }
        for name in NUMERIC_FEATURES:
            row[name] = round(float(feats.get(name, 0.0) or 0.0), 6)
        rows.append(row)
    return rows


def build_swe_rows(
    path: str,
    *,
    cache_dir: Path,
    max_windows_per_trace: int,
) -> list[dict[str, Any]]:
    trace = load_swe_trace(path, cache_dir)
    if not trace:
        return []
    info = trace.get("info") or {}
    messages = trace.get("messages") or []
    events = events_from_swe_messages(messages)
    if len(events) < 4:
        return []
    parts = path.split("/")
    model = parts[1] if len(parts) > 2 else str(info.get("model", ""))
    task_id = str(trace.get("instance_id") or (parts[2] if len(parts) > 2 else path))
    task = ""
    for msg in messages:
        if msg.get("role") == "user":
            task = str(msg.get("content") or "")
            break
    resolved = info.get("resolved")
    if resolved is None:
        resolved = (info.get("scores") or {}).get("resolved")
    return trace_window_rows(
        source="swebench",
        task_id=task_id,
        model=model,
        trace_id=path,
        final_success=bool(resolved),
        task=task,
        events=events,
        max_windows_per_trace=max_windows_per_trace,
    )


def build_terminal_rows(
    row: dict[str, Any],
    *,
    ordinal: int,
    max_windows_per_trace: int,
) -> list[dict[str, Any]]:
    events = events_from_terminal_steps(row.get("steps"))
    if len(events) < 4:
        return []
    return trace_window_rows(
        source="terminalbench",
        task_id=str(row.get("task_name") or row.get("trial_name") or ordinal),
        model=str(row.get("model") or ""),
        trace_id=str(row.get("trial_id") or row.get("trial_name") or ordinal),
        final_success=bool(row.get("reward")),
        task=str(row.get("task_name") or ""),
        events=events,
        max_windows_per_trace=max_windows_per_trace,
    )


def write_rows(
    writer: csv.DictWriter,
    rows: list[dict[str, Any]],
    *,
    by_source: dict[str, list[int]],
) -> tuple[int, int]:
    total = 0
    positives = 0
    for row in rows:
        writer.writerow(row)
        total += 1
        positives += int(row.get("bad_next") or 0)
        source = str(row.get("source") or "")
        by_source.setdefault(source, [0, 0])
        by_source[source][0] += 1
        by_source[source][1] += int(row.get("bad_next") or 0)
    return total, positives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data_public/session-health-windows.csv")
    parser.add_argument("--cache-dir", default="data_public/hf_cache")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--swe-max-traces", type=int, default=800)
    parser.add_argument("--terminal-max-rows", type=int, default=1200)
    parser.add_argument("--max-windows-per-trace", type=int, default=10)
    parser.add_argument("--swe-workers", type=int, default=8)
    parser.add_argument("--terminal-page-size", type=int, default=100)
    parser.add_argument("--no-swe", action="store_true")
    parser.add_argument("--no-terminal", action="store_true")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    fieldnames = [
        "source",
        "task_id",
        "model",
        "trace_id",
        "turn_idx",
        "final_success",
        "bad_next",
        "window_text",
        *NUMERIC_FEATURES,
    ]
    total = 0
    positives = 0
    by_source: dict[str, list[int]] = {}

    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        if not args.no_swe:
            paths = swe_paths(max_traces=args.swe_max_traces, seed=args.seed)
            with cf.ThreadPoolExecutor(max_workers=max(1, args.swe_workers)) as ex:
                futures = [
                    ex.submit(
                        build_swe_rows,
                        path,
                        cache_dir=cache_dir,
                        max_windows_per_trace=args.max_windows_per_trace,
                    )
                    for path in paths
                ]
                for i, fut in enumerate(cf.as_completed(futures), 1):
                    rows = fut.result()
                    n, p = write_rows(writer, rows, by_source=by_source)
                    total += n
                    positives += p
                    if i % 100 == 0:
                        print(f"swebench {i}/{len(paths)} traces -> {total} windows")

        if not args.no_terminal:
            rows = terminal_rows(max_rows=args.terminal_max_rows, page_size=args.terminal_page_size)
            for i, row in enumerate(rows, 1):
                n, p = write_rows(
                    writer,
                    build_terminal_rows(
                        row,
                        ordinal=i,
                        max_windows_per_trace=args.max_windows_per_trace,
                    ),
                    by_source=by_source,
                )
                total += n
                positives += p

    os.replace(tmp, out)

    print(f"wrote {total} windows -> {out}")
    print(f"positive bad_next={positives} ({100 * positives / max(1, total):.1f}%)")
    for source, (n, p) in sorted(by_source.items()):
        print(f"  {source}: {n} windows, {p} positive ({100 * p / max(1, n):.1f}%)")


if __name__ == "__main__":
    main()
