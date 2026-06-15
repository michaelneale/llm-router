#!/usr/bin/env python3
"""Build the public full-spectrum router training set.

Sources:
  - RouterBench: general Q&A with per-model correctness.
  - SWE-bench trajectories: code-fixing tasks with verified resolved labels.
  - terminal-bench trajectories: terminal/ops tasks with verified reward labels.

Each source has its own model names, so this script maps source models onto the
router's fixed pool slots by observed capability tier. The output stays in the
long format consumed by `model-router train`:

    question,model,isCorrect,output_tokens

It also preserves provenance columns (`source`, `task_id`, `source_model`) so
offline sweeps can break regret/savings down by general QA vs. code-fixing vs.
terminal/ops tasks. The train/eval readers ignore unknown columns.

The source matrices are intentionally kept under data_public/ because they are
external research artifacts, not private Goose traces.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

SLOTS = [
    "nemotron-3-nano-reasoning",
    "gpt-oss-20b-high",
    "gpt-4-1-nano-high",
    "gpt-oss-120b-high",
    "gpt-4-1-mini-high",
    "claude-sonnet-4-6-high",
    "claude-haiku-4-5-high",
    "claude-opus-4-6-high",
]

ROUTERBENCH_MODELS = [
    "mistralai/mistral-7b-chat",
    "WizardLM/WizardLM-13B-V1.2",
    "mistralai/mixtral-8x7b-chat",
    "gpt-3.5-turbo-1106",
    "claude-instant-v1",
    "claude-v2",
    "claude-v1",
    "gpt-4-1106-preview",
]

SWE_MAP = {
    "gpt-5-mini": "nemotron-3-nano-reasoning",
    "claude-4.5-haiku-high": "gpt-oss-20b-high",
    "gemini-3-flash-high": "gpt-4-1-nano-high",
    "glm-5-high": "gpt-oss-120b-high",
    "minimax-m2.5-high": "gpt-4-1-mini-high",
    "claude-4.5-opus-high": "claude-sonnet-4-6-high",
    "gpt-5.2-high": "claude-haiku-4-5-high",
    "claude-opus-4.6": "claude-opus-4-6-high",
}


def emit(
    rows: list[dict],
    seen: set[tuple[str, str]],
    *,
    source: str,
    task_id: str,
    source_model: str,
    slot: str,
    correct: bool,
    text: str,
) -> None:
    key = (task_id, slot)
    if key in seen or not text or len(text) < 10:
        return
    seen.add(key)
    rows.append(
        {
            "question": text[:4000],
            "model": slot,
            "isCorrect": int(bool(correct)),
            "output_tokens": 0,
            "source": source,
            "task_id": task_id,
            "source_model": source_model,
        }
    )


def evenly_map_by_accuracy(
    scores: dict[str, float],
    *,
    slots: list[str] = SLOTS,
) -> dict[str, str]:
    """Map source models weak->strong onto router slots weak->strong."""
    ranked = sorted(scores, key=lambda m: scores[m])
    if not ranked:
        return {}
    if len(ranked) >= len(slots):
        idxs = [round(i * (len(ranked) - 1) / (len(slots) - 1)) for i in range(len(slots))]
        return {ranked[idx]: slots[i] for i, idx in enumerate(idxs)}
    return {model: slots[i] for i, model in enumerate(ranked)}


def load_routerbench(path: Path):
    candidates = [path, Path("/tmp/rb.pkl")]
    for candidate in candidates:
        if candidate.exists():
            try:
                import pandas as pd
            except ModuleNotFoundError as exc:
                raise SystemExit(
                    "RouterBench pickle is present, but pandas is not installed. "
                    "Install pandas or remove/skip the RouterBench pickle."
                ) from exc
            return pd.read_pickle(candidate), pd
    return None, None


def add_routerbench(
    rows: list[dict],
    seen: set[tuple[str, str]],
    *,
    path: Path,
    sample_per_eval: int,
    manifest: dict,
) -> None:
    df, pd = load_routerbench(path)
    if df is None:
        manifest["routerbench"] = {"status": "missing", "path": str(path)}
        return

    rb_models = [m for m in ROUTERBENCH_MODELS if m in df.columns]
    scores = {m: float(df[m].mean()) for m in rb_models}
    mapping = evenly_map_by_accuracy(scores)
    manifest["routerbench"] = {
        "status": "included",
        "rows_before_sampling": int(len(df)),
        "models": scores,
        "mapping": mapping,
    }
    if not mapping:
        return

    if "eval_name" in df.columns:
        sampled = df.groupby("eval_name", group_keys=False).apply(
            lambda g: g.sample(min(len(g), sample_per_eval), random_state=1)
        )
    else:
        sampled = df.sample(min(len(df), sample_per_eval * 40), random_state=1)
    manifest["routerbench"]["sampled_prompts"] = int(len(sampled))

    for _, row in sampled.iterrows():
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or len(prompt) < 10:
            continue
        sample_id = row.get("sample_id", row.name)
        for source_model, slot in mapping.items():
            score = row.get(source_model)
            if pd.notna(score):
                emit(
                    rows,
                    seen,
                    source="routerbench",
                    task_id=f"rb::{sample_id}",
                    source_model=source_model,
                    slot=slot,
                    correct=float(score) > 0.5,
                    text=prompt,
                )


def add_swebench(
    rows: list[dict],
    seen: set[tuple[str, str]],
    *,
    path: Path,
    manifest: dict,
) -> None:
    if not path.exists():
        manifest["swebench"] = {"status": "missing", "path": str(path)}
        return
    source_rows = [json.loads(line) for line in path.open()]
    task_text = {
        r["instance_id"]: r.get("task", "")
        for r in source_rows
        if r.get("instance_id") and r.get("task")
    }
    before = len(rows)
    for row in source_rows:
        slot = SWE_MAP.get(row.get("model"))
        if not slot:
            continue
        emit(
            rows,
            seen,
            source="swebench",
            task_id=f"swe::{row['instance_id']}",
            source_model=str(row.get("model", "")),
            slot=slot,
            correct=bool(row.get("resolved")),
            text=task_text.get(row["instance_id"], ""),
        )
    manifest["swebench"] = {
        "status": "included",
        "source_rows": len(source_rows),
        "tasks_with_text": len(task_text),
        "mapping": SWE_MAP,
        "emitted_rows": len(rows) - before,
    }


def add_terminalbench(
    rows: list[dict],
    seen: set[tuple[str, str]],
    *,
    path: Path,
    min_rows_per_model: int,
    manifest: dict,
) -> None:
    if not path.exists():
        manifest["terminalbench"] = {"status": "missing", "path": str(path)}
        return
    source_rows = [json.loads(line) for line in path.open()]
    by_model = defaultdict(lambda: [0, 0])
    for row in source_rows:
        model = row.get("model")
        if not model:
            continue
        by_model[model][0] += int(row.get("reward", 0))
        by_model[model][1] += 1
    scores = {
        model: correct / total
        for model, (correct, total) in by_model.items()
        if total >= min_rows_per_model
    }
    mapping = evenly_map_by_accuracy(scores)
    before = len(rows)
    for row in source_rows:
        slot = mapping.get(row.get("model"))
        if not slot:
            continue
        emit(
            rows,
            seen,
            source="terminalbench",
            task_id=f"tb::{row['task_name']}",
            source_model=str(row.get("model", "")),
            slot=slot,
            correct=bool(row.get("reward")),
            text=row.get("task", ""),
        )
    manifest["terminalbench"] = {
        "status": "included",
        "source_rows": len(source_rows),
        "eligible_models": scores,
        "mapping": mapping,
        "emitted_rows": len(rows) - before,
    }


def summarize(rows: list[dict]) -> dict:
    by_model = defaultdict(lambda: [0, 0])
    coverage = defaultdict(set)
    for row in rows:
        by_model[row["model"]][0] += int(row["isCorrect"])
        by_model[row["model"]][1] += 1
        coverage[row["question"]].add(row["model"])
    return {
        "rows": len(rows),
        "unique_tasks": len(coverage),
        "coverage": {str(k): v for k, v in sorted(Counter(len(v) for v in coverage.values()).items())},
        "by_model": {
            model: {"rows": total, "accuracy": correct / total if total else 0.0}
            for model, (correct, total) in sorted(by_model.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routerbench-pickle", default="data_public/routerbench.pkl")
    parser.add_argument("--swebench-jsonl", default="data_public/swebench_matrix.jsonl")
    parser.add_argument(
        "--terminalbench-jsonl",
        default="data_public/terminalbench_matrix_text.jsonl",
    )
    parser.add_argument("--output", default="data/full-train.csv")
    parser.add_argument("--manifest", default="data/public-training-manifest.json")
    parser.add_argument("--sample-per-eval", type=int, default=120)
    parser.add_argument("--terminal-min-rows-per-model", type=int, default=30)
    args = parser.parse_args()

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    manifest: dict = {"slot_order": SLOTS}

    add_routerbench(
        rows,
        seen,
        path=Path(args.routerbench_pickle),
        sample_per_eval=args.sample_per_eval,
        manifest=manifest,
    )
    add_swebench(rows, seen, path=Path(args.swebench_jsonl), manifest=manifest)
    add_terminalbench(
        rows,
        seen,
        path=Path(args.terminalbench_jsonl),
        min_rows_per_model=args.terminal_min_rows_per_model,
        manifest=manifest,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "model",
                "isCorrect",
                "output_tokens",
                "source",
                "task_id",
                "source_model",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    manifest["summary"] = summary
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"wrote {len(rows)} rows -> {output}")
    print(f"manifest -> {manifest_path}")
    print(f"unique tasks: {summary['unique_tasks']} | coverage: {summary['coverage']}")
    for model in SLOTS:
        stats = summary["by_model"].get(model, {"rows": 0, "accuracy": 0.0})
        print(f"  {model:28s} {100 * stats['accuracy']:5.1f}%  ({stats['rows']} rows)")


if __name__ == "__main__":
    main()
