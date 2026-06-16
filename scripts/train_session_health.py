#!/usr/bin/env python3
"""Train a public-trace session-health classifier."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from model_router_toolkit.session_health import NUMERIC_FEATURES, feature_vector


def load_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    required = {"task_id", "window_text", "bad_next", *NUMERIC_FEATURES}
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(f"{path} missing columns: {sorted(missing)}")
    return rows


def split_by_task(rows: list[dict], *, test_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row)
    tasks = list(by_task)
    random.Random(seed).shuffle(tasks)
    n_test = max(1, int(len(tasks) * test_ratio))
    test_tasks = set(tasks[:n_test])
    train = [row for task, vals in by_task.items() if task not in test_tasks for row in vals]
    test = [row for task, vals in by_task.items() if task in test_tasks for row in vals]
    return train, test


def matrix(rows: list[dict], vectorizer: TfidfVectorizer, scaler: StandardScaler, *, fit: bool):
    texts = [row["window_text"] for row in rows]
    nums = np.asarray([feature_vector(row) for row in rows], dtype=float)
    if fit:
        x_text = vectorizer.fit_transform(texts)
        x_num = scaler.fit_transform(nums)
    else:
        x_text = vectorizer.transform(texts)
        x_num = scaler.transform(nums)
    return hstack([x_text, x_num])


def choose_threshold(y_true: np.ndarray, prob: np.ndarray, *, min_precision: float) -> tuple[float, dict]:
    best = (0.5, {"precision": 0.0, "recall": 0.0, "f1": 0.0})
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = prob >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            pred,
            average="binary",
            zero_division=0,
        )
        if precision >= min_precision and recall >= best[1]["recall"]:
            best = (
                float(threshold),
                {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                },
            )
    if best[1]["recall"] == 0.0:
        for threshold in np.linspace(0.05, 0.95, 91):
            pred = prob >= threshold
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true,
                pred,
                average="binary",
                zero_division=0,
            )
            if f1 >= best[1]["f1"]:
                best = (
                    float(threshold),
                    {
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1),
                    },
                )
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_public/session-health-windows.csv")
    parser.add_argument("--output", default="checkpoints/session_health_public.pkl")
    parser.add_argument("--report", default="data_public/session-health-report.json")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--min-precision", type=float, default=0.80)
    parser.add_argument("--max-features", type=int, default=40000)
    args = parser.parse_args()

    rows = load_rows(args.data)
    train_rows, test_rows = split_by_task(rows, test_ratio=args.test_ratio, seed=args.seed)
    y_train = np.asarray([int(row["bad_next"]) for row in train_rows])
    y_test = np.asarray([int(row["bad_next"]) for row in test_rows])
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        raise SystemExit("train/test split needs both positive and negative examples")

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=2,
        lowercase=True,
        strip_accents="unicode",
    )
    scaler = StandardScaler()
    x_train = matrix(train_rows, vectorizer, scaler, fit=True)
    x_test = matrix(test_rows, vectorizer, scaler, fit=False)

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=args.seed,
    )
    classifier.fit(x_train, y_train)
    prob = classifier.predict_proba(x_test)[:, 1]
    threshold, threshold_metrics = choose_threshold(
        y_test,
        prob,
        min_precision=args.min_precision,
    )
    pred = prob >= threshold
    report = {
        "data": args.data,
        "rows": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "threshold": threshold,
        "threshold_metrics": threshold_metrics,
        "accuracy": float(accuracy_score(y_test, pred)),
        "average_precision": float(average_precision_score(y_test, prob)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "classification_report": classification_report(
            y_test,
            pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    checkpoint = {
        "kind": "session_health",
        "version": 1,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "classifier": classifier,
        "threshold": threshold,
        "numeric_features": NUMERIC_FEATURES,
        "report": report,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(checkpoint, f)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"wrote checkpoint -> {args.output}")
    print(f"wrote report     -> {args.report}")
    print(
        "threshold "
        f"{threshold:.2f}: precision={threshold_metrics['precision']:.3f} "
        f"recall={threshold_metrics['recall']:.3f} f1={threshold_metrics['f1']:.3f}"
    )
    print(f"average_precision={report['average_precision']:.3f} roc_auc={report['roc_auc']:.3f}")


if __name__ == "__main__":
    main()

