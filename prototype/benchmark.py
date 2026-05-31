"""Stdlib-only benchmark: rule + tfidf routers. Intent/model accuracy, latency, size."""
import time
import random
import statistics
from collections import defaultdict

from dataset import get_dataset, route_to_model
import rule_router
from tfidf_lr_router import TfidfLogReg


def latency_percentiles(predict_fn, samples, repeats=50):
    times = []
    for _ in range(repeats):
        for text, _, img in samples:
            t0 = time.perf_counter()
            predict_fn(text, img)
            times.append((time.perf_counter() - t0) * 1000.0)  # ms
    times.sort()
    p50 = statistics.median(times)
    p95 = times[int(0.95 * len(times)) - 1]
    return p50, p95


def eval_predictions(pred_pairs):
    """pred_pairs: list of (true_intent, pred_intent)"""
    n = len(pred_pairs)
    intent_correct = sum(1 for t, p in pred_pairs if t == p)
    model_correct = sum(1 for t, p in pred_pairs if route_to_model(t) == route_to_model(p))
    return intent_correct / n, model_correct / n


def stratified_folds(data, k=5, seed=0):
    rnd = random.Random(seed)
    by_label = defaultdict(list)
    for row in data:
        by_label[row[1]].append(row)
    folds = [[] for _ in range(k)]
    for label, rows in by_label.items():
        rnd.shuffle(rows)
        for i, row in enumerate(rows):
            folds[i % k].append(row)
    return folds


def run_rule(data):
    pairs = [(intent, rule_router.predict(text, img)) for text, intent, img in data]
    ia, ma = eval_predictions(pairs)
    p50, p95 = latency_percentiles(rule_router.predict, data)
    return ia, ma, p50, p95, 0


def run_tfidf_cv(data, k=5):
    folds = stratified_folds(data, k=k)
    all_pairs = []
    last_model = None
    for i in range(k):
        test = folds[i]
        train = [row for j, f in enumerate(folds) if j != i for row in f]
        m = TfidfLogReg().fit([r[0] for r in train], [r[2] for r in train], [r[1] for r in train])
        for text, intent, img in test:
            all_pairs.append((intent, m.predict(text, img)))
        last_model = m
    ia, ma = eval_predictions(all_pairs)
    # train one final model on all data for latency + size
    full = TfidfLogReg().fit([r[0] for r in data], [r[2] for r in data], [r[1] for r in data])
    p50, p95 = latency_percentiles(full.predict, data)
    return ia, ma, p50, p95, full.num_params(), full


def fmt_row(name, ia, ma, p50, p95, params, footprint, gpu):
    return (f"{name:<26} {ia*100:6.1f}%   {ma*100:6.1f}%   "
            f"{p50:7.3f}  {p95:7.3f}   {params:>12}  {footprint:>10}  {gpu}")


def main():
    data = get_dataset()
    print(f"Dataset: {len(data)} curated examples across 6 intents "
          f"(ground truth = repo's route_config + MAP_INTENT_TO_PIPELINE)\n")

    print(f"{'router':<26} {'intent':>7} {'model':>8}   {'p50ms':>7}  {'p95ms':>7}   "
          f"{'params':>12}  {'footprint':>10}  gpu")
    print("-" * 100)

    ia, ma, p50, p95, _ = run_rule(data)
    print(fmt_row("rule baseline (0-ML)", ia, ma, p50, p95, 0, "~0", "no"))

    ia, ma, p50, p95, params, full = run_tfidf_cv(data)
    # estimate footprint: params * 8 bytes (float) + vocab strings
    kb = (params * 8 + sum(len(t) for t in full.vocab) ) / 1024.0
    print(fmt_row("tfidf+logreg (5-fold CV)", ia, ma, p50, p95, params, f"~{kb:.0f} KB", "no"))

    # Qwen reference row (characteristics, not measured here)
    print(fmt_row("Qwen3-1.7B (reference)*", float('nan'), float('nan'),
                  float('nan'), float('nan'), "1,700,000,000", "~3.4 GB", "REQUIRED"))

    print("\n* Qwen row = model characteristics for context (size/GPU). Its accuracy is the")
    print("  implicit 'ground truth' our curated labels approximate; latency ~150-600ms/turn")
    print("  (full LLM generation) per earlier analysis. Not benchmarked live (no server).")
    print("\nIntent = exact 6-way route match. Model = same downstream model chosen (3-way,")
    print("the decision that actually drives cost/latency).")


if __name__ == "__main__":
    main()
