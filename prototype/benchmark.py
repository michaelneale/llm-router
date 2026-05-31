"""Benchmark the router: accuracy (5-fold CV), latency, model size."""
import time, random, statistics
from collections import defaultdict

from dataset import get_dataset, route_to_model
from router import Router


def stratified_folds(data, k=5, seed=0):
    rnd = random.Random(seed)
    by = defaultdict(list)
    for row in data:
        by[row[1]].append(row)
    folds = [[] for _ in range(k)]
    for rows in by.values():
        rnd.shuffle(rows)
        for i, row in enumerate(rows):
            folds[i % k].append(row)
    return folds


def main():
    data = get_dataset()
    print(f"{len(data)} labeled examples\n")

    # Shared frozen embedder across folds (only the head is retrained).
    base = Router()
    folds = stratified_folds(data, k=5)
    pairs = []
    for i in range(5):
        test = folds[i]
        train = [r for j, f in enumerate(folds) if j != i for r in f]
        m = Router.__new__(Router)
        m.embedder = base.embedder
        m.clf = None
        m.classes_ = None
        m.fit([r[0] for r in train], [r[2] for r in train], [r[1] for r in train])
        for text, intent, img in test:
            pairs.append((intent, m.predict(text, img)))

    n = len(pairs)
    intent_acc = sum(1 for t, p in pairs if t == p) / n
    model_acc = sum(1 for t, p in pairs if route_to_model(t) == route_to_model(p)) / n

    # Latency on a model trained on all data.
    full = base.fit([r[0] for r in data], [r[2] for r in data], [r[1] for r in data])
    times = []
    for _ in range(20):
        for text, _, img in data:
            t0 = time.perf_counter()
            full.predict(text, img)
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()

    print(f"Intent accuracy (6-way):  {intent_acc*100:.1f}%")
    print(f"Model accuracy (3-way):   {model_acc*100:.1f}%   <- the cost-driving decision")
    print(f"Latency p50 / p95:        {statistics.median(times):.2f} / {times[int(0.95*len(times))-1]:.2f} ms (CPU)")
    print(f"Embedder params:          {full.embedder_params()/1e6:.1f}M (frozen, ~90 MB)")
    print(f"Trained head params:      {full.head_params()}")
    print(f"\nReference: Qwen3-1.7B router = 1.7B params, ~3.4 GB, GPU required, ~150-600 ms/turn")


if __name__ == "__main__":
    main()
