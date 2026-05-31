"""
Unified benchmark: 5 router strategies vs the 1.7B Qwen reference.

Routers:
  1. rule baseline (0-ML)                  - heuristics, no model
  2. tfidf + logreg (from scratch)         - pure stdlib
  3. embedding + sklearn logreg head       - frozen MiniLM + trained head (Option B)
  4. embedding zero-shot (desc matching)   - frozen MiniLM, NO training (Option C)
  5. Qwen3-1.7B (reference)                - the existing LLM router (characteristics only)

Metrics: intent acc (6-way), model acc (3-way, drives cost), latency p50/p95,
params/footprint, GPU need, and FLEXIBILITY (can add a route with zero labeled data?).

Trained models (2,3) use stratified 5-fold CV for honest held-out accuracy.
Zero-shot (4) and rule (1) need no training, evaluated on the full set.
"""
import time, random, statistics
from collections import defaultdict

from dataset import get_dataset, route_to_model
import rule_router
from tfidf_lr_router import TfidfLogReg
from embedding_zeroshot_router import EmbeddingZeroShotRouter
from embedding_trained_router import EmbeddingTrainedRouter


def latency_ms(predict_fn, samples, repeats):
    times = []
    for _ in range(repeats):
        for text, _, img in samples:
            t0 = time.perf_counter()
            predict_fn(text, img)
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return statistics.median(times), times[int(0.95 * len(times)) - 1]


def eval_pairs(pairs):
    n = len(pairs)
    ia = sum(1 for t, p in pairs if t == p) / n
    ma = sum(1 for t, p in pairs if route_to_model(t) == route_to_model(p)) / n
    return ia, ma


def stratified_folds(data, k=5, seed=0):
    rnd = random.Random(seed); by = defaultdict(list)
    for row in data: by[row[1]].append(row)
    folds = [[] for _ in range(k)]
    for rows in by.values():
        rnd.shuffle(rows)
        for i, row in enumerate(rows): folds[i % k].append(row)
    return folds


def cv_eval(make_and_fit, data, k=5):
    folds = stratified_folds(data, k)
    pairs = []
    for i in range(k):
        test = folds[i]; train = [r for j, f in enumerate(folds) if j != i for r in f]
        m = make_and_fit(train)
        for text, intent, img in test:
            pairs.append((intent, m.predict(text, img)))
    return eval_pairs(pairs)


def row(name, ia, ma, p50, p95, params, foot, gpu, flex):
    def acc(x): return f"{x*100:5.1f}%" if x == x else "   -- "
    def ms(x): return f"{x:7.3f}" if x == x else "    -- "
    print(f"{name:<30} {acc(ia)} {acc(ma)}  {ms(p50)} {ms(p95)}  {params:>14}  {foot:>9}  {gpu:<4} {flex}")


def main():
    data = get_dataset()
    print(f"Dataset: {len(data)} curated examples - 6 intents -> 3 models "
          f"(ground truth = repo route_config + MAP_INTENT_TO_PIPELINE)\n")
    hdr = f"{'router':<30} {'intent':>6} {'model':>6}  {'p50ms':>7} {'p95ms':>7}  {'params':>14}  {'foot':>9}  gpu  zero-shot?"
    print(hdr); print("-" * len(hdr))

    # 1. rule
    pairs = [(i, rule_router.predict(t, im)) for t, i, im in data]
    ia, ma = eval_pairs(pairs)
    p50, p95 = latency_ms(rule_router.predict, data, repeats=50)
    row("rule baseline (0-ML)", ia, ma, p50, p95, "0", "~0", "no", "no (edit regex)")

    # 2. tfidf + logreg (from scratch)
    ia, ma = cv_eval(lambda tr: TfidfLogReg().fit([r[0] for r in tr], [r[2] for r in tr], [r[1] for r in tr]), data)
    full = TfidfLogReg().fit([r[0] for r in data], [r[2] for r in data], [r[1] for r in data])
    p50, p95 = latency_ms(full.predict, data, repeats=50)
    row("tfidf+logreg (from scratch)", ia, ma, p50, p95, f"{full.num_params():,}", "~46 KB", "no", "no (retrain)")

    # 3. embedding + trained head (Option B) -- share one embedder instance across folds
    shared = EmbeddingTrainedRouter()
    def make_b(tr):
        m = EmbeddingTrainedRouter.__new__(EmbeddingTrainedRouter)
        m.embedder = shared.embedder
        m.clf = None; m.classes_ = None
        return m.fit([r[0] for r in tr], [r[2] for r in tr], [r[1] for r in tr])
    ia, ma = cv_eval(make_b, data)
    fullb = shared.fit([r[0] for r in data], [r[2] for r in data], [r[1] for r in data])
    p50, p95 = latency_ms(fullb.predict, data, repeats=10)
    tot = fullb.embedder_params() + fullb.head_params()
    row("embedding+logreg head (B)", ia, ma, p50, p95, f"~{tot/1e6:.1f}M", "~90 MB", "no", "no (retrain head)")

    # 4. embedding zero-shot (Option C) -- reuse embedder
    zs = EmbeddingZeroShotRouter()
    zs.model = shared.embedder  # reuse loaded embedder
    zs.set_routes(zs.routes and __import__("embedding_zeroshot_router").ROUTE_DESCRIPTIONS)
    pairs = [(i, zs.predict(t, im)) for t, i, im in data]
    ia, ma = eval_pairs(pairs)
    p50, p95 = latency_ms(zs.predict, data, repeats=10)
    row("embedding zero-shot (C)", ia, ma, p50, p95, f"~{zs.num_params()/1e6:.1f}M", "~90 MB", "no", "YES (just a desc)")

    # 5. Qwen reference
    row("Qwen3-1.7B (reference)*", float('nan'), float('nan'), float('nan'), float('nan'),
        "1,700,000,000", "~3.4 GB", "YES", "YES (just a desc)")

    print("\n* Qwen = characteristics only (not benchmarked live; ~150-600ms/turn full LLM gen).")
    print("intent = exact 6-way; model = same downstream model (3-way, the cost-driving decision).")
    print("Trained rows (tfidf, embedding+head) = stratified 5-fold CV.")
    print("foot = approx on-disk/in-memory footprint of the routing model itself.")


if __name__ == "__main__":
    main()
