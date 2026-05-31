# Lightweight router prototype

A CPU-only alternative to the Qwen 1.7B intent router.

**Old:** a 1.7B LLM classifies each prompt (GPU required, on the hot path of every turn).
**This:** a small frozen BERT embedder (`all-MiniLM-L6-v2`, 22.7M params) turns the prompt
into a vector, and a logistic-regression head — trained on labeled examples — picks the route.

```
prompt -> MiniLM embedding (384-d) -> LogisticRegression -> route -> downstream model
```

Same output as the original router (a model name). No LLM, no GPU.

## Results (128 labeled examples, 5-fold CV)

| | this router | Qwen3-1.7B (original) |
|---|---|---|
| Intent accuracy | 96% | — (the implicit ground truth) |
| Model accuracy* | 98% | — |
| Latency | ~6 ms (CPU) | ~150–600 ms (GPU) |
| Size | 22.7M frozen + 2.3k trained | 1.7B |

\* whether it picks the same downstream model — the decision that drives cost.

## Trade-off

The original LLM is **zero-shot**: add a route by writing a description, no data.
This router needs **labeled examples** to learn a route, but is ~75× smaller and CPU-only.

## Caveats

- The 128 examples (`dataset.py`) are hand-written, not real traffic — accuracy is
  **directional**. In production you'd train on many real `(prompt -> route)` examples.
- `all-MiniLM-L6-v2` is text-only; image routes here use the `has_image` flag, not actual
  image understanding (a real multimodal version needs a vision encoder).

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --index-url https://pypi.org/simple \
  scikit-learn numpy sentence-transformers
.venv/bin/python benchmark.py
```

## Files

- `router.py` — the router (embedder + trained head)
- `dataset.py` — labeled example prompts
- `benchmark.py` — accuracy / latency / size
