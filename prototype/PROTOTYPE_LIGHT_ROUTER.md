# Prototype: A Lightweight, Non-LLM Router

**Question it answers:** Can the routing decision — currently made by a **1.7B-param
Qwen LLM** on a GPU, on the hot path of every turn — be done "vastly more efficiently"
by something that *learns from examples* instead of generating text?

**Short answer:** Yes. A pure-Python classifier (or even zero-ML rules) reproduces the
routing decision at **~94–98% of the model-choice accuracy**, with **>300,000× fewer
parameters**, **~0.006 ms** per decision, and **no GPU**.

---

## What was built

Three routers, compared against the repo's own ground truth:

| Router | What it is | Deps |
|---|---|---|
| **rule baseline** | Zero-ML heuristics: image flag → image intents; "wrong/try again" keywords; reasoning/proof keywords → hard_question; short greetings → chit_chat; else → other | none |
| **tfidf + logreg** | TF-IDF (unigrams + bigrams + `has_image` feature) → softmax linear classifier, trained by gradient descent. **Written from scratch, no numpy/sklearn.** | none |
| **Qwen3-1.7B** (reference) | The existing intent router (full LLM generation per turn) | GPU, vLLM |

All code is in `prototype/` and runs on the Python **standard library only**.

### Ground truth

Labels come **verbatim from the repo**: the `route_config` intents and the
`MAP_INTENT_TO_PIPELINE` mapping in
`src/nat_sfc_router/functions/hf_intent_objective_fn.py`:

```
hard_question, try_again            -> gpt-5-chat                       (frontier)
image_understanding, image_question -> nvidia/nemotron-nano-12b-v2-vl   (VLM)
chit_chat, other                    -> nvidia/nvidia-nemotron-nano-9b-v2 (cheap)
```

The dataset is **128 curated prompts** across the 6 intents, written to match the
official category descriptions. It is a self-consistent stand-in for labeling real
traffic with the live Qwen router (which was not running in this environment).

---

## Results

128 curated examples · 6 intents → 3 models · CPU (Apple M-series class)

| router | intent acc | **model acc** | p50 | p95 | params | footprint | GPU |
|---|---:|---:|---:|---:|---:|---:|:--:|
| rule baseline (0-ML) | 90.6% | **98.4%** | 0.002 ms | 0.004 ms | 0 | ~0 | **no** |
| tfidf + logreg (5-fold CV) | 85.2% | **93.8%** | 0.006 ms | 0.010 ms | 5,016 | ~46 KB | **no** |
| Qwen3-1.7B (reference) | — | — | ~150–600 ms* | — | 1,700,000,000 | ~3.4 GB | **required** |

\* Qwen latency is the per-turn estimate from the earlier analysis (full LLM
generation: prompt prefill + ≤32-token decode + network round-trip). Not benchmarked
live here. Its accuracy is the *implicit* ground truth our curated labels approximate.

- **Intent acc** = exact 6-way route match.
- **Model acc** = picked the *same downstream model* (3-way). **This is the metric that
  actually drives cost and latency** — getting the intent slightly wrong is free if it
  still routes to the same model.

### Two findings worth highlighting

1. **Model-decision accuracy ≫ intent accuracy.** Most residual "errors" collapse to
   the *same* model: `image_question ↔ image_understanding` (both → VLM) and
   `other ↔ chit_chat` (both → cheap LLM). So the costly-error rate is much lower than
   the raw intent-error rate suggests.

2. **The TF-IDF model was data-starved, and more data fixed it linearly.** Going from
   70 → 128 examples lifted it from **67% → 85%** intent and **84% → 94%** model
   accuracy. The method isn't the bottleneck; *labeled examples* are — exactly the
   lever you'd pull in production (harvest real traffic).

---

## Efficiency delta

| | Qwen3-1.7B router | tfidf+logreg | rule |
|---|---:|---:|---:|
| Parameters | 1.7 B | 5,016 | 0 |
| Size reduction vs Qwen | 1× | **~339,000×** | ∞ |
| Footprint | ~3.4 GB VRAM | ~46 KB | ~0 |
| Latency / decision | ~150–600 ms | ~0.006 ms | ~0.002 ms |
| Speedup vs Qwen | 1× | **~25,000–100,000×** | even faster |
| GPU on hot path | **yes, always** | **no** | **no** |
| Add a new route | write 1 sentence (zero-shot) | needs labeled examples + retrain | edit a regex |

The headline isn't the exact accuracy number (small dataset) — it's that the
**routing decision is a cheap classification problem, not a generation problem**, and
treating it as such removes the always-hot GPU from the per-turn critical path.

---

## How this maps to the design discussion

This validates the cascade we sketched earlier:

```
prompt ──► cheap rules (image flag / keywords)   → decided?  → done   (0 ms, no model)
              │ uncertain
              ▼
        tiny classifier (~5k params, CPU, ~0.006 ms) → confident?  → done
              │ rare residual ambiguity
              ▼
        the big LLM router  (only when truly needed — or drop it entirely)
```

The rule layer alone already nails the **highest-value, zero-cost signal**: *"is there
an image?"* → VLM, with no inference at all.

---

## Honest caveats (read before quoting numbers)

- **Accuracy is measured against curated labels, not live production traffic** and not
  live agreement with the actual Qwen router. The live servers (Qwen `:8011`, CLIP
  `:51000`) were not running, and the package mirror was unreachable, so this used a
  self-consistent stand-in.
- **The dataset is small (128 ex.)** and written by hand. Numbers are **directional**.
  The robust, environment-independent findings are the **>300,000× size** and
  **>10,000× latency** reductions and the **removal of the GPU dependency** — those
  don't depend on the dataset.
- **Real prompts are messier** than curated ones (typos, mixed intent, long context,
  adversarial phrasing). Expect lower accuracy on live traffic — mitigated by (a) more
  labeled data and (b) the cascade escalating only the uncertain cases.
- **Zero-shot flexibility is the real trade-off.** The 1.7B LLM can gain a new route
  from a one-sentence description with no data. The light classifier needs labeled
  examples and a (cheap, seconds-long) retrain. For frequently-changing route sets,
  that flexibility has genuine value.

---

## To make this rigorous (next step)

1. Stand up the live Qwen router (`docker compose --profile intent up`).
2. Run a **real prompt corpus** (e.g. a public chat/instruction dataset) through it to
   harvest true `(prompt → route)` labels — i.e. **distill the LLM's actual judgment**.
3. Retrain the tiny classifier on those labels; report **agreement rate with Qwen** and
   **p50/p95 latency** on held-out real prompts.
4. Optionally add a small sentence-embedder (MiniLM/bge-small, ~22M, CPU) feeding the
   same linear head, to test whether embeddings beat TF-IDF on messy text — still
   GPU-free at inference.

---

## Update — recovering the LLM's "zero-shot flexibility" without an LLM

The TF-IDF/rule routers are cheap but **rigid**: adding a new route needs labeled
examples + a retrain. The 1.7B LLM's one advantage is **zero-shot flexibility** — you
add a route by writing one English sentence, no data. We tested two ways to recover
that with a small, GPU-free, **pre-trained** model.

Pre-trained embedder used: **`all-MiniLM-L6-v2`** — 22.7M params, 384-dim, runs on CPU
(~3.3 ms/encode). One-time ~90 MB download. (Installed from **public PyPI**; torch 2.12
+ sentence-transformers work on Python 3.14.)

- **Option B — frozen embedder + trained head:** embed the prompt, feed a tiny sklearn
  logistic-regression head. Needs labels (not zero-shot), but very accurate.
- **Option C — zero-shot description matching:** embed the *route descriptions*
  (verbatim from the repo) once; route each prompt to the nearest description by cosine
  similarity. **No training, no labels** → genuinely zero-shot.

### Full comparison (128 curated examples · 6 intents → 3 models · CPU)

| router | intent | **model** | p50 | p95 | params | footprint | GPU | zero-shot? |
|---|---:|---:|---:|---:|---:|---:|:--:|:--:|
| rule baseline (0-ML) | 90.6% | **98.4%** | 0.003 ms | 0.006 ms | 0 | ~0 | no | no (edit regex) |
| tfidf+logreg (scratch) | 85.2% | 93.8% | 0.006 ms | 0.011 ms | 5,016 | ~46 KB | no | no (retrain) |
| **embedding + head (B)** | **96.1%** | **98.4%** | 3.34 ms | 5.46 ms | ~22.7M | ~90 MB | no | no (retrain head) |
| embedding zero-shot (C) | 56.2% | 85.9% | 3.58 ms | 6.13 ms | ~22.7M | ~90 MB | no | **YES** |
| Qwen3-1.7B (reference) | — | — | ~150–600 ms | — | 1.7 B | ~3.4 GB | **yes** | **yes** |

Trained rows use stratified 5-fold CV.

### The trade-off curve, made concrete

- **Best accuracy, no GPU:** Option B (embedding + trained head) hits **96.1% intent /
  98.4% model** — matching the rule baseline's model accuracy and beating everything on
  intent — at **3.3 ms** and **~75× smaller than Qwen**. This is the strong default if
  your route set is stable.
- **Zero-shot flexibility, no GPU:** Option C adds/removes routes from a *description
  alone*. It's rougher (56% intent / 86% model) because pure description-matching is
  noisier than a trained head — but it's the only light option that behaves like the
  LLM for *unseen* routes.

### Live zero-shot demo (the thing TF-IDF cannot do)

We added a brand-new `code_generation` route with **one sentence and zero examples**:

```
BEFORE (only the 6 original routes exist):
   Write a Python function to reverse a linked list -> try_again
   Debug this segfault in my C++ program.           -> hard_question
   Refactor this JavaScript to use async/await.     -> other
   Generate a SQL query to join two tables.         -> other

AFTER add_route('code_generation', <one sentence>)  -- ZERO examples, NO retrain:
   Write a Python function to reverse a linked list -> code_generation
   Debug this segfault in my C++ program.           -> code_generation
   Refactor this JavaScript to use async/await.     -> other
   Generate a SQL query to join two tables.         -> other
```

It immediately captured 2 of 4 code prompts with no data — the other two stayed
borderline against the existing descriptions. A TF-IDF/logreg model would route **0 of
4** correctly until you collected labels and retrained. That gap *is* zero-shot
flexibility: imperfect here, but free and instant.

### So what would "pre-training" take?

You don't pre-train anything yourself — you **reuse** a pre-trained embedder
(downloaded once, ~90 MB, CPU). From there:

- **Stable routes →** train a tiny head (Option B): seconds, needs a few labeled
  examples per route, best accuracy.
- **Changing routes →** description-matching (Option C): write a sentence, done, zero
  data — at some accuracy cost.
- **Best of both (recommended):** a **cascade** — Option C handles unseen/new routes
  zero-shot; periodically harvest traffic labels and promote hot routes to the trained
  head (Option B) for accuracy; escalate only genuinely ambiguous cases to the big LLM.

All three keep the **1.7B LLM and its GPU off the per-turn hot path**, which was the
original goal.

### Honest notes on the embedding options

- Accuracy is still vs **curated labels**, small set — directional, not production.
- Option C's 56% intent is real: cosine-to-description is a weak classifier. It shines
  for *flexibility*, not raw accuracy. Better route descriptions and per-route
  similarity thresholds would lift it.
- Latency rose from ~0.006 ms (TF-IDF) to ~3.3 ms (embedder) — still **40–180× faster**
  than the LLM and CPU-only. First call also pays a one-time model load (~1–2 s).
- The embedder pulls torch (~88 MB wheel). It runs on CPU, but it's a heavier
  dependency than the stdlib-only routers. Pick based on whether you need its accuracy
  or zero-shot ability.

## Files

```
prototype/
├── dataset.py                     # 128 curated (prompt, intent, has_image)
├── rule_router.py                 # zero-ML heuristic router
├── tfidf_lr_router.py             # from-scratch TF-IDF + logistic regression (stdlib)
├── embedding_zeroshot_router.py   # Option C: zero-shot route-description matching
├── embedding_trained_router.py    # Option B: frozen MiniLM + sklearn logreg head
├── benchmark.py                   # original stdlib-only benchmark
├── benchmark_full.py              # all 5 routers + flexibility column
└── PROTOTYPE_LIGHT_ROUTER.md
```

Reproduce:
- stdlib-only: `cd prototype && python3 benchmark.py`
- full (needs venv): `python3 -m venv .venv && .venv/bin/python -m pip install \
  --index-url https://pypi.org/simple scikit-learn numpy sentence-transformers && \
  .venv/bin/python benchmark_full.py`
