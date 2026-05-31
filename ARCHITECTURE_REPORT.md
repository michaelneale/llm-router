# LLM Router v2 — Architecture & Technical Report

> **NVIDIA AI Blueprint: LLM Router v2 (Experimental)**

---

## 1. What This Is

This project is an **experimental, next-generation LLM/VLM router**. Given a user
prompt (text and/or images), it analyzes the request and returns the **name of the
most appropriate model** to handle it — trading off **accuracy vs. cost vs. latency**.

The core idea: in agentic AI systems you don't want to send *every* request to your
most expensive frontier model. A cheap, fast model is fine for chit-chat; a frontier
model is worth it for hard reasoning; a vision-language model (VLM) is needed for
images. This router automates that decision.

### Key distinction vs. v1

| Aspect | v1 (main branch) | v2 (this, experimental) |
|---|---|---|
| Server | Rust proxy | **NVIDIA NeMo Agent Toolkit** (FastAPI) |
| Inference backend | BERT + Triton | **Qwen3 1.7B LLM** *or* **CLIP + Neural Network** |
| Behavior | Classify **and proxy** to the LLM | **Classify only** — returns a model name |
| Input | Text only | **Text + Images (multimodal)** |
| Routing | Task / complexity classification | **Intent-based** or **Auto-routing (NN)** |

> ⚠️ **Important architectural fact:** v2 does **not** proxy. It only returns the
> recommended model name in an OpenAI-compatible response. The *caller* (e.g. the
> demo app) is responsible for actually invoking the chosen model.

### Default model pool

| Model | Type | Provider | Role |
|---|---|---|---|
| `gpt-5-chat` | Frontier LLM | Azure OpenAI / OpenAI | Complex reasoning, hard questions |
| `nvidia/nemotron-nano-12b-v2-vl` | Open VLM | NVIDIA Build API | Image / multimodal understanding |
| `nvidia/nvidia-nemotron-nano-9b-v2` | Small open LLM | NVIDIA Build API | Simple text, chit-chat |

These are examples — you can reconfigure intent mappings or retrain the NN to route
to any models.

---

## 2. High-Level Architecture

```
                        ┌──────────────────────────────────────────────┐
                        │            Demo App (Gradio UI)                │
                        │            demo/app.py · port 7860             │
                        │                                                │
   User ──text/image──► │  1. POST request to router                    │
                        │  2. Receive recommended model name            │
                        │  3. Call that model's real API directly        │
                        │  4. Stream answer back to user                 │
                        └───────────────┬───────────────▲────────────────┘
                                        │               │
                  (1) /sfc_router/chat/completions       (3) real model calls
                                        │               │   (Azure OpenAI / NVIDIA Build)
                                        ▼               │
   ┌─────────────────────────────────────────────────────────────────────┐
   │              Router Backend  (NeMo Agent Toolkit / FastAPI)           │
   │                       src/nat_sfc_router · port 8001                  │
   │                                                                       │
   │   sfc_router_fn  ──► objective_fn  (one of two, chosen in config)     │
   │                                                                       │
   │   ┌──────────────────────────┐      ┌──────────────────────────────┐ │
   │   │ hf_intent_objective_fn   │  OR  │ nn_objective_fn               │ │
   │   │ (Intent-based routing)   │      │ (Auto-routing)                │ │
   │   └────────────┬─────────────┘      └───────────────┬──────────────┘ │
   └────────────────┼────────────────────────────────────┼────────────────┘
                    │                                     │
        prompt → route classification          text+image → CLIP embedding
                    │                                     │     → NN prediction
                    ▼                                     ▼
   ┌──────────────────────────┐        ┌─────────────────────────────────┐
   │  Qwen3-1.7B (vLLM)        │        │  CLIP Embedding Server          │
   │  qwen-router · port 8011  │        │  clip-as-service · port 51000   │
   │  profile: intent          │        │  profile: nn                    │
   └──────────────────────────┘        └─────────────────────────────────┘
```

There are **three deployable services** (the 2nd is profile-dependent):

1. **router-backend** (`:8001`) — always on. The brain.
2. **qwen-router** (`:8011`) — only with `--profile intent`. Runs Qwen3-1.7B via vLLM.
3. **clip-server** (`:51000`) — only with `--profile nn`. CLIP embeddings.
4. **demo-app** (`:7860`) — interactive Gradio UI.

---

## 3. Tech Stack

| Layer | Technology |
|---|---|
| **Router framework** | NVIDIA NeMo Agent Toolkit (`nvidia-nat==1.3.1`), FastAPI, Uvicorn |
| **API contract** | OpenAI Chat Completions schema (Pydantic models generated from OpenAI's `CompletionCreateParamsNonStreaming` TypedDict) |
| **Intent router model** | Qwen3-1.7B served by **vLLM** (OpenAI-compatible), Hugging Face `transformers` tokenizer |
| **Auto-router model** | **CLIP** (via `clip-client` / Jina `clip-as-service`) for 1024-D embeddings + a **PyTorch** feed-forward NN |
| **ML / data** | PyTorch, NumPy, scikit-learn, XGBoost (`xgb_multi_router.joblib` as alt), joblib, datasets |
| **Demo UI** | Gradio |
| **Downstream model clients** | `openai` SDK (`AzureOpenAI` for GPT-5, `OpenAI` w/ NVIDIA base_url for Nemotron) |
| **Packaging / deploy** | Python 3.12, `uv`, Docker + Docker Compose, NVIDIA CUDA base image |
| **Config** | YAML (`config.yml`) drives the NeMo workflow |

---

## 4. Components in Detail

### 4.1 Router Backend (`src/nat_sfc_router/`)

A NeMo Agent Toolkit application. It is **config-driven** — `configs/config.yml`
declares a FastAPI front-end with endpoints and wires up "functions":

**Endpoints exposed:**
- `POST /sfc_router/chat/completions` → main router (returns chosen model)
- `GET  /health` → health check
- `POST /nn_objective_fn/chat/completions` → call NN router directly (debug)
- `POST /hf_intent_objective_fn/chat/completions` → call intent router directly (debug)

**Functions (registered NeMo components):**

| File | Function | Purpose |
|---|---|---|
| `functions/router_fn.py` | `sfc_router` | Thin orchestrator. Calls the configured `objective_fn`, wraps the returned model name in an OpenAI `ChatCompletion` response. |
| `functions/hf_intent_objective_fn.py` | `hf_intent_objective_fn` | **Intent-based routing** strategy. |
| `functions/nn_objective_fn.py` | `nn_objective_fn` | **Auto-routing** strategy (CLIP + NN). |
| `functions/healthcheck_fn.py` | `healthcheck` | Liveness. |
| `schema/openai_chat_request.py` | `OpenAIChatRequest` | Dynamically builds a Pydantic model from OpenAI's TypedDict, with logic to coerce OpenAI `ValidatorIterator` objects into plain lists for clean serialization. |

The selected strategy is chosen by the `objective_fn:` key in `config.yml`
(`hf_intent_objective_fn` or `nn_objective_fn`). The output is always:

```json
{ "choices": [ { "message": { "content": "<model-name>", "role": "assistant" } } ],
  "model": "<objective_fn-name>" }
```

The chosen model is in `choices[0].message.content`.

---

### 4.2 Routing Strategy A — Intent-Based (`hf_intent_objective_fn`)

**No training required.** Uses Qwen3-1.7B as a classifier.

Flow:
1. Take the **last** message of the conversation.
2. **Redact images** from it (the intent model is text-only) — but textual cues like
   "what's in this image?" survive, so it can still infer that vision is needed.
3. Build a prompt embedding a list of **routes** (intent categories with descriptions)
   and ask Qwen to return JSON `{"route": "..."}`.
4. Call the remote vLLM server (`ROUTER_MODEL_URL`, `/v1/chat/completions`,
   `max_tokens=32`, `temperature=0.3`).
5. Parse the route and map it to a model via a static dictionary.

Intent categories → model mapping:

```python
route_config = [hard_question, chit_chat, try_again,
                image_understanding, image_question]

MAP_INTENT_TO_PIPELINE = {
    "other":               "nvidia/nvidia-nemotron-nano-9b-v2",
    "chit_chat":           "nvidia/nvidia-nemotron-nano-9b-v2",
    "hard_question":       "gpt-5-chat",
    "image_understanding": "nvidia/nemotron-nano-12b-v2-vl",
    "image_question":      "nvidia/nemotron-nano-12b-v2-vl",
    "try_again":           "gpt-5-chat",
}
```

Performance touches: cached routes JSON, `lru_cache` on response parsing, lazy
tokenizer loading, detailed per-stage timing logs.

**Pros:** zero training, semantic understanding, trivially reconfigurable.
**Needs:** the Qwen vLLM service running on `:8011` (GPU, ~16 GB, T4+).

---

### 4.3 Routing Strategy B — Auto-Routing (`nn_objective_fn`)

**Learning-based, cost-optimized.** Uses CLIP embeddings + a trained neural net.

Flow:
1. Extract all text and images across the conversation
   (`extract_text_and_images_from_messages` — robust to dicts, Pydantic objects,
   multimodal content lists, base64 data URIs).
2. **Generate a 1024-D embedding** via CLIP:
   - text → 512-D, image → 512-D, concatenated.
   - text-only → 512-D text + 512 zeros padding.
   - Done async (`asyncio.to_thread`) with an asyncio-policy reset to avoid
     uvloop/Jina event-loop conflicts.
3. Feed the embedding to the **RouterNetwork** (PyTorch), getting an independent
   sigmoid probability per candidate model.
4. **Cost-based selection** (`select_best_model_by_cost`):
   - keep models whose probability ≥ their confidence threshold,
   - among those, pick the **lowest cost factor** (prob as tiebreaker),
   - fall back to highest probability if none qualify.
5. Map the router's model label to the target model name and return it (plus the
   probability dict).

Thresholds & costs come from `config.yml`:

```yaml
model_thresholds: { gpt-5-chat: 0.5, ...12b-v2-vl: 0.65, ...9b-v2: 0.6 }
model_costs:      { gpt-5-chat: 1.0, ...12b-v2-vl: 0.5,  ...9b-v2: 0.3 }
```

The router is loaded **once at startup** into a module-global (`_load_router`),
not per request.

**Neural network architecture** (`training/nn_router.py · RouterNetwork`):

```
Input(1024)
  → [Linear → BatchNorm1d → ReLU → Dropout(0.3)] × hidden_dims [512, 256, 128]
  → Linear(→ N_models)
  → Sigmoid   (independent per-model probability — multi-label, not softmax)
```

Trained with weighted BCE loss (class weights handle data imbalance), with
Optuna-style random hyperparameter tuning. Multi-label sigmoid means "how likely is
*each* model to be correct," which pairs naturally with the threshold + cost selection.

**Pros:** data-driven, learns from real usage, optimizes quality/latency/cost,
multimodal-native, inference runs on CPU once trained.
**Needs:** the CLIP server running (GPU for embeddings); training needs PyTorch (GPU optional).

---

### 4.4 Training Pipeline (`src/nat_sfc_router/training/`)

| File | Role |
|---|---|
| `prepare_hf_data.py` | Build training data from HF datasets (labels models via API calls). |
| `generate_embeddings.py` | Produce CLIP embeddings for the dataset. |
| `nn_router.py` | Defines `RouterNetwork`, training loop, class weights, hyperparameter tuning, evaluation, threshold sweeps; `load_router` / `route_embeddings` helpers. |
| `model_router.py` | `ModelRouter` — production wrapper combining CLIP client + trained NN; handles path resolution (dev vs. packaged), sync/async embedding, preset threshold configs (conservative/balanced/aggressive), cost-aware logic. |
| `router_usage_examples.py` | Usage examples. |
| `router_artifacts/` | Pre-trained artifacts shipped with the repo: `nn_router.pth`, `model_names.joblib`, `best_hyperparameters.joblib`, `xgb_multi_router.joblib` (XGBoost alternative). |

Three notebooks document the journey:
`1_IntentRouter_Example.ipynb`, `2_Embedding_NN_Training.ipynb`,
`3_Embedding_NN_Usage.ipynb`.

---

### 4.5 Demo Application (`demo/app.py`)

A Gradio web UI that demonstrates **end-to-end** routing (the only component that
actually calls downstream models). Key functions:

- `call_router(messages)` → POSTs to `ROUTER_ENDPOINT`, gets the model name.
- `call_model_azure_openai(...)` → invokes `gpt-5-chat` via `AzureOpenAI`.
- `call_model_nvidia(...)` → invokes Nemotron models via `OpenAI` SDK pointed at the
  NVIDIA Build API base URL.
- `call_model(model_name, messages)` → dispatches to the right provider.
- `encode_image_to_base64(...)` → prepares images (resized) as data URIs.
- `chat(...)` / `create_demo()` → the Gradio chat loop and NVIDIA-themed UI.

So the demo: takes user input → asks router which model → calls that model's real
API → renders the answer. This is the reference for how a consuming app integrates
a "classify-only" router.

---

## 5. Deployment

**Build & packaging:** Python 3.12, `uv` for dependency management, built on an
NVIDIA CUDA Ubuntu base image. The backend container runs:

```bash
uv run nat serve --config_file config.yml --host 0.0.0.0 --port 8001
```

**Docker Compose profiles** select the routing backend:

```bash
# Intent-based (default; brings up qwen-router on :8011)
docker compose --profile intent up -d --build

# Neural-network auto-routing (brings up clip-server on :51000)
docker compose --profile nn up -d --build
```

You must keep `objective_fn` in `config.yml` consistent with the profile:
- intent → `objective_fn: hf_intent_objective_fn`
- nn → `objective_fn: nn_objective_fn`

**Service wiring (from `docker-compose.yml`):**
- `router-backend` env: `ROUTER_MODEL_URL=http://qwen-router:8000`,
  `ROUTER_MODEL_NAME=Qwen/Qwen3-1.7B`, `CLIP_SERVER=grpc://clip-server:51000`.
- `qwen-router`: `vllm/vllm-openai` image, custom `qwen3_nonthinking.jinja` chat
  template, HF cache mounted, GPU reserved.
- `clip-server`: `jinaai/clip-as-service` image, GPU reserved.
- `demo-app`: needs `OPENAI_API_KEY`, `NVIDIA_API_KEY`, `AZURE_OPENAI_ENDPOINT`;
  talks to router over the internal Docker network.
- Healthchecks + `depends_on` ensure ordered, healthy startup.
- All services share a bridge network `router-network`.

Access the demo at **http://localhost:7860**.

---

## 6. Request Flow (End to End)

1. User sends text (+ optional image) in the Gradio UI.
2. Demo app POSTs an OpenAI-style chat request to
   `http://router-backend:8001/sfc_router/chat/completions`.
3. `sfc_router` invokes the configured `objective_fn`:
   - **Intent:** redact images → Qwen classifies intent → map to model.
   - **NN:** CLIP-embed text+image → NN probabilities → threshold + cost selection.
4. Backend returns `{choices[0].message.content = "<model-name>"}`.
5. Demo app reads the model name and calls that model's **real** API
   (Azure OpenAI for GPT-5, NVIDIA Build for Nemotron).
6. The downstream model's answer is streamed back to the user.

---

## 7. Notable Design Decisions & Observations

- **Separation of routing from serving.** The router is a pure decision service; it
  never holds model credentials or proxies traffic. This keeps it lightweight and
  lets each consumer manage its own auth and providers. (Trade-off: callers must do
  the model dispatch themselves, as the demo illustrates.)
- **Two interchangeable strategies behind one interface.** Swappable via a single
  config key, both conforming to the same `objective_fn` contract
  `(request) -> (model_name, probabilities)`.
- **Cost-aware multi-label NN.** Independent per-model sigmoids + thresholds + cost
  factors implement an explicit quality/cost optimization rather than a plain argmax.
- **Multimodal handling.** The intent router cleverly *redacts* images but keeps
  textual cues; the NN router embeds images natively via CLIP.
- **OpenAI compatibility via codegen.** The request schema is generated from OpenAI's
  own TypedDict at import time, with careful `ValidatorIterator → list` coercion.
- **Async safety.** CLIP/Jina blocking calls are pushed to worker threads with an
  asyncio-policy reset to coexist with FastAPI/uvloop.

### Minor issues spotted
- In `nn_objective_fn.py`, the `MODEL_ROUTER_TO_TARGET` mapping has a **duplicate
  key** `'nemotron-nano-12b-v2-vl'` (defined twice with different targets — the second
  wins), and the partial-match fallback references `'openai/gpt-oss-120b'`, which
  isn't in the documented default model pool. Worth reconciling.
- On the error path, `nn_objective_fn` returns a bare string `DEFAULT_FALLBACK_MODEL`
  while the success path returns a `(model, probabilities)` tuple; `sfc_router` does
  handle both via try/except, but it's a slightly fragile contract.

---

## 8. File Map (Quick Reference)

```
llm-router/
├── docker-compose.yml          # 3–4 services, intent/nn profiles
├── Dockerfile                  # backend image (uv, CUDA base, nat serve)
├── pyproject.toml              # deps; registers nat component entrypoint
├── qwen3_nonthinking.jinja     # Qwen chat template for vLLM
├── 1/2/3_*.ipynb               # intent demo, NN training, NN usage notebooks
├── src/nat_sfc_router/
│   ├── register.py             # imports functions to register NeMo components
│   ├── configs/config.yml      # FastAPI endpoints + function wiring + objective_fn
│   ├── functions/
│   │   ├── router_fn.py         # sfc_router orchestrator
│   │   ├── hf_intent_objective_fn.py   # Strategy A: Qwen intent routing
│   │   ├── nn_objective_fn.py           # Strategy B: CLIP+NN auto-routing
│   │   └── healthcheck_fn.py
│   ├── schema/openai_chat_request.py    # OpenAI-compatible Pydantic schema
│   └── training/
│       ├── nn_router.py         # RouterNetwork + train/eval/tune
│       ├── model_router.py      # ModelRouter production wrapper
│       ├── generate_embeddings.py, prepare_hf_data.py, router_usage_examples.py
│       └── router_artifacts/    # nn_router.pth, model_names, hyperparams, xgb
└── demo/
    ├── app.py                  # Gradio UI; calls router then real models
    ├── Dockerfile, requirements.txt, run.sh, env_template.txt
```
