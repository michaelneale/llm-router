# Routing findings: what a personalized router learned from real traces

A study of training a prefill complexity router on real agent session history
(goose + codex), and an account of what the data does and does **not** prove
about when a frontier model is actually needed.

## TL;DR

- On **single-turn, pass/fail-judged** prompts from real traffic, cheap and
  mid-tier models clear the bar on almost everything. Measured "need the frontier
  model" rate: **0 / 300** labeled tasks across two corpora.
- **This is not the same as "you never need the frontier model."** The labeling
  has three structural blind spots that hide exactly the quality the frontier
  model provides. A concrete real failure (below) confirms it.
- Conclusion: route cheap↔mid with the *learned* router; escalate to the top
  tier for sustained-loop / high-stakes tasks via an explicit **policy overlay**,
  not by hoping the prefill classifier learns a need that's invisible to it.

## What we measured

Two corpora extracted with context-aware preambles (`turn N | tools:K | ...`):

| Corpus | Prompts | Median tools/turn | Character |
|---|---|---|---|
| goose | 1,258 | 49 | PR/git/repo ops, shorter sessions |
| codex | 680 | 2,114 | long debugging / architecture marathons |

Balanced 150-question pilots from each were labeled with `model-router collect`
(LLM-as-judge, 9-model pool) and analyzed for the **gap**: questions where the
top model is right and every cheaper model is wrong.

| Pilot | Opus right & ALL cheap wrong | Opus right & mid wrong | depth-correlated? |
|---|---|---|---|
| goose | 0 / 150 | 23 / 150 (15%) | no (shallow 13% ≈ deep 17%) |
| codex | 0 / 150 | 1 / 150 (1%) | no (deep 0%) |

On the *deep* codex work, cheap/mid accuracy was actually **higher** (mini 99.3%,
gpt-5.1/5.2 = 100%) — because rich agentic context in the prompt gives mid models
enough to work with. The real discrimination is at the **bottom** (gpt-4.1-nano
84.7%), i.e. "cheapest vs mid," not "mid vs frontier."

## Why "0 escalation need" is misleading — three blind spots

1. **The judge is a pass/fail gate, not a quality ranker.** In 69% of codex
   questions, *all 9 models* scored "correct." A `gpt-4.1-mini` judge can tell
   "acceptable" from "broken" but not "deeply right" from "plausibly shallow."
   The frontier model's nuance advantage is binary-collapsed to `1 == 1`.

2. **Single-turn labeling can't see agentic-loop quality.** Real frontier-model
   value often lives in *trajectory coherence over a long tool-using loop* — not
   in any one prompt's answer. A model that's 99% per-turn but compounds errors
   over 50 turns fails the **session** while passing every isolated prompt.

3. **We asked for short answers.** Median Opus answer ≈ 400 tokens; real work is
   long, multi-step, tool-using output. The model was tested with one hand tied.

## A concrete failure (blind spot #2, observed)

A real goose session (`20260613_22`) used the router for:

> "Monitor GitHub Actions … Poll every 60 seconds … continue until all workflow
> runs are completed … maximum runtime 24h … be silent until finished, then
> output the JSON result."

The **routed cheap model misunderstood the task**: it wrote a `monitor.sh`
script, ran it once with `MAX_RUNTIME=1 POLL_INTERVAL=1`, checked tool versions,
and **stopped**. It offloaded the patience into a script and bailed. With a
frontier model the agent simply sustains the loop — sleeps, polls, reports at the
end. Single-turn labeling would have scored "wrote a reasonable polling script"
as **correct**, completely missing the failure.

Sustained-loop intent is rare (~1–2% of prompts) — too rare to learn a robust
boundary from — but high-stakes when it occurs. That is the textbook case for a
**policy rule**, not statistical learning.

## What we shipped as a result

- **`configs/goose-mix-v2.yaml`** — re-tiered pool: cheap → mid workhorse →
  strong → top, tolerance tightened to `0.03` (more mid-tier usage), plus an
  `escalation:` overlay.
- **Escalation policy overlay** (`config.py` `EscalationConfig`,
  `strategy.py` `_escalates`): forces the top tier when the prompt matches
  sustained-loop / high-stakes patterns (`poll`, `monitor`, `continue until`,
  `every N seconds`, `be silent until`, or an explicit `!hard` prefix), checked
  *before* the learned router scores the prompt. Rare, transparent, not overfit.

## Takeaways for anyone reusing this

- A personalized router genuinely captures **most** savings — your traffic is
  mostly mid-tier-bound, and that's real.
- **Do not trust a "never escalates" result as proof you don't need a frontier
  model.** Single-turn binary judging is blind to the agentic quality that
  justifies it. Measure session-level outcomes, or keep a policy escalation.
- The cost dashboard measures **cost**, not **quality** — a truncated/empty
  reasoning-model answer counts as a cheap "win." Pair savings with head-to-head
  quality spot-checks (`scripts/headtohead.py`).
