# Router Tuning Knobs

The trained checkpoint ranks models by predicted success. The pool config turns
that into a cost decision. Use these knobs to explore the cost/quality frontier
without retraining.

The live proxy also exposes the effective settings on `/dashboard` under
Routing knobs, including env tolerance overrides, the weighted cost ladder,
switching margins, the top-tier override, and the route log path.

## Main Quality/Cost Knob

```yaml
routing:
  tolerance: 0.145
```

Higher tolerance allows cheaper models farther below the top predicted success
probability. Lower tolerance is more conservative.

Useful starting points:

| tolerance | use when |
| ---: | --- |
| `0.11` | small-loss check, safer first rollback |
| `0.145` | aggressive high-savings test |
| `0.18` | very aggressive, only after real quality checks |

You can override without editing YAML:

```bash
ROUTER_TOLERANCE=0.11 scripts/router-service.sh restart
```

## Tier Weights

```yaml
routing:
  output_token_weight: 0.25

models:
  - name: claude-haiku-4-5-high
    routing_cost_multiplier: 1.0
```

`output_token_weight` decides how much output pricing affects routing order:

- `0.0`: input-price-only routing.
- `0.25`: approximate agent blend used by the replay tools.
- higher values penalize high-output-cost models more.

`routing_cost_multiplier` is routing-only. It changes how cheap/dear a slot
looks to the selector, but does not change real savings accounting.

- `> 1.0`: penalize this tier, route to it less often.
- `< 1.0`: favor this tier, route to it more often.

Example: make Sonnet less attractive without changing its real price:

```yaml
  - name: claude-sonnet-4-6-high
    display_name: Claude Sonnet 4.6
    litellm_model: anthropic/claude-sonnet-4-6
    cost_per_m_input_tokens: 3.00
    cost_per_m_output_tokens: 15.00
    routing_cost_multiplier: 1.5
```

## Tier Mappings

The checkpoint slot `name` must stay fixed, but the real model behind it can
change:

```yaml
  - name: claude-haiku-4-5-high
    display_name: GPT-5.4 Mini Cap
    litellm_model: openai/gpt-5.4-mini
```

This is the highest-impact knob. It is also the riskiest because the public loss
calibration was learned for the checkpoint slot, not for every later remap.

Current aggressive test:

- top forced slot: `openai/gpt-5.5`
- high automatic slot: `openai/gpt-5.4-mini`

## Starting Cheap

```yaml
utility:
  cheap_when_prompt_matches:
    - "^\\s*(hi|hello|hey|thanks|thank you)\\s*[!.?]*\\s*$"
    - "^\\s*(ls|pwd|whoami|date)\\b\\s*$"
```

These only apply at cold session start. They are for obvious utility turns where
model quality is not the scarce resource. Keep this narrow.

## Explicit Burst To Frontier

```yaml
escalation:
  top_tier_model: claude-opus-4-6-high
  force_top_tier_when_prompt_matches:
    - "(^|\\s)!hard\\b"
```

`!hard` is the only explicit manual override in the current setup. It routes the
turn to the top slot, currently `openai/gpt-5.5`.

## Going Down Vs Up

```yaml
switching:
  enabled: true
  up_margin: 0.0
  down_margin: 0.06
  down_margin_per_100k: 0.04
  max_down_margin: 0.25
```

This is the cache-aware switching gate:

- Going up to a stronger/dearer model is easy: `up_margin`.
- Going down to a cheaper model is harder: `down_margin`.
- Big cached contexts make down-switching stickier:
  `down_margin_per_100k`.

Runtime overrides:

```bash
ROUTER_DISABLE_SWITCHING=1 scripts/router-service.sh restart
ROUTER_DISABLE_SWITCHING=0 scripts/router-service.sh restart
```

## Per-Request Controls

The proxy also accepts:

- request body or metadata `tolerance`: override tolerance for one request.
- metadata `pin_model`: force one internal slot for subagent-style flows.
- metadata `models`: restrict allowed internal slots for one request.

Goose may not expose all of these directly, but OpenAI-compatible clients can.

## Current Rollback Ladder

For the aggressive pool:

1. Keep pool, set `ROUTER_TOLERANCE=0.11`.
2. Keep tolerance, map `claude-haiku-4-5-high` back to `openai/gpt-5.4`.
3. Disable switching if cache stickiness hides savings:
   `ROUTER_DISABLE_SWITCHING=1`.
4. Force a task to frontier with `!hard`.
