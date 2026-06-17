# TODO

## Retrain Public-Trace Router On Real Provider Ladder

Train the next checkpoint against this real model ladder, using public
HF/benchmark prompts as the task source and fresh correctness labels from the
actual models below.

| Tier | Served model | Provider/key |
| --- | --- | --- |
| Cheapest | `openrouter/openai/gpt-oss-120b` | OpenRouter, `OPENROUTER_API_KEY` |
| Cheap open middle | `openrouter/qwen/qwen3.5-35b-a3b` | OpenRouter, `OPENROUTER_API_KEY` |
| Middle | `openai/gpt-5-mini` | Direct OpenAI, `OPENAI_API_KEY` |
| High | `anthropic/claude-sonnet-4-6` | Direct Anthropic, `ANTHROPIC_API_KEY` |
| Frontier | `anthropic/claude-opus-4-8` | Direct Anthropic, `ANTHROPIC_API_KEY` |

Notes:

- Use these real model identities in labels/config/dashboard; do not remap
  through legacy checkpoint slot names.
- Use OpenRouter only for the open/OSS-style lower tiers above.
- Use direct OpenAI/Anthropic credentials for `gpt-5-mini`, Sonnet, and Opus.
- After labeling, train a fresh checkpoint, sweep tolerance for the
  cost-vs-quality knee, then validate with Goose sessions as spot checks.
