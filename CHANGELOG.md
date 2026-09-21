# Changelog

## 0.2.0

- `OpenAICompatThinker` accepts base URLs with or without a `/v1` suffix.
- Reasoning-model side channels (`reasoning_content` / `reasoning`) are folded into
  candidate text inside `<think>` tags (`include_reasoning=False` to disable).
- Transient 408/429/5xx/522/524 responses retried with backoff (`max_retries`).
- `best_of_n` never commits to an unfinished candidate — finished paths outrank
  higher-scored truncated ones.
- Live multi-model demo + self-contained HTML renderer in `examples/`
  (`live_demo.py`, `render_demo.py`).

## 0.1.0

- Initial release: stepwise and best-of-N metacognition loop, OpenAI-compatible and
  local-HF thinkers, Jev/reflex System-One judge (noul + choice strategies),
  HumanEval harness.
