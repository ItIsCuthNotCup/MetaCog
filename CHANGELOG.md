# Changelog

## Unreleased

- `answer_prior`: Jev also judges each distinct bare final answer; that prior is
  added to the text score (offline on saved disagreeing pools: 83→93 of 119 correct
  picks, +14/−4). Judge default `max_chars_per_path` raised 6000→24000.
- `mode="adaptive"` (experimental): the greedy answer is the root; the judge's
  uncertainty `u = 1 − score` scales branching between `n_min`/`n_max`, and an
  optional sketch level (`sketch_tokens`) is judged and pruned (`expand_max`,
  `prune_margin`) so only kept sketches are expanded to full solutions. A greedy
  path at `stop_confidence` returns immediately; sketches are never the answer.
- adaptive `max_rounds`: repeat sketch → prune → expand levels (re-sketching from the best answer so far) until the judge scores a full answer ≥ `stop_confidence`.

## 0.2.0

- `OpenAICompatThinker` accepts base URLs with or without a `/v1` suffix.
- Reasoning-model side channels (`reasoning_content` / `reasoning`) are folded into
  candidate text inside `<think>` tags (`include_reasoning=False` to disable).
- Transient 408/429/5xx/522/524 responses retried with backoff (`max_retries`).
- Optional SSE streaming (`stream=True`) for generation requests — survives
  proxies that kill long-idle responses; chunks are folded back into the
  standard response shape.
- `best_of_n` never commits to an unfinished candidate — finished paths outrank
  higher-scored truncated ones.
- `Config.greedy_anchor` keeps a temperature-0 sample in the candidate pool
  (source `"greedy"`), so the judge can never do worse than the baseline for
  lack of the option.
- `Config.cascade_confidence` (requires `greedy_anchor`): scores the greedy path
  with a noul first and skips the remaining samples when it clears the threshold
  (measured ≥0.95 → 226/229 correct).
- `Config.finish_paths` (stepwise): after the last branching step, each surviving
  prefix is expanded into `n_paths` full completions and the judge decides among
  those — a pruned decision tree ending on full thoughts. Stepwise ranking is now
  finished-first at every level.
- When a thinker endpoint rejects `n>1`, remaining samples are fetched
  concurrently (up to 8 workers) instead of strictly sequentially.
- Live multi-model demo + self-contained HTML renderer in `examples/`
  (`live_demo.py`, `render_demo.py`).

## 0.1.0

- Initial release: stepwise and best-of-N metacognition loop, OpenAI-compatible and
  local-HF thinkers, Jev/reflex System-One judge (noul + choice strategies),
  HumanEval harness.
