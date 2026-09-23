# Locked evaluation sets

Two frozen problem lists for v0.4 experiments, built by `runs/make_eval_sets.py`
(they contain ids and sha1s only — no problem text, so the sets can't be
re-read as training data and stay within the datasets' licenses).

- `confirm_set.json` — fresh problems (sha1 of the problem text does not appear
  in any saved `runs/` row). Currently **GPQA-only (60)**: every AIME problem in
  `runs/bench/` already appears in saved rows, so no unseen AIME items exist.
  **Read once per promotion decision** — a config that wins on the dev set or
  quick screen gets one run here before it can become the default. Never tune
  on it.
- `quick_screen.json` — the ~40-row first gate for generation-side ideas: rows
  where the v0.3 default (adaptive + answer_prior) was wrong or picked with a
  judge score < 0.8, balanced across thinkers. Run a new arm on these rows
  before spending a full paired run.
