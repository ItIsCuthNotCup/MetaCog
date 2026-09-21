# Roadmap: next additions and how they are tested

Status of the default: best-of-N + greedy anchor + Jev confidence cascade
(v0.2). Measured winner not yet promoted: `answer_prior=0.5` (+ adaptive
paths), ~+5 pts over the default on paired live rows, p≈0.07 at 125 rows.

## Test protocol (iterate fast without fooling ourselves)

1. **Dev set** — every saved row where finished candidates *disagree* on the
   final answer (`runs/devset.py`). Only these rows can be flipped by a change
   to picking, so they carry ~10× the information per Jev call of random rows.
   Split A/B by a hash of the problem text; tune on A, read B once per idea.
2. **Picking-side ideas** (how to choose between paths) are replayed fully
   offline on the dev set: no thinker calls, only cached Jev calls.
3. **Generation-side ideas** (what paths to write) need live rows: run them
   paired against the current best arm on the *same fixed problem list*
   (`runs/problems_*.json`), and stop when fixed/lost separates (exact McNemar)
   or when 150 rows are in — not at an arbitrary N.
4. **Confirm set** — a config that wins on dev gets one run on fresh problems
   before it is promoted to default. Never promote on the tuning rows.

Metric everywhere: fixed / lost vs the reference arm on identical problems,
exact McNemar p; plus time and token multipliers.

## Additions

| # | Idea | Side | Status |
|---|------|------|--------|
| 1 | Diversity prompts: each sampled thought path gets a different angle (eliminate options / check edge cases / work backwards) | generation | to run live |
| 2 | Answer-group voting: sum text+prior scores per distinct answer, pick inside the winning group | picking | **tested offline, no gain** (all: 130 vs 133 right, +17/−20) — dropped |
| 3 | Thinker escalation: if Jev is still < 0.95 after branching, one path from a stronger model instead of more paths from the same one | generation | to run live |
| 4 | Pairwise tie-break: Jev `choice` between the top-2 when totals are within 0.05 | picking | **tested offline, no gain** (all: 132 vs 133, +3/−4) — dropped |
| 5 | Distil MetaCog picks back into the thinker (SFT/GRPO via Halo) | training | later; needs GPUs |

Dev-set baseline (213 disagreeing pools, ceiling 178): text-only pick 123,
text + 0.5·prior 133 (+17/−7 vs text, p=0.06) — the prior is confirmed as the
picking lever; further picking tricks on top of it did not move the number.
Remaining misses are mostly "no path was right" (35/213), which is why #1 and
#3 target coverage rather than judging.
