# Roadmap: next additions and how they are tested

Status of the default (v0.3): `mode="adaptive"` + `answer_prior=0.5` — promoted
on 180 paired rows (DeepSeek-v4-flash, Kimi-K2.5, GLM-5.3-flash; GPQA Diamond
60 + AIME): 86.7 % vs 81.7 % for the v0.2 default (+11/−2, McNemar p=0.02),
0.89× wall time, 1.39× thinker tokens; model alone 76.7 %.

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
| 1 | Diversity prompts: each sampled thought path gets a different angle (eliminate options / check edge cases / work backwards) | generation | **tested live, no gain** (+2/−3 vs adaptive+prior, n=69, 0.89× tokens) — opt-in via `DIVERSITY=1` |
| 2 | Answer-group voting: sum text+prior scores per distinct answer, pick inside the winning group | picking | **tested offline, no gain** (all: 130 vs 133 right, +17/−20) — dropped |
| 3 | Thinker escalation: if Jev is still < 0.95 after branching, one path from a stronger model instead of more paths from the same one | generation | **tested live, no gain** (+3/−3, n=50; fired on 19/50 rows at 1.19× time) — opt-in via `ESCALATE_MODEL` |
| 4 | Pairwise tie-break: Jev `choice` between the top-2 when totals are within 0.05 | picking | **tested offline, no gain** (all: 132 vs 133, +3/−4) — dropped |
| 5 | Distil MetaCog picks back into the thinker (SFT/GRPO via Halo) | training | later; needs GPUs |
| 6 | Open-weights local judge (`LogitJudge`: P(yes) from next-token logits, SemIf-style) | picking | **shipped** (v0.3.x, PR #8). Replay on 127 disagreeing pools: 68/127 vs Jev 72/127, +8/−12, p=0.50 — statistical tie, slightly weaker. Next: GPU/llama-server latency (≈2 min/4-path pool on CPU), shared-state batching, per-workload calibration |

Dev-set baseline (213 disagreeing pools, ceiling 178): text-only pick 123,
text + 0.5·prior 133 (+17/−7 vs text, p=0.06) — the prior is confirmed as the
picking lever; further picking tricks on top of it did not move the number.
Remaining misses are mostly "no path was right" (35/213), which is why #1 and
#3 target coverage rather than judging.
