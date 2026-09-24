# MetaCog — full results and experiment ledger

Everything measured so far, in one place. The README carries only the headline
numbers; this file keeps the tables, provenance and every configuration tried.

Conventions: rows are *paired* (same model, same problem, with and without the
change). "fixed / lost" counts problems the change got right that the baseline
missed, and vice versa. p-values are exact McNemar. "Ceiling" is the share of
problems where at least one generated path was correct — the best any judge could
do on those paths.

## 1. v0.3 headline (adaptive + answer prior)

Live, 180 paired rows (GPQA Diamond first 60 + AIME 60; DeepSeek V4 Flash,
Kimi K2.5, GLM-5.3 Flash), Jev judging.

| arm | accuracy |
|---|---|
| thinker alone (one greedy answer) | 76.7 % |
| v0.2 default (best-of-N + greedy anchor + cascade) | 81.7 % |
| **v0.3 default (adaptive + answer prior)** | **86.7 %** |

v0.3 vs v0.2: +11 / −2, p = 0.02, 0.89× wall time, 1.39× thinker tokens.

| | |
|---|---|
| ![Accuracy: baseline → MetaCog → ceiling](charts/accuracy.svg) | ![Why MetaCog still misses](charts/misses.svg) |
| Gap MetaCog→ceiling is what a better judge could recover; gap ceiling→100 % needs more/better paths. | "No path right" → generate more or more diverse paths. "Jev picked wrong" → improve the judge. Judge errors are the minority everywhere. |
| ![Cost vs one greedy answer](charts/cost.svg) | ![Cascade: problems stopped after the greedy path](charts/cascade.svg) |
| Generations ≈ token cost. Time exceeds generations because greedy → samples → judge run in sequence. | Share of problems where Jev was ≥ 0.95 on the greedy answer so no samples were made, and how often that answer was right. A noul ≥ 0.95 was correct 226/229 times. |

Charts regenerate from the run files with `python runs/dashboard.py --export`.

## 2. Standard benchmarks, 10 hosted thinkers (v0.2 configuration)

Best-of-N with `greedy_anchor` + cascade, 3 paths (5 on AIME), raw chat
completions, no tools; 10 CommandCode-hosted thinkers in parallel, Jev judging.
Denominators are partial — provider usage caps stopped the runs — so these are
not full benchmark scores; the comparison is baseline vs MetaCog on identical rows.

| benchmark | rows | baseline | MetaCog | ceiling | fixed / lost | p |
|---|---|---|---|---|---|---|
| custom harder counting | 134 | 0.642 | **0.701** | 0.709 | 10 / 2 | 0.039 |
| MATH-500 level 5 (integer) | 159 | 0.931 | **0.969** | 0.975 | 7 / 1 | 0.070 |
| GPQA Diamond | 372 | 0.707 | **0.742** | 0.772 | 23 / 10 | 0.035 |
| AIME 2024+2025 | 191 | 0.759 | **0.817** | 0.859 | 14 / 3 | 0.013 |
| pooled | 856 | 0.750 | **0.794** | — | 54 / 16 | 6e-6 |

Where the paths disagreed and at least one was right, Jev picked a correct one
224/245 times.

## 3. Experiment ledger

Every configuration tested. Live rows are fixed / lost vs the **v0.2 default**
unless stated; opt-in additions were also measured against the v0.3 default they
sit on top of, and that number decided their verdict. Offline rows replay saved
candidate pools through the judge only (no thinker calls) — see `runs/devset.py`.

| idea | what it changes | rows | result (fixed / lost) | verdict |
|---|---|---|---|---|
| adaptive + answer prior | branch 2–6 full paths sized by judge uncertainty; judge also scores each bare final answer | 180 | 86.7 % vs 81.7 %, +11 / −2, p=0.02 | **promoted — the v0.3 default** |
| adaptive paths alone | the branching without the prior | 263 | +15 / −8, p=0.21 | folded into default |
| answer prior alone | the prior on top of best-of-N (offline: 83→93/119 picks) | 191 | +8 / −4, p=0.39 | folded into default |
| answer-group voting | sum scores per distinct answer, pick inside the winner | 213 pools (offline) | 130 vs 133 picks, +17/−20 | dropped |
| pairwise tie-break | `choice` call between top-2 within 0.05 | 213 pools (offline) | 132 vs 133 picks, +3/−4 | dropped |
| prefix judging / early pruning | judge 256–2048-token prefixes, prune early | 47 pools (offline) | worse: 31–36 vs 38 correct | dropped |
| stepwise 2-level tree | continuations at the leaves instead of full answers | 20 (pilot) | tie: 12/20 vs 12/20, +7 % tokens, 1.8× judge calls | kept as `mode="stepwise"`, not default |
| local judge (LogitJudge, Qwen3.5-4B) | P(yes) from next-token logits, no API key | 127 pools (offline) | 68 vs Jev 72, +8/−12, p=0.50 | shipped, opt-in |
| calibrated judge ensemble | logistic meta-ranker over Jev text, bare-answer, answer-frequency and local-judge scores | 250 pools (offline; fit on A, read once on B) | no gain — learned weights reproduce the fixed 0.5 prior (B: 74 vs 73/98); answer frequency hurts held-out (70/98); local-judge fusion adds nothing; Jev already calibrated (Brier 0.16) | dropped |
| diversity hints | each extra path gets a different approach prompt | 69 | +4 / −0 vs v0.2; **+2 / −3 vs v0.3** | opt-in `diversity_hints` |
| escalate thinker | a stronger model writes one path when still unsure | 50 | +5 / −1 vs v0.2; **+3 / −3 vs v0.3**, 1.19× time | opt-in `escalate_thinker` |
| triage | judge rates the bare problem; hard ones branch immediately | 205 | +12 / −5, p=0.14, 0.69× time | opt-in `triage` |
| adaptive sketches | 800-token outlines, kept 1–3 expanded | 256 | +14 / −8, p=0.29, 1.09× time | opt-in `sketch_tokens` |
| cheaper sketches | 500-token outlines, ≤2 expanded | 218 | +9 / −8, p=1.00 | dropped |
| wider & earlier stop | stop at 0.90, branch up to 8 | 23 | +0 / −0 | dropped |
| deep tree | prune sketches, re-branch up to 3 levels | 81 | +3 / −1, p=0.62, 1.51× time | dropped |

### What limits further gains

On most benchmarks only 1–4 % of problems are lost to a wrong pick, while up to
~29 % have *no correct path at all*. The judge is near its ceiling; the next gains
are on the generation side (more or more-diverse thought paths — repair,
verification, tools; see [ROADMAP.md](ROADMAP.md)) and on scheduling (wall time
runs ~5× a single answer vs ~2.8× generations because the stages run in sequence).

## 4. Local judge vs Jev

Offline replay on 127 saved pools (tuning split) where the thought paths disagree
and both judges have scores. `LogitJudge` (Qwen3.5-4B Q4, CPU) picks the correct
path 68/127 vs Jev 72/127 (ceiling 100). Head-to-head +8/−12, p = 0.50 —
statistically a tie, slightly weaker, in line with SemIf's reported Jev-agreement
gap. `answer_prior` gives no gain with the local judge (68→68); answer-group
voting helps it modestly (73/127). Latency ≈ 2 min per 4-path pool on 8 CPU cores
vs seconds for Jev — use a GPU or llama-server for real use.

## 5. Live sanity pass, 11 hosted thinkers (Sep 2026)

`examples/live_demo.py`: 12 harder counting problems, 3 candidates each, Jev
judging, no cherry-picking. A check that the loop works on live models, not a
benchmark — most problems are too easy to separate the arms.

| thinker | baseline | MetaCog (Jev pick of 3) | ceiling |
|---|---|---|---|
| deepseek/deepseek-v4-flash | 10/12 | **11/12** | 11/12 |
| xiaomi/mimo-v2.5 | 10/12 | **11/12** | 11/12 |
| z-ai/glm-5.3-flash | 10/12 | **11/12** | 12/12 |
| MiniMaxAI/MiniMax-M2.5 | 10/12 | 10/12 | 10/12 |
| meituan/LongCat-2.0 | 10/12 | 10/12 | 10/12 |
| stepfun/Step-3.5-Flash | 10/12 | 10/12 | 10/12 |
| moonshotai/Kimi-K2.5 | 11/12 | 11/12 | 11/12 |
| Qwen/Qwen3.8-Flash | 11/11 | 11/11 | 11/11 |
| zai-org/GLM-5.1 (provider dropped out) | 6/6 | 6/6 | 6/6 |
| Qwen/Qwen3.6-Plus (provider dropped out) | 3/3 | 3/3 | 3/3 |
| MiniCPM5-2B, local on a DGX Spark (12 easier problems) | 11/12 | **12/12** | 12/12 |

MetaCog never did worse than the baseline; every rescue was a problem where the
baseline was wrong and Jev picked a correct candidate out of a disagreeing set.

## 6. v0.1: MiniCPM5-2B with a separate judge (HumanEval / GSM8K)

The original experiment that motivated the project. MiniCPM5-2B thinker,
separate judge, oracle grading.

| arm | HumanEval (164 tasks, 6 candidates) | GSM8K (84/200, 10 candidates) |
|---|---|---|
| thinker's single answer (no judge) | 0.555 | 0.738 |
| majority vote over candidates (no judge) | — | 0.833 |
| thinker judging its own candidates | 0.524 (worse than no judge) | — |
| **Reflex judge, per-candidate noul** (local, MIT, $0) | **0.689** | not run |
| Reflex judge, one-shot choice | 0.671 | — |
| Jev judge, one-shot choice | 0.726 | 0.869 |
| **Jev judge, per-candidate noul** | **0.756** | **0.905** |
| ceiling: any candidate passes | 0.793 | 0.976 |

The gain comes from *choosing*, not from thinking more: majority vote helps, a
separate judge helps more, and the thinker grading itself is worse than no judge
at all. Reflex–Jev gap within noise (p = 0.11, n = 164). Reflex rescued 36 % of
items where the first answer failed. Judge confidence was *not* a usable
"send to a human" signal. Provenance and limits in [METHOD.md](METHOD.md).
