# MetaCog: System-One arbitration over parallel thought paths

*Status: method write-up for v0.1. The numbers in §1 and §4 come from the original
hand-run experiment (HumanEval-164, MiniCPM5-2B, 6 candidates per item, oracle-graded by
unit tests) and have not yet been re-run with this package; see [Reproducing](#reproducing).*

## 1. Observation

Small reasoning models often do not think in a single line. MiniCPM5-2B, asked a coding
problem, regularly produces a thinking trace that contains **several alternative
approaches side by side** ("Approach 1 … Alternatively … Another way …") and then commits
to one of them, frequently the wrong one. The model has the right idea *somewhere* in its
own output; what it lacks is a reliable way to choose.

Choosing between fixed alternatives is exactly the job a *System One* decision model is
built for. TypeSafe's [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
and its open re-creation [Reflex](https://github.com/kshetrajna12/reflex) (Qwen3.5-4B)
take a `state` and a `choice` question and return a calibrated probability for every
option in one forward pass, in ~100 ms, without generating any text.

Putting the two together — the small thinker proposes paths, the judge picks — raised
MiniCPM5-2B's HumanEval accuracy from **0.555 to 0.689** with a local Reflex judge and to
**0.756** with Jev (coverage ceiling 0.793: at least one of the 6 candidates passes).
Reflex rescued **36 %** of the items where the model's own first answer failed.
MetaCog is the formalisation of that loop, made model-agnostic.

## 2. Roles

| role | what it does | requirements |
|---|---|---|
| **Thinker** \(T\) | Generates reasoning text. Any autoregressive LM: an OpenAI-compatible endpoint (vLLM, llama.cpp, Ollama, OpenAI, Anthropic via proxy) or a local `transformers` model. | Must be able to sample \(n\) continuations of a given prefix. |
| **Judge** \(J\) | Given the problem and \(k\) candidate paths, returns a probability distribution over which path to continue. Never writes text. | Any `POST /v1/systemone` server: Jev, Reflex, or your own. |
| **Controller** | The MetaCog loop below. Pure Python, no model weights. | — |

The judge is deliberately *not* asked to solve the problem. It is asked a yes/no or
comparative question ("is this path correct?", "which of these is most likely to lead to a
correct answer?"), which is a far easier task than generation and one a 4B model does well. This is the sense in which
MetaCog is *metacognition*: a cheap process that monitors and steers an expensive one.

## 3. The loop

Let \(P\) be the problem, \(n\) the branching factor, \(k\) the beam width, \(s\) the
step size in tokens.

```
prefixes ← [""]
for step in 1..max_steps:
    C ← ∅
    for p in prefixes:
        for c in T.sample(P, prefix=p, n=n, max_tokens=s):        # branch
            C ← C ∪ {p + c}
            C ← C ∪ split_paths(p + c)                            # optional: harvest paths the
                                                                  # thinker already wrote side by side
    π ← J.score(P, C)                                             # judge: P(correct) per path (noul)
                                                                  # or J.choose(P, C) for one-shot choice
    kept ← top-k of C by π
    if kept[0] is finished:            return kept[0]             # commit
    if confidence(π) ≥ τ:              return kept[0] + T.greedy(P, prefix=kept[0])
    prefixes ← kept
return kept[0]                                                    # budget exhausted
```

Three knobs matter:

- **`n_paths`** (\(n\)) — how many alternatives the judge sees per step. 4 is the default
  (the original experiment used 6; `choice` accepts up to 26).
- **`step_tokens`** (\(s\)) — how often the judge intervenes. Small \(s\) means tight
  steering and more judge calls; \(s = \infty\) collapses the loop to **best-of-\(n\)**
  (mode `best_of_n`), which is the cheapest configuration and the one used in the original
  experiment.
- **`commit_confidence`** (\(\tau\)) — when the judge is this sure, stop paying for
  branching and let the thinker finish greedily.

Cost: per step, \(n \cdot k\) thinker samples of \(s\) tokens plus \(n \cdot k\) judge
calls (`noul`) or **one** judge call (`choice`). With Reflex on a consumer GPU each judge
call is ~100–300 ms; with Jev it is one HTTP round trip (~$0.00006 per decision).

### 3.1 Splitting

`split_paths` is the piece specific to the observation in §1. When a single sample already
contains several approaches, it is cut at the approach markers ("Alternatively", "Approach
2:", "Another way", …), and each piece is prefixed with the shared preamble so it is
self-contained. The pieces enter the candidate pool with `source="split"`. This lets the
judge select among the model's *own* alternatives, not only among resampled ones, at zero
extra thinker cost. It can be disabled (`split_generations=False`) for thinkers that reason
linearly.

### 3.2 What the judge sees

Two judging strategies are implemented; `noul` is the default because it measured better.

**`noul` (per-candidate, isolated).** One request per candidate, the judge never sees the
others:

```json
{
  "model": "jev-latest",
  "state": {"problem": "<the original problem>", "path": "<one candidate>"},
  "questions": {"is_correct": {"type": "noul", "instructions": "`path` is one candidate solution or reasoning path for `problem`. Is its final answer correct (or, if it is unfinished, is it on track to reach a correct answer)? Judge the substance, not the length or style."}}
}
```

The \(k\) yes-probabilities are normalised into the selection distribution \(\pi\); the raw
values are kept in the trace. Cost: \(k\) judge calls per step (each ~100 ms on Reflex).

**`choice` (one-shot, comparative).** All candidates in one request:

```json
{
  "model": "jev-latest",
  "state": {
    "problem": "<the original problem>",
    "paths": {"A": "<candidate A>", "B": "<candidate B>", "C": "..."}
  },
  "questions": {
    "best_path": {
      "type": "choice",
      "instructions": "Several candidate reasoning paths for the same problem are shown under `paths`. Which path is most likely to lead to a correct, complete final answer if it is continued (or, if already finished, is most likely correct)? Judge the soundness of the reasoning and how directly it addresses the problem, not its length or style.",
      "criteria": {"A": "path A", "B": "path B", "C": "path C"}
    }
  }
}
```

Long paths are **tail-truncated** (the most recent reasoning is kept), because that is
where a path is going. Reflex users should set `permutations=2` so option order does not
bias the `choice` readout. Optionally, a finished winner is double-checked with two `noul`
questions (`is_complete`, `is_correct`), recorded in the trace for downstream thresholds.

A third variant, **refute** ("is this candidate wrong?"), was measured and is harmful
(0.402, below the no-judge baseline); it is deliberately not implemented.

## 4. Why this should work (and when it won't)

- **Selection is easier than generation.** The thinker's per-sample accuracy \(a\) may be
  low, but if at least one of \(n\) samples is right with probability
  \(1-(1-a)^n\) (the *coverage* ceiling) and the judge picks a right one with probability
  \(q\), accuracy rises toward \(q \cdot \text{coverage}\). Measured: coverage@6 = 0.793,
  Reflex-noul reached 0.689 (\(q \approx 0.87\)), Jev-noul 0.756 (\(q \approx 0.95\)).
- **The judge must be a different model.** MiniCPM5-2B judging its own candidates scored
  0.524 — *below* its unrouted first answer (0.555). A separate, calibrated System-One
  judge is the whole point; never let the thinker grade itself.
- **Calibrated probabilities make the loop tunable.** Because System-One outputs are
  probabilities rather than rankings, `commit_confidence` and beam pruning are principled
  thresholds, not heuristics, and they can be re-fitted on your own tasks with Reflex's
  calibration tooling.
- **It is model-agnostic by construction.** The controller only needs "sample \(n\)
  continuations of a prefix" from the thinker and "answer a choice question" from the
  judge. Swapping MiniCPM for GPT-class API models, or Reflex for Jev, is a constructor
  argument.

Limits:

- Gains are bounded by *coverage*: if no path is right, no judge can help. Stronger
  thinkers with high single-sample accuracy see smaller gains.
- The judge must know the domain well enough to discriminate. Reflex's frozen Qwen3.5-4B is
  good at code and general reasoning; for exotic domains, fine-tune or calibrate it.
- Stepwise mode with prefix continuation needs a thinker endpoint that supports
  continuing an assistant message (vLLM `continue_final_message`, or the raw
  `/v1/completions` API). Chat-only endpoints should use `best_of_n`.
- Truncation to 26 candidates (`choice` only) and to `max_chars_per_path` is lossy by design.
- **Confidence margins are not yet usable for abstention.** On the HumanEval pool, deferring
  the lowest-margin quartile *reduced* accuracy (0.659 vs 0.689). Treat `commit_confidence`
  as a cost knob, not a correctness guarantee, until calibrated on your task.
- Measured on one task family (HumanEval, n=164). The Reflex–Jev gap (8–2 discordant items,
  McNemar p=0.11) is within noise at that size. Generalisation to math and open-ended
  reasoning is untested.

## 5. Relation to prior work

MetaCog is a **beam search with a learned, calibrated, non-generative value function**.
It differs from self-consistency (majority vote needs a parseable final answer; MetaCog
works on partial reasoning) and from Tree-of-Thoughts / LLM-as-judge (the judge there is
the same expensive generative model; here it is a purpose-built System-One model that
returns probabilities in one pass). It is closest to process-reward-model guided search,
with the PRM replaced by a general-purpose decision model that needs no task-specific
training.

## 6. Reproducing

```bash
# thinker: MiniCPM5-2B behind vLLM on :8000; judge: Reflex on :8008
metacog-eval humaneval --thinker-url http://localhost:8000 --thinker-model openbmb/MiniCPM5-2B \
    --judge none --out runs/baseline.jsonl                           # baseline pass@1
metacog-eval humaneval --thinker-url http://localhost:8000 --thinker-model openbmb/MiniCPM5-2B \
    --judge reflex --mode best_of_n --n 4 --out runs/metacog_bon.jsonl
metacog-eval humaneval --thinker-url http://localhost:8000 --thinker-model openbmb/MiniCPM5-2B \
    --judge jev --mode stepwise --n 4 --step-tokens 256 --out runs/metacog_step.jsonl
```

The harness executes model-written code; run it in a container. Always report the
coverage ceiling next to any accuracy, and compare arms with paired tests (McNemar) on the
same pool. Results, configs, and traces are collected in `docs/results/`.

Planned: a *repair* step (judge picks, thinker repairs the pick), which can exceed the
coverage ceiling; a GSM8K arm with exact-number grading; a calibration module for
abstention.
