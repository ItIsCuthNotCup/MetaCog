# MetaCog

**Let a small, fast decision model steer a bigger model's reasoning.**

MetaCog wraps any language model (local or API) in a *metacognition* loop: the model
branches into several thought paths, a System-One judge — [Jev](https://typesafe.ai) or
the open [Reflex](https://github.com/kshetrajna12/reflex) (4B, runs on your GPU) — picks
the most promising path in one ~100 ms forward pass, and the model continues down it.

```
problem ──▶ thinker samples n paths ──▶ judge picks one ──▶ thinker continues it ──▶ … ──▶ answer
                    ▲                         │
                    └─────────────────────────┘   (repeat until the winning path finishes)
```

## Measured

MiniCPM5-2B thinker, separate judge, oracle grading. Two task families, same pattern twice.

| arm | HumanEval (164 tasks, 6 candidates) | GSM8K (84/200 so far, 10 candidates) |
|---|---|---|
| thinker's single answer (no judge) | 0.555 | 0.738 |
| majority vote over candidates (no judge) | — | 0.833 |
| thinker judging its own candidates | 0.524 (worse than no judge) | — |
| **Reflex judge, per-candidate noul** (local, MIT, $0) | **0.689** | not yet run |
| Reflex judge, one-shot choice | 0.671 | — |
| Jev judge, one-shot choice | 0.726 | 0.869 |
| **Jev judge, per-candidate noul** | **0.756** | **0.905** |
| ceiling: any candidate passes | 0.793 | 0.976 |

The gain comes from *choosing*, not from thinking more: majority vote helps, a separate
judge helps more, and the thinker grading itself is worse than no judge at all. The
Reflex–Jev gap is within noise (McNemar p=0.11, n=164). Reflex rescued 36 % of the items
where the first answer failed. Judge confidence was *not* a usable "send to a human"
signal. Samples are modest — directional, not gospel. Full HumanEval table, provenance and
limits in [docs/METHOD.md](docs/METHOD.md).

## Live demo (any hosted model + Jev)

`examples/live_demo.py` wraps any OpenAI-compatible endpoint
(env `THINKER_URL` / `THINKER_MODEL` / `THINKER_KEY`, `PROBLEM_SET=hard`,
`OUT=file.json`) in MetaCog with Jev, and records baseline vs candidates vs the
judge's pick with full traces. `examples/render_demo.py 'runs/*.json' report.html
"title"` renders a self-contained HTML report.

One pass (Sep 2026) over 12 harder counting problems, 3 candidates each, Jev judging, no
cherry-picking of models or problems. Sanity check that the loop works on live models, not a
benchmark — most problems are too easy to separate the arms.

| thinker | baseline (1 greedy) | MetaCog (Jev pick of 3) | ceiling |
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

MetaCog never did worse than the baseline; every rescue was a problem where the baseline
was wrong and Jev picked a correct candidate out of a disagreeing set.
`greedy_anchor=True` now keeps the greedy answer in the candidate pool, so the judge can
never do worse than baseline for lack of the option — and the confidence cascade
(`cascade_confidence`, default on in the demo at 0.95) scores the greedy path first and
skips sampling entirely when it's that confident: measured, a noul ≥ 0.95 is correct
226/229 times.

## Install

```bash
pip install -e .              # core: httpx + pydantic only
pip install -e ".[eval]"      # + HumanEval harness
pip install -e ".[hf]"        # + local transformers thinker
```

## Use

```python
from metacog import MetaCog, Config, OpenAICompatThinker, SystemOneJudge

thinker = OpenAICompatThinker(
    "http://localhost:8000", model="openbmb/MiniCPM5-2B"
)  # vLLM / llama.cpp / Ollama / OpenAI
judge = SystemOneJudge.reflex(
    "http://localhost:8008"
)  # or SystemOneJudge.jev()  (uses $TYPESAFE_API_KEY)

mc = MetaCog(thinker, judge, Config(mode="stepwise", n_paths=4, step_tokens=256))
result = mc.run(
    "Write a Python function that returns the k-th smallest element of an unsorted list."
)

print(result.answer)
for r in result.trace.rounds:
    print(r.step, [f"{p:.2f}" for p in r.verdict.probabilities], "kept", r.kept)
```

Two modes:

| mode | what happens | when |
|---|---|---|
| `best_of_n` | sample `n_paths` full answers, judge picks one | cheapest; works with any chat endpoint |
| `stepwise` | sample `n_paths` continuations of `step_tokens`, judge picks, extend, repeat | tighter steering; needs prefix continuation (vLLM `continue_final_message` or `/v1/completions`) |

Judging strategy: `strategy="noul"` (default; one isolated "is this correct?" call per
candidate — the measured winner) or `strategy="choice"` (one comparative call, cheaper).
Set `split_generations=True` (default) to also harvest the alternative approaches a
thinker writes inside a single sample, and `commit_confidence=0.9` to stop branching once
the judge is sure (a cost knob — judge margins were *not* a usable abstention signal in the
measured pool).

CLI:

```bash
metacog solve "…problem…" --thinker-url http://localhost:8000 --thinker-model X --judge reflex
```

## Bring your own model

MetaCog never touches weights; it only needs something that can sample text.

| you have | use |
|---|---|
| vLLM / llama.cpp / Ollama / LM Studio / any OpenAI-compatible server | `OpenAICompatThinker(base_url, model)` |
| OpenAI, Together, Groq, OpenRouter, … | `OpenAICompatThinker("https://api.openai.com", model="gpt-4o-mini", api_key=…)` |
| a Hugging Face checkpoint in-process | `TransformersThinker("openbmb/MiniCPM5-2B")` (`pip install -e ".[hf]"`) |
| anything else | any object with `generate(problem, prefix, *, n, max_tokens, temperature, stop) -> list[Generation]` |

```python
from metacog import Generation


class MyThinker:
    def generate(self, problem, prefix, *, n, max_tokens, temperature, stop=None):
        texts = my_model.sample(
            problem + prefix, n=n, max_new_tokens=max_tokens, temperature=temperature
        )
        return [Generation(text=t, finished=True) for t in texts]


mc = MetaCog(MyThinker(), SystemOneJudge.jev(), Config(mode="best_of_n", n_paths=4))
```

`OpenAICompatThinker` tolerates imperfect servers: ones that ignore `n` (falls back to
sequential sampling), report `finish_reason="stop"` on truncated output (repaired from
`usage`), return HTTP 200 with an `{"error": …}` body, or drop the connection (retried).
If your server cannot continue a partial assistant message, use `mode="best_of_n"` or
`prefix_mode="prompt"`.

The judge is likewise pluggable: any object with `score(problem, candidates) -> Verdict`
(and `choose` for the comparative strategy) works. The one rule: **the judge must not be
the thinker** — self-grading measured worse than no judge.

## Evaluate

```bash
metacog-eval humaneval --thinker-url http://localhost:8000 --thinker-model X --judge none        # baseline
metacog-eval humaneval --thinker-url http://localhost:8000 --thinker-model X --judge reflex --mode best_of_n --n 6
```

The harness runs model-generated code — use a container or VM. Always report the coverage
ceiling alongside accuracy and never let the thinker grade itself.

## Judges

Anything that speaks `POST /v1/systemone` (TypeSafe's schema):

- **Jev** — `SystemOneJudge.jev()`, hosted, `TYPESAFE_API_KEY` in the environment.
- **Reflex** — `uv run reflex-serve --model Qwen/Qwen3.5-4B --port 8008`, then
  `SystemOneJudge.reflex()`. MIT, Apache-2.0 weights, ~8 GB GPU.

## Develop

```bash
pip install -e ".[dev,eval]"
ruff check . && ruff format --check . && pytest -q
```

## License

MIT. Reflex is MIT (Kshetrajna Raghavan); Jev is a TypeSafe product. Not affiliated with
either.
