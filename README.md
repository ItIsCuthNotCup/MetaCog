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

## Measured (HumanEval-164, MiniCPM5-2B thinker, 6 candidates/item, oracle unit tests)

| arm | accuracy |
|---|---|
| thinker's first answer (no judge) | 0.555 |
| thinker judging its own candidates | 0.524 (worse than no judge) |
| **Reflex judge, per-candidate noul** (local, MIT, $0) | **0.689** |
| Reflex judge, one-shot choice | 0.671 |
| Jev judge, one-shot choice | 0.726 |
| **Jev judge, per-candidate noul** | **0.756** |
| ceiling: any of 6 candidates passes | 0.793 |

One task family, n=164; the Reflex–Jev gap is within noise (McNemar p=0.11). Reflex
rescued 36 % of the items where the first answer failed. Details, provenance and limits in
[docs/METHOD.md](docs/METHOD.md).

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
