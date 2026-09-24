# MetaCog

**Let a small, fast judge steer a bigger model's reasoning.**

MetaCog wraps any language model — local or API — in an inference-time
*metacognition* loop. The model writes one answer; a separate judge scores it.
If the judge is confident, that is the answer. If not, the model writes several
more **thought paths** in parallel, the judge scores each path and each distinct
final answer, and the best one is returned. No training, no weights touched.

```
problem ──▶ greedy thought path ──▶ judge confident? ──yes──▶ answer
                                          │ no
                                          ▼
                       2–6 more thought paths (sized by uncertainty), in parallel
                                          │
                       judge scores every path + every distinct final answer
                                          │
                                          ▼
                                    best path ──▶ answer
```

**Measured:** on 180 paired GPQA Diamond + AIME rows across three hosted
thinkers, the thinker alone scores 76.7 %, MetaCog **86.7 %** (+11 fixed / −2 lost
vs the previous MetaCog default, p = 0.02, 0.89× wall time, 1.39× thinker tokens).
Pooled over 856 rows on four benchmarks, the earlier configuration lifted
75.0 % → 79.4 %. Full tables, ledger of every idea tried and charts:
[docs/RESULTS.md](docs/RESULTS.md).

Looking for the agent? **[Heuristic](https://github.com/ItIsCuthNotCup/heuristic)**
is an async coding-agent harness (fork of Unreal Agent) with MetaCog built in.

## Install

```bash
pip install -e .              # core: httpx + pydantic only
pip install -e ".[local]"     # + local logprob judge (llama-cpp)
pip install -e ".[eval]"      # + HumanEval harness
pip install -e ".[hf]"        # + in-process transformers thinker
```

## Quick start

```python
from metacog import MetaCog, Config, OpenAICompatThinker, SystemOneJudge

thinker = OpenAICompatThinker("http://localhost:8000", model="openbmb/MiniCPM5-2B")
judge = SystemOneJudge.jev()  # hosted; reads $TYPESAFE_API_KEY

mc = MetaCog(thinker, judge, Config())  # defaults = the measured v0.3 pipeline
result = mc.run("How many positive integers below 1000 have digits summing to 9?")

print(result.answer)
for r in result.trace.rounds:
    print(r.step, [f"{p:.2f}" for p in r.verdict.probabilities], "kept", r.kept)
```

Or from the shell:

```bash
metacog solve "…problem…" --thinker-url http://localhost:8000 --thinker-model X --judge jev --mode best_of_n
```

(`metacog solve` currently exposes `best_of_n` and `stepwise`; the adaptive default
is available from Python and `metacog-eval`.)

Works with any OpenAI-compatible thinker (vLLM, llama.cpp, Ollama, LM Studio,
OpenAI, OpenRouter, Together, Groq, …). No judge key? See
[Local judge](#local-judge-no-api-key).

## How it works

`Config()` defaults are the measured v0.3 pipeline (`mode="adaptive"`):

1. **Greedy path.** The thinker answers once.
2. **Stop check.** The judge scores it; a score ≥ `stop_confidence` (0.95) is
   returned immediately. Measured: a Jev score ≥ 0.95 was correct 226/229 times.
3. **Branch.** Otherwise `u = 1 − score` sets how many more thought paths to
   sample, `round(n_min + u·(n_max − n_min))` (defaults 2..6), all in parallel.
4. **Judge.** Every full path is scored independently ("is this correct?" — one
   call per path). Each *distinct* bare final answer is scored too
   (`Final answer: X`, no reasoning) and added with weight `answer_prior` (0.5).
5. **Pick.** Highest combined score wins. Only finished paths can win.

Modes:

| mode | what happens | when |
|---|---|---|
| `adaptive` | above; optional sketch level and multi-round tree | **default** |
| `best_of_n` | sample `n_paths` full answers, judge picks one | cheapest; any chat endpoint |
| `stepwise` | sample `n_paths` continuations of `step_tokens`, judge picks, extend, repeat | tighter steering; needs prefix continuation (vLLM `continue_final_message` or `/v1/completions`) |

Opt-in knobs (all measured; none beat the default significantly — see the
[ledger](docs/RESULTS.md#3-experiment-ledger)):

| knob | effect |
|---|---|
| `triage=0.5` | judge rates the bare problem first; hard ones skip the greedy stage and branch immediately (0.69× time) |
| `sketch_tokens=800`, `expand_max=3` | branches are cheap outlines, judged and pruned before expansion |
| `max_rounds>1` | keep branching from the best path until the judge is confident |
| `diversity_hints=["solve by elimination", …]` | each extra path gets a different approach prompt |
| `escalate_thinker=Thinker` | a stronger model writes one path when still unsure |
| `strategy="choice"` | one comparative judge call instead of one per path (cheaper, less accurate) |
| `answer_prior=None` | disable the bare-answer score (recommended with the local judge) |

## Judges

The one rule: **the judge must not be the thinker** — self-grading measured
worse than no judge at all.

| judge | class | needs |
|---|---|---|
| **Jev** (TypeSafe, hosted) — default, best measured | `SystemOneJudge.jev()` | `TYPESAFE_API_KEY` |
| **Reflex** (MIT, 4B, local) | `SystemOneJudge.reflex("http://localhost:8008")` | `uv run reflex-serve --model Qwen/Qwen3.5-4B --port 8008`, ~8 GB GPU |
| **Local logprob judge** (any open model, no key) | `LogitJudge` | a `.gguf` or any server returning `top_logprobs` |
| your own | any object with `score(problem, candidates) -> Verdict` | — |

### Local judge (no API key)

`LogitJudge` reads P(yes) straight from a frozen open model's next-token logits —
one forward pass per path, no text generated (idea credited to
[SemIf](https://github.com/TheoLeeCJ/SemIf)):

```python
from metacog import LogitJudge

judge = LogitJudge.llama_cpp("Qwen_Qwen3.5-4B-Q4_K_M.gguf")           # in-process
judge = LogitJudge.openai_compat("http://localhost:8000", model="qwen")  # vLLM / llama-server
```

Measured on 127 saved pools: 68/127 correct picks vs Jev 72/127 (p = 0.50 — a
statistical tie, slightly weaker). Slow on CPU (~2 min per pool); use a GPU.

## Bring your own model

MetaCog only needs something that can sample text.

| you have | use |
|---|---|
| any OpenAI-compatible server or API | `OpenAICompatThinker(base_url, model, api_key=…)` |
| a Hugging Face checkpoint in-process | `TransformersThinker("openbmb/MiniCPM5-2B")` |
| anything else | an object with `generate(problem, prefix, *, n, max_tokens, temperature, stop) -> list[Generation]` |

```python
from metacog import Generation


class MyThinker:
    def generate(self, problem, prefix, *, n, max_tokens, temperature, stop=None):
        texts = my_model.sample(problem + prefix, n=n, max_new_tokens=max_tokens)
        return [Generation(text=t, finished=True) for t in texts]
```

`OpenAICompatThinker` tolerates imperfect servers (ignored `n`, wrong
`finish_reason`, HTTP 200 error bodies, dropped connections). If your server
cannot continue a partial assistant message, use `mode="best_of_n"` or
`OpenAICompatThinker(..., prefix_mode="prompt")`.

## Evaluate

```bash
metacog-eval humaneval --thinker-url http://localhost:8000 --thinker-model X --judge none   # baseline
metacog-eval humaneval --thinker-url http://localhost:8000 --thinker-model X --judge jev    # v0.3 default
metacog-eval humaneval --thinker-url http://localhost:8000 --thinker-model X --judge local --judge-model Qwen3.5-4B.gguf
```

Every `Config` knob passes through (`--stop-confidence`, `--n-min/--n-max`,
`--sketch-tokens`, `--triage`, `--answer-prior`, `--mode`, …). The harness runs
model-generated code — use a container. Always report the coverage ceiling next
to accuracy, and never let the thinker grade itself.

Live paired runs and offline replays of saved pools live under `runs/`
(`runs/dashboard.py` serves the results dashboard; `--export` regenerates the
charts in `docs/charts/`).

## Docs

- [docs/RESULTS.md](docs/RESULTS.md) — all numbers, charts, experiment ledger
- [docs/METHOD.md](docs/METHOD.md) — the method, formally, with provenance and limits
- [docs/ROADMAP.md](docs/ROADMAP.md) — what to try next and the test protocol
- [docs/explainer.html](docs/explainer.html) — animated walkthrough of one real problem

## Develop

```bash
pip install -e ".[dev,eval]"
ruff check . && ruff format --check . && pytest -q
```

## License

MIT. Reflex is MIT (Kshetrajna Raghavan); Jev is a TypeSafe product; Unreal Agent
is MIT (Unreal Labs). Not affiliated with any of them.
