"""Local judge: read P(yes) from a frozen open model's next-token logits.

Idea credited to SemIf (https://github.com/TheoLeeCJ/SemIf): score a statement
by reading P(option) directly from the model's first predicted token — one
forward pass per candidate, no text generation. `LogitJudge` implements the
same `Judge` protocol as `SystemOneJudge`, so MetaCog needs no Jev API key.

Backends:
- `LogitJudge.openai_compat(url, model, ...)`: any OpenAI-compatible server that
  returns `top_logprobs` (vLLM, llama-server).
- `LogitJudge.llama_cpp(path, ...)`: a local GGUF via llama-cpp-python
  (``pip install -e ".[local]"``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import httpx

from .judge import (
    DEFAULT_SCORE_INSTRUCTIONS,
    JudgeError,
    truncate_path,
)
from .types import Verdict

YES_TOKENS = ("yes", " yes", "Yes", " Yes")
NO_TOKENS = ("no", " no", "No", " No")

_SYSTEM = "You are a strict grader. Answer with exactly one word: yes or no."


def yes_probability(logprobs: dict[str, float]) -> float:
    """P(yes) / (P(yes) + P(no)) over the yes/no token variants in top-k
    logprobs; 0.5 when neither family is present."""
    p_yes = sum(math.exp(lp) for t, lp in logprobs.items() if t in YES_TOKENS)
    p_no = sum(math.exp(lp) for t, lp in logprobs.items() if t in NO_TOKENS)
    if p_yes + p_no == 0.0:
        return 0.5
    return p_yes / (p_yes + p_no)


def build_prompt(problem: str, path: str, instructions: str) -> list[dict]:
    """Chat messages for the one-word grading readout (path tail-truncated)."""
    return [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"{instructions}\n\nproblem:\n{problem}\n\npath:\n{path}\n\nAnswer yes or no."
            ),
        },
    ]


class LogitJudge:
    """``Judge`` protocol implementation backed by next-token logprob readouts.

    ``backend`` maps chat messages to a dict of top-k next-token logprobs.
    Backends are sequential; no concurrency inside.
    """

    def __init__(
        self,
        backend: Callable[[list[dict]], dict[str, float]],
        max_chars: int = 24000,
    ) -> None:
        self._backend = backend
        self.max_chars = max_chars

    # -- backends ------------------------------------------------------------

    @classmethod
    def openai_compat(
        cls,
        url: str,
        model: str,
        api_key: str | None = None,
        top_logprobs: int = 20,
        timeout: float = 60.0,
        max_chars: int = 24000,
        client: httpx.Client | None = None,
    ) -> LogitJudge:
        """Any OpenAI-compatible server returning ``top_logprobs`` (vLLM,
        llama-server). NOTE: the MiniCPM Spark server does not return logprobs."""
        base = url.rstrip("/")
        base = base[: -len("/v1")] if base.endswith("/v1") else base
        http = client or httpx.Client(timeout=timeout)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        def backend(messages: list[dict]) -> dict[str, float]:
            resp = http.post(
                f"{base}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 1,
                    "temperature": 0,
                    "logprobs": True,
                    "top_logprobs": top_logprobs,
                },
                headers=headers,
            )
            if resp.status_code >= 400:
                raise JudgeError(f"HTTP {resp.status_code}: {resp.text}")
            data = resp.json()
            try:
                tops = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            except (KeyError, IndexError, TypeError) as e:
                raise JudgeError(
                    "server did not return top_logprobs (logprobs unsupported?)"
                ) from e
            return {t["token"]: t["logprob"] for t in tops}

        return cls(backend, max_chars=max_chars)

    @classmethod
    def llama_cpp(
        cls,
        model_path: str,
        n_ctx: int = 16384,
        n_threads: int | None = None,
        max_chars: int = 24000,
        top_logprobs: int = 20,
        **kw: Any,
    ) -> LogitJudge:
        """Local GGUF via llama-cpp-python (``pip install -e ".[local]"``)."""
        try:
            import llama_cpp
            import numpy as np
        except ImportError as e:
            raise JudgeError("llama-cpp-python is not installed; pip install -e '.[local]'") from e
        try:
            import jinja2
        except ImportError as e:
            raise JudgeError("jinja2 is required for the llama_cpp backend") from e
        llama = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            logits_all=False,
            verbose=False,
            **kw,
        )
        env = jinja2.Environment()
        template = llama.metadata.get("tokenizer.chat_template")
        bos = llama.detokenize([llama.token_bos()]).decode(errors="ignore")

        def render(messages: list[dict]) -> str:
            if not template:
                return (
                    "".join(
                        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
                    )
                    + "<|im_start|>assistant\n"
                )
            return env.from_string(template).render(
                messages=messages, bos_token=bos, add_generation_prompt=True
            )

        # Thinking models emit <think> as the first token no matter the prompt, so
        # yes/no never appears in the readout. Detect that once and switch to a raw
        # completion with the think block already closed; the next token is then
        # literally "yes"/"no".
        thinking = {"chat"}

        def next_token_logprobs(prompt: str) -> dict[str, float]:
            # Only the last position's logits are computed (logits_all=False), so
            # memory stays at n_batch x vocab instead of n_ctx x vocab.
            tokens = llama.tokenize(prompt.encode(), add_bos=False, special=True)
            tokens = tokens[-(n_ctx - 1) :]
            llama.reset()
            llama.eval(tokens)
            logits = np.ctypeslib.as_array(llama._ctx.get_logits(), shape=(llama.n_vocab(),))
            logits = logits.astype(np.float64)
            logprobs = logits - (np.log(np.sum(np.exp(logits - logits.max()))) + logits.max())
            top = np.argpartition(-logprobs, top_logprobs)[:top_logprobs]
            return {
                llama.detokenize([int(i)]).decode(errors="ignore"): float(logprobs[i]) for i in top
            }

        def backend(messages: list[dict]) -> dict[str, float]:
            suffix = "" if "chat" in thinking else "<think></think>\n\n"
            result = next_token_logprobs(render(messages) + suffix)
            top = max(result, key=result.get) if result else None
            if "chat" not in thinking or top in YES_TOKENS + NO_TOKENS:
                return result
            thinking.discard("chat")  # thinking model: use the closed-think readout
            return backend(messages)

        return cls(backend, max_chars=max_chars)

    # -- Judge protocol --------------------------------------------------------

    def _p(self, problem: str, path: str, instructions: str | None, default: str) -> float:
        path = truncate_path(path, self.max_chars)
        return yes_probability(self._backend(build_prompt(problem, path, instructions or default)))

    def score(
        self,
        problem: str,
        candidates: list[str],
        *,
        instructions: str | None = None,
    ) -> Verdict:
        raw = [self._p(problem, c, instructions, DEFAULT_SCORE_INSTRUCTIONS) for c in candidates]
        total = sum(raw)
        probs = [p / total for p in raw] if total else [1.0 / len(raw)] * len(raw)
        choice = max(range(len(raw)), key=lambda i: raw[i]) if raw else 0
        return Verdict(
            probabilities=probs,
            choice=choice,
            confidence=max(raw) if raw else 0.0,
            raw=raw,
        )

    def choose(
        self,
        problem: str,
        candidates: list[str],
        *,
        instructions: str | None = None,
    ) -> Verdict:
        # Isolation scoring, not a listwise prompt: the choose instructions
        # describe a `paths` list this readout never shows, so score() wording is used.
        return self.score(
            problem, candidates, instructions=instructions or DEFAULT_SCORE_INSTRUCTIONS
        )

    def assess(
        self, problem: str, candidate: str, *, questions: dict[str, str]
    ) -> dict[str, float]:
        return {
            key: self._p(problem, candidate, instr, DEFAULT_SCORE_INSTRUCTIONS)
            for key, instr in questions.items()
        }
