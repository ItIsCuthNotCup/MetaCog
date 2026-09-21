"""Thinker backends: any OpenAI-compatible HTTP endpoint, or a local HF model."""

from __future__ import annotations

import time
from typing import Literal, Protocol

import httpx

from .types import Generation


class ThinkerError(Exception):
    """Raised when a thinker endpoint returns an HTTP error."""


PROMPT_PREFIX_TEMPLATE = (
    "{problem}\n\nContinue the following partial solution exactly from where it stops. "
    "Output only the continuation, do not repeat it.\n\n{prefix}"
)


class Thinker(Protocol):
    """The base model that produces reasoning paths."""

    def generate(
        self,
        problem: str,
        prefix: str,
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None = None,
    ) -> list[Generation]: ...


RETRY_STATUSES = {408, 409, 429, 500, 502, 503, 504, 522, 524}


class OpenAICompatThinker:
    """Thinker backed by an OpenAI-compatible server (vLLM, llama.cpp, Ollama, OpenAI).

    Two API shapes:

    * ``api="completions"`` -- ``POST {base_url}/v1/completions``. The prefix is simply
      concatenated onto the rendered prompt. Works on every server; prefer this when the
      server has no chat-prefill support.
    * ``api="chat"`` -- ``POST {base_url}/v1/chat/completions``. How a non-empty
      ``prefix`` is presented depends on ``prefix_mode``:
      ``"assistant"`` (default) appends a trailing ``assistant`` message and asks the
      server to continue it via the vLLM convention
      ``{"continue_final_message": true, "add_generation_prompt": false}``.
      ``"prompt"`` instead folds the prefix into the user turn
      (``PROMPT_PREFIX_TEMPLATE``) for servers that ignore or mishandle prefill.
      Servers without prefill support should use ``prefix_mode="prompt"`` or stick to
      ``mode="best_of_n"`` (which never sends a prefix).

    ``base_url`` may or may not end in ``/v1`` (``https://api.openai.com/v1`` and
    ``http://localhost:8000`` both work). Hosted gateways that need a key take
    ``api_key`` (sent as a bearer token).

    Quirks tolerated automatically: servers that silently ignore ``n`` (detected once,
    then requests go sequential), ``finish_reason="stop"`` on token-truncated output
    (repaired from ``usage``), HTTP 200 bodies carrying an ``{"error": ...}``, and
    transient 429/5xx responses (retried with backoff). Reasoning models' side-channel
    chain of thought (``reasoning_content`` / ``reasoning``) is folded into the
    candidate text inside ``<think>`` tags so the judge can read it
    (``include_reasoning=False`` to disable).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        api: Literal["chat", "completions"] = "chat",
        system_prompt: str | None = None,
        timeout: float = 600.0,
        extra_body: dict | None = None,
        prefix_mode: Literal["assistant", "prompt"] = "assistant",
        supports_n: bool = True,
        max_retries: int = 2,
        include_reasoning: bool = True,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.model = model
        self.api_key = api_key
        self.api = api
        self.system_prompt = system_prompt
        self.extra_body = extra_body or {}
        self.prefix_mode = prefix_mode
        self.supports_n = supports_n
        self.max_retries = max_retries
        self.include_reasoning = include_reasoning
        self._client = client or httpx.Client(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, path: str, body: dict) -> dict:
        resp = None
        # Retry transient transport failures (disconnects, read timeouts, refused
        # connections) with a short backoff: 1s, 2s, ... up to max_retries retries.
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post(
                    f"{self.base_url}{path}", json=body, headers=self._headers()
                )
                if resp.status_code in RETRY_STATUSES and attempt < self.max_retries:
                    time.sleep(2 * (attempt + 1))
                    continue
                break
            except httpx.TransportError as e:
                if attempt >= self.max_retries:
                    raise ThinkerError(f"{path} -> transport error: {e}") from e
                time.sleep(attempt + 1)
        if resp.status_code >= 400:
            raise ThinkerError(f"{path} -> HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        # Some servers return HTTP 200 with an {"error": ...} body instead of a status code.
        if isinstance(data, dict) and "error" in data and "choices" not in data:
            raise ThinkerError(f"{path} -> error response: {data['error']}")
        return data

    def _render(self, problem: str) -> str:
        if self.system_prompt:
            return f"{self.system_prompt}\n\n{problem}\n\n"
        return f"{problem}\n\n"

    def _to_generations(
        self,
        data: dict,
        chat: bool,
        max_tokens: int,
        stop: list[str] | None,
    ) -> list[Generation]:
        choices = data.get("choices", [])
        total = (data.get("usage") or {}).get("completion_tokens")
        # Only an aggregate is reported; split it evenly across the returned choices.
        per_choice = total // len(choices) if total is not None and choices else 0
        gens = []
        for choice in choices:
            if chat:
                msg = choice.get("message") or {}
                text = msg.get("content") or ""
                # Reasoning models (DeepSeek, Qwen, GLM, ...) return their chain of
                # thought in a side field; fold it in so the judge sees the reasoning
                # and a token-truncated response is not an empty string.
                reasoning = msg.get("reasoning_content") or msg.get("reasoning")
                if reasoning and self.include_reasoning:
                    text = f"<think>\n{reasoning}\n</think>\n{text}"
            else:
                text = choice.get("text") or ""
            finished = choice.get("finish_reason") == "stop"
            # Quirky servers say "stop" even when output hit max_tokens; treat a
            # per-choice token count >= max_tokens as truncated when usage is known.
            if total is not None and per_choice >= max_tokens:
                finished = False
            for s in stop or []:
                if s in text:
                    text = text[: text.index(s)]
                    finished = True
            gens.append(Generation(text=text, finished=finished, tokens=per_choice))
        return gens

    def _fill(
        self,
        path: str,
        make_body,
        *,
        n: int,
        max_tokens: int,
        chat: bool,
        stop: list[str] | None,
    ) -> list[Generation]:
        """Request n generations, compensating for servers that ignore ``n`` or
        reject ``n > 1`` (400): once seen, flip ``supports_n`` off permanently and
        go sequential."""
        gens: list[Generation] = []
        while len(gens) < n:
            want = (n - len(gens)) if self.supports_n else 1
            try:
                data = self._post(path, make_body(want))
            except ThinkerError as e:
                if want > 1 and "HTTP 400" in str(e):
                    self.supports_n = False
                    continue
                raise
            new = self._to_generations(data, chat=chat, max_tokens=max_tokens, stop=stop)
            if want > 1 and len(new) < want:
                # Server silently returned fewer choices than requested.
                self.supports_n = False
            gens.extend(new)
            if not new:
                break  # avoid an infinite loop on empty responses
        return gens

    def _generate_completions(
        self,
        problem: str,
        prefix: str,
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
    ) -> list[Generation]:
        body: dict = {
            "model": self.model,
            "prompt": self._render(problem) + prefix,
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **self.extra_body,
        }
        if stop:
            body["stop"] = stop

        def make_body(want: int) -> dict:
            return {**body, "n": want}

        return self._fill(
            "/v1/completions",
            make_body,
            n=n,
            max_tokens=max_tokens,
            chat=False,
            stop=stop,
        )

    def _chat_body(
        self,
        problem: str,
        prefix: str,
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
    ) -> dict:
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        user_content = problem
        if prefix and self.prefix_mode == "prompt":
            user_content = PROMPT_PREFIX_TEMPLATE.format(problem=problem, prefix=prefix)
        messages.append({"role": "user", "content": user_content})
        extra = dict(self.extra_body)
        if prefix and self.prefix_mode == "assistant":
            messages.append({"role": "assistant", "content": prefix})
            extra = {
                "continue_final_message": True,
                "add_generation_prompt": False,
                **extra,
            }
        body: dict = {
            "model": self.model,
            "messages": messages,
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **extra,
        }
        if stop:
            body["stop"] = stop
        return body

    def _generate_chat(
        self,
        problem: str,
        prefix: str,
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
    ) -> list[Generation]:
        def make_body(want: int) -> dict:
            return self._chat_body(
                problem,
                prefix,
                n=want,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )

        return self._fill(
            "/v1/chat/completions",
            make_body,
            n=n,
            max_tokens=max_tokens,
            chat=True,
            stop=stop,
        )

    def generate(
        self,
        problem: str,
        prefix: str,
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None = None,
    ) -> list[Generation]:
        if self.api == "completions":
            return self._generate_completions(
                problem,
                prefix,
                n=n,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
        return self._generate_chat(
            problem,
            prefix,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )


class TransformersThinker:
    """Thinker backed by a local Hugging Face model.

    ``transformers`` and ``torch`` are imported lazily in ``__init__`` so the package
    remains importable without them (install the ``hf`` extra to use this class).
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        dtype: str | None = None,
        chat_template: bool = True,
    ) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=getattr(torch, dtype) if dtype else "auto",
            device_map=device,
        )
        self.chat_template = (
            chat_template and getattr(self.tokenizer, "chat_template", None) is not None
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _render(self, problem: str, prefix: str) -> str:
        if self.chat_template:
            messages = [{"role": "user", "content": problem}]
            if prefix:
                messages.append({"role": "assistant", "content": prefix})
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    continue_final_message=True,
                    add_generation_prompt=False,
                )
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"{problem}\n\n{prefix}"

    def generate(
        self,
        problem: str,
        prefix: str,
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None = None,
    ) -> list[Generation]:
        prompt = self._render(problem, prefix)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[-1]
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            num_return_sequences=n,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        eos_ids = {self.tokenizer.eos_token_id}
        gens: list[Generation] = []
        for seq in out:
            new_tokens = seq[prompt_len:]
            finished = any(t.item() in eos_ids for t in new_tokens)
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            for s in stop or []:
                if s in text:
                    text = text[: text.index(s)]
                    finished = True
            gens.append(Generation(text=text, finished=finished, tokens=len(new_tokens)))
        return gens
