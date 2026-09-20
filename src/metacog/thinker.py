"""Thinker backends: any OpenAI-compatible HTTP endpoint, or a local HF model."""

from __future__ import annotations

from typing import Literal, Protocol

import httpx

from .types import Generation


class ThinkerError(Exception):
    """Raised when a thinker endpoint returns an HTTP error."""


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


class OpenAICompatThinker:
    """Thinker backed by an OpenAI-compatible server (vLLM, llama.cpp, Ollama, OpenAI).

    Two API shapes:

    * ``api="completions"`` -- ``POST {base_url}/v1/completions``. The prefix is simply
      concatenated onto the rendered prompt. Works on every server; prefer this when the
      server has no chat-prefill support.
    * ``api="chat"`` -- ``POST {base_url}/v1/chat/completions``. A non-empty ``prefix`` is
      sent as a trailing ``assistant`` message and the server is asked to continue it via
      the vLLM convention ``{"continue_final_message": true, "add_generation_prompt": false}``
      (merged into ``extra_body``). Servers that reject trailing-assistant messages
      (e.g. stock llama.cpp) should use ``api="completions"`` instead.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        api: Literal["chat", "completions"] = "chat",
        system_prompt: str | None = None,
        timeout: float = 120.0,
        extra_body: dict | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.api = api
        self.system_prompt = system_prompt
        self.extra_body = extra_body or {}
        self._client = client or httpx.Client(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, path: str, body: dict) -> dict:
        resp = self._client.post(f"{self.base_url}{path}", json=body, headers=self._headers())
        if resp.status_code >= 400:
            raise ThinkerError(f"{path} -> HTTP {resp.status_code}: {resp.text}")
        return resp.json()

    def _render(self, problem: str) -> str:
        if self.system_prompt:
            return f"{self.system_prompt}\n\n{problem}\n\n"
        return f"{problem}\n\n"

    def _usage_tokens(self, usage: dict | None, n: int) -> int:
        total = (usage or {}).get("completion_tokens", 0) or 0
        # When only the aggregate is reported, split it evenly across the n samples.
        return total // n if n else total

    def _to_generations(self, data: dict, chat: bool, n: int) -> list[Generation]:
        tokens_each = self._usage_tokens(data.get("usage"), n)
        gens = []
        for choice in data.get("choices", []):
            if chat:
                text = (choice.get("message") or {}).get("content") or ""
            else:
                text = choice.get("text") or ""
            gens.append(
                Generation(
                    text=text,
                    finished=choice.get("finish_reason") == "stop",
                    tokens=tokens_each,
                )
            )
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
        return self._to_generations(self._post("/v1/completions", body), chat=False, n=n)

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
        messages.append({"role": "user", "content": problem})
        extra = dict(self.extra_body)
        if prefix:
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
        try:
            data = self._post(
                "/v1/chat/completions",
                self._chat_body(
                    problem,
                    prefix,
                    n=n,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                ),
            )
            return self._to_generations(data, chat=True, n=n)
        except ThinkerError as e:
            # Some servers reject n>1 on chat; fall back to n sequential requests.
            if n <= 1 or "HTTP 400" not in str(e):
                raise
        gens: list[Generation] = []
        for _ in range(n):
            data = self._post(
                "/v1/chat/completions",
                self._chat_body(
                    problem,
                    prefix,
                    n=1,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                ),
            )
            gens.extend(self._to_generations(data, chat=True, n=1))
        return gens

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
