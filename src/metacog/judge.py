"""System-One judge client.

Speaks the `POST /v1/systemone` protocol shared by TypeSafe's Jev
(https://api.typesafe.ai) and a local reflex server (http://localhost:8008).
"""

from __future__ import annotations

import os
import string
from typing import Protocol

import httpx

from .types import Verdict

DEFAULT_CHOOSE_INSTRUCTIONS = (
    "Several candidate reasoning paths for the same problem are shown under `paths`. "
    "Which path is most likely to lead to a correct, complete final answer if it is continued "
    "(or, if already finished, is most likely correct)? Judge the soundness of the reasoning "
    "and how directly it addresses the problem, not its length or style."
)

DEFAULT_SCORE_INSTRUCTIONS = (
    "`path` is one candidate solution or reasoning path for `problem`. Is its final answer "
    "correct (or, if it is unfinished, is it on track to reach a correct answer)? Judge the "
    "substance, not the length or style."
)

ANSWER_PRIOR_INSTRUCTIONS = (
    "`path` states only a proposed final answer to `problem`, with no reasoning. "
    "Is that final answer correct?"
)

TRIAGE_INSTRUCTIONS = (
    "`path` is empty. Judge `problem` itself: is this problem EASY enough that a "
    "strong language model would almost certainly answer it correctly on its first "
    "attempt?"
)

FINISHED_QUESTIONS = {
    "is_complete": (
        "Does `path` reach a definite final answer to `problem` (not just a plan or partial work)?"
    ),
    "is_correct": "Is the final answer given in `path` correct for `problem`?",
}

TYPESAFE_BASE_URL = "https://api.typesafe.ai"


def truncate_path(text: str, max_chars: int) -> str:
    """Keep the LAST ``max_chars`` chars; the most recent reasoning matters."""
    if len(text) <= max_chars:
        return text
    return "…" + text[-max_chars:]


class JudgeError(Exception):
    """Raised when the judge endpoint returns an HTTP error."""


class Judge(Protocol):
    """The System-One judge that scores candidate paths."""

    def choose(
        self,
        problem: str,
        candidates: list[str],
        *,
        instructions: str | None = None,
    ) -> Verdict: ...

    def score(
        self,
        problem: str,
        candidates: list[str],
        *,
        instructions: str | None = None,
    ) -> Verdict: ...

    def assess(
        self, problem: str, candidate: str, *, questions: dict[str, str]
    ) -> dict[str, float]: ...


class SystemOneJudge:
    """Client for `POST {base_url}/v1/systemone`.

    ``api_key`` defaults to the ``TYPESAFE_API_KEY`` env var. When ``base_url`` is not
    TypeSafe's and no key is configured, no Authorization header is sent (a local
    reflex server is usually unauthenticated).
    """

    def __init__(
        self,
        base_url: str = TYPESAFE_BASE_URL,
        api_key: str | None = None,
        model: str = "jev-latest",
        permutations: int | None = None,
        timeout: float = 60.0,
        max_chars_per_path: int = 24000,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # api_key defaults to env TYPESAFE_API_KEY, but only for the TypeSafe endpoint:
        # a non-TypeSafe base_url with no explicit key sends no Authorization header
        # (a local reflex server is usually open).
        if api_key is None and self.base_url == TYPESAFE_BASE_URL:
            api_key = os.environ.get("TYPESAFE_API_KEY")
        self.api_key = api_key
        self.model = model
        self.permutations = permutations
        self.max_chars_per_path = max_chars_per_path
        self._client = client or httpx.Client(timeout=timeout)

    @classmethod
    def jev(cls, api_key: str | None = None, model: str = "jev-latest") -> SystemOneJudge:
        return cls(base_url=TYPESAFE_BASE_URL, api_key=api_key, model=model)

    @classmethod
    def reflex(
        cls,
        base_url: str = "http://localhost:8008",
        model: str = "reflex-latest",
        permutations: int = 2,
    ) -> SystemOneJudge:
        return cls(base_url=base_url, model=model, permutations=permutations, api_key=None)

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _truncate(self, text: str) -> str:
        """Keep the LAST ``max_chars_per_path`` chars; the most recent reasoning matters."""
        return truncate_path(text, self.max_chars_per_path)

    def _post(self, body: dict) -> dict:
        last_exc: Exception | None = None
        for _attempt in range(2):  # retry once on 5xx / timeouts
            try:
                resp = self._client.post(
                    f"{self.base_url}/v1/systemone", json=body, headers=self._headers()
                )
            except httpx.TimeoutException as e:
                last_exc = e
                continue
            if resp.status_code >= 500:
                last_exc = JudgeError(f"HTTP {resp.status_code}: {resp.text}")
                continue
            if resp.status_code >= 400:
                raise JudgeError(f"HTTP {resp.status_code}: {resp.text}")
            return resp.json()
        if isinstance(last_exc, JudgeError):
            raise last_exc
        raise JudgeError(f"request failed: {last_exc}")

    def choose(
        self,
        problem: str,
        candidates: list[str],
        *,
        instructions: str | None = None,
    ) -> Verdict:
        if len(candidates) > 26:
            raise ValueError("choose supports at most 26 candidates (labels A..Z)")
        labels = list(string.ascii_uppercase[: len(candidates)])
        body: dict = {
            "model": self.model,
            "state": {
                "problem": problem,
                "paths": {
                    label: self._truncate(text)
                    for label, text in zip(labels, candidates, strict=True)
                },
            },
            "questions": {
                "best_path": {
                    "type": "choice",
                    "instructions": instructions or DEFAULT_CHOOSE_INSTRUCTIONS,
                    "criteria": {label: f"path {label}" for label in labels},
                }
            },
        }
        if self.permutations is not None:
            body["permutations"] = self.permutations
        data = self._post(body)
        answer = data["answers"]["best_path"]
        probs = [answer["probabilities"][label] for label in labels]
        return Verdict(
            probabilities=probs,
            choice=labels.index(answer["choice"]),
            confidence=answer.get("confidence", 0.0),
        )

    def score(
        self,
        problem: str,
        candidates: list[str],
        *,
        instructions: str | None = None,
    ) -> Verdict:
        """Score each candidate in isolation with a noul question (one request per
        candidate, so the judge never sees the other paths). Returns a Verdict whose
        ``probabilities`` are the raw P(correct) values normalised to sum 1 (uniform
        if all zero); ``raw`` holds the unnormalised values."""
        raw: list[float] = []
        for cand in candidates:
            body: dict = {
                "model": self.model,
                "state": {"problem": problem, "path": self._truncate(cand)},
                "questions": {
                    "is_correct": {
                        "type": "noul",
                        "instructions": instructions or DEFAULT_SCORE_INSTRUCTIONS,
                    }
                },
            }
            if self.permutations is not None:
                body["permutations"] = self.permutations
            data = self._post(body)
            raw.append(data["answers"]["is_correct"]["noul"])
        total = sum(raw)
        probs = [p / total for p in raw] if total else [1.0 / len(raw)] * len(raw)
        choice = max(range(len(raw)), key=lambda i: raw[i]) if raw else 0
        return Verdict(
            probabilities=probs,
            choice=choice,
            confidence=max(raw) if raw else 0.0,
            raw=raw,
        )

    def assess(
        self, problem: str, candidate: str, *, questions: dict[str, str]
    ) -> dict[str, float]:
        body = {
            "model": self.model,
            "state": {"problem": problem, "path": self._truncate(candidate)},
            "questions": {
                key: {"type": "noul", "instructions": instr} for key, instr in questions.items()
            },
        }
        if self.permutations is not None:
            body["permutations"] = self.permutations
        data = self._post(body)
        return {key: data["answers"][key]["noul"] for key in questions}
