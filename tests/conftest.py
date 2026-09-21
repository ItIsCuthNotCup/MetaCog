"""Shared fakes for controller tests. No network, no GPU."""

from __future__ import annotations

import pytest

from metacog.types import Generation, Verdict


class FakeThinker:
    """Scripted thinker: pops one list of generations per generate() call."""

    def __init__(self, script: list[list[Generation]]):
        self.script = list(script)
        self.calls: list[dict] = []

    def generate(self, problem, prefix, *, n, max_tokens, temperature, stop=None):
        self.calls.append(
            {
                "problem": problem,
                "prefix": prefix,
                "n": n,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
            }
        )
        if not self.script:
            return [Generation(text="", finished=False, tokens=0)]
        return self.script.pop(0)


class FakeJudge:
    """Prefers the candidate containing "CORRECT"; records calls."""

    def __init__(self, confidence: float = 0.7, scores: list[list[float]] | None = None):
        self.confidence = confidence
        # optional queue: per-score()-call raw noul values, overriding CORRECT matching
        self.scores = list(scores) if scores else None
        self.calls: list[dict] = []
        self.assess_calls: list[dict] = []

    def choose(self, problem, candidates, *, instructions=None):
        self.calls.append({"problem": problem, "candidates": list(candidates)})
        idx = next(
            (i for i, c in enumerate(candidates) if "CORRECT" in c),
            0,
        )
        probs = [0.0] * len(candidates)
        probs[idx] = 1.0
        return Verdict(probabilities=probs, choice=idx, confidence=self.confidence)

    def score(self, problem, candidates, *, instructions=None):
        self.calls.append({"kind": "score", "problem": problem, "candidates": list(candidates)})
        if self.scores is not None:
            raw = self.scores.pop(0)
        else:
            raw = [1.0 if "CORRECT" in c else 0.1 for c in candidates]
        total = sum(raw)
        choice = max(range(len(raw)), key=lambda i: raw[i])
        return Verdict(
            probabilities=[p / total for p in raw],
            choice=choice,
            confidence=self.confidence,
            raw=raw,
        )

    def assess(self, problem, candidate, *, questions):
        self.assess_calls.append({"problem": problem, "questions": dict(questions)})
        return {k: 0.9 for k in questions}


def gen(text: str, finished: bool = False, tokens: int = 5) -> Generation:
    return Generation(text=text, finished=finished, tokens=tokens)


@pytest.fixture(autouse=True)
def _no_typesafe_env(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
