"""Core data model for metacog traces and results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Generation(BaseModel):
    """One raw sample returned by a thinker."""

    text: str
    finished: bool
    tokens: int = 0


class Verdict(BaseModel):
    """A judge's choice over an ordered list of candidates."""

    probabilities: list[float]
    choice: int
    confidence: float
    raw: list[float] | None = None  # unnormalised scores, when produced by noul scoring


class Candidate(BaseModel):
    """A path considered in one round."""

    text: str
    finished: bool
    # sketch: short outline of an approach (adaptive mode, never a final answer);
    # expand: full solution written by following a kept sketch.
    source: Literal["sample", "split", "greedy", "sketch", "expand"]
    parent: int | None = None


class Round(BaseModel):
    """One generate -> judge -> keep step of the loop."""

    step: int
    prefix: str
    candidates: list[Candidate]
    verdict: Verdict
    kept: list[int]
    assessment: dict[str, float] | None = None


class Trace(BaseModel):
    rounds: list[Round] = Field(default_factory=list)
    thinker_calls: int = 0
    judge_calls: int = 0
    thinker_tokens: int = 0


class Result(BaseModel):
    answer: str
    finished: bool
    trace: Trace
