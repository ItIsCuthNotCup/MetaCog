"""Extract a candidate's final answer for the answer-prior signal."""

from __future__ import annotations

import re

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER = re.compile(
    r"(?:final answer|answer)\s*(?:is|:)\s*\**\s*\(?\s*([^\n]+?)\s*\)?\s*\**\s*\.?\s*$",
    re.I | re.M,
)


def extract_answer(text: str) -> str | None:
    """Last ``\\boxed{...}`` or last ``Answer: ...`` line; None if neither is present."""
    m = _BOXED.findall(text) or _ANSWER.findall(text)
    if not m:
        return None
    answer = m[-1].strip().strip("$").strip()
    return answer or None
