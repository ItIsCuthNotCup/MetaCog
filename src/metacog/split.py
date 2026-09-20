"""Split a single generation that wanders through several approaches into paths.

Some models (e.g. MiniCPM) produce "Alternatively, ... / Approach 2: ..." inside one
sample rather than committing. ``split_paths`` cuts those into self-contained paths,
each prefixed with the shared preamble before the first marker. Pure and deterministic.
"""

from __future__ import annotations

import re

DEFAULT_MARKERS = [
    "Alternatively,",
    "Another approach",
    "Another way",
    "Approach 1",
    "Approach 2",
    "Approach 3",
    "Option 1",
    "Option 2",
    "Path 1",
    "Path 2",
    "Wait, alternatively",
    "Or we could",
    "Let me try a different",
]

_NUMBERED_RE = re.compile(r"^(?:Approach|Path|Option|Method)\s*\d+[:.]", re.IGNORECASE)
MIN_PIECE_CHARS = 40


def _marker_at(line: str, markers: list[str]) -> bool:
    stripped = line.lstrip()
    if _NUMBERED_RE.match(stripped):
        return True
    low = stripped.lower()
    return any(low.startswith(m.lower()) for m in markers)


def _find_marker_offsets(text: str, markers: list[str]) -> list[int]:
    """Offsets where a marker starts a line, or follows sentence-final punctuation."""
    offsets: list[int] = []
    # Line-start markers.
    pos = 0
    for line in text.splitlines(keepends=True):
        if _marker_at(line, markers):
            offsets.append(pos + (len(line) - len(line.lstrip())))
        pos += len(line)
    # Mid-line markers right after a sentence end: ". Alternatively," etc.
    for m in re.finditer(r"[.!?]\s+", text):
        rest = text[m.end() :]
        low = rest.lower()
        if any(low.startswith(mk.lower()) for mk in markers) or _NUMBERED_RE.match(rest):
            offsets.append(m.end())
    return sorted(set(offsets))


def split_paths(text: str, markers: list[str] | None = None) -> list[str]:
    markers = markers if markers is not None else DEFAULT_MARKERS
    offsets = _find_marker_offsets(text, markers)
    if len(offsets) < 2:
        return [text]
    preamble = text[: offsets[0]].rstrip()
    pieces = [text[a:b].strip() for a, b in zip(offsets, offsets[1:] + [len(text)], strict=True)]
    pieces = [p for p in pieces if p]
    if len(pieces) < 2 or any(len(p) < MIN_PIECE_CHARS for p in pieces):
        return [text]
    return [f"{preamble}\n\n{p}" if preamble else p for p in pieces]
