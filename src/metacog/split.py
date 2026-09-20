"""Split a single generation that wanders through several approaches into paths.

Some models (e.g. MiniCPM) produce "Alternatively, ... / Approach 2: ..." inside one
sample rather than committing. ``split_paths`` cuts those into self-contained paths.
If the text before the first marker is long enough to stand alone it is itself path 1;
otherwise it is a shared preamble prepended to each marker piece (the judge sees the
problem separately anyway). Pure and deterministic.
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
    if not offsets:
        return [text]
    first = text[: offsets[0]].strip()
    pieces = [
        p
        for p in (
            text[a:b].strip() for a, b in zip(offsets, offsets[1:] + [len(text)], strict=True)
        )
        if p
    ]
    if len(first) >= MIN_PIECE_CHARS:
        # The opening segment is itself a complete approach -> path 1, no preamble.
        paths = [first, *pieces]
        preamble = ""
    else:
        paths = pieces
        preamble = first
    if len(paths) < 2 or any(len(p) < MIN_PIECE_CHARS for p in paths):
        return [text]
    return [f"{preamble}\n\n{p}" for p in paths] if preamble else paths
