"""Chunking strategies.

A strategy is a function ``(text: str, max_words: int) -> list[str]``.
Register your own with ``@register_strategy("name")`` and it shows up in the
CLI and the benchmark runner automatically.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List

Strategy = Callable[[str, int], List[str]]

_REGISTRY: Dict[str, Strategy] = {}

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def register_strategy(name: str) -> Callable[[Strategy], Strategy]:
    """Decorator that adds a chunking strategy to the registry."""

    def decorator(fn: Strategy) -> Strategy:
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_strategy(name: str) -> Strategy:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown strategy {name!r}. Available: {', '.join(available_strategies())}"
        ) from None


def available_strategies() -> List[str]:
    return sorted(_REGISTRY)


def _sentences(text: str) -> List[str]:
    parts = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            parts.extend(s for s in _SENTENCE_END.split(line) if s)
    return parts


def _pack(units: List[str], max_words: int) -> List[str]:
    """Greedily pack text units into chunks of at most ``max_words`` words.

    A single unit longer than the budget is split on word boundaries.
    """
    chunks: List[str] = []
    current: List[str] = []
    count = 0
    for unit in units:
        n = len(unit.split())
        if n > max_words:
            if current:
                chunks.append(" ".join(current))
                current, count = [], 0
            words = unit.split()
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i : i + max_words]))
            continue
        if count + n > max_words and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(unit)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


@register_strategy("fixed")
def fixed(text: str, max_words: int) -> List[str]:
    """Sliding window of ``max_words`` words with 20% overlap."""
    words = text.split()
    if not words:
        return []
    step = max(1, max_words - max(1, max_words // 5))
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):
            break
    return chunks


@register_strategy("sentence")
def sentence(text: str, max_words: int) -> List[str]:
    """Pack whole sentences up to the word budget, ignoring document structure."""
    return _pack(_sentences(text), max_words)


@register_strategy("recursive")
def recursive(text: str, max_words: int) -> List[str]:
    """Paragraph-first splitting: never merge across blank lines, then fall back
    to sentences, then words (the usual 'recursive character splitter' idea)."""
    chunks: List[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph.split()) <= max_words:
            chunks.append(" ".join(paragraph.split()))
        else:
            chunks.extend(_pack(_sentences(paragraph), max_words))
    return chunks


@register_strategy("structure")
def structure(text: str, max_words: int) -> List[str]:
    """Heading-aware baseline: split on markdown headings, keep the heading
    path as a prefix of every chunk, and sub-split long sections by sentence.

    This is a simple structure-aware *baseline*, not a research algorithm.
    More advanced structure-aware methods can be added with
    ``@register_strategy``.
    """
    path: List[str] = []
    sections: List[tuple] = []  # (heading_path, body_lines)
    body: List[str] = []

    def flush() -> None:
        if body or path:
            sections.append((" > ".join(path), list(body)))
        body.clear()

    for line in text.splitlines():
        match = _HEADING.match(line.strip())
        if match:
            flush()
            level = len(match.group(1))
            path[:] = path[: level - 1] + [match.group(2).strip()]
        else:
            body.append(line)
    flush()

    chunks: List[str] = []
    for heading, lines in sections:
        sentences = _sentences("\n".join(lines))
        if not sentences:
            continue
        for piece in _pack(sentences, max_words):
            chunks.append(f"{heading}: {piece}" if heading else piece)
    return chunks
