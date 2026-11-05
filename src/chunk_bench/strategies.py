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


def _sections(text: str) -> List[tuple]:
    """Parse markdown into ``(breadcrumb_list, paragraphs)`` sections.

    Paragraphs are runs of non-blank lines; a heading line always starts a new
    section, even without a blank line before it.
    """
    path: List[str] = []
    sections: List[tuple] = []
    paragraphs: List[str] = []
    lines: List[str] = []

    def end_paragraph() -> None:
        if lines:
            paragraphs.append(" ".join(" ".join(lines).split()))
            lines.clear()

    def end_section() -> None:
        end_paragraph()
        if paragraphs:
            sections.append((list(path), list(paragraphs)))
        paragraphs.clear()

    for raw in text.splitlines():
        line = raw.strip()
        match = _HEADING.match(line)
        if match:
            end_section()
            level = len(match.group(1))
            path[:] = path[: level - 1] + [match.group(2).strip()]
        elif not line:
            end_paragraph()
        else:
            lines.append(line)
    end_section()
    return sections


@register_strategy("sdca_lite")
def sdca_lite(text: str, max_words: int) -> List[str]:
    """Markdown re-implementation of the *ideas* behind structure-aware chunking.

    Not the full SDCA algorithm: that one reads PDF font geometry, which plain
    text does not have. This keeps what carries over to markdown:

    1. Section boundaries are hard: chunks never span two sections.
    2. Paragraphs are kept whole; a paragraph is only split (at sentence
       boundaries) when it alone exceeds the budget.
    3. The heading breadcrumb is bound to every chunk of the section, and it
       counts against the word budget (so budgets stay comparable with the
       other strategies).
    4. A short tail chunk is merged into the previous chunk of the same
       section when the result still fits (minimum size = budget / 3).
    """
    min_words = max(1, max_words // 3)
    chunks: List[str] = []
    for crumbs, paragraphs in _sections(text):
        prefix = " > ".join(crumbs)
        if len(prefix.split()) > max_words // 2:
            prefix = crumbs[-1] if crumbs else ""
        if len(prefix.split()) > max_words // 2:
            prefix = ""
        budget = max(1, max_words - len(prefix.split()))

        units: List[str] = []
        for paragraph in paragraphs:
            if len(paragraph.split()) > budget:
                units.extend(_pack(_sentences(paragraph), budget))
            else:
                units.append(paragraph)

        bodies: List[str] = []
        current: List[str] = []
        count = 0
        for unit in units:
            n = len(unit.split())
            if current and count + n > budget:
                bodies.append(" ".join(current))
                current, count = [], 0
            current.append(unit)
            count += n
        if current:
            bodies.append(" ".join(current))

        if len(bodies) > 1 and len(bodies[-1].split()) < min_words:
            if len(bodies[-2].split()) + len(bodies[-1].split()) <= budget:
                bodies[-2] = bodies[-2] + " " + bodies.pop()

        chunks.extend(f"{prefix}: {b}" if prefix else b for b in bodies)
    return chunks
