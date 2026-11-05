"""A dependency-free BM25 retriever so the benchmark runs anywhere."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, List, Sequence, Tuple

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.tokens]
        self.avg_len = (sum(self.doc_len) / len(docs)) if docs else 0.0
        self.tf = [Counter(t) for t in self.tokens]
        df: Counter = Counter()
        for t in self.tokens:
            df.update(set(t))
        n = len(docs)
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def score(self, query_tokens: Sequence[str], index: int) -> float:
        total = 0.0
        tf, dl = self.tf[index], self.doc_len[index]
        for term in query_tokens:
            if term not in tf:
                continue
            f = tf[term]
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1.0))
            total += self.idf[term] * f * (self.k1 + 1) / denom
        return total

    def top_k(self, query: str, k: int) -> List[Tuple[int, float]]:
        q = tokenize(query)
        scored = [(i, self.score(q, i)) for i in range(len(self.tokens))]
        # Stable, deterministic tie-break on index.
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]


# --------------------------------------------------------------------------- #
# Dense and hybrid retrieval (optional)                                        #
# --------------------------------------------------------------------------- #

RETRIEVERS = ("bm25", "dense", "hybrid")
DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60  # standard reciprocal-rank-fusion constant

Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def load_embedder(name: str = DEFAULT_EMBEDDER):
    """Return ``embed(texts) -> list of vectors`` backed by sentence-transformers.

    Needs the optional extra: ``pip install "chunk-bench[dense]"``. The model
    weights are downloaded on first use, so this requires network access.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            'dense retrieval needs sentence-transformers: pip install "chunk-bench[dense]"'
        ) from exc
    model = SentenceTransformer(name)

    def embed(texts: Sequence[str]):
        return model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False).tolist()

    return embed


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class DenseIndex:
    """Cosine-similarity retrieval over embeddings (vectors are L2-normalised here)."""

    def __init__(self, docs: Sequence[str], embed):
        self.embed = embed
        self.vectors = [self._unit(v) for v in embed(list(docs))]

    @staticmethod
    def _unit(vec: Sequence[float]) -> List[float]:
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def top_k(self, query: str, k: int) -> List[Tuple[int, float]]:
        q = self._unit(self.embed([query])[0])
        scored = [(i, _dot(q, v)) for i, v in enumerate(self.vectors)]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]


class HybridIndex:
    """BM25 + dense fused with reciprocal rank fusion (score = sum 1 / (60 + rank))."""

    def __init__(self, docs: Sequence[str], embed):
        self.bm25 = BM25(docs)
        self.dense = DenseIndex(docs, embed)
        self.n = len(docs)

    def top_k(self, query: str, k: int) -> List[Tuple[int, float]]:
        fused: Counter = Counter()
        for index in (self.bm25, self.dense):
            for rank, (i, _) in enumerate(index.top_k(query, self.n), 1):
                fused[i] += 1.0 / (RRF_K + rank)
        ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:k]


def build_retriever(name: str, docs: Sequence[str], embed=None):
    """Build a retriever by name. ``dense``/``hybrid`` need an ``embed`` callable;
    when omitted, the sentence-transformers default is loaded lazily."""
    if name == "bm25":
        return BM25(docs)
    if name not in RETRIEVERS:
        raise ValueError(f"unknown retriever {name!r}; choose from {', '.join(RETRIEVERS)}")
    if embed is None:
        embed = load_embedder()
    return DenseIndex(docs, embed) if name == "dense" else HybridIndex(docs, embed)
