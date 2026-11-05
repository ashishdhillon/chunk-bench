"""A dependency-free BM25 retriever so the benchmark runs anywhere."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Sequence, Tuple

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
