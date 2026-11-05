"""Benchmark runner: chunk -> index -> retrieve -> score."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .datasets import normalize
from .retrieval import BM25
from .strategies import get_strategy

MAX_RANK = 5


def run_benchmark(
    dataset: Dict[str, Any],
    strategies: Sequence[str],
    max_words: int = 40,
    ks: Sequence[int] = (1, 3),
) -> List[Dict[str, Any]]:
    """Return one result row per strategy.

    Metrics:
      * ``hit@k``: share of questions whose evidence span is fully contained in
        a single chunk among the top-k retrieved. Evidence split across chunk
        boundaries therefore counts as a miss, which is the failure mode that
        bad chunking causes.
      * ``mrr``: mean reciprocal rank of the first evidence-bearing chunk
        within the top 5.
    """
    rows: List[Dict[str, Any]] = []
    for name in strategies:
        strategy = get_strategy(name)
        chunks: List[str] = []
        for doc in dataset["documents"]:
            chunks.extend(strategy(doc["text"], max_words))
        index = BM25(chunks)
        norm_chunks = [normalize(c) for c in chunks]

        hits = {k: 0 for k in ks}
        rr_total = 0.0
        top = max(max(ks), MAX_RANK)
        for q in dataset["questions"]:
            evidence = normalize(q["evidence"])
            ranked = [i for i, _ in index.top_k(q["question"], top)]
            first = next(
                (r for r, i in enumerate(ranked, 1) if evidence in norm_chunks[i]), None
            )
            for k in ks:
                if first is not None and first <= k:
                    hits[k] += 1
            if first is not None and first <= MAX_RANK:
                rr_total += 1.0 / first

        n_q = len(dataset["questions"])
        lengths = [len(c.split()) for c in chunks]
        row: Dict[str, Any] = {
            "strategy": name,
            "chunks": len(chunks),
            "avg_words": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        }
        for k in ks:
            row[f"hit@{k}"] = round(hits[k] / n_q, 3)
        row["mrr"] = round(rr_total / n_q, 3)
        rows.append(row)
    return rows
