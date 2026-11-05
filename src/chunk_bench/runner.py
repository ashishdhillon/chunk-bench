"""Benchmark runner: chunk -> index -> retrieve -> score (+ optional bootstrap CIs)."""

from __future__ import annotations

import random
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence

from .datasets import normalize
from .retrieval import BM25
from .strategies import get_strategy

MAX_RANK = 5

Stat = Callable[[Optional[int]], float]


def _hit(k: int) -> Stat:
    return lambda first: 1.0 if first is not None and first <= k else 0.0


def _rr(first: Optional[int]) -> float:
    return 1.0 / first if first is not None and first <= MAX_RANK else 0.0


def _mean(firsts: Sequence[Optional[int]], idx: Sequence[int], stat: Stat) -> float:
    return sum(stat(firsts[i]) for i in idx) / len(idx)


def _interval(values: List[float]) -> List[float]:
    values = sorted(values)
    lo = values[int(0.025 * (len(values) - 1))]
    hi = values[int(round(0.975 * (len(values) - 1)))]
    return [round(lo, 3), round(hi, 3)]


def run_benchmark(
    dataset: Dict[str, Any],
    strategies: Sequence[str],
    max_words: int = 40,
    ks: Sequence[int] = (1, 3),
    bootstrap: int = 0,
    seed: int = 0,
    baseline: Optional[str] = None,
    cluster: bool = False,
) -> List[Dict[str, Any]]:
    """Return one result row per strategy.

    Metrics:
      * ``hit@k``: share of questions whose evidence span is fully contained in
        a single chunk among the top-k retrieved. Evidence split across chunk
        boundaries therefore counts as a miss, which is the failure mode that
        bad chunking causes.
      * ``mrr``: mean reciprocal rank of the first evidence-bearing chunk
        within the top 5.

    With ``bootstrap=N`` each metric also gets a ``<metric>_ci`` entry: a 95%
    percentile interval from N resamples of the questions. All strategies share
    the same resamples, so with ``baseline`` set, ``d_hit@<k>`` (k = largest) is
    a *paired* difference against the baseline strategy with its own interval.
    With ``cluster=True`` the resampling unit is the *document*: each resample
    draws documents with replacement and keeps all of their questions, which
    respects the dependence between questions from one document. With few
    documents these intervals are wide, and that is the honest answer.
    The intervals only capture sampling noise for this one dataset.
    """
    if baseline is not None and baseline not in strategies:
        raise ValueError(f"baseline {baseline!r} must be one of the benchmarked strategies")

    n_q = len(dataset["questions"])
    top = max(max(ks), MAX_RANK)

    firsts_by_strategy: Dict[str, List[Optional[int]]] = {}
    stats_rows: Dict[str, Dict[str, Any]] = {}
    for name in strategies:
        strategy = get_strategy(name)
        chunks: List[str] = []
        for doc in dataset["documents"]:
            chunks.extend(strategy(doc["text"], max_words))
        index = BM25(chunks)
        norm_chunks = [normalize(c) for c in chunks]

        firsts: List[Optional[int]] = []
        for q in dataset["questions"]:
            evidence = normalize(q["evidence"])
            ranked = [i for i, _ in index.top_k(q["question"], top)]
            firsts.append(
                next((r for r, i in enumerate(ranked, 1) if evidence in norm_chunks[i]), None)
            )
        firsts_by_strategy[name] = firsts
        lengths = [len(c.split()) for c in chunks]
        stats_rows[name] = {
            "strategy": name,
            "chunks": len(chunks),
            "avg_words": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        }

    metrics: Dict[str, Stat] = {f"hit@{k}": _hit(k) for k in ks}
    metrics["mrr"] = _rr
    all_idx = list(range(n_q))

    rng = random.Random(seed)
    if cluster:
        by_doc: Dict[Any, List[int]] = {}
        for i, q in enumerate(dataset["questions"]):
            by_doc.setdefault(q["doc_id"], []).append(i)
        groups = [by_doc[d] for d in sorted(by_doc, key=str)]
        if len(groups) < 10:
            warnings.warn(
                f"--cluster with only {len(groups)} documents: there are very few distinct "
                "resamples, so intervals can be spuriously narrow or wide. Treat them as rough.",
                stacklevel=2,
            )
        resamples = [
            [i for g in (rng.choice(groups) for _ in groups) for i in g]
            for _ in range(bootstrap)
        ]
    else:
        resamples = [[rng.randrange(n_q) for _ in range(n_q)] for _ in range(bootstrap)]

    rows: List[Dict[str, Any]] = []
    for name in strategies:
        firsts = firsts_by_strategy[name]
        row = dict(stats_rows[name])
        for metric, stat in metrics.items():
            row[metric] = round(_mean(firsts, all_idx, stat), 3)
        if bootstrap:
            for metric, stat in metrics.items():
                row[f"{metric}_ci"] = _interval([_mean(firsts, idx, stat) for idx in resamples])
            if baseline is not None:
                key = f"hit@{max(ks)}"
                stat = metrics[key]
                base = firsts_by_strategy[baseline]
                row[f"d_{key}"] = round(
                    _mean(firsts, all_idx, stat) - _mean(base, all_idx, stat), 3
                )
                row[f"d_{key}_ci"] = _interval(
                    [_mean(firsts, idx, stat) - _mean(base, idx, stat) for idx in resamples]
                )
        rows.append(row)
    return rows
