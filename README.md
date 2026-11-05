# chunk-bench

A small, dependency-free, reproducible benchmark for **RAG chunking strategies**.

Chunking decides what your retriever can ever find. `chunk-bench` measures it directly: chunk a corpus, index it with BM25, retrieve for each question, and check whether the evidence span survives intact in a retrieved chunk.

```bash
pip install -e .
chunk-bench list
chunk-bench run --max-words 40 --format md
```

## Built-in strategies

| name | idea |
|---|---|
| `fixed` | sliding word window, 20% overlap |
| `sentence` | pack whole sentences up to the budget |
| `recursive` | paragraph-first, then sentences, then words |
| `structure` | heading-aware baseline: split on markdown headings, prefix chunks with the heading path |

## Metrics

- **hit@k**: share of questions whose evidence span is fully inside one of the top-k chunks. Evidence cut across a chunk boundary is a miss.
- **mrr**: mean reciprocal rank of the first evidence-bearing chunk (top 5).
- **chunks / avg_words**: index size and chunk granularity.

## Add your own strategy

```python
from chunk_bench import register_strategy, run_benchmark, load_dataset

@register_strategy("my_chunker")
def my_chunker(text: str, max_words: int) -> list[str]:
    ...

print(run_benchmark(load_dataset("sample"), ["my_chunker", "fixed"]))
```

It also appears in `chunk-bench list` and `chunk-bench run`.

## Bring your own dataset

JSON with `documents` (`id`, `text`) and `questions` (`question`, `doc_id`, `evidence`). `evidence` must appear verbatim in the document; the loader validates this. See `src/chunk_bench/data/sample.json`.

## Datasets

| name | contents |
|---|---|
| `sample` | 3 short synthetic docs, 15 questions. A smoke test only: every strategy scores about 1.0. |
| `k8s-docs` | 5 real Kubernetes concept pages (ConfigMaps, Pods, Deployments, Services, Persistent Volumes), 113 questions with verbatim evidence spans. Source and license: see [NOTICE](NOTICE). |

## First results (`k8s-docs`, BM25)

```bash
chunk-bench run --dataset k8s-docs --max-words 40,80,160 --bootstrap 2000 --baseline fixed --format md
```

| max_words | strategy | chunks | avg_words | hit@1 | hit@3 | mrr | d_hit@3 |
|---|---|---|---|---|---|---|---|
| 40 | fixed | 695 | 39.9 | 0.522 [0.425, 0.619] | 0.708 [0.619, 0.788] | 0.619 [0.539, 0.700] | +0.000 [+0.000, +0.000] |
| 40 | recursive | 1078 | 20.6 | 0.584 [0.496, 0.673] | 0.743 [0.664, 0.823] | 0.677 [0.601, 0.749] | +0.035 [-0.062, +0.124] |
| 40 | sentence | 643 | 34.5 | 0.513 [0.416, 0.611] | 0.655 [0.566, 0.743] | 0.594 [0.512, 0.676] | -0.053 [-0.133, +0.027] |
| 40 | structure | 686 | 40.9 | 0.566 [0.478, 0.655] | 0.743 [0.664, 0.823] | 0.653 [0.573, 0.728] | +0.035 [-0.062, +0.124] |
| 80 | fixed | 349 | 79.4 | 0.664 [0.575, 0.752] | 0.841 [0.770, 0.903] | 0.753 [0.687, 0.820] | +0.000 [+0.000, +0.000] |
| 80 | recursive | 860 | 25.8 | 0.611 [0.522, 0.699] | 0.823 [0.752, 0.894] | 0.722 [0.650, 0.791] | -0.018 [-0.080, +0.044] |
| 80 | sentence | 298 | 74.5 | 0.655 [0.566, 0.743] | 0.788 [0.717, 0.858] | 0.726 [0.655, 0.797] | -0.053 [-0.106, -0.009]* |
| 80 | structure | 362 | 69.0 | 0.690 [0.602, 0.770] | 0.867 [0.805, 0.929] | 0.785 [0.720, 0.844] | +0.027 [-0.027, +0.080] |
| 160 | fixed | 175 | 158.0 | 0.717 [0.628, 0.796] | 0.858 [0.788, 0.920] | 0.785 [0.716, 0.850] | +0.000 [+0.000, +0.000] |
| 160 | recursive | 823 | 27.0 | 0.619 [0.531, 0.708] | 0.832 [0.761, 0.894] | 0.726 [0.654, 0.795] | -0.027 [-0.097, +0.035] |
| 160 | sentence | 146 | 152.1 | 0.690 [0.611, 0.779] | 0.876 [0.814, 0.929] | 0.777 [0.713, 0.842] | +0.018 [-0.027, +0.062] |
| 160 | structure | 226 | 104.7 | 0.761 [0.681, 0.841] | 0.903 [0.841, 0.956] | 0.833 [0.774, 0.892] | +0.044 [+0.000, +0.088] |

Brackets are 95% bootstrap intervals (2,000 resamples of the questions, seed 0). `d_hit@3` is the paired difference against `fixed` on the same resamples; `*` marks an interval that excludes zero.

**What the intervals say.**
- Almost nothing is separable. Of 9 comparisons against `fixed`, only one excludes zero: `sentence` is worse than `fixed` at 80 words (-0.053, [-0.106, -0.009]).
- `structure` at 160 words is the best-looking result (hit@3 0.903, +0.044 over `fixed`), but its interval just touches zero ([+0.000, +0.088]), so it is a hint, not a finding.
- The one pattern that holds for every strategy is that larger chunks score higher on this dataset. That is partly an artifact of the metric: bigger chunks make it easier for the evidence span and the query keywords to land together, and they say nothing about answer quality or cost.
- The intervals only cover question-sampling noise on one documentation family with one retriever (BM25). They do not cover other corpora, other retrievers, or other chunk-size settings, and the ranking has already changed as the dataset grew.

## Roadmap

- [x] First real dataset (`k8s-docs`, 113 questions over 5 pages)
- [x] Bootstrap confidence intervals and paired differences (`--bootstrap`, `--baseline`)
- [ ] More and larger structured corpora (more doc pages, HTML, PDF-to-markdown)
- [ ] Embedding-based retrieval backend (optional extra)
- [ ] Answer-level evaluation with an LLM judge (optional extra)
- [ ] Results table and hosted leaderboard
- [ ] PyPI release

## Development

```bash
pip install -e ".[dev]"
pytest
```

MIT licensed.
