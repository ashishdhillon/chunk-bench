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
| `sdca_lite` | markdown re-implementation of the *ideas* behind structure-aware chunking (see below): hard section boundaries, whole paragraphs, heading breadcrumb counted in the budget, short-tail merge |

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
| `docker-docs` | 4 real Docker documentation pages (Compose networking, networking overview, volumes, build best practices), 114 questions with verbatim evidence spans. Apache-2.0, see [NOTICE](NOTICE). |
| `k8s-docs` | 5 real Kubernetes concept pages (ConfigMaps, Pods, Deployments, Services, Persistent Volumes), 113 questions with verbatim evidence spans. Source and license: see [NOTICE](NOTICE). |

## First results (`k8s-docs`, BM25)

```bash
chunk-bench run --dataset k8s-docs --max-words 40,80,160 --bootstrap 2000 --baseline fixed --format md
chunk-bench run --dataset k8s-docs --max-words 40,80,160 --bootstrap 2000 --baseline fixed --cluster   # resample documents
```

| max_words | strategy | chunks | avg_words | hit@1 | hit@3 | mrr | d_hit@3 |
|---|---|---|---|---|---|---|---|
| 40 | fixed | 695 | 39.9 | 0.522 [0.425, 0.619] | 0.708 [0.619, 0.788] | 0.619 [0.539, 0.700] | +0.000 [+0.000, +0.000] |
| 40 | recursive | 1078 | 20.6 | 0.584 [0.496, 0.673] | 0.743 [0.664, 0.823] | 0.677 [0.601, 0.749] | +0.035 [-0.062, +0.124] |
| 40 | sdca_lite | 992 | 31.7 | 0.602 [0.513, 0.690] | 0.717 [0.628, 0.796] | 0.669 [0.587, 0.745] | +0.009 [-0.080, +0.097] |
| 40 | sentence | 643 | 34.5 | 0.513 [0.416, 0.611] | 0.655 [0.566, 0.743] | 0.594 [0.512, 0.676] | -0.053 [-0.133, +0.027] |
| 40 | structure | 932 | 33.4 | 0.504 [0.407, 0.593] | 0.690 [0.602, 0.770] | 0.590 [0.507, 0.671] | -0.018 [-0.124, +0.088] |
| 80 | fixed | 349 | 79.4 | 0.664 [0.575, 0.752] | 0.841 [0.770, 0.903] | 0.753 [0.687, 0.820] | +0.000 [+0.000, +0.000] |
| 80 | recursive | 860 | 25.8 | 0.611 [0.522, 0.699] | 0.823 [0.752, 0.894] | 0.722 [0.650, 0.791] | -0.018 [-0.080, +0.044] |
| 80 | sdca_lite | 449 | 57.6 | 0.673 [0.584, 0.761] | 0.876 [0.814, 0.929] | 0.776 [0.712, 0.838] | +0.035 [-0.018, +0.088] |
| 80 | sentence | 298 | 74.5 | 0.655 [0.566, 0.743] | 0.788 [0.717, 0.858] | 0.726 [0.655, 0.797] | -0.053 [-0.106, -0.009]* |
| 80 | structure | 405 | 62.9 | 0.611 [0.522, 0.708] | 0.850 [0.788, 0.912] | 0.722 [0.652, 0.791] | +0.009 [-0.044, +0.062] |
| 160 | fixed | 175 | 158.0 | 0.717 [0.628, 0.796] | 0.858 [0.788, 0.920] | 0.785 [0.716, 0.850] | +0.000 [+0.000, +0.000] |
| 160 | recursive | 823 | 27.0 | 0.619 [0.531, 0.708] | 0.832 [0.761, 0.894] | 0.726 [0.654, 0.795] | -0.027 [-0.097, +0.035] |
| 160 | sdca_lite | 242 | 98.5 | 0.752 [0.673, 0.832] | 0.885 [0.823, 0.938] | 0.830 [0.769, 0.887] | +0.027 [-0.018, +0.080] |
| 160 | sentence | 146 | 152.1 | 0.690 [0.611, 0.779] | 0.876 [0.814, 0.929] | 0.777 [0.713, 0.842] | +0.018 [-0.027, +0.062] |
| 160 | structure | 235 | 101.1 | 0.743 [0.655, 0.823] | 0.885 [0.823, 0.938] | 0.819 [0.756, 0.879] | +0.027 [-0.018, +0.071] |

Brackets are 95% bootstrap intervals (2,000 resamples of the questions, seed 0; add `--cluster` to resample documents). `d_hit@3` is the paired difference against `fixed` on the same resamples; `*` marks an interval that excludes zero.

**What the intervals say.**
- Almost nothing is separable. Of 12 comparisons against `fixed`, only one excludes zero when resampling questions: `sentence` is worse than `fixed` at 80 words (-0.053, [-0.106, -0.009]).
- That one result does not survive `--cluster`, which resamples whole documents because questions from one page are not independent: its interval becomes [-0.113, +0.009]. Under clustering no comparison excludes zero (`sdca_lite` at 160 words touches it: +0.027 [+0.000, +0.052]). With only 5 documents even the clustered intervals are rough.
- `sdca_lite` has the highest hit@3 at 80 words (0.876), and `sdca_lite` and `structure` tie at 0.885 at 160, but their intervals against `fixed` include zero. These are hints, not findings.
- The `structure` prefix is counted inside the word budget, like `sdca_lite`, so chunk sizes are comparable across strategies.
- The one pattern that holds for every strategy is that larger chunks score higher on this dataset. That is partly an artifact of the metric: bigger chunks make it easier for the evidence span and the query keywords to land together, and they say nothing about answer quality or cost.
- The intervals only cover sampling noise on one documentation family with one retriever (BM25). They do not cover other corpora, other retrievers, or other chunk-size settings, and the ranking has already changed as the dataset grew and as the budget accounting changed.

## Second corpus (`docker-docs`, BM25)

```bash
chunk-bench run --dataset docker-docs --max-words 40,80,160 --bootstrap 2000 --baseline fixed --format md
```

| max_words | strategy | chunks | avg_words | hit@1 | hit@3 | mrr | d_hit@3 |
|---|---|---|---|---|---|---|---|
| 40 | fixed | 330 | 39.8 | 0.535 [0.447, 0.623] | 0.728 [0.640, 0.807] | 0.636 [0.559, 0.713] | +0.000 [+0.000, +0.000] |
| 40 | recursive | 521 | 20.2 | 0.561 [0.474, 0.649] | 0.772 [0.693, 0.851] | 0.659 [0.582, 0.734] | +0.044 [-0.035, +0.123] |
| 40 | sdca_lite | 465 | 31.0 | 0.605 [0.509, 0.693] | 0.746 [0.658, 0.825] | 0.680 [0.596, 0.757] | +0.018 [-0.061, +0.096] |
| 40 | sentence | 300 | 35.0 | 0.588 [0.500, 0.675] | 0.719 [0.632, 0.798] | 0.655 [0.575, 0.732] | -0.009 [-0.088, +0.070] |
| 40 | structure | 426 | 33.1 | 0.544 [0.447, 0.632] | 0.675 [0.588, 0.754] | 0.606 [0.521, 0.688] | -0.053 [-0.140, +0.035] |
| 80 | fixed | 165 | 79.3 | 0.658 [0.570, 0.746] | 0.886 [0.825, 0.939] | 0.763 [0.697, 0.825] | +0.000 [+0.000, +0.000] |
| 80 | recursive | 426 | 24.7 | 0.640 [0.553, 0.728] | 0.833 [0.763, 0.904] | 0.749 [0.683, 0.813] | -0.053 [-0.105, +0.000] |
| 80 | sdca_lite | 214 | 56.6 | 0.737 [0.649, 0.816] | 0.877 [0.807, 0.930] | 0.802 [0.735, 0.864] | -0.009 [-0.053, +0.035] |
| 80 | sentence | 142 | 74.0 | 0.684 [0.596, 0.763] | 0.851 [0.781, 0.912] | 0.764 [0.695, 0.829] | -0.035 [-0.096, +0.018] |
| 80 | structure | 196 | 61.1 | 0.658 [0.561, 0.746] | 0.807 [0.728, 0.877] | 0.739 [0.664, 0.805] | -0.079 [-0.140, -0.026]* |
| 160 | fixed | 83 | 157.1 | 0.728 [0.640, 0.807] | 0.912 [0.860, 0.965] | 0.820 [0.760, 0.877] | +0.000 [+0.000, +0.000] |
| 160 | recursive | 420 | 25.0 | 0.623 [0.535, 0.711] | 0.833 [0.763, 0.904] | 0.739 [0.673, 0.806] | -0.079 [-0.140, -0.018]* |
| 160 | sdca_lite | 123 | 91.7 | 0.737 [0.649, 0.816] | 0.886 [0.825, 0.939] | 0.817 [0.754, 0.875] | -0.026 [-0.079, +0.026] |
| 160 | sentence | 70 | 150.2 | 0.728 [0.640, 0.807] | 0.877 [0.816, 0.939] | 0.803 [0.735, 0.864] | -0.035 [-0.088, +0.009] |
| 160 | structure | 119 | 94.4 | 0.711 [0.623, 0.798] | 0.868 [0.798, 0.930] | 0.795 [0.728, 0.855] | -0.044 [-0.105, +0.009] |

**Does the ranking replicate?** Not really.
- On `k8s-docs` the heading-aware strategies were nominally ahead of `fixed`. Here `fixed` has the highest hit@3 at 80 and 160 words, and `structure` is worse at 80 words (-0.079, [-0.140, -0.026]) and `recursive` is worse at 160 (-0.079, [-0.140, -0.018]) when resampling questions.
- `sdca_lite` is within noise of `fixed` at every budget on both corpora.
- The Docker pages are hard-wrapped and have a looser heading structure, so heading-aware chunking has less to exploit. This is a possible explanation, not something these runs test.
- Resampling documents (`--cluster`) with only 4 documents gives degenerate intervals (a few distinct resamples), so it is not shown. The tool warns when you use it with fewer than 10 documents.

The honest reading of the two corpora together is that no chunking strategy dominates, and the choice depends on the document family. Four to five pages per corpus is small, so treat all of this as a demonstration of the method.

### About `sdca_lite`

It is **not** the SDCA algorithm. SDCA (Structure-Aware Document Chunking, from an M.Tech thesis) detects headings from PDF font geometry; plain markdown has no font sizes, so it cannot run here. `sdca_lite` keeps only the parts that carry over to markdown, and exists so that a heading-aware chunker with paragraph integrity is on the table next to the simpler ones. Results for the real algorithm on PDFs will be reported separately.

## Roadmap

- [x] First real dataset (`k8s-docs`, 113 questions over 5 pages)
- [x] Bootstrap confidence intervals and paired differences (`--bootstrap`, `--baseline`)
- [x] Document-clustered bootstrap (`--cluster`)
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
