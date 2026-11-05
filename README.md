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
chunk-bench run --dataset k8s-docs --max-words 40,80,160 --format md
```

| max_words | strategy | chunks | avg_words | hit@1 | hit@3 | mrr |
|---|---|---|---|---|---|---|
| 40 | fixed | 695 | 39.9 | 0.522 | 0.708 | 0.619 |
| 40 | recursive | 1078 | 20.6 | 0.584 | 0.743 | 0.677 |
| 40 | sentence | 643 | 34.5 | 0.513 | 0.655 | 0.594 |
| 40 | structure | 686 | 40.9 | 0.566 | 0.743 | 0.653 |
| 80 | fixed | 349 | 79.4 | 0.664 | 0.841 | 0.753 |
| 80 | recursive | 860 | 25.8 | 0.611 | 0.823 | 0.722 |
| 80 | sentence | 298 | 74.5 | 0.655 | 0.788 | 0.726 |
| 80 | structure | 362 | 69.0 | 0.69 | 0.867 | 0.785 |
| 160 | fixed | 175 | 158.0 | 0.717 | 0.858 | 0.785 |
| 160 | recursive | 823 | 27.0 | 0.619 | 0.832 | 0.726 |
| 160 | sentence | 146 | 152.1 | 0.69 | 0.876 | 0.777 |
| 160 | structure | 226 | 104.7 | 0.761 | 0.903 | 0.833 |

**Read these with care.** 113 questions means one question is about 0.9 points of hit rate, and the dataset is one documentation family, so this says little about other corpora. The ranking has been unstable as the dataset grew (it changed from 2 to 3 to 5 pages), which is a warning against reading much into any single table. In this run the heading-aware `structure` baseline is ahead at 80 and 160 words (hit@3 0.867 and 0.903 against 0.841 and 0.858 for `fixed`), `recursive` is ahead at 40 words, and all strategies improve with larger chunks. I have not tested why; one untested guess is that the heading prefix adds useful keywords for BM25. Bootstrap confidence intervals and more corpora are the next step before any claim is made.

## Roadmap

- [x] First real dataset (`k8s-docs`, 113 questions over 5 pages)
- [ ] Bootstrap confidence intervals on every metric
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
