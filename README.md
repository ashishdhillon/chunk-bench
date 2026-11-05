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
| `k8s-docs` | 2 real Kubernetes concept pages (ConfigMaps, Pods), 31 questions with verbatim evidence spans. Source and license: see [NOTICE](NOTICE). |

## First results (`k8s-docs`, BM25)

```bash
chunk-bench run --dataset k8s-docs --max-words 40,80,160 --format md
```

| max_words | strategy | chunks | avg_words | hit@1 | hit@3 | mrr |
|---|---|---|---|---|---|---|
| 40 | fixed | 149 | 39.9 | 0.645 | 0.774 | 0.722 |
| 40 | recursive | 215 | 22.2 | 0.742 | 0.774 | 0.791 |
| 40 | sentence | 137 | 34.8 | 0.581 | 0.71 | 0.658 |
| 40 | structure | 145 | 41.0 | 0.516 | 0.742 | 0.648 |
| 80 | fixed | 75 | 79.1 | 0.71 | 0.935 | 0.818 |
| 80 | recursive | 169 | 28.2 | 0.71 | 0.935 | 0.828 |
| 80 | sentence | 64 | 74.4 | 0.677 | 0.871 | 0.791 |
| 80 | structure | 75 | 70.7 | 0.677 | 0.871 | 0.774 |
| 160 | fixed | 38 | 155.7 | 0.774 | 0.935 | 0.858 |
| 160 | recursive | 159 | 30.0 | 0.742 | 0.935 | 0.844 |
| 160 | sentence | 32 | 148.8 | 0.774 | 0.935 | 0.852 |
| 160 | structure | 47 | 107.7 | 0.742 | 0.935 | 0.834 |

**Read these with care.** There are only 31 questions, so one question is about 3 points of hit rate; gaps of a few points are noise, not a ranking. `recursive` leads at the smallest budget, and every strategy converges as chunks get larger. The heading-prefix baseline does not win here (its prefix adds words beyond the budget and can dilute BM25 scores). More documents and questions are needed before drawing conclusions.

## Roadmap

- [x] First real dataset (`k8s-docs`)
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
