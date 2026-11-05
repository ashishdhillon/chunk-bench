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
| `k8s-docs` | 3 real Kubernetes concept pages (ConfigMaps, Pods, Deployments), 60 questions with verbatim evidence spans. Source and license: see [NOTICE](NOTICE). |

## First results (`k8s-docs`, BM25)

```bash
chunk-bench run --dataset k8s-docs --max-words 40,80,160 --format md
```

| max_words | strategy | chunks | avg_words | hit@1 | hit@3 | mrr |
|---|---|---|---|---|---|---|
| 40 | fixed | 331 | 39.9 | 0.5 | 0.683 | 0.616 |
| 40 | recursive | 525 | 20.1 | 0.583 | 0.717 | 0.654 |
| 40 | sentence | 310 | 34.1 | 0.567 | 0.717 | 0.651 |
| 40 | structure | 327 | 40.3 | 0.533 | 0.717 | 0.63 |
| 80 | fixed | 166 | 79.4 | 0.633 | 0.817 | 0.733 |
| 80 | recursive | 429 | 24.7 | 0.65 | 0.783 | 0.728 |
| 80 | sentence | 143 | 74.0 | 0.667 | 0.85 | 0.759 |
| 80 | structure | 167 | 70.6 | 0.65 | 0.85 | 0.758 |
| 160 | fixed | 84 | 156.8 | 0.717 | 0.867 | 0.792 |
| 160 | recursive | 407 | 26.0 | 0.667 | 0.8 | 0.738 |
| 160 | sentence | 70 | 151.1 | 0.7 | 0.867 | 0.792 |
| 160 | structure | 100 | 111.9 | 0.75 | 0.9 | 0.832 |

**Read these with care.** There are only 60 questions, so one question is about 1.7 points of hit rate; gaps of a few points are noise, not a ranking. Going from 2 to 3 documents already reshuffled the order: `recursive` led at small budgets and `structure` leads at 160 words here, but neither gap is large. The consistent signal is that hit@k rises with chunk size for every strategy, and that the strategies stay within a few points of each other. More documents and questions are needed before drawing conclusions.

## Roadmap

- [x] First real dataset (`k8s-docs`, 60 questions)
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
