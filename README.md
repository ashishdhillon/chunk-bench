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

## Status and honesty note

The bundled `sample` dataset (3 short documents, 15 questions) is a **smoke test**, not a leaderboard: all strategies score near 1.0 on it, so it cannot separate them. Meaningful comparisons need larger, structure-rich corpora (long manuals, papers, legal and API docs). Adding such datasets and a results table is the next milestone.

## Roadmap

- [ ] Larger public datasets with structured documents (HTML, PDF-to-markdown)
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
