# Changelog

## 0.2.0
- Document-clustered bootstrap (`--cluster`), with a warning when there are fewer than 10 documents.
- `structure` now counts its heading prefix inside the word budget.
- New `docker-docs` dataset (4 pages, 114 questions, Apache-2.0).
- Optional dense and hybrid (RRF) retrievers via `--retriever` and the `[dense]` extra.

## 0.1.0
- Strategy registry, BM25 retrieval, `k8s-docs` dataset, bootstrap CIs with paired differences, `sdca_lite`.
