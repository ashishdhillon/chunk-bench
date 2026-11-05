import json

import pytest

from chunk_bench import available_strategies, load_dataset, register_strategy, run_benchmark
from chunk_bench.cli import main
from chunk_bench.retrieval import BM25
from chunk_bench.strategies import get_strategy

TEXT = "# Title\n\n## A\n\nOne two three. Four five six. Seven eight nine.\n\n## B\n\nTen eleven twelve."


def test_builtin_strategies_present():
    assert {"fixed", "sentence", "recursive", "structure"} <= set(available_strategies())


@pytest.mark.parametrize("name", ["fixed", "sentence", "recursive", "structure"])
def test_strategies_respect_budget_and_keep_content(name):
    chunks = get_strategy(name)(TEXT, 5)
    assert chunks
    joined = " ".join(chunks).lower()
    for word in ["one", "nine", "twelve"]:
        assert word in joined
    if name != "structure":  # structure adds a heading prefix on top of the budget
        assert all(len(c.split()) <= 5 + 1 for c in chunks)


def test_structure_prefixes_heading_path():
    chunks = get_strategy("structure")(TEXT, 50)
    assert chunks[0].startswith("Title > A:")
    assert chunks[1].startswith("Title > B:")


DOC = (
    "# Guide\n\n## Alpha\n\nAlpha one two three four. Alpha five six seven eight.\n\n"
    "Alpha nine ten eleven twelve.\n\n## Beta\n\nBeta one two. Beta three four.\n"
)


def test_sdca_lite_never_crosses_sections_and_binds_breadcrumb():
    chunks = get_strategy("sdca_lite")(DOC, 20)
    assert all(("Alpha" in c) != ("Beta" in c) for c in chunks)
    assert chunks[0].startswith("Guide > Alpha:")
    assert chunks[-1].startswith("Guide > Beta:")


def test_sdca_lite_keeps_paragraphs_whole_when_they_fit():
    chunks = get_strategy("sdca_lite")(DOC, 14)
    joined = " | ".join(chunks)
    assert "Alpha nine ten eleven twelve." in joined
    assert not any(c.endswith("Alpha nine") for c in chunks)


def test_sdca_lite_budget_counts_the_breadcrumb():
    for c in get_strategy("sdca_lite")(DOC, 12):
        assert len(c.replace(">", " ").replace(":", " ").split()) <= 12 + 1


def test_sdca_lite_merges_short_tail_within_section():
    text = "## S\n\nOne two three four five six seven eight nine.\n\nTen.\n"
    assert len(get_strategy("sdca_lite")(text, 40)) == 1


def test_sdca_lite_splits_oversized_paragraph_at_sentences():
    text = "## S\n\n" + " ".join(f"Sentence {i} has five words." for i in range(10))
    chunks = get_strategy("sdca_lite")(text, 20)
    assert len(chunks) > 1
    assert all(c.rstrip().endswith(".") for c in chunks)


def test_empty_text():
    assert get_strategy("fixed")("", 10) == []


def test_bm25_ranks_relevant_first():
    index = BM25(["red apples are sweet", "blue cars are fast", "green apples are sour"])
    top = [i for i, _ in index.top_k("sweet red apples", 3)]
    assert top[0] == 0


def test_sample_dataset_is_valid():
    data = load_dataset("sample")
    assert len(data["questions"]) >= 10


def test_k8s_dataset_is_valid_and_separates_budgets():
    data = load_dataset("k8s-docs")
    assert len(data["questions"]) >= 100
    small = run_benchmark(data, ["fixed"], max_words=40)[0]
    large = run_benchmark(data, ["fixed"], max_words=160)[0]
    assert small["chunks"] > large["chunks"]


def test_cli_budget_sweep(capsys):
    assert main(["run", "--strategies", "fixed", "--max-words", "20,40", "--format", "json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["max_words"] for r in rows] == [20, 40]


def test_bootstrap_intervals_are_sane_and_seeded():
    data = load_dataset("sample")
    names = ["fixed", "structure"]
    a = run_benchmark(data, names, max_words=40, bootstrap=200, seed=1, baseline="fixed")
    b = run_benchmark(data, names, max_words=40, bootstrap=200, seed=1, baseline="fixed")
    assert a == b
    for row in a:
        for key in ("hit@1", "hit@3", "mrr"):
            lo, hi = row[f"{key}_ci"]
            assert 0.0 <= lo <= hi <= 1.0
    base = a[0]
    assert base["d_hit@3"] == 0.0 and base["d_hit@3_ci"] == [0.0, 0.0]


def test_baseline_must_be_benchmarked():
    with pytest.raises(ValueError):
        run_benchmark(load_dataset("sample"), ["fixed"], baseline="structure", bootstrap=10)


def test_cli_bootstrap_table(capsys):
    argv = ["run", "--strategies", "fixed,structure", "--bootstrap", "50", "--baseline", "fixed"]
    assert main(argv + ["--format", "md"]) == 0
    out = capsys.readouterr().out
    assert "d_hit@3" in out and "[" in out


def test_invalid_evidence_rejected(tmp_path):
    bad = {
        "documents": [{"id": "d", "text": "hello world"}],
        "questions": [{"question": "q", "doc_id": "d", "evidence": "missing"}],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_dataset(str(path))


def test_benchmark_is_deterministic_and_bounded():
    data = load_dataset("sample")
    a = run_benchmark(data, available_strategies(), max_words=40)
    b = run_benchmark(data, available_strategies(), max_words=40)
    assert a == b
    for row in a:
        assert 0.0 <= row["hit@1"] <= row["hit@3"] <= 1.0
        assert 0.0 <= row["mrr"] <= 1.0


def test_custom_strategy_plugs_in():
    @register_strategy("whole_doc")
    def whole_doc(text, max_words):
        return [text]

    rows = run_benchmark(load_dataset("sample"), ["whole_doc"], max_words=40)
    assert rows[0]["chunks"] == 3


def test_cli_json(capsys):
    assert main(["run", "--strategies", "fixed,structure", "--format", "json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["strategy"] for r in rows] == ["fixed", "structure"]


def test_cli_unknown_strategy():
    assert main(["run", "--strategies", "nope"]) == 2


def test_structure_prefix_counts_toward_budget():
    from chunk_bench.strategies import get_strategy

    for doc in load_dataset("k8s-docs")["documents"]:
        for chunk in get_strategy("structure")(doc["text"], 40):
            # allow one oversized unsplittable sentence (packer emits it alone)
            assert len(chunk.split()) <= 40 or ": " in chunk and len(chunk.split()) <= 80


def test_clustered_bootstrap_is_seeded_and_valid():
    data = load_dataset("k8s-docs")
    names = ["fixed", "structure"]
    kw = dict(max_words=80, bootstrap=200, seed=1, baseline="fixed")
    a = run_benchmark(data, names, cluster=True, **kw)
    b = run_benchmark(data, names, cluster=True, **kw)
    assert a == b
    for row in a:
        lo, hi = row["hit@3_ci"]
        assert lo <= row["hit@3"] <= hi
    assert a[0]["d_hit@3"] == 0.0


def test_cli_cluster_flag(capsys):
    argv = ["run", "--strategies", "fixed,structure", "--bootstrap", "50",
            "--baseline", "fixed", "--cluster"]
    assert main(argv) == 0


def test_docker_docs_dataset_loads_and_validates():
    data = load_dataset("docker-docs")
    assert len(data["documents"]) == 4 and len(data["questions"]) >= 100
    rows = run_benchmark(data, ["fixed", "sdca_lite"], max_words=80)
    assert all(0.0 < r["hit@3"] <= 1.0 for r in rows)


def test_cluster_warns_with_few_documents():
    import pytest

    with pytest.warns(UserWarning, match="few distinct"):
        run_benchmark(load_dataset("docker-docs"), ["fixed"], bootstrap=20, cluster=True)
