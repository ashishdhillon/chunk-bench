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
    assert len(data["questions"]) >= 60
    small = run_benchmark(data, ["fixed"], max_words=40)[0]
    large = run_benchmark(data, ["fixed"], max_words=160)[0]
    assert small["chunks"] > large["chunks"]


def test_cli_budget_sweep(capsys):
    assert main(["run", "--strategies", "fixed", "--max-words", "20,40", "--format", "json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["max_words"] for r in rows] == [20, 40]


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
