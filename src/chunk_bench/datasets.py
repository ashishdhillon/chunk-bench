"""Dataset loading and validation.

Dataset format (JSON)::

    {
      "name": "my-dataset",
      "documents": [{"id": "doc1", "text": "# Title\\n\\n..."}],
      "questions": [
        {"question": "...", "doc_id": "doc1", "evidence": "exact span from the doc"}
      ]
    }

``evidence`` is the exact text span a retrieved chunk must contain for the
question to count as answerable from that chunk (whitespace/case-insensitive).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

BUILTIN = {"sample": "sample.json", "k8s-docs": "k8s-docs.json", "docker-docs": "docker-docs.json"}


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def validate(data: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("documents", "questions"):
        if key not in data or not data[key]:
            raise ValueError(f"dataset is missing a non-empty {key!r} list")
    docs = {d["id"]: normalize(d["text"]) for d in data["documents"]}
    if len(docs) != len(data["documents"]):
        raise ValueError("duplicate document ids")
    for q in data["questions"]:
        if q["doc_id"] not in docs:
            raise ValueError(f"question references unknown doc_id {q['doc_id']!r}")
        if normalize(q["evidence"]) not in docs[q["doc_id"]]:
            raise ValueError(f"evidence not found verbatim in {q['doc_id']!r}: {q['evidence']!r}")
    return data


def load_dataset(name_or_path: str = "sample") -> Dict[str, Any]:
    if name_or_path in BUILTIN:
        path = Path(__file__).parent / "data" / BUILTIN[name_or_path]
    else:
        path = Path(name_or_path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("name", path.stem)
    return validate(data)
