"""Command line interface: ``chunk-bench list`` and ``chunk-bench run``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .datasets import load_dataset
from .runner import run_benchmark
from .strategies import available_strategies


def _format_rows(rows: List[dict], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(rows, indent=2)
    headers = list(rows[0].keys())
    if fmt == "md":
        lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        lines += ["| " + " | ".join(str(r[h]) for h in headers) + " |" for r in rows]
        return "\n".join(lines)
    widths = [max(len(h), *(len(str(r[h])) for r in rows)) for h in headers]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths))]
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(str(r[h]).ljust(w) for h, w in zip(headers, widths)) for r in rows]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="chunk-bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list available chunking strategies")

    run = sub.add_parser("run", help="run the benchmark")
    run.add_argument("--dataset", default="sample", help="'sample' or path to a JSON dataset")
    run.add_argument("--strategies", default="", help="comma-separated; default: all")
    run.add_argument("--max-words", type=int, default=40, help="chunk word budget")
    run.add_argument("--format", choices=["table", "md", "json"], default="table")

    args = parser.parse_args(argv)

    if args.command == "list":
        print("\n".join(available_strategies()))
        return 0

    names = [s for s in args.strategies.split(",") if s] or available_strategies()
    dataset = load_dataset(args.dataset)
    try:
        rows = run_benchmark(dataset, names, max_words=args.max_words)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 2
    print(_format_rows(rows, args.format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
