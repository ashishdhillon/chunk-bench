"""Command line interface: ``chunk-bench list`` and ``chunk-bench run``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .datasets import load_dataset
from .runner import run_benchmark
from .strategies import available_strategies


def _humanize(rows: List[dict]) -> List[dict]:
    """Fold ``<metric>_ci`` entries into the metric cell, e.g. ``0.903 [0.85, 0.95]``.

    Paired differences (``d_*``) get a ``*`` when the interval excludes zero.
    """
    out = []
    for row in rows:
        new = {}
        for key, value in row.items():
            if key.endswith("_ci"):
                continue
            ci = row.get(f"{key}_ci")
            if ci is None:
                new[key] = value
            elif key.startswith("d_"):
                star = "*" if ci[0] > 0 or ci[1] < 0 else ""
                new[key] = f"{value:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]{star}"
            else:
                new[key] = f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
        out.append(new)
    return out


def _format_rows(rows: List[dict], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(rows, indent=2)
    rows = _humanize(rows)
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
    run.add_argument(
        "--dataset", default="sample", help="'sample', 'k8s-docs' or path to a JSON dataset"
    )
    run.add_argument("--strategies", default="", help="comma-separated; default: all")
    run.add_argument(
        "--max-words", default="40", help="chunk word budget; comma-separated to sweep, e.g. 40,80,160"
    )
    run.add_argument(
        "--bootstrap", type=int, default=0, help="bootstrap resamples for 95%% CIs (0 = off)"
    )
    run.add_argument("--seed", type=int, default=0, help="seed for the bootstrap resamples")
    run.add_argument(
        "--baseline", default=None, help="strategy for paired hit@k differences (needs --bootstrap)"
    )
    run.add_argument(
        "--cluster",
        action="store_true",
        help="bootstrap over documents instead of questions (needs --bootstrap)",
    )
    run.add_argument("--format", choices=["table", "md", "json"], default="table")

    args = parser.parse_args(argv)

    if args.command == "list":
        print("\n".join(available_strategies()))
        return 0

    names = [s for s in args.strategies.split(",") if s] or available_strategies()
    dataset = load_dataset(args.dataset)
    try:
        budgets = [int(b) for b in args.max_words.split(",") if b]
    except ValueError:
        print("--max-words must be integers, e.g. 40 or 40,80,160", file=sys.stderr)
        return 2
    rows: List[dict] = []
    try:
        for budget in budgets:
            for row in run_benchmark(
                dataset,
                names,
                max_words=budget,
                bootstrap=args.bootstrap,
                seed=args.seed,
                baseline=args.baseline,
                cluster=args.cluster,
            ):
                rows.append({"max_words": budget, **row})
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 2
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(_format_rows(rows, args.format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
