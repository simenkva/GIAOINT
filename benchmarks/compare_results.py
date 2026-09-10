#!/usr/bin/env python3
"""Compare two JSON-lines benchmark runs without hiding unmatched cases."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

KEY_FIELDS = ("case", "operator", "field", "shell_class", "primitive_shape", "threads")


def load_results(path: Path) -> dict[tuple[object, ...], dict[str, object]]:
    results: dict[tuple[object, ...], dict[str, object]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = tuple(record[field] for field in KEY_FIELDS)
                float(record["integrals_per_second"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid benchmark record"
                ) from error
            if key in results:
                raise ValueError(
                    f"{path}:{line_number}: duplicate benchmark case {key!r}"
                )
            results[key] = record
    if not results:
        raise ValueError(f"{path}: no benchmark records")
    return results


def compare(
    baseline: dict[tuple[object, ...], dict[str, object]],
    current: dict[tuple[object, ...], dict[str, object]],
) -> Iterable[tuple[str, float, float, float]]:
    for key in sorted(baseline.keys() & current.keys(), key=str):
        old = float(baseline[key]["integrals_per_second"])
        new = float(current[key]["integrals_per_second"])
        if old <= 0.0 or new <= 0.0:
            raise ValueError(f"non-positive throughput for {key!r}")
        yield str(key[0]), old, new, new / old


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument(
        "--minimum-ratio",
        type=float,
        default=0.0,
        help=(
            "fail if a matched case falls below this throughput ratio "
            "(default: report only)"
        ),
    )
    arguments = parser.parse_args(argv)

    try:
        baseline = load_results(arguments.baseline)
        current = load_results(arguments.current)
        rows = list(compare(baseline, current))
    except ValueError as error:
        parser.error(str(error))

    missing = sorted(baseline.keys() - current.keys(), key=str)
    added = sorted(current.keys() - baseline.keys(), key=str)
    print("case\tbaseline int/s\tcurrent int/s\tratio")
    for name, old, new, ratio in rows:
        print(f"{name}\t{old:.6g}\t{new:.6g}\t{ratio:.4f}")
    for key in missing:
        print(f"missing current case: {key!r}", file=sys.stderr)
    for key in added:
        print(f"new current case: {key!r}", file=sys.stderr)

    regressed = [name for name, _, _, ratio in rows if ratio < arguments.minimum_ratio]
    if missing or not rows:
        return 2
    if regressed:
        print("throughput threshold failed: " + ", ".join(regressed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
