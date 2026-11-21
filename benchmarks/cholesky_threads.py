"""Ne aug-cc-pCVQZ threading benchmark with optional per-line update profiling.

Set VECLIB_MAXIMUM_THREADS=1, then run:
python benchmarks/cholesky_threads.py --threads 1 4 8 12 --profile
Each JSON line includes a vector/pivot hash to check bitwise thread invariance.
Run on a quiet host for controlled performance comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import resource
import sys
from importlib import import_module
from time import perf_counter

import basis_set_exchange as bse
import giao_integrals as gi


def run(threads: int, profile: bool, atom="Ne", threshold=1e-8) -> dict:
    data = bse.get_basis("aug-cc-pCVQZ", elements=[atom])
    element = next(iter(data["elements"].values()))
    exponents = {}
    for shell in element["electron_shells"]:
        for angular in shell["angular_momentum"]:
            exponents.setdefault(angular, set()).update(map(float, shell["exponents"]))
    element["electron_shells"] = [
        {
            "angular_momentum": [angular],
            "exponents": [str(exponent)],
            "coefficients": [["1.0"]],
        }
        for angular, values in sorted(exponents.items())
        for exponent in sorted(values, reverse=True)
    ]
    mol = gi.Molecule([(atom, (0, 0, 0))], data, field=gi.MagneticField((0, 0, 0.1)))
    assert mol.ao_count == 152
    function = import_module("giao_integrals.cholesky").cholesky_eri
    source, first = inspect.getsourcelines(function)
    update = next(
        first + i
        for i, line in enumerate(source)
        if "column = columns[int(match[0])].copy()" in line
    )
    residual = next(
        first + i
        for i, line in enumerate(source)
        if "residual -= np.abs(column) ** 2" in line
    )
    stop = next(
        first + i for i, line in enumerate(source) if "residual[pivot] = 0" in line
    )
    times = {"vector_update_seconds": 0.0, "residual_update_seconds": 0.0}
    last = [0, 0.0]

    def trace(frame, event, arg):
        if frame.f_code is not function.__code__:
            return None
        now = perf_counter()
        if event in {"line", "return"}:
            line, previous = last
            if update <= line < residual:
                times["vector_update_seconds"] += now - previous
            elif residual <= line <= stop:
                times["residual_update_seconds"] += now - previous
        last[:] = [frame.f_lineno, now]
        return trace

    if profile:
        sys.settrace(trace)
    start = perf_counter()
    try:
        result = mol.cholesky_eri(threshold=threshold, threads=threads)
    finally:
        elapsed = perf_counter() - start
        if profile:
            sys.settrace(None)
    digest = hashlib.sha256(memoryview(result.vectors)).hexdigest()
    pivots = hashlib.sha256(memoryview(result.pivots)).hexdigest()
    return {
        "profiled": bool(profile),
        "vectors_bytes": result.vectors.nbytes,
        "atom": atom,
        "threshold": threshold,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "threads": threads,
        "seconds": elapsed,
        "nao": mol.ao_count,
        "rank": len(result.vectors),
        "max_residual_diagonal": result.max_residual_diagonal,
        "vectors_sha256": digest,
        "pivots_sha256": pivots,
        "build": gi.build_info(),
        **(times if profile else {}),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8, 12])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--profile-serial", action="store_true", help="profile only threads=1"
    )
    parser.add_argument("--atom", choices=["Ne", "Be"], default="Ne")
    parser.add_argument("--threshold", type=float, default=1e-8)
    args = parser.parse_args()
    for count in args.threads:
        print(
            json.dumps(
                run(
                    count,
                    args.profile or (args.profile_serial and count == 1),
                    args.atom,
                    args.threshold,
                )
            ),
            flush=True,
        )
