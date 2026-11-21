"""Measure molecular Cholesky phases and exact integral-work counts.

Run each case in a separate process. Limit BLAS threads before starting Python.
The --profile option wraps native calls and line-times Cholesky updates; it
counts selected AO primitive-kernel evaluations from the actual requests.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import platform
import resource
import sys
from time import perf_counter

import basis_set_exchange as bse
import giao_integrals as gi
import numpy as np

WATER = [("O", (0.0, 0.1, 0.2)), ("H", (0.1, 1.4, -0.9)), ("H", (-0.2, -1.3, -1.0))]
FIELD = gi.MagneticField((0.03, -0.02, 0.05), (0.7, -0.4, 1.1))


def molecule(case, finite):
    if case in ("Ne", "Na"):
        data = bse.get_basis(
            "aug-cc-pCVQZ",
            elements=[case],
            uncontract_general=True,
            uncontract_segmented=True,
            uncontract_spdf=True,
        )
        return gi.Molecule(
            [(case, (0, 0, 0))], data, unit="bohr", field=gi.MagneticField((0, 0, 0.1))
        )
    if case == "benzene":
        atoms = []
        for element, radius in (("C", 2.64), ("H", 4.70)):
            for angle in np.arange(6) * np.pi / 3:
                atoms.append(
                    (element, (radius * np.cos(angle), radius * np.sin(angle), 0))
                )
        basis = "cc-pVDZ"
    else:
        atoms = WATER
        basis = "aug-cc-pVTZ" if case == "water-aug" else "cc-pVTZ"
    return gi.Molecule(atoms, basis, unit="bohr", field=FIELD if finite else None)


def run(args):
    mol = molecule(args.case, args.field)
    shells, offsets, n = mol.basis.shells, mol.basis.ao_offsets, mol.ao_count
    owners = np.repeat(np.arange(len(shells)), [s.ao_count for s in shells])
    prim = [s.primitive_count for s in shells]
    cart = [(s.angular_momentum + 1) * (s.angular_momentum + 2) // 2 for s in shells]
    real = args.case in ("Ne", "Na") or not args.field
    counts = {
        "column_calls": 0,
        "column_shell_quartet_visits": 0,
        "column_ao_integrals": 0,
        "column_primitive_integrals": 0,
        "diagonal_shell_quartets": 0,
        "diagonal_primitive_integrals": 0,
        "dense_shell_quartets": 0,
        "dense_primitive_integrals": 0,
        "column_tasks": 0,
    }
    for a in range(len(shells)):
        for b in range(len(shells)):
            for c in range(len(shells)):
                for d in range(len(shells)):
                    q = (a, b, c, d)
                    if q == min(q, (c, d, a, b), (b, a, d, c), (d, c, b, a)):
                        counts["dense_shell_quartets"] += 1
                        counts["dense_primitive_integrals"] += int(
                            np.prod([prim[x] * cart[x] for x in q])
                        )
    module = importlib.import_module("giao_integrals.cholesky")
    function = module.cholesky_eri
    times = {"diagonal_seconds": 0.0, "columns_seconds": 0.0, "updates_seconds": 0.0}
    original_diag, original_columns = (
        module._cholesky_diagonal,
        module._cholesky_columns,
    )
    seen = set()
    reuse = "candidate_count =" in inspect.getsource(function)
    counts["ao_primitive_integrals_without_reuse"] = 0

    def diagonal(basis, pairs, **kw):
        counts["diagonal_shell_quartets"] += len(pairs)
        counts["diagonal_primitive_integrals"] += sum(
            shells[a].ao_count * shells[b].ao_count * prim[a] ** 2 * prim[b] ** 2
            for a, b in pairs
        )
        start = perf_counter()
        result = original_diag(basis, pairs, **kw)
        times["diagonal_seconds"] += perf_counter() - start
        return result

    def columns(basis, pivots, **kw):
        sa, sb = (int(owners[x]) for x in pivots[0])
        counts["column_calls"] += 1
        for a in range(len(shells)):
            for b in range(a if real else 0, len(shells)):
                seen.add((a, b, sb, sa))
                size = len(pivots) * shells[a].ao_count * shells[b].ao_count
                counts["column_shell_quartet_visits"] += 1
                primitive_product = prim[a] * prim[b] * prim[sb] * prim[sa]
                contracted = reuse and any(
                    shells[x].contraction_count > 1 for x in (a, b, sa, sb)
                )
                nominal = size * primitive_product * (2 if real and a == b else 1)
                counts["ao_primitive_integrals_without_reuse"] += nominal
                if contracted:
                    components = {
                        (
                            int(mu - offsets[sa]) % cart[sa],
                            int(nu - offsets[sb]) % cart[sb],
                        )
                        for mu, nu in pivots
                    }
                    counts["column_tasks"] += 1
                    counts["column_primitive_integrals"] += (
                        cart[a] * cart[b] * len(components) * primitive_product
                    )
                else:
                    counts["column_tasks"] += (size + 63) // 64
                    counts["column_primitive_integrals"] += nominal
                counts["column_ao_integrals"] += size * (2 if real and a == b else 1)
        start = perf_counter()
        result = original_columns(basis, pivots, **kw)
        times["columns_seconds"] += perf_counter() - start
        return result

    lines, first = inspect.getsourcelines(function)
    begin = next(
        first + i
        for i, line in enumerate(lines)
        if "column = columns[int(match[0])].copy()" in line
    )
    end = next(
        first + i for i, line in enumerate(lines) if "residual[pivot] = 0" in line
    )
    previous = [0, 0.0]

    def trace(frame, event, arg):
        if frame.f_code is not function.__code__:
            return None
        now = perf_counter()
        if event in ("line", "return") and begin <= previous[0] <= end:
            times["updates_seconds"] += now - previous[1]
        previous[:] = [frame.f_lineno, now]
        return trace

    if args.profile:
        module._cholesky_diagonal, module._cholesky_columns = diagonal, columns
        sys.settrace(trace)
    load_start = os.getloadavg()
    start = perf_counter()
    try:
        chol = mol.cholesky_eri(threshold=1e-8, threads=args.threads)
    finally:
        elapsed = perf_counter() - start
        sys.settrace(None)
        module._cholesky_diagonal, module._cholesky_columns = (
            original_diag,
            original_columns,
        )
    output = {
        "case": args.case,
        "finite_field": args.field or args.case in ("Ne", "Na"),
        "threads": args.threads,
        "nao": n,
        "rank": len(chol.vectors),
        "seconds": elapsed,
        "residual": chol.max_residual_diagonal,
        "vectors_sha256": hashlib.sha256(memoryview(chol.vectors)).hexdigest(),
        "pivots_sha256": hashlib.sha256(memoryview(chol.pivots)).hexdigest(),
        "rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "load_start": load_start,
        "load_end": os.getloadavg(),
        "profiled": args.profile,
        "build": gi.build_info(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    if args.profile:
        counts["unique_column_shell_quartets"] = len(seen)
        times["python_overhead_seconds"] = elapsed - sum(times.values())
        output.update(times=times, counts=counts)
    if args.dense:
        start = perf_counter()
        dense = mol.eri(max_bytes=2**35, threads=args.threads)
        output["dense_seconds"] = perf_counter() - start
        # Slice reconstruction to bound checking workspace, including benzene.
        error = 0.0
        for a in range(n):
            block = chol.reconstruct(
                (slice(a, a + 1), slice(None), slice(None), slice(None))
            )
            error = max(error, float(np.max(np.abs(block[0] - dense[a]))))
        output["max_error"] = error
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=["water", "water-aug", "benzene", "Ne", "Na"], default="water"
    )
    parser.add_argument("--field", action="store_true")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dense", action="store_true")
    print(json.dumps(run(parser.parse_args())), flush=True)
