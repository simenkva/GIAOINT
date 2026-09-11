"""Compare giao_integrals wall-clock timings against PySCF/libcint across a
small ladder of Basis Set Exchange basis sets, on the same water geometry
used in basis_set_exchange_comparison.py.

This file is deliberately standalone, like
``tests/external/pyscf_zero_field.py``. It requires the optional
``basis_set_exchange`` and ``pyscf`` packages, which are not part of this
project's dependencies:

    pip install basis_set_exchange pyscf

This is a quick illustrative comparison, not a controlled benchmark: it runs
a handful of repetitions on whatever machine happens to run it, with no
attempt to pin CPU frequency, isolate cores, or average away system noise.
For reproducible, machine-tagged performance records see ``benchmarks/``.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import TypeVar

import giao_integrals as gi
import pyscf
from basis_set_exchange_comparison import ATOMS, build_pyscf_molecule, load_bse_basis

T = TypeVar("T")

# Basis sets in increasing size, all defined for H and O on the Basis Set
# Exchange. Exact zero field uses the collapsed real MD kernel; finite fields
# use the general complex kernel. This example measures zero field only.
BASIS_NAMES = ["STO-3G", "6-31G", "cc-pVDZ"]
ONE_ELECTRON_REPEATS = 5
ERI_REPEATS = 3
ERI_MAX_BYTES = 512 * 1024**2


def _best_of(build: Callable[[], T], repeats: int) -> tuple[T, float]:
    """Return the last result and the minimum wall time over ``repeats`` runs.

    One untimed warm-up call runs first to absorb one-off allocation costs;
    the minimum of the remaining runs is a standard, noise-resistant estimate
    for a wall-clock microbenchmark.
    """

    result = build()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = build()
        best = min(best, time.perf_counter() - start)
    return result, best


def _benchmark_basis(
    basis_name: str, *, threads: int = 1, screening_threshold: float = 0.0
) -> dict[str, object]:
    basis = load_bse_basis(basis_name, ATOMS)
    mol = build_pyscf_molecule(basis_name, ATOMS)
    assert basis.ao_count == mol.nao
    nuclei = [
        gi.Nucleus(float(mol.atom_charge(i)), tuple(mol.atom_coord(i)))
        for i in range(mol.natm)
    ]

    _, giao_overlap = _best_of(lambda: gi.overlap(basis), ONE_ELECTRON_REPEATS)
    _, pyscf_overlap = _best_of(
        lambda: mol.intor("int1e_ovlp_cart"), ONE_ELECTRON_REPEATS
    )
    _, giao_kinetic = _best_of(lambda: gi.kinetic(basis), ONE_ELECTRON_REPEATS)
    _, pyscf_kinetic = _best_of(
        lambda: mol.intor("int1e_kin_cart"), ONE_ELECTRON_REPEATS
    )
    _, giao_attraction = _best_of(
        lambda: gi.nuclear_attraction(basis, nuclei), ONE_ELECTRON_REPEATS
    )
    _, pyscf_attraction = _best_of(
        lambda: mol.intor("int1e_nuc_cart"), ONE_ELECTRON_REPEATS
    )
    _, giao_eri = _best_of(
        lambda: gi.eri(
            basis,
            storage="full",
            max_bytes=ERI_MAX_BYTES,
            threads=threads,
            screening_threshold=screening_threshold,
        ),
        ERI_REPEATS,
    )
    _, pyscf_eri = _best_of(lambda: mol.intor("int2e_cart"), ERI_REPEATS)

    return {
        "basis": basis_name,
        "aos": basis.ao_count,
        "overlap": (giao_overlap, pyscf_overlap),
        "kinetic": (giao_kinetic, pyscf_kinetic),
        "nuclear attraction": (giao_attraction, pyscf_attraction),
        "ERI": (giao_eri, pyscf_eri),
    }


def _print_table(rows: list[dict[str, object]]) -> None:
    columns = ("basis", "AOs", "operator", "giao_integrals", "PySCF", "ratio")
    header = (
        f"{columns[0]:<10}{columns[1]:>5}  {columns[2]:<18}"
        f"{columns[3]:>15}{columns[4]:>13}{columns[5]:>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        for operator in ("overlap", "kinetic", "nuclear attraction", "ERI"):
            giao_seconds, pyscf_seconds = row[operator]
            ratio = giao_seconds / pyscf_seconds
            print(
                f"{row['basis']:<10}{row['aos']:>5}  {operator:<18}"
                f"{giao_seconds * 1e3:>12.3f} ms{pyscf_seconds * 1e3:>10.3f} ms"
                f"{ratio:>9.1f}x"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", nargs="+", default=BASIS_NAMES)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--screening-threshold", type=float, default=0.0)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.threads > 1 and not gi.openmp_enabled():
        parser.error("--threads > 1 requires an OpenMP-enabled giao_integrals build")
    # Give both engines the same requested thread count. Screening-bound setup
    # stays inside each timed call, so this measures the complete public API.
    pyscf.lib.num_threads(args.threads)
    print(f"giao-integrals {gi.__version__} vs PySCF {pyscf.__version__}")
    print(f"threads={args.threads}; screening_threshold={args.screening_threshold:g}")
    print(
        "Best-of timings on the water geometry from basis_set_exchange_comparison.py."
    )
    print(
        "Ratio is giao_integrals / PySCF wall time "
        "(>1x means giao_integrals is slower).\n"
    )
    rows = [
        _benchmark_basis(
            name, threads=args.threads, screening_threshold=args.screening_threshold
        )
        for name in args.basis
    ]
    _print_table(rows)


if __name__ == "__main__":
    main()
