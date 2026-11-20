r"""Time streamed Cholesky ERI generation for one atom across thread counts.

Install with: python -m pip install '.[molecule]'
Run from the checkout, limiting BLAS threads before Python starts:

    OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
        python examples/atom_cholesky_timing.py --threads 1 2 4

The default Ne/cc-pVDZ case is small. For the larger 152-AO Ne benchmark:

    OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
        python examples/atom_cholesky_timing.py --basis aug-cc-pCVQZ \
        --uncontracted --threads 1 4 8 12

Large cases can take minutes per run. Each thread count is measured once;
speedups are relative to the first requested count. Basis loading, hashing,
and printing are outside the timing; diagonal generation and factorization
are included. First-call setup and caches can affect these illustrative timings;
small cases may be slower in parallel because of overhead.

ERI[mu,nu,lam,sig] ~= sum_P L[P,mu,nu] * conj(L[P,sig,lam]). No dense tensor
is formed. For a single atom, the common London phase cancels in each pair
density, so ERIs are field-independent even though a finite field is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
from math import isfinite
from time import perf_counter

import basis_set_exchange as bse
import giao_integrals as gi


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--atom", default="Ne", help="element symbol (default: Ne)")
    parser.add_argument("--basis", default="cc-pVDZ", help="BSE basis name")
    parser.add_argument(
        "--uncontracted", action="store_true", help="uncontract all BSE shell types"
    )
    parser.add_argument("--threshold", type=float, default=1e-8)
    parser.add_argument(
        "--field",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.1),
        metavar=("BX", "BY", "BZ"),
        help="magnetic field in atomic units",
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        help="counts in timing order (default: 1 and the configured parallel default)",
    )
    args = parser.parse_args()
    if not isfinite(args.threshold) or args.threshold < 0:
        parser.error("--threshold must be finite and non-negative")
    if not all(isfinite(value) for value in args.field):
        parser.error("--field components must be finite")

    info = gi.build_info()
    counts = args.threads
    if counts is None:
        counts = [1]
        if info["default_threads"] > 1:
            counts.append(info["default_threads"])
    if any(count < 1 for count in counts):
        parser.error("--threads counts must be positive")
    if any(count > 1 for count in counts) and not info["openmp_enabled"]:
        parser.error("parallel counts require an OpenMP build; use --threads 1")

    data = bse.get_basis(
        args.basis,
        elements=[args.atom],
        uncontract_general=args.uncontracted,
        uncontract_segmented=args.uncontracted,
        uncontract_spdf=args.uncontracted,
    )
    mol = gi.Molecule(
        [(args.atom, (0.0, 0.0, 0.0))],
        data,
        field=gi.MagneticField(args.field),
    )
    print(
        f"Atom: {mol.atoms[0][0]}; basis: {args.basis}; "
        f"uncontracted: {args.uncontracted}"
    )
    print(
        f"Cartesian AOs: {mol.ao_count}; B={tuple(args.field)}; "
        f"threshold={args.threshold:g}"
    )
    print(
        f"OpenMP: {info['openmp_enabled']}; "
        f"configured threads: {info['default_threads']}"
    )
    # complex128 is 16 bytes per integral; estimate without allocating the tensor.
    print(f"Dense complex128 ERI estimate: {mol.ao_count**4 * 16 / 1024**2:,.2f} MiB")
    print(
        f"Speedup relative to {counts[0]} thread(s); bitwise checks against that run."
    )
    print(
        f"{'threads':>7} {'seconds':>10} {'speedup':>8} {'M':>6} {'M/nao':>8} "
        f"{'vectors MiB':>12} {'max residual':>14} {'identical':>10}",
        flush=True,
    )

    reference_hashes = None
    reference_seconds = None
    for count in counts:
        start = perf_counter()
        chol = mol.cholesky_eri(threshold=args.threshold, threads=count)
        seconds = perf_counter() - start

        # Keep only hashes between runs, rather than retaining another factor tensor.
        hashes = (
            hashlib.sha256(memoryview(chol.vectors)).hexdigest(),
            hashlib.sha256(memoryview(chol.pivots)).hexdigest(),
        )
        if reference_hashes is None:
            reference_hashes = hashes
            reference_seconds = seconds
            identical = "reference"
        else:
            if hashes != reference_hashes:
                raise RuntimeError(f"factors/pivots differ at threads={count}")
            identical = "yes"
        rank = len(chol.vectors)
        print(
            f"{count:7d} {seconds:10.3f} {reference_seconds / seconds:8.2f} "
            f"{rank:6d} {rank / mol.ao_count:8.2f} "
            f"{chol.vectors.nbytes / 1024**2:12.3f} "
            f"{chol.max_residual_diagonal:14.3e} {identical:>10}",
            flush=True,
        )
        del chol


if __name__ == "__main__":
    main()
