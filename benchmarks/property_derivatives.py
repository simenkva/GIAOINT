"""Time one-electron derivative matrices; limit BLAS threads before starting.

    OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
        python benchmarks/property_derivatives.py --threads 1 4

Defaults to Ne/aug-cc-pVQZ (105 Cartesian AOs), B=(0.05,-0.03,0.2).
Only the derivative call is timed; basis loading and equality checks are not.
"""

import argparse
from time import perf_counter

import giao_integrals as gi
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", nargs="+", type=int, default=[1])
    parser.add_argument("--basis", default="aug-cc-pVQZ")
    args = parser.parse_args()
    mol = gi.Molecule(
        [("Ne", (0.1, -0.2, 0.3))],
        args.basis,
        unit="bohr",
        field=gi.MagneticField((0.05, -0.03, 0.2), (0.37, -0.29, 0.43)),
    )
    print(
        f"Ne/{args.basis}: {mol.ao_count} Cartesian AOs; "
        f"{len(mol.basis.shells)} shells",
        flush=True,
    )
    reference = None
    for threads in args.threads:
        start = perf_counter()
        result = gi.moment_nuclear_derivatives(
            mol.basis,
            (1, 0, 0),
            origin=(0.2, 0.3, -0.1),
            field=mol.field,
            threads=threads,
        )
        seconds = perf_counter() - start
        if reference is not None:
            for actual, expected in zip(result, reference, strict=True):
                np.testing.assert_array_equal(actual, expected)
        else:
            reference = result
        print(
            f"moment nuclear: threads={threads}, seconds={seconds:.6f}, "
            f"output MiB={sum(x.nbytes for x in result)/1024**2:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
