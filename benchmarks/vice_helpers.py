"""Reproduce VICE atom/density/TD4 timings; set BLAS thread limits before Python.

    OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
        python benchmarks/vice_helpers.py --threads 1 4

Times exclude basis loading and verification. Ordered density reference includes
its Python quartet loop; native times include streamed generation and contraction.
"""

import argparse
import json
from itertools import product
from time import perf_counter

import giao_integrals as gi
import numpy as np


def ordered_density_gradient(mol, p, q):
    result = np.zeros((len(mol.atoms), 3), complex)
    shells = mol.basis.shells
    offsets = mol.basis.ao_offsets
    for indices in product(range(len(shells)), repeat=4):
        s = [slice(offsets[k], offsets[k] + shells[k].ao_count) for k in indices]
        a, b, c, d = s
        weight = np.einsum("nm,sl->mnls", p[b, a], q[d, c]) + np.einsum(
            "nl,sm->mnls", p[b, c], q[d, a]
        )
        derivative = gi.eri_center_derivatives_shell(
            *[shells[k] for k in indices], field=mol.field
        )
        for slot, shell in enumerate(indices):
            result[mol.shell_atoms[shell]] += np.einsum(
                "rmnls,mnls->r", derivative[slot], weight
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    args = parser.parse_args()
    field = gi.MagneticField((0.05, -0.03, 0.2), (0.37, -0.29, 0.43))
    mol = gi.Molecule(
        [("O", (0.1, -0.2, 0.3)), ("H", (0.2, 0.4, 1.6)), ("H", (1.5, -0.1, -0.3))],
        "STO-3G",
        unit="bohr",
        field=field,
    )
    rng = np.random.default_rng(192)
    n = mol.ao_count
    p = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    start = perf_counter()
    reference = ordered_density_gradient(mol, p, q)
    seconds = perf_counter() - start
    print(
        json.dumps(
            {
                "case": "H2O/STO-3G J+K",
                "implementation": "Python ordered",
                "nao": n,
                "seconds": seconds,
            }
        ),
        flush=True,
    )
    for threads in args.threads:
        start = perf_counter()
        actual = mol.eri_derivative_contraction(
            coulomb_pairs=[(p, q)],
            exchange_pairs=[(p, q)],
            threads=threads,
            target_bytes=1024**2,
        )
        seconds = perf_counter() - start
        error = float(np.max(np.abs(actual - reference)))
        np.testing.assert_allclose(actual, reference, atol=5e-12, rtol=5e-12)
        print(
            json.dumps(
                {
                    "case": "H2O/STO-3G J+K",
                    "implementation": "native streamed",
                    "nao": n,
                    "threads": threads,
                    "seconds": seconds,
                    "max_error": error,
                }
            ),
            flush=True,
        )
    atom = gi.Molecule(
        [("Ne", (0.1, -0.2, 0.3))], "aug-cc-pVQZ", unit="bohr", field=field
    )
    calls = [
        (
            "atom moment nuclear",
            lambda t: atom.moment_nuclear_derivatives(
                (1, 0, 0), origin=(0.2, 0.3, -0.1), threads=t
            ),
        ),
        (
            "kinetic angular x",
            lambda t: (
                atom.kinetic_angular_momentum("x", origin=(0.2, 0.3, -0.1), threads=t),
            ),
        ),
        (
            "independent magnetic kinetic",
            lambda t: (
                atom.magnetic_kinetic_operator(
                    operator_field=(-0.12, 0.08, 0.19),
                    origin=(0.2, 0.3, -0.1),
                    threads=t,
                ),
            ),
        ),
    ]
    for name, function in calls:
        reference = None
        for threads in args.threads:
            start = perf_counter()
            result = function(threads)
            seconds = perf_counter() - start
            if reference is not None:
                for actual, expected in zip(result, reference, strict=True):
                    np.testing.assert_array_equal(actual, expected)
            else:
                reference = result
            print(
                json.dumps(
                    {
                        "case": name,
                        "basis": "Ne/aug-cc-pVQZ",
                        "nao": atom.ao_count,
                        "threads": threads,
                        "seconds": seconds,
                        "bytes": sum(x.nbytes for x in result),
                        "identical": True,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
