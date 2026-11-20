"""Reproducible single-process Be Cholesky timing and peak-RSS benchmark.

Run separately for each threshold. For the recorded serial budget, use:
    GIAO_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python benchmarks/cholesky_eri.py 1e-8
"""

import json
import platform
import resource
import sys
from time import perf_counter

import basis_set_exchange as bse
import giao_integrals as gi


def uncontracted_be():
    """Unique primitive exponents per L: 16s 10p 6d 4f 2g, 152 Cartesian AOs."""
    data = bse.get_basis("aug-cc-pCVQZ", elements=["Be"])
    exponents = {}
    for shell in data["elements"]["4"]["electron_shells"]:
        for angular in shell["angular_momentum"]:
            exponents.setdefault(angular, set()).update(map(float, shell["exponents"]))
    data["elements"]["4"]["electron_shells"] = [
        {
            "angular_momentum": [angular],
            "exponents": [str(exponent)],
            "coefficients": [["1.0"]],
        }
        for angular, values in sorted(exponents.items())
        for exponent in sorted(values, reverse=True)
    ]
    return gi.Molecule([("Be", (0, 0, 0))], data, field=gi.MagneticField((0, 0, 0.1)))


if __name__ == "__main__":
    mol = uncontracted_be()
    assert mol.ao_count == 152
    start = perf_counter()
    result = mol.cholesky_eri(threshold=float(sys.argv[1]))
    seconds = perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "threshold": result.threshold,
                "nao": mol.ao_count,
                "rank": len(result.vectors),
                "rank_per_ao": len(result.vectors) / mol.ao_count,
                "seconds": seconds,
                "vectors_bytes": result.vectors.nbytes,
                "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
                "max_residual_diagonal": result.max_residual_diagonal,
                "platform": platform.platform(),
                "openmp": gi.openmp_enabled(),
            }
        ),
        flush=True,
    )
