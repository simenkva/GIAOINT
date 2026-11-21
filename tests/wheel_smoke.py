"""Run against the installed/repaired wheel, never the checkout's Python package."""

from importlib.metadata import version
from pathlib import Path

import giao_integrals as gi
import numpy as np

info = gi.build_info()
print(info)
assert info["version"] == version("giao-integrals")
assert info["openmp_enabled"], "release wheels must include OpenMP"
runtime = Path(info["openmp_runtime_path"]).resolve()
assert runtime.is_file(), runtime
# Repaired wheels must not depend on a host's Homebrew/compiler installation.
# Source archive smoke checks may link the system runtime, unlike wheel tests.
if any(part.endswith((".dylibs", ".libs")) for part in runtime.parts):
    print("Bundled OpenMP runtime:", runtime)
elif (
    "site-packages" in str(Path(gi.__file__).resolve())
    and __import__("os").environ.get("GIAO_REQUIRE_BUNDLED_OMP") == "1"
):
    raise AssertionError(f"OpenMP runtime was not bundled: {runtime}")

basis = gi.Basis(
    [
        gi.Shell((0, 0, 0), 0, [0.8], [1.0]),
        gi.Shell((0.7, 0.1, -0.2), 1, [0.6], [1.0]),
    ]
)
field = gi.MagneticField((0.05, -0.03, 0.2), (0.37, -0.29, 0.43))
serial = gi.eri(basis, field=field, storage="full", max_bytes=4096, threads=1)
parallel = gi.eri(basis, field=field, storage="full", max_bytes=4096, threads=2)
np.testing.assert_array_equal(parallel, serial)
a = gi.cholesky_eri(basis, field=field, threads=1)
b = gi.cholesky_eri(basis, field=field, threads=2)
np.testing.assert_array_equal(a.pivots, b.pivots)
np.testing.assert_array_equal(a.vectors, b.vectors)
np.testing.assert_allclose(b.reconstruct(), serial, atol=1e-8, rtol=0)

# Priority 1: moment centre/origin response in the installed wheel.
centres, origin = gi.moment_nuclear_derivatives(
    basis, (1, 0, 0), origin=(0.2, -0.1, 0.3), field=field, threads=1
)
assert centres.shape == (2, 3, 4, 4)
np.testing.assert_allclose(origin[0], -gi.overlap(basis, field=field), atol=2e-14)
np.testing.assert_array_equal(origin[1:], 0)

# Priority 2: all London terms under canonical momentum's spatial operator.
dp = gi.momentum_magnetic_derivatives(basis, "x", field=field, threads=2)
dg = gi.gradient_magnetic_derivatives(basis, "x", field=field, threads=1)
np.testing.assert_array_equal(dp, -1j * dg)

# Priority 3: cross response checked against field differences of first dR.
mixed = gi.overlap_mixed_derivatives(basis, field=field, threads=2)
h = 2e-4


def centre_response(step):
    changed = gi.MagneticField(np.array(field.B) + (0, 0, step), field.gauge_origin)
    return gi.overlap_nuclear_derivatives(basis, field=changed)


coarse = (centre_response(h) - centre_response(-h)) / (2 * h)
fine = (centre_response(h / 2) - centre_response(-h / 2)) / h
np.testing.assert_allclose(mixed[:, :, 2], (4 * fine - coarse) / 3, atol=2e-10)

# Priority 4: packed native centre derivatives, evaluated with two threads.
q = (0, 1, 0, 1)
batch = next(gi.eri_derivative_batches(basis, quartets=[q], field=field, threads=2))
np.testing.assert_array_equal(
    batch.block(0),
    gi.eri_center_derivatives_shell(*[basis.shells[k] for k in q], field=field),
)
print("Installed derivative priorities 1–4: passed")
