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
