"""macOS conda regression: NumPy and GIAOINT must share one OpenMP runtime.

Run in a conda environment with OpenMP OpenBLAS, using each import order:
    python tests/conda_openmp_smoke.py numpy-first
    python tests/conda_openmp_smoke.py giao-first
"""

import ctypes
import sys
from importlib import import_module
from pathlib import Path


def main():
    mode = sys.argv[1]
    orders = {
        "numpy-first": ("numpy", "giao_integrals"),
        "giao-first": ("giao_integrals", "numpy"),
    }
    for name in orders[mode]:
        import_module(name)
    np = import_module("numpy")
    gi = import_module("giao_integrals")
    prefix = Path(sys.prefix).resolve()
    assert (prefix / "conda-meta").is_dir()
    info = gi.build_info()
    print(mode, info, flush=True)
    assert info["openmp_enabled"]
    runtime = Path(info["openmp_runtime_path"]).resolve()
    assert runtime.is_relative_to(prefix), runtime

    # Initialize BLAS as well as GIAO's OpenMP team: a second runtime aborts here.
    matrix = np.ones((64, 64))
    np.testing.assert_array_equal(matrix @ matrix, np.full((64, 64), 64.0))
    basis = gi.Basis(
        [
            gi.Shell((0, 0, 0), 0, [0.8], [1.0]),
            gi.Shell((0.7, 0.1, -0.2), 1, [0.6], [1.0]),
        ]
    )
    field = gi.MagneticField((0.05, -0.03, 0.2))
    serial = gi.cholesky_eri(basis, field=field, threads=1)
    parallel = gi.cholesky_eri(basis, field=field, threads=2)
    np.testing.assert_array_equal(serial.vectors, parallel.vectors)
    np.testing.assert_array_equal(serial.pivots, parallel.pivots)

    # Inspect loaded images without initializing any additional runtime.
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    runtimes = set()
    for index in range(dyld._dyld_image_count()):
        path = Path(dyld._dyld_get_image_name(index).decode())
        if path.name in {"libomp.dylib", "libiomp5.dylib", "libgomp.dylib"}:
            runtimes.add(path.resolve())
    assert runtimes == {runtime}, runtimes
    print("One shared runtime; serial/parallel factors and pivots match", flush=True)


if __name__ == "__main__":
    main()
