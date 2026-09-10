# Contributing

The numerical core is C++20 and the Python extension uses pybind11. Runtime
code must not depend on the pure-Python reference implementation.

For Python development in the requested conda environment:

```console
conda activate pyscf
python -m pip install -r requirements-dev.txt
python -m pip install --no-build-isolation -e .
python -m pytest -q
python -m ruff check python reference tests examples benchmarks/compare_results.py
python -m black --check python reference tests examples benchmarks/compare_results.py
```

For the standalone core:

```console
cmake --preset release
cmake --build --preset release
ctest --preset release
cmake --preset sanitizers
cmake --build --preset sanitizers
ctest --preset sanitizers
```

Ninja is required by the checked-in presets. Direct CMake configuration works
with other generators. Keep changes to phase, normalization, ordering, units,
and symmetry coupled to mathematical documentation and independent tests.
Randomized failures should become deterministic regressions.

Build distributions with `python -m build`. Wheels are produced by the tag or
manual GitHub workflow; it intentionally uploads artifacts without publishing
them. Follow `docs/release_checklist.md` for a release.
