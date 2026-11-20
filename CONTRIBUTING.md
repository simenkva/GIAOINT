# Contributing

The numerical core is C++20 and the Python extension uses pybind11. Runtime
code must not depend on the pure-Python reference implementation.

For Python development in a virtual environment:

```console
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pytest -q
python -m ruff check python reference tests examples benchmarks/compare_results.py benchmarks/cholesky_threads.py
python -m black --check python reference tests examples benchmarks/compare_results.py benchmarks/cholesky_threads.py
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

Install the frontend with `python -m pip install build`, then build
distributions with `python -m build`.
The tag/manual workflow prepares repaired wheels and an sdist. Its publication
job requires approval for the exact tag, successful builds, and the protected
`pypi` environment; a tag alone does not publish. Follow the
[release checklist](docs/release_checklist.md).

Editable installs update Python sources immediately but do not rebuild native
code automatically. Re-run `pip install -e .` after C++ edits. Explicitly test
`-Ccmake.define.GIAO_ENABLE_OPENMP=OFF` and `ON`; AUTO is the install default.
Wheel smoke tests must run against the repaired, installed artifact outside
the source package, with `GIAO_REQUIRE_BUNDLED_OMP=1`.
