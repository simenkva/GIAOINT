# Release checklist

1. Update the version in pyproject.toml and CMakeLists.txt; update CHANGELOG,
   STATUS and the validation report. Do not create a publication tag until
   the maintainer explicitly approves the release.
2. Run Ruff, Black and the complete Python and C++ suites for OpenMP OFF,
   OpenMP ON and ASan/UBSan (-O1, debug symbols). Verify thread equality with
   and without screening and fields, including Cholesky vectors and pivots.
3. Verify plain isolated `pip install .` and `pip install -e .` in fresh venvs.
   Record build_info(). Test AUTO fallback and strict ON when unavailable.
   Editable C++ changes require a repeat install, as documented.
4. Build an sdist; install it with isolated pip in a clean environment.
5. Run wheels.yml manually. Inspect every CPython 3.11–3.14 Linux x86_64/aarch64
   and macOS arm64/x86_64 artifact. Runtime repair and installed smoke tests
   must pass on every target. Windows wheels are omitted for 0.9. macOS wheel
   baseline is 15; Linux wheels use manylinux_2_28.
6. Check that auditwheel/delocate bundled the runtime and that license notices
   accompany it. Verify runtime paths point inside the installed wheel.
   Document the duplicate-libomp risk; never suggest KMP_DUPLICATE_LIB_OK.
7. Record controlled-host Ne Cholesky timings using
   benchmarks/cholesky_threads.py; keep BLAS thread settings and profiling
   overhead with the results. Do not compare unlike or busy hosts as a speedup
   guarantee. Investigate numerical or bitwise thread-equality regressions.
8. Confirm PyPI project ownership/name (`giao-integrals`) and configure its
   trusted publisher for simenkva/GIAOINT, wheels.yml, environment pypi.
   Protect that environment with required reviewers.
9. Ask the maintainer for explicit approval of the exact release version.
   Only after approval set PYPI_APPROVED_TAG to vX.Y.Z and create/push that
   tag. Publication also requires all wheel/sdist jobs and environment approval.
   A tag alone does not authorize publication. Remove the approval variable
   afterward if desired; a different tag always requires a new value.

The optional PySCF/libcint oracle and native CMake install/downstream-consumer
check remain useful additional release checks. Scheduled benchmarks are
informational rather than release approval.
