# Release checklist

1. Update the version in `pyproject.toml` and `CMakeLists.txt`; the generated
   header supplies the Python extension and benchmark output. Update
   `CHANGELOG.md` and `STATUS.md`.
2. Run Ruff and Black checks, the complete Python suite, and release plus
   ASan/UBSan C++ tests from a clean checkout.
3. Run the optional PySCF/libcint zero-field oracle in the `pyscf` environment.
4. Build an sdist and wheel without build isolation, install the wheel into a
   clean environment, and run import plus example smoke tests.
5. Install the C++ library into a staging prefix and compile a downstream
   `find_package(giao_integrals CONFIG REQUIRED)` consumer.
6. Record benchmark JSONL on a quiet machine. Compare against the matching
   machine/compiler baseline and investigate material changes; do not compare
   unlike hosts.
7. Confirm the release notes state the numerical domain, maximum tested angular
   momentum, benchmark environment, supported platforms, and known limitations.
8. Tag `vX.Y.Z`. Inspect wheel and sdist artifacts from the tag workflow before
   publishing. Publication remains an explicit maintainer action.

The scheduled benchmark is informational; release decisions use archived
controlled-host results.
