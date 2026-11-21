# 0.10.0 release candidate

Prepared on 21 November 2026. This release contains the derivative priorities
1–4 and VICE's requested docstring corrections. PyPI publication requires a
new explicit approval of **v0.10.0**; no tag or upload is created by preparation.
Later Cholesky/assembly/TD4.0 changes are excluded from this release candidate.

## Evidence

- Local full Python suite: 621 passed, seven existing BSE/jsonschema warnings.
- Installed native smoke test covers moment nuclear/origin response, property
  field response, mixed nuclear/field response and streamed ERI derivatives.
- Docstring regression tests check actual arguments, shapes and included terms.
- Black, Ruff, local documentation links/anchors/Python fences and diff checks pass.
- Prior derivative numerical, libcint and memory/timing evidence is in
  [the derivative validation report](derivative_validation.md).

VICE independently reported verification of main `4e2bb57`: H2O/cc-pVTZ,
B=(0.05,-0.03,0.2), off-centre gauge and off-atom moment origin. Its derivative
families agreed with central differences to 2e-9–2e-8 relative; moment-origin
response agreed to 9e-13, streamed ERI derivatives matched shell calls exactly,
and an RHF electric-field gradient agreed to 1.8e-9 after using the moment
response. These are downstream measurements supplied with the request, not
additional GIAOINT measurements.

## Artifact and publication checks

The existing matrix builds CPython 3.11–3.14 for macOS 15+ arm64/x86_64 and
manylinux x86_64/aarch64, plus an sdist. Every installed wheel/source artifact
runs `tests/wheel_smoke.py` with one derivative call per priority. [CI run 44890477090](https://github.com/simenkva/GIAOINT/actions/runs/44890477090)
and [artifact run 44890479022](https://github.com/simenkva/GIAOINT/actions/runs/44890479022)
both pass at source commit `7910655`. This covers serial/OpenMP Python and
native tests, Linux sanitizers, conda, source installs and all four repaired
wheel architectures. Each wheel and the source archive passed the expanded
installed derivative smoke test. Publication was skipped on the branch run.
All five artifact groups (four wheel platforms plus sdist) are available on
the artifact run for review.

The existing trusted publisher and protected `pypi` environment are retained.
Publication still requires an exact `PYPI_APPROVED_TAG=v0.10.0`, an approved
version tag and the environment review. Those gates are left unset pending
explicit approval. Version 0.9.0 remains the published release during preparation.
