# Compatibility and deprecation policy

`giao-integrals` follows semantic versioning. The project is still below 1.0,
but releases from 0.8 onward treat documented Python functions, installed C++
headers, result shapes, Cartesian ordering, and exception categories as the
public API.

Patch releases preserve that API. Before 1.0, a minor release may make a
breaking change when the numerical contract cannot be repaired compatibly.
Where practical, the old spelling or behavior remains for at least one minor
release and emits a warning. From 1.0 onward, breaking public changes require a
major release.

The following are not compatibility promises:

- symbols beginning with an underscore;
- implementation details in `src/` and the pure-Python reference package;
- exact floating-point bits, performance, or diagnostic error estimates;
- benchmark JSON fields added in a backward-compatible way;
- experimental features explicitly labeled as such in their documentation.

Numerical changes that remain within the documented error envelope are not API
breaks. A change to phase convention, normalization, units, Cartesian order,
tensor axis order, or finite-field symmetry is a breaking change even if the
function signature is unchanged.

## Deprecation mechanism

Python compatibility shims use `giao_integrals._compat.deprecated`, which
emits `FutureWarning` with the replacement and earliest removal version. C++
declarations use `GIAO_DEPRECATED` from `giao_integrals/deprecation.hpp` with
the same information. Every deprecation must appear in `CHANGELOG.md` and have
a test for the old route. Internal symbols may be removed without a
deprecation period.

## Supported platforms

Configured release CI targets CPython 3.11–3.14 manylinux_2_28 x86_64/aarch64
and macOS 15+ arm64/x86_64. Local 0.9 validation covers CPython 3.13 on
macOS arm64. Remote wheel CI has passed for every configured platform and
Python version; see [release validation](release_0_9_report.md). The C++ core requires a C++20
compiler and CMake 3.20 or newer. Native Windows is not currently a supported release platform. PySCF/libcint checks are
optional external-oracle tests and are not runtime dependencies.

## Version 0.9 performance defaults

Version 0.9 deliberately changes ERI threading from opt-in `threads=1` to
`threads=None`, resolved through the process override, GIAO_NUM_THREADS,
OMP_NUM_THREADS, and physical cores. Screening remains `0.0`. Explicit
`threads=1` preserves serial execution. Every threaded Python entry point is
checked for bitwise equality at one, two, and the runtime maximum thread
count, with/without screening and with/without a field. Cholesky vectors and
pivots are compared exactly. Primitive reduction order, AO conventions,
normalization, dtype, and shapes are unchanged.

CMake OpenMP support defaults to AUTO, with explicit ON and OFF available.
See [installation](installation.md) for fallback warnings, runtime conflicts,
and independent BLAS thread settings. C++ `EriEvaluationOptions` retains its
explicit `thread_count` field and serial default; `None` is a Python policy.
