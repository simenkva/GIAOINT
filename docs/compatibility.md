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

Release CI covers CPython 3.11--3.14 on current GitHub-hosted Linux and macOS
runners. The C++ core requires a C++20 compiler and CMake 3.20 or newer. Native
Windows is not currently a supported release platform. PySCF/libcint checks are
optional external-oracle tests and are not runtime dependencies.

## Milestone 9 performance defaults

Milestone 9 keeps Python `eri()` defaults at `screening_threshold=0.0` and
`threads=1`. Callers opt into approximation and parallel execution. Faster
shell kernels do not establish a suitable error threshold or thread budget
for all applications. Exact-zero-field dispatch changes the internal
algorithm while preserving the documented numerical tolerance, complex output
dtype, shapes, and conventions.
