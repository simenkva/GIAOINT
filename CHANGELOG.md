# Changelog

## 0.8.0 — 2026-09-10

Milestone 8 hardens the 0.7 integral engine for repeatable distribution. It
adds BSD-3-Clause licensing, package metadata, examples, a compatibility and
deprecation policy, CMake presets, Linux/macOS CI, sanitizer and clang-tidy
jobs, CPython 3.11--3.14 wheel builds, source-distribution validation, benchmark
history tooling, and a release checklist. The optional spherical layer is
designed as a transform over Cartesian blocks and is not yet implemented.

The numerical domain is unchanged from 0.7. Complex Boys orders 0--32 are
supported within \(|z|\le160\) or the documented conservative positive
asymptotic sector. Nuclear attraction and ERIs accept combined Cartesian order
through 32, subject to the 4,000,000-entry ERI workspace cap. See `STATUS.md`
for the operator-specific tested angular momenta.

The most recent controlled performance record is the Milestone 6 AppleClang 21
arm64 run: p-p-p-p ERIs reached 706.6 blocks/s at zero field and 696.9 blocks/s
at finite field, with 6.48x batch scaling at eight OpenMP threads. These are
machine-specific observations, not promised performance. No kernel algorithm
changed in 0.8.0.

Remaining limitations include Cartesian-only output, no atom-to-shell ownership
map, no mixed nuclear/magnetic derivatives, a correctness-first serial
derivative implementation, scalar work within ERI blocks, and the documented
high-angular-momentum cancellation risks.

## 0.7.0 — 2026-09-10

Added analytic basis-center, attraction-center, and magnetic-field first
derivatives for the baseline Hamiltonian integrals and ERI shell quartets.

## 0.6.0 — 2026-09-09

Added ERI caching, proved complex Schwarz screening, optional OpenMP block
parallelism, and reproducible JSON-lines benchmarks.
