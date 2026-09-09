# Status

## Completed

- Milestone 0 repository assessment.
- Frozen GIAO phase, gauge-origin, normalization, contraction, and Cartesian
  ordering conventions.
- Bra-ket London-phase and complex Gaussian product derivations.
- Coulomb-integral generalization and complex symmetry identities.
- Initial McMurchie-Davidson/Hermite recurrence decision.
- Complex Boys-function development strategy.
- Proposed C++ shell API, NumPy-facing Python API, repository layout,
  dependencies, verification gates, and milestone plan.
- Milestone 1 standard-library reference data model and overlap engine.
- Arbitrary Cartesian primitive overlap using direct complex-center moments.
- Segmented contraction normalization and complete shell-pair blocks.
- 80-digit `mpmath` evaluation plus independent real-axis quadrature checks.
- Analytic, regression, property-based, gauge-origin, translation,
  Hermiticity, zero-field, and continuity tests.
- Milestone 2 C++20 value types, validated shell/basis models, and installable
  CMake target.
- Arbitrary-angular-momentum MD/Hermite primitive overlap with
  Fourier-transformed London Hermites and reusable scratch storage.
- Segmented and general contraction normalization, caller-buffer shell-pair
  evaluation, and basis overlap assembly.
- scikit-build-core packaging, pybind11 bindings, typed NumPy API,
  `complex128` output, strict `out=` validation, and GIL release.
- C++ unit tests, exhaustive and randomized Python-reference comparisons,
  80-digit spot checks, sanitizer verification, and a wheel install smoke test.
- Milestone 3 reusable angular-momentum raising/lowering machinery for
  arbitrary Cartesian moments, gradients, and canonical momentum.
- Canonical kinetic energy including ordinary-Gaussian and London-phase
  derivative terms.
- Physical magnetic kinetic energy
  \(\tfrac12(\mathbf p+\mathbf A_{\mathbf O})^2\), assembled from
  canonical, paramagnetic, and diamagnetic contributions.
- Primitive, shell, and basis-level C++/NumPy APIs for every Milestone 3
  operator, with general contractions, `out=`, and GIL release.
- Independent direct-moment property reference, randomized comparisons,
  analytic cases, symmetry/covariance tests, and real-space finite-difference
  quadrature validation.

## Current limitations

- Nuclear-attraction and electron-repulsion integrals are not yet implemented.
- The reference engine supports one segmented contraction per shell, while
  the production engine supports one or more contraction rows for overlap and
  every Milestone 3 operator.
- The double-precision direct-moment formula prioritizes clarity and can lose
  accuracy through cancellation for high angular momentum or extreme input.
- The production MD one-electron paths are scalar and serial. Shell-pair
  caching, screening, parallelism, and tuned/vectorized kernels are deferred
  to Milestone 6.
- Milestone 3 operators intentionally compose multiple shifted MD overlaps for
  auditability. This repeats pair setup and is not yet performance-tuned.
- Complex Boys-function production algorithm remains a gated Milestone 4
  decision after prototypes are tested over the reachable complex domain.
- ERI screening bounds with London factors have not been proved or enabled.
- Cartesian functions only are planned for the initial engine.

## Maximum tested angular momentum

Reference and C++ primitive overlap have exhaustive Cartesian-component
comparison and Hermiticity coverage through total angular momentum 4, plus
randomized double- and 80-digit-reference coverage through 6. Primitive
normalization is tested for every component through 6. Shell-block comparisons
include pairs through \(L=5\). Milestone 3 primitive property comparisons are
randomized through \(L=6\), with contracted property blocks through f-d. The
implementation has no hard-coded angular-momentum ceiling.

## Known numerical issues

- Complex Boys arguments can have negative real parts.
- Separating London pair prefactors from Boys values can overflow or lose
  relative accuracy even when their product is finite.
- Upward and downward Boys recurrences have different stability regions.
- High angular momentum and diffuse/tight exponent combinations can amplify
  cancellation in Hermite contractions.
- Direct complex-center polynomial expansion can suffer catastrophic
  cancellation long before overflow; it remains an oracle for the tested
  domain, with difficult cases compared separately at high precision.
- The scalar MD recurrence can also amplify cancellation for high angular
  momentum, extreme exponent ratios, or large geometry/field products. No
  guaranteed production envelope has yet been established outside the tested
  domain.
- High-order moments introduce binomial growth and additional cancellation;
  their reliable double-precision domain has not yet been mapped.

## Benchmark status

The first release-build overlap baseline is recorded on the development
machine with AppleClang 21: a finite-field f-f shell pair with 4x4 primitives
evaluated 20,000 times at approximately 8.2k shell blocks/s and 0.82M
integrals/s. This is an informational scalar baseline, not a CI threshold.
No separate Milestone 3 property benchmark is recorded; the shifted-overlap
composition is deliberately correctness-first. Milestone 6 adds stable
performance-regression tracking and optimization.

## Next milestone

Milestone 4: implement and map a stable complex Boys-function algorithm, then
build nuclear-attraction MD auxiliaries and contracted shell-pair/nucleus
drivers with analytic, high-precision, and zero-field validation.
