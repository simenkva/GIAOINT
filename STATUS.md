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
- Milestone 4 dual 80+-digit complex Boys references using defining
  quadrature and the entire hypergeometric representation.
- Production Boys sequences through order 32 with compensated series,
  adaptive embedded quadrature, positive-sector asymptotics, scaled values,
  dispatch/error diagnostics, and typed numerical failures.
- Cancellation-safe pairing of scaled Boys values with London damping for
  negative-real arguments.
- Complex McMurchie--Davidson nuclear-attraction auxiliaries, arbitrary
  Cartesian primitive integrals, contracted shell-pair/nucleus drivers, and
  basis matrices.
- Independent high-precision Obara--Saika attraction reference, analytic s-s,
  randomized finite-field, negative-real, Hermiticity, gauge-origin,
  zero-field, and allocation-reuse tests.
- Version 0.4.0 release and ASan/UBSan C++ builds, 225 Python tests, and a
  clean-wheel installation smoke test.
- Zero-field normalized Cartesian s/p/d comparison against PySCF 2.8.0 and
  bundled libcint, with maximum overlap and attraction errors of
  \(4.44\times10^{-16}\) and \(3.95\times10^{-14}\).
- Milestone 5 unscreened four-center McMurchie--Davidson ERIs with the full
  six-index complex London auxiliary, arbitrary Cartesian primitives, general
  contractions, caller-buffer shell quartets, and reusable workspace.
- Exact finite-field quartet canonicalization using only pair exchange and
  conjugate double reversal, plus a C++ block-consumer interface.
- Packed canonical Python batch iteration by default and an opt-in full
  `(nao, nao, nao, nao)` tensor protected by an explicit byte limit.
- Independent 80-digit Obara--Saika ERI reference, an exhaustive 91-case
  Cartesian core through combined degree two, selected cases through combined
  degree ten, scaled negative-real validation, symmetry-negative tests, and
  streamed/full equality.
- Version 0.5.0 release and ASan/UBSan C++ builds, 238 Python tests, and a
  clean-wheel installation smoke test.
- PySCF 2.8.0/libcint zero-field normalized Cartesian s/p/d ERI comparison
  with maximum error \(1.72\times10^{-14}\).

## Current limitations

- The reference engine supports one segmented contraction per shell, while
  the production engine supports one or more contraction rows for overlap and
  every implemented one-electron operator, including nuclear attraction.
- The double-precision direct-moment formula prioritizes clarity and can lose
  accuracy through cancellation for high angular momentum or extreme input.
- The production MD one-electron paths are scalar and serial. Shell-pair
  caching, screening, parallelism, and tuned/vectorized kernels are deferred
  to Milestone 6.
- Milestone 3 operators intentionally compose multiple shifted MD overlaps for
  auditability. This repeats pair setup and is not yet performance-tuned.
- Complex Boys orders above 32 and arguments outside either \(|z|\le160\) or
  the conservative positive asymptotic sector are rejected diagnostically.
- The correctness-first adaptive Boys quadrature is not yet performance-tuned;
  an exponential-sum implementation remains a possible Milestone 6 backend.
- ERI screening bounds with London factors have not been proved or enabled.
- ERIs are correctness-first, scalar, serial, and unscreened. The full
  six-index auxiliary has a 4,000,000-entry workspace cap; combined Cartesian
  order above 32 is rejected.
- Cartesian functions only are planned for the initial engine.

## Maximum tested angular momentum

Reference and C++ primitive overlap have exhaustive Cartesian-component
comparison and Hermiticity coverage through total angular momentum 4, plus
randomized double- and 80-digit-reference coverage through 6. Primitive
normalization is tested for every component through 6. Shell-block comparisons
include pairs through \(L=5\). Milestone 3 primitive property comparisons are
randomized through \(L=6\), with contracted property blocks through f-d. The
overlap and Milestone 3 implementations have no hard-coded angular-momentum
ceiling.

Complex Boys values are tested through order 32. Nuclear attraction is
randomized against the independent reference through \(L=4\) on each primitive
center (combined auxiliary order 8), with contracted p--d shell blocks. The
production attraction path accepts combined order through 32.

Primitive ERIs are exhaustively compared with the high-precision reference for
all Cartesian power distributions through combined degree two (91 cases),
with selected finite-field cases through combined degree ten. Production
accepts combined order through 32 subject to the auxiliary workspace cap.

## Known numerical issues

- Very strong fields can produce negative-real Boys arguments outside the
  verified direct disk; these fail rather than returning an unmeasured value.
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

Milestone 6: establish measured ERI and one-electron performance baselines,
then add pair caching, contraction-loop tuning, proved screening, and optional
shell-block parallelism without changing the Milestone 5 unscreened oracle.
