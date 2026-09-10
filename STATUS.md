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
- Milestone 6 cached Boys sequences and pair Hermite products, contiguous
  three-axis contraction coefficients, and generation-tagged ERI auxiliary
  scratch without changing the audited MD recurrence.
- Proved complex Coulomb-space Schwarz screening with cached shell-pair bounds,
  an explicit threshold, conservative floating margin, screening statistics,
  and an exact threshold-zero path.
- Optional OpenMP shell-quartet evaluation with static scheduling, thread-local
  workspaces, input-ordered serial callbacks, and runtime capability queries.
- Reproducible JSON-lines benchmarks for all baseline operators, ERI backend
  comparison, and thread scaling, with a checked-in AppleClang arm64 report.
- Benchmark-only direct Cartesian OS prototype agreeing with MD within
  \(5.6\times10^{-17}\) on representative zero- and finite-field primitives.
- Version 0.6.0 release and ASan/UBSan/OpenMP C++ builds, 244 Python tests,
  an OpenMP Python smoke test, and a clean-wheel installation smoke test.

## Current limitations

- The reference engine supports one segmented contraction per shell, while
  the production engine supports one or more contraction rows for overlap and
  every implemented one-electron operator, including nuclear attraction.
- The double-precision direct-moment formula prioritizes clarity and can lose
  accuracy through cancellation for high angular momentum or extreme input.
- The production one-electron paths remain scalar and serial; Milestone 6
  parallelism targets independent ERI shell quartets.
- Milestone 3 operators intentionally compose multiple shifted MD overlaps for
  auditability. This repeats pair setup and is not yet performance-tuned.
- Complex Boys orders above 32 and arguments outside either \(|z|\le160\) or
  the conservative positive asymptotic sector are rejected diagnostically.
- The correctness-first adaptive Boys quadrature is not yet performance-tuned;
  an exponential-sum implementation remains a possible future backend.
- ERIs remain correctness-first and scalar within each shell block. Screening
  and shell-block OpenMP are opt-in; the default remains unscreened and serial.
  The full
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
Milestone 6 adds a JSON-lines release harness for overlap, kinetic, attraction,
and ERIs at zero and finite field. On the recorded AppleClang 21 arm64 run,
p--p--p--p ERI throughput improved from 580.4 to 706.6 blocks/s at zero field
(+21.7%) and from 536.4 to 696.9 blocks/s at finite field (+29.9%). A
76-quartet finite-field p-shell batch scaled to 1.84x, 3.64x, and 6.48x at 2,
4, and 8 OpenMP threads. Full conditions and the MD/OS prototype comparison
are in `benchmarks/results/m6_macos_arm64.md`. These remain informational,
machine-specific baselines rather than noisy CI thresholds.

## Next milestone

Milestone 7: implement analytic nuclear-coordinate and magnetic-field
derivatives with separate ordinary-Gaussian and London-phase terms, then
validate them by multi-step finite differences and symmetry/covariance tests.
