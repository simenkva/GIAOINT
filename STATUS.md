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
- Milestone 7 analytic first derivatives for overlap, canonical kinetic,
  physical magnetic kinetic, nuclear attraction, and four-center ERIs using
  normalization-preserving angular raising/lowering identities.
- Separate ordinary-Gaussian and London-phase center response, analytic field
  response of every London factor, and the explicit angular-momentum and
  diamagnetic response of the physical magnetic Hamiltonian.
- Separate attraction basis-center and potential-center derivatives, with the
  latter evaluated analytically through integration by parts and spatial
  orbital gradients.
- Primitive, contracted shell-block, one-electron basis-matrix, and ERI
  shell-quartet derivative APIs with leading center/axis dimensions, strict
  NumPy `out=` validation, and GIL release.
- Two-step central finite differences with Richardson extrapolation,
  Hermiticity, zero-field translation, gauge-origin, contribution-separation,
  and C++ smoke tests for the derivative paths.
- Version 0.7.0 release with 270 Python tests, clean release and
  ASan/UBSan/OpenMP C++ builds, a clean-wheel installation smoke test, and
  unchanged PySCF 2.8.0/libcint zero-field agreement.
- Milestone 8 BSD-3-Clause licensing, expanded package metadata, stable
  examples, CMake presets, and documented source, wheel, and downstream CMake
  consumption paths.
- GitHub Actions matrices for CPython 3.11--3.14 on Linux and macOS, native
  release, ASan/UBSan, OpenMP, clang-tidy, tag-triggered wheel/sdist artifacts,
  and scheduled informational benchmarks.
- A pre-1.0 compatibility policy with tested Python warnings and an installed
  C++ deprecation macro; units, ordering, ownership, shapes, and exception
  categories are part of the documented compatibility boundary.
- Machine-readable benchmark comparison with optional regression thresholds,
  explicit missing/malformed-case failures, a release checklist, changelog,
  contribution guide, and a decoupled spherical-transform design.
- Version 0.8.0 validation with 280 Python tests, Ruff and Black checks,
  release, ASan/UBSan, and OpenMP C++ builds, clean sdist and wheel builds,
  clean-wheel example smoke tests, an installed-package downstream CMake
  consumer, and unchanged PySCF 2.8.0/libcint zero-field agreement.

- Milestone 9 iterative six-index ERI auxiliaries, collapsed real arithmetic
  at exact zero field, a direct ssss seed, and Gaussian-pair reuse outside
  Cartesian loops. The general path retains the finite-field recurrence and
  primitive accumulation order; screening/threading defaults remain unchanged.
- Milestone 9 validation with 376 Python tests, release/ASan/UBSan/OpenMP C++
  checks, Ruff/Black, 192 frozen recursive values, 128 randomized real/general
  comparisons, and PySCF/libcint agreement with maximum ERI error `1.704e-14`.
  This work is unreleased; the package version remains 0.8.0.

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
- ERIs use an iterative complex six-index kernel at finite field and a
  collapsed real kernel at exact zero field. Work within each block remains
  serial; screening and shell-block OpenMP are opt-in. Both paths retain the
  original 4,000,000-entry six-index admission cap and combined order 32
  limit. The optimized general kernel still accounts for 93% of finite-field
  dddd top-of-stack samples; see `benchmarks/results/m9_macos_arm64.md`.
  Basis-scale zero-field ERIs remain slower than libcint.
- Derivative shell drivers are correctness-first compositions of shifted
  production kernels. They are serial, do not yet use derivative screening,
  and may repeat primitive setup.
- `Basis` has no atom-to-shell ownership map. Basis-center and attraction
  potential-center derivatives are therefore returned separately for callers
  to assemble into atom derivatives.
- Mixed nuclear/magnetic derivatives and derivatives of the general
  moment/gradient/momentum property APIs are not yet exposed.
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

Milestone 7 overlap center and field derivatives are randomized through
angular momentum 4 on each primitive. Canonical and physical kinetic and
nuclear-attraction primitive derivatives are checked through combined angular
degree 3, with contracted s--p blocks and one-electron basis matrices. ERI
derivatives are checked for finite-field primitive and s--s--p--s shell cases
through combined degree 2. Since analytic differentiation raises one angular
component, derivative calls require the corresponding base kernel to accept
one additional order.

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

Milestone 9 records matched before/after runs on Apple M2 Pro, macOS 26.6.2,
AppleClang 21, Release, one thread, screening disabled. The dddd workload
improved from 5.47 to 89.37 blocks/s at zero field (16.34x), and 5.48 to
23.24 blocks/s at finite field (4.24x). The water/cc-pVDZ ERI call improved
from 691.7 to 238.8 ms (2.90x). New OpenMP runs reach 5.86x at eight threads
on the 76-quartet finite-field p-shell batch. Raw JSONL, profiling counts,
intermediate measurements, and the basis-scale comparison are in
`benchmarks/results/m9_macos_arm64.md`. These are local measurements without
CPU pinning, not portable performance guarantees.

## Next milestone

Milestones 0--9 are complete. A separately validated spherical transformation
layer and preparation of a 1.0 API freeze remain candidate work. Mixed
nuclear/magnetic derivatives remain deferred. Further finite-field ERI work
can use the remaining costs recorded in `docs/eri_performance_plan.md`.
