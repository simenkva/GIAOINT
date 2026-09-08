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

## Current limitations

- There is no C++ core, pybind11 extension, NumPy production API, or CMake
  build yet.
- The reference engine supports one segmented contraction per shell and
  overlap integrals only.
- The double-precision direct-moment formula prioritizes clarity and can lose
  accuracy through cancellation for high angular momentum or extreme input.
- Complex Boys-function production algorithm remains a gated Milestone 4
  decision after prototypes are tested over the reachable complex domain.
- ERI screening bounds with London factors have not been proved or enabled.
- Cartesian functions only are planned for the initial engine.

## Maximum tested angular momentum

Primitive overlap has exhaustive Cartesian-component Hermiticity coverage
through total angular momentum 4 and randomized/high-precision coverage
through 6. Primitive normalization is tested for every component through 6.
The implementation has no hard-coded angular-momentum ceiling.

## Known numerical issues

- Complex Boys arguments can have negative real parts.
- Separating London pair prefactors from Boys values can overflow or lose
  relative accuracy even when their product is finite.
- Upward and downward Boys recurrences have different stability regions.
- High angular momentum and diffuse/tight exponent combinations can amplify
  cancellation in Hermite contractions.
- Direct complex-center polynomial expansion can suffer catastrophic
  cancellation long before overflow; Milestone 2 comparisons must classify
  such cases rather than weaken global tolerances.

## Benchmark status

No benchmark applies to the deliberately slow reference engine. Milestone 2
establishes the first production overlap baseline; Milestone 6 adds
performance-regression tracking.

## Next milestone

Milestone 2: implement the C++20 MD/Hermite overlap engine, shell data model,
CMake/scikit-build-core packaging, pybind11 bindings, NumPy API, and exhaustive
comparisons against the Milestone 1 reference.
