# Milestone implementation plan

## Project rules

Each milestone begins by naming its formulas and ends only after its independent
checks pass. A later milestone may refine an earlier interface through a
documented compatibility change, but it may not silently alter the GIAO phase,
normalization, ordering, or complex-conjugation rules.

`STATUS.md` is updated with completed scope, limitations, maximum tested
angular momentum, numerical issues, benchmark state, and the next milestone.

## Milestone 0: specification

### Delivered

- repository assessment and proposed layout;
- frozen mathematical conventions and intermediate derivations;
- recurrence and complex Boys strategy;
- C++ and Python API proposal;
- verification hierarchy and tolerances;
- references, risks, open decisions, and this milestone plan.

### Exit gate

The maintainer reviews the frozen phase convention, contraction convention,
Cartesian ordering, MD-first decision, public API shapes, and unresolved
decisions below. No numerical source is included in this milestone.

## Milestone 1: independent Python reference

Status: completed on 2026-09-08.

### Scope

- immutable Python records for primitives and magnetic context;
- primitive normalization and Cartesian enumeration;
- London wave vectors and pair data;
- arbitrary-angular-momentum primitive overlap by direct complex-center
  polynomial moments;
- simple segmented contraction and shell-pair assembly;
- mpmath s-s and direct-quadrature checks.

### Formulas

Sections 2–6 of `mathematical_specification.md`, especially
\(\mathbf q=\boldsymbol\kappa_B-\boldsymbol\kappa_A\),
\(\mathbf P'=\mathbf P-i\mathbf q/(2p)\), and the even complex-centered
Gaussian moments.

### Tests and exit gate

- analytic s-s values and normalized self-overlaps;
- randomized double vs 80-digit calculations;
- Hermiticity, zero-field, gauge-origin, translation, and continuity tests;
- exhaustive Cartesian components through \(L=4\), randomized through
  \(L=6\);
- no C++ extension import in the reference package.

## Milestone 2: C++ overlap engine and binding

Status: completed on 2026-09-09.

### Scope

- CMake/scikit-build-core project and C++20 value types;
- shell validation, component enumeration, primitive normalization, and
  contraction normalization;
- MD Hermite coefficients and Fourier-transformed London Hermites;
- primitive contraction, caller-buffer shell-pair driver, basis overlap
  matrix, reusable workspace;
- pybind11 shell and basis APIs with `complex128`, `out=`, and GIL release.

### Tests and exit gate

- C++ building-block tests and all Milestone 1 comparisons;
- zero-field checks against at least one external engine when available;
- ASan/UBSan clean test runs;
- wheel installation and Python API tests on supported platforms;
- overlap benchmark baseline with no heap allocations in primitive inner
  loops.

The exit gate passed on macOS arm64 with AppleClang 21 and Python 3.13. The
release and ASan/UBSan C++ tests passed, all 83 Python/reference tests passed,
and a built wheel installed and imported in a clean virtual environment. The
test suite includes exhaustive component comparisons through (L=4),
randomized production-vs-80-digit checks through (L=6), and shell blocks
through (L=5). PySCF, libcint, and Libint were not installed, so the optional
zero-field external-engine comparison was not run. The informational
finite-field f-f 4x4-primitive baseline was about 8.2k shell blocks/s on the
development machine.

## Milestone 3: kinetic and property machinery

Status: completed on 2026-09-09.

### Scope

- shared multiplication and derivative raising/lowering operators;
- canonical kinetic, coordinate moments through caller-selected low order,
  gradient, and momentum;
- physical magnetic one-electron kinetic combination, including paramagnetic
  and diamagnetic terms, so gauge behavior can be tested as a whole.

### Formulas

Differentiate both the ordinary Gaussian and
\(e^{-i\boldsymbol\kappa\cdot r}\). Express coordinate powers and derivatives
as shifted angular-momentum combinations; avoid operator-specific shell cases.

### Tests and exit gate

- low-order analytic cases and Python reference comparisons;
- Hermiticity for self-adjoint operators;
- canonical-kinetic gauge dependence documented and full magnetic operator
  gauge-origin behavior verified;
- numerical differentiation of primitive functions at sampled points;
- no duplicated s/p/d/f kernels.

The exit gate passed with analytic normalized and displaced s-Gaussian cases,
randomized primitive comparisons through \(L=6\), contracted shell comparisons
through f-d, and general-contraction ordering checks. Matrix tests cover
Hermiticity of moments, momentum, canonical kinetic energy, and physical
magnetic kinetic energy; gradient anti-Hermiticity; the zero-field reduction;
physical gauge-origin invariance; canonical gauge dependence; coordinate
origin identities; and translation rephasing. Independent Gauss-Hermite
real-space quadrature with finite-difference first and second derivatives
checks both canonical and physical magnetic kinetic values. All 111 Python
tests and the release and ASan/UBSan C++ tests pass on the development
platform. The optional zero-field external-engine check remains unavailable
because PySCF, libcint, and Libint are not installed.

## Milestone 4: nuclear attraction and complex Boys

### Scope

- dual high-precision Boys references;
- production `F_0..F_nmax` prototype with region dispatch, error diagnostics,
  and scaled path;
- MD complex Coulomb auxiliaries for nuclear attraction;
- contracted shell-pair/nucleus driver.

### Tests and exit gate

- complex-plane error maps, boundary continuity, recurrence residuals, and
  conjugation checks;
- negative-real-part and cancellation stress tests;
- analytic s-s nuclear attraction and zero-field external comparisons;
- randomized reference comparisons through the stated angular momentum;
- the production Boys algorithm and supported domain documented from measured
  errors. If no prototype meets the target, the milestone remains open rather
  than relaxing tolerances.

The exit gate passed with two 80--100 digit reference definitions, production
region diagnostics and scaled/unscaled tests through order 32, a reproducible
120-point reachable-argument map, and adversarial complex-plane cases. The
physical map's observed maximum absolute and relative errors were
\(2.28\times10^{-15}\) and \(1.06\times10^{-14}\). Nuclear attraction uses
the London MD auxiliary recurrence and a cancellation-safe scaled seed; an
independent Obara--Saika oracle validates randomized primitives through
\(L=4\) per center and contracted p--d blocks. Analytic s-s, negative-real,
Hermiticity, zero-field, gauge-origin, validation, output-buffer, and warmed
allocation tests pass. A zero-field Cartesian s/p/d comparison against PySCF
2.8.0 and its bundled libcint observed maximum overlap and attraction errors
of \(4.44\times10^{-16}\) and \(3.95\times10^{-14}\), respectively, after
explicitly converting PySCF's shared radial Cartesian normalization.
The final gate contains 225 passing Python tests plus passing release and
ASan/UBSan C++ test builds and a 0.4.0 wheel-install smoke test.

## Milestone 5: four-center ERIs

### Scope

- unscreened primitive MD ERIs, contraction, and caller-buffer shell quartet;
- basis quartet scheduler using only derived complex symmetries;
- packed batch iterator and C++ consumer interface;
- opt-in full NumPy tensor with allocation guard.

### Tests and exit gate

- ssss formula, high-precision small cases, and Python reference cases;
- pair exchange, conjugate reversal, and finite-field negative symmetry tests;
- zero-field comparisons with a conventional engine;
- full-tensor and streamed-block equality for small bases;
- arbitrary Cartesian recurrence structure, exhaustively tested through the
  milestone's documented limit;
- no screening until unscreened correctness passes.

The exit gate passed with the unscreened six-index London MD recurrence,
segmented and general contraction, caller-buffer shell quartets, the exact
four-member complex-symmetry scheduler, a C++ consumer, packed Python batches,
and an explicitly byte-guarded full tensor. An independent 80-digit OS oracle
covers every Cartesian distribution through combined degree two plus selected
cases through degree ten. The PySCF 2.8.0/libcint s/p/d tensor comparison
observed a maximum error of \(1.72\times10^{-14}\). The final gate contains 238
passing Python tests plus passing release and ASan/UBSan C++ builds and a 0.5.0
wheel-install smoke test.

## Milestone 6: measured performance baseline

### Scope

- scratch reuse, shell-pair caches, contraction-loop tuning, and compact table
  extents;
- proved modulus screening bound and threshold-controlled implementation;
- MD vs OS/HGP prototype comparison for representative shell classes;
- optional OpenMP at shell-block level with thread-local workspaces;
- stable benchmark schema and dedicated-runner regression reports.

### Tests and exit gate

- every optimization reproduces scalar reference blocks;
- tightened-screening convergence to unscreened values;
- deterministic serial output and documented parallel reproducibility;
- zero-field and finite-field throughput for overlap, kinetic, attraction, and
  ERIs across low and moderate angular momentum;
- thread-scaling report and allocation profile.

The exit gate passed with cached Boys and pair-Hermite data, generation-tagged
auxiliary scratch, contiguous pair contraction coefficients, the proved
complex Coulomb Schwarz screen, and opt-in OpenMP with thread-local workspaces
and input-ordered callbacks. The JSON-lines suite covers every baseline
operator at zero and finite field. On the recorded AppleClang arm64 run,
p--p--p--p ERIs improved by 21.7% at zero field and 29.9% at finite field;
the 76-quartet OpenMP workload reached 1.84x, 3.64x, and 6.48x on 2, 4, and 8
threads. A benchmark-only direct OS prototype agreed within
\(5.6\times10^{-17}\) and was mixed across shell classes, so MD remains the
production oracle while a contracted HGP prototype remains a future measured
option.

## Milestone 7: derivatives

### Scope

- first nuclear derivatives;
- first magnetic-field derivatives;
- derivative shell blocks and basis drivers;
- mixed nuclear/magnetic derivatives only after the first-order paths pass.

### Tests and exit gate

- separate ordinary-Gaussian and London-phase term tests;
- multi-step central finite differences with extrapolation;
- translational and gauge covariance of derivative tensors;
- derivative symmetry identities and zero-field limits;
- maximum tested angular momentum and difficult numerical regions recorded.

## Milestone 8: production hardening

### Scope

- stable documentation, examples, package metadata, license, CI, wheels, and
  versioning;
- API compatibility policy and deprecation mechanism;
- fuzz/property expansion, sanitizer and static-analysis coverage;
- benchmark history and release checklist;
- optional spherical transformation design, without coupling it to Cartesian
  kernels.

### Exit gate

- clean source and wheel builds from documented prerequisites;
- all test levels pass on supported platforms;
- public API docs contain shapes, units, ordering, ownership, and errors;
- release notes state numerical domain, maximum tested angular momentum,
  performance environment, and remaining limitations.

## Highest-risk issues

| Risk | Failure mode | Planned control |
|---|---|---|
| Phase, charge, or gauge sign mismatch | Plausible complex values with wrong imaginary signs or broken response properties | Freeze one Hamiltonian and phase convention; s-s analytic tests, conjugation, gauge shift, translation covariance, and zero-field checks start in Milestone 1. |
| Dropped translational phase | Energies may appear correct in special geometries while AO tensors transform incorrectly | Keep \(e^{-i\mathbf q\cdot\mathbf P}\) in the pair prefactor and test non-axis-aligned translations. |
| Complex Boys instability | Cancellation, branch discontinuity, overflow, or inaccurate high orders | Define Boys by an entire integral/hypergeometric form; use two high-precision references, region maps, recurrence residuals, and scaled auxiliaries. |
| Pair-prefactor/Boys cancellation | Intermediate overflow despite a representable ERI | Design a combined scaled Coulomb seed and test extreme negative-real arguments before production acceptance. |
| Invalid real-ERI symmetry reuse | Wrong finite-field quartets and Fock matrices | Encode only pair exchange and conjugate double reversal; include tests where one-pair swaps differ. |
| Full finite-field MD auxiliary growth | Excess memory and low throughput at moderate angular momentum | Size tables before implementation, reuse workspace, benchmark shell classes, and retain OS/HGP as a backend option. |
| Derivative omission of phase terms | Incorrect gradients and magnetic response | Treat phase derivatives as separate terms and validate each before combined finite differences. |
| Screening without a valid complex bound | Silent loss of significant integrals | Ship unscreened first; prove bounds on the original modulus; test monotone threshold convergence. |
| Reference correlation | Shared bug passes both implementations | Python overlap uses direct moments, C++ uses Hermites, Boys uses quadrature plus hypergeometric evaluation, and zero-field cases use external engines. |
| Normalization/import ambiguity | Correct kernels disagree with common basis formats | Public coefficients multiply normalized primitives; importers convert explicitly and tests use hand-computed contractions. |
| Unbounded full ERI allocation | Accidental \(O(N^4)\) memory exhaustion | Default to batches/direct consumers and require an explicit byte guard for full tensors. |

## Unresolved decisions for review

These decisions do not block the mathematical reference, but they affect later
engineering:

Milestone 4 resolved the initial Boys questions: the guaranteed direct disk is
\(|z|\le160\), the conservative positive asymptotic sector is stated in
`docs/algorithms.md`, and the accepted kernel is an accuracy-first
series/quadrature/asymptotic dispatcher. Beylkin--Sharma remains an optional
performance replacement rather than an open correctness dependency.

1. **License.** Choose a permissive license before accepting external
   contributors or distributing wheels. MIT or BSD-3-Clause fits the intended
   library use.
2. **Next optimized ERI backend.** Milestone 6 found the direct London OS
   prototype competitive but mixed across its small shell matrix. Decide
   whether a contracted HGP prototype is worthwhile after broader profiling.
3. **External oracle in required CI.** PySCF/libcint is convenient for optional
   zero-field validation. Decide whether a pinned external-engine job is
   mandatory or periodic because it increases wheel and CI cost.
5. **Initial platform matrix.** Linux and macOS are proposed. Native Windows
   support should be accepted only with a maintained CI runner.

## Decisions already resolved

- Atomic units and electron charge \(-1\).
- Symmetric-gauge vector potential and explicit gauge origin.
- London phase \(e^{-i\boldsymbol\kappa_A\cdot r}\) with
  \(\boldsymbol\kappa_A=\tfrac12\boldsymbol{\mathcal B}\times(\mathbf A-\mathbf O)\).
- Cartesian primitives and the stated component order.
- Deterministic serial execution remains the default. Optional OpenMP uses an
  explicit thread count, static scheduling, thread-local workspaces, and
  input-order callback delivery.
- Input coefficients multiply normalized primitives; contraction
  normalization is explicit.
- `complex128` throughout the public integral API.
- Shell blocks and batches as the computational boundary.
- General contraction rows in the production shell model from Milestone 2;
  contraction index remains outermost in AO ordering.
- Caller-provided output buffers in C++; validated `out=` arrays in Python.
- MD/Hermite as the first auditable C++ recurrence path.
- Independent direct-moment Python overlap reference.
- Unscreened ERIs before any complex screening optimization.
- Streaming/direct ERI consumption as the default architecture.
