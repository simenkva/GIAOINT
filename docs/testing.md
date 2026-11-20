# Verification strategy

Correctness gates precede optimization and new operator families. The Python
reference package never imports the C++ extension.

## 1. Validation hierarchy

### Level 1: analytic special cases

- normalized primitive self-overlap;
- finite-field s-s overlap from the closed expression;
- coincident centers, equal exponents, zero phase vector, and a field parallel
  to the center separation;
- low-order kinetic and Coulomb s seeds;
- exact zero-field reduction;
- known odd/even Cartesian moment zeros.

### Level 2: independent Python reference

Use direct complex-center polynomial expansion and Gaussian moments for
overlap, explicit raising/lowering expressions for low-order operators, and
mpmath arithmetic for difficult cases. Compare randomized primitive,
contracted, shell-pair, and shell-quartet results. Preserve failing random
examples as named regression cases.

### Level 3: conventional external engines at zero field

Optional scripts in `tests/external/` compare shell blocks and molecular
matrices with PySCF/libcint after explicit normalization and Cartesian-order
conversion. No Libint oracle is currently implemented. These scripts record
the reference engine/version; external packages remain optional validation
dependencies rather than runtime or required-CI dependencies.

### Level 4: high precision

Use at least 80-decimal mpmath calculations for complex Boys values, severe
exponent ratios, diffuse functions, and cancellation-prone finite-field cases.
Increase precision until the reference stabilizes by more digits than the
double-precision acceptance target.

### Level 5: numerical integration

For low angular momentum, compare overlap and local one-electron operators
with direct adaptive quadrature or high-order Gauss-Hermite integration. Use
independent coordinate transformations and convergence studies. Numerical
quadrature is a diagnostic where it cannot reach the analytic tolerance.

### Level 6: finite differences

Check analytic nuclear and magnetic derivatives with central differences over
several step sizes. Require the expected truncation-error regime before
comparing the extrapolated result. Test ordinary Gaussian and London-phase
derivative contributions separately.

Milestone 7 uses centered differences at two step sizes followed by Richardson
extrapolation. Primitive checks cover overlap, canonical and physical magnetic
kinetic energy, nuclear attraction basis and potential centers, and ERIs at
finite field. Contracted shell and basis checks exercise leading derivative
axis ordering, general contraction, strict `out=` handling, Hermiticity,
zero-field translation identities, and local-operator gauge-origin behavior.
The ordinary Gaussian and London terms are also assembled separately for an
overlap case and required to be individually nonzero before comparison with
the combined analytic response.

Milestone 8 adds boundary tests for empty bases and empty nucleus lists,
explicit derivative shapes at zero size, diagnostic failure at the Coulomb
angular-order boundary, randomized finite-field conjugation of values and
responses, executable examples, and malformed/missing benchmark-history data.
The current CI configuration runs the full Python suite on CPython 3.11–3.14
under Linux and macOS with both OpenMP OFF and ON. Separate native OFF/ON and
ASan/UBSan jobs cover C++; the sanitizer job also runs Python with the ASan
runtime preloaded. clang-tidy provides static-analysis coverage. The
[0.9 report](release_0_9_report.md) distinguishes completed local checks from
remote release-wheel verification that is still pending. Optional PySCF/libcint
validation remains a release check because it is not a runtime dependency.

## 2. Invariant and covariance tests

Every operator milestone includes:

- one-electron Hermiticity, `I == I.conj().T`, where the operator is
  self-adjoint;
- exact complex ERI relations
  `(ab|cd) == (cd|ab)` and `(ab|cd).conj() == (ba|dc)`;
- a negative test showing that a one-pair ERI swap is not treated as a symmetry
  at finite field;
- recovery of real values and conventional symmetries at zero field;
- continuity along a logarithmic sequence of field magnitudes tending to zero;
- local-operator independence under a gauge-origin shift;
- canonical-kinetic covariance only when combined with the matching magnetic
  operator terms;
- AO tensor rephasing under a common translation of geometry and gauge origin;
- continuity as centers coalesce and exponents approach equality.

Property-based tests generate non-axis-aligned fields and centers so cross
products do not vanish by accident.

## 3. Boys-function matrix

The complex Boys suite crosses:

- orders `n = 0..32`, with a smaller exhaustive core and larger sampled set;
- small, moderate, and large magnitudes;
- positive and negative real axes approached from both half-planes;
- nearly imaginary and genuinely complex arguments;
- points around every algorithm dispatch boundary;
- complex arguments collected from randomized nuclear-attraction and ERI
  shell data.

Tests check values, recurrence residuals, conjugation
`F_n(z.conjugate()) == F_n(z).conjugate()`, continuity, and scaled/unscaled
agreement when both are representable. High-precision direct integration and
hypergeometric evaluation must first agree with one another.

Milestone 4 implements this matrix through order 32. The fixed-seed physical
map uses exponents approximately 0.05--20, coordinates in \([-2,2]\) bohr,
and field magnitudes through 1 atomic unit. Its 600 sampled comparisons
observed \(2.28\times10^{-15}\) maximum absolute error and
\(1.06\times10^{-14}\) maximum relative error. Primitive
nuclear-attraction comparisons use an independent 75--100 digit Obara--Saika
reference through \(L=4\) on each center; contracted blocks are tested through
p--d.

The optional external gate is implemented as
`tests/external/pyscf_zero_field.py`. With PySCF 2.8.0/libcint it compares
normalized Cartesian s, p, and d shells after explicit normalization
conversion; observed maximum overlap and nuclear-attraction errors are
\(4.44\times10^{-16}\) and \(3.95\times10^{-14}\). Milestone 5 extends the
same basis comparison to the full zero-field ERI tensor, with maximum error
\(1.72\times10^{-14}\).

Finite-field primitive ERIs are compared with an independent 80-digit
Obara--Saika reference. The exhaustive core covers all 91 distributions of
combined Cartesian degree zero through two across the four centers and three
axes; selected cases extend to combined degree 10. Separate tests cover the
scaled negative-real Boys seed, exact pair exchange and conjugate reversal,
the invalid one-pair swap, general contraction ordering, warmed workspace
reuse, and equality of canonical streamed blocks with the guarded full tensor.

Milestone 6 checks every finite-field AO block against its cached shell Schwarz
product, verifies that screening error decreases monotonically to exactly zero
as the threshold reaches zero, and compares threshold-zero evaluation with the
Milestone 5 path. Serial and OpenMP builds run the same C++ suite. The OpenMP
suite also requires input-ordered callbacks and bitwise-equal blocks at one and
two threads. Python validates screening statistics, option failures, and
capability reporting. Release benchmark checksums are identical before and
after the cache and layout changes.

## 4. Tolerance policy

Tests use a mixed condition

```text
abs(actual - reference) <= atol + rtol * abs(reference)
```

and report absolute error, relative error, reference scale, and condition
category. Initial targets are:

| Category | `atol` | `rtol` | Notes |
|---|---:|---:|---|
| well-conditioned primitive overlap vs analytic double | `2e-14` | `2e-13` | through tested angular momentum |
| C++ primitive/shell overlap vs 80-digit reference | `5e-14` | `1e-12` | contraction error scales with primitive count |
| Hermiticity and finite-field conjugation | `5e-14` | `5e-13` | compare full blocks |
| zero-field external engine | `2e-12` | `2e-11` | after normalization/order conversion |
| C++ moment/gradient/momentum/kinetic vs direct reference | `2e-13` | `2e-11` | randomized primitives through tested angular momentum |
| contracted property shell blocks | `1e-12` | `2e-11` | includes derivative and magnetic combinations |
| complex Boys, representable well-conditioned region | `2e-14` | `5e-13` | per order and region |
| finite-field primitive ERI vs 80-digit OS reference | `2e-12` | `2e-11` | exhaustive combined degree 0--2, sampled through 10 |
| finite-difference real-space kinetic quadrature | `1e-8` | `1e-8` | three-step convergence diagnostic, not a production tolerance |
| other numerical quadrature diagnostics | convergence-based | convergence-based | no loose global fallback |
| finite-difference derivatives | step-study based | `1e-6` initially | tightened after analytic implementation |
| M7 analytic derivatives vs Richardson extrapolation | `8e-9` | `8e-8` | primitive, contracted shell, basis, and ERI shell-quartet paths |

These are starting gates. A test may use a different tolerance only with a
named numerical reason and an error study. Near-zero references emphasize
absolute error; no test divides by a tiny reference to manufacture a large
relative failure. Difficult regions receive separate expected-error records,
not a global tolerance increase.

## 5. Test partitioning

- C++ unit tests cover value types, enumeration, normalization, product data,
  recurrence tables, buffer validation, and kernels.
- Python API tests cover shape, dtype, ordering, `out=`, error mapping, GIL
  release, and NumPy lifetime behavior.
- Reference tests validate the Python oracle against analytic and
  high-precision routes before it validates C++.
- Regression tests hold minimal cases for every fixed sign, branch,
  cancellation, or symmetry bug.
- Property tests use fixed top-level seeds while recording minimized failing
  examples.
- Sanitizer jobs exercise extreme shell metadata and workspace growth.
- OpenMP jobs compare thread-local shell blocks with serial results before any
  scaling result is accepted.

## 6. Performance verification

Benchmarks consume results through a checksum so the compiler cannot remove
the work. Records include commit, compiler and flags, CPU, thread count, shell
class, primitive counts, contraction counts, field, iteration count, checksum,
wall time, integrals per second, and shell blocks per second.

Milestone 2 records overlap throughput without a pass/fail threshold.
Milestone 6 establishes a stable JSON-lines schema and checked-in reference
reports. The current reports are informational; a future dedicated runner may
introduce noise-aware regression thresholds once enough history exists.
Numerical tests run after every optimization; matching a performance baseline
never excuses changed values.

## 7. Milestone gate template

Each milestone PR or review record states:

1. formulas and conventions used;
2. smallest implemented scope and explicit exclusions;
3. analytic and independent reference tests added;
4. commands and environments used to run tests;
5. maximum angular momentum and numerical domain tested;
6. failures, error envelopes, and unresolved issues;
7. benchmark change, if optimization occurred.

## Milestone 9 optimization checks

`tests/data/eri_recursive_m8.jsonl` freezes 192 primitive values from the M8
recursive kernel at commit `2d5b3b13c85fb1258788d698bff551bebbde4b93`.
The 96 fixed-seed quartets cover Cartesian powers through g on each center,
signed coefficients, normalized and unnormalized primitives, and zero and
finite fields. The iterative-only phase passed these checks before adding
the real path; the final implementation runs the same regression test.
The independent 80-digit OS comparisons remain unchanged.

C++ tests compare 128 randomized zero-field quartets with the general complex
path, reuse one workspace across both paths, and check signed zero,
nonzero gauge origins, tiny fields on all axes, and the smallest positive
subnormal field. Warmed zero-field and finite-field shell calls must allocate
no heap memory. The release, ASan/UBSan, and OpenMP builds run these checks.

## Molecule and basis-format wrapper checks

`tests/python/test_molecule.py` checks XYZ paths/text, atom counts and symbols,
ångström/bohr conversion, finite coordinates, shell ownership and ordering,
general contractions and SP splitting, standard basis-file round trips,
optional-dependency errors, ECP rejection, and full/streamed ERI limits.
Zero- and finite-field matrices are compared with separately constructed
low-level bases and nuclei. The JSON tests block the BSE import to verify that
this path works without the optional extra. Development dependencies include
BSE so the format-reader tests run in CI.

Run `python tests/external/molecule_pyscf.py` with PySCF installed for an
independent end-to-end check. It compares water/STO-3G, 6-31G, and cc-pVDZ,
then cc-pVDZ read from Gaussian94, NWChem, and JSON files. It accounts for
PySCF's Cartesian normalization and supplies bohr coordinates after conversion
to avoid differences between physical-constant versions. The local PySCF
2.8.0 / BSE 0.11 run agreed to at most `4.352e-14` for ERIs; the Python suite
also passed with BSE 0.12.

## 0.9 Cholesky, threading and distribution checks

`tests/python/test_cholesky.py` compares the streamed factors against exact
small-system ERIs for water STO-3G/6-31G, an O atom and displaced d/f shells,
at zero and finite field, with off-symmetry gauge origins and thresholds from
1e-4 through 1e-10 plus numerical exhaustion. It checks residual diagonals,
ERI symmetry error bounds, gauge independence, real symmetric zero-field
vectors, independent real Cholesky, output caps and traced memory. A separate
single-center d/f check verifies finite-field phase cancellation. See
[Cholesky validation](cholesky_eri.md) for recorded measurements.

`tests/python/test_threading.py` checks policy precedence, invalid values,
serial fallback warnings, explicit unsupported requests and build diagnostics.
Each ERI entry point, including Molecule integrals, packed/native APIs and
Cholesky vectors/pivots, is compared bit for bit at one, two and the runtime
maximum thread count on OpenMP builds. Zero/finite fields and screening off/on
are covered. A distant center ensures quartets actually get screened; a close
pair retains complex finite-field integrals. Serial builds still run the same
suite and test rejection of explicit parallel requests.

`tests/wheel_smoke.py` runs against the installed artifact, not the checkout.
For repaired wheels, `GIAO_REQUIRE_BUNDLED_OMP=1` requires the loaded runtime to
be bundled. The smoke test checks version metadata, OpenMP, serial/parallel
ERI equality, Cholesky factor/pivot equality, and reconstruction. The release
workflow also installs the sdist in an isolated fresh environment. Source and
editable-install CI checks AUTO, missing-runtime fallback and strict ON failure.

The sanitizer preset uses ASan/UBSan with -O1 and debug symbols. Python needs
that compiler's ASan runtime preloaded (LD_PRELOAD on Linux; on macOS launch
the actual interpreter binary with DYLD_INSERT_LIBRARIES, since launch wrappers
may strip it). Leak detection is disabled for the Python host, but address and
undefined-behavior checks remain enabled. The GIL test has a 60-second
completion timeout to accommodate instrumentation; its GIL-release assertion
is separate from that timeout. See the [release checklist](release_checklist.md)
for the complete publication gate.
