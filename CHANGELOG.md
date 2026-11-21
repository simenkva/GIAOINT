# Changelog

## Unreleased

### Priority 2 — gradient/momentum and property field derivatives

Added literal nuclear derivatives for all components of gradient and canonical
momentum, plus magnetic derivatives of general moments, gradient and momentum.
Primitive/shell/basis APIs include all London-phase terms, release the GIL,
accept strict outputs and use configured shell-pair threading at basis level.
Field derivatives hold the origins fixed and include the ket wave-vector
response under the spatial operator; no vector-potential term is added to
canonical momentum. Validation includes Richardson differences through g,
anti-/Hermiticity and independent reference/libcint comparisons.

### Priority 1 — moment nuclear derivatives

Added analytic primitive, shell and basis-centre moment derivatives, including
London-phase response and separate moment-origin derivatives for general
Cartesian powers. The basis API supports `out=(shell_out, origin_out)`, releases
the GIL, and uses the configured thread policy with bitwise-equal shell-pair
parallelism. Derivatives are literal coordinate derivatives, with no force or
electronic-charge sign. See docs/derivative_validation.md for finite-difference,
reference-engine, libcint and 105-AO timing results. These additions require a
source build; PyPI 0.9.0 is unchanged.

## 0.9.0 — 20 November 2026

Fixed macOS conda source builds to prefer the target interpreter's OpenMP
runtime and headers ahead of Homebrew, avoiding duplicate-runtime aborts with
conda NumPy/OpenBLAS. Explicit runtime overrides remain supported. Added an
installed-rpath fix and a conda regression job exercising both import orders
and serial/parallel Cholesky equality with OpenMP OpenBLAS.

Fixed Linux sanitizer CI startup by preloading libasan followed by libstdc++
from the build compiler, keeping C++ exception interception available when
Python loads the extension. Native sanitizer output is no longer hidden by
pytest FD capture; ASan and UBSan both halt on reported errors.

OpenMP now defaults to AUTO at build time, including automatic Homebrew libomp
lookup on macOS, with a visible serial fallback and strict ON/OFF overrides.
Python ERI, batching, molecular-integral, and Cholesky calls now default to
`threads=None`, resolving `set_num_threads`, GIAO_NUM_THREADS, OMP_NUM_THREADS,
then physical cores. `build_info()` exposes compiler, version, loaded OpenMP
runtime, and default threads. Explicit unsupported parallel requests raise;
unavailable parallel defaults warn and run serially.

This intentionally reverses the old opt-in threading policy. Tests check
bitwise equality across entry points, screening and field settings, including
Cholesky pivots and vectors, at 1, 2, and the runtime maximum thread count.
The Ne profile locates the bottleneck in native ERI batches, not NumPy updates;
dynamic shell-block scheduling improves load balance without changing any
primitive reduction order. BLAS remains outside OpenMP regions.

Prepared CPython 3.11–3.14 wheels for Linux x86_64/aarch64 and macOS arm64/x86_64,
with repaired/bundled OpenMP runtimes and install smoke tests, plus a source
archive. Trusted publication is gated by explicit approval for each version
tag. Windows wheels remain out of scope. See docs/installation.md and the
0.9 validation report. Source and editable installs use standard pip build
isolation; C++ editable changes require reinstalling.

Added streamed pivoted Coulomb Cholesky via `Molecule.cholesky_eri()` and
`cholesky_eri(basis, ...)`, with complex ordered-pair factors, residual and
pivot metadata, block reconstruction, and an optional output cap. No dense
AO ERI tensor or pair matrix is allocated. `Molecule.eri()` and
`integrals(eri=True)` now default to no output cap (`None`); explicit caps
remain supported. See `docs/cholesky_eri.md` for conventions and benchmarks.

Added `Molecule.from_xyz()` and `MolecularIntegrals` for computing integrals
from XYZ files/text and named, Gaussian94, NWChem, or BSE JSON bases. The
optional `molecule` extra supplies Basis Set Exchange's local database and
format readers. The wrapper handles coordinate conversion and shell ownership,
provides a physical core Hamiltonian, and offers full ERIs with an optional
output limit or streaming batches. All output remains Cartesian;
unsupported ECP data raises an error. See `docs/molecule_api.md`.

Milestone 9 replaced the recursive ERI auxiliary with an iterative six-index
fill and added a collapsed real-arithmetic path for exact zero field. Shell evaluation
reuses Gaussian-pair data across Cartesian components; ssss evaluation uses
the Boys seed directly. Public numerical conventions, output shapes, and
angular/workspace admission limits remain unchanged.

Milestone 9 originally kept both screening and threading opt-in. Version 0.9
retains `screening_threshold=0.0` but changes the Python thread default to
`None`, as described above. The performance-comparison CLI still explicitly
defaults to one thread and sets the same requested count for PySCF. Before/after measurements, randomized cross-checks,
and profiling results are in `benchmarks/results/m9_macos_arm64.md`.

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
