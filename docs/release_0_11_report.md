# VICE follow-ups and 0.11.0 release preparation

Prepared on 21 November 2026. Publication of 0.10.0 and 0.11.0 each requires
explicit approval. No new release tag or PyPI upload has been created.
The independent 0.10.0 candidate and its passing artifacts are recorded in
[the 0.10 report](release_0_10_report.md); it excludes all items 3–5 below.

## Item 1: derivative documentation

Commit `6e7f064` replaces copied text with accurate per-function arguments,
return shapes, held origins and response terms. It also audits primitive,
shell and mixed derivative text and adds nine installed-documentation checks.
No numerical kernel changes. Included in 0.10.0.

## Item 2: 0.10.0 preparation

Source candidate `7910655`, merged/report recorded on main `511e68f`, passes
621 tests, full serial/OpenMP/sanitizer CI and the complete repaired wheel/sdist
matrix. Every installed artifact exercises derivative priorities 1–4.
Approval was requested as soon as these checks passed, independently of later
work. No publication approval is inferred from an unanswered request.

## Item 3: Cholesky generation

Commit `816b6ec` profiles and replaces the full-quartet extraction bottleneck
with native selected diagonal/column evaluation and deterministic task-level
parallelism. The original ERI value kernel, primitive contraction sequence,
pivot selection and NumPy/BLAS projection reduction order are retained.
Before/after 152-AO Ne and Be measurements and hashes are recorded below
from matched profiling runs. The serial Ne baseline is 438.320 s; vector updates take
3.978 s and residual updates 0.060 s, under 1% combined. The suspected serial
NumPy dominance is not supported by this profile. Selected-entry tests and
small complete-factor comparisons are bitwise equal to the previous backend.
Memory remains O(rank*nao²) plus bounded column/task/native workspace.

## Item 4: gradient assembly

Commit `e045008` adds native per-atom one-electron nuclear/mixed assembly,
including separate moment origin and attraction potential ownership, plus
streamed Coulomb/exchange/dense-Gamma contractions. Inputs are held fixed;
there is no implicit density conjugation or spin/force factor. Canonical AO
and centre-slot permutations are explicitly expanded with duplicate shell
images removed. Native reductions have fixed quartet/slot order.

37 focused tests cover value differences, explicit all-ordered-quartet sums,
complex non-Hermitian densities, repeated slots, native threading, batch-size
independence, output alias/shape validation, opt-in screening and Gamma identity
(no copy). The final full suite includes all 37 checks and passes 681 tests. Exact formulae are in [the API](api.md).
The largest absolute comparison discrepancy in these 37 checks was
4.6491e-11, including finite differences. H2O/STO-3G (7 AOs), general complex
J+K: explicit Python ordered-shell contraction 0.84295 s; streamed native
1/4 threads 0.27330/0.08702 s, maximum error 1.9301e-14. Ne/aug-cc-pVQZ
(105 AOs), atom moment nuclear/origin assembly: 0.10314/0.02836 s at 1/4
threads, 1,058,400 output bytes. Inputs use B=(0.05,-0.03,0.2), gauge
(0.37,-0.29,0.43) and moment origin (0.2,0.3,-0.1) bohr.

## Item 5: TD4.0 operators

Commit `b7f8063` adds kinetic angular momentum about independent G with
pi0=p+B0 cross (r-G0)/2, and magnetic kinetic with independent operator C/G at
fixed London B0/G0. Existing moment powers give all second moments about G.
Matching operator and phase fields delegate to the old kinetic kernel bitwise.

18 focused checks cover independent reference values through g, fixed-phase
operator-field differences, Hermiticity, gauge consistency and origin sum
rules. The optional `tests/external/td4_pyscf.py` compares H2O/cc-pVDZ (25
Cartesian AOs) with libcint angular momentum, quadrupole and kinetic matrices
at B0=0: maximum discrepancy 1.776e-14. The final isolated conda source
installation passes all 681 tests. The largest absolute discrepancy across
the 18 TD4 tests (including finite differences) was 5.2695e-12.

Ne/aug-cc-pVQZ, 105 AOs: kinetic angular x takes 0.06211/0.01709 s
at 1/4 threads; independent-field magnetic kinetic takes 0.13890/0.03750 s.
Each matrix uses 176,400 bytes; both and the atom moment assembly are bitwise
equal across these thread counts. Raw helper timings and the reproducer are
[stored with the benchmarks](../benchmarks/results/vice_helpers_macos_arm64.jsonl)
and [in the helper script](../benchmarks/vice_helpers.py).

## Item 6: release preparation

Version 0.11.0 includes items 3–5 on top of the independently frozen 0.10.0
candidate. Wheel/source smoke tests exercise atom assembly, streamed density
contraction, TD4.0 operators and the derivative priorities. The existing
CPython 3.11–3.14 macOS/Linux architecture matrix and protected trusted
publication workflow are retained. Publishing requires a separate explicit
v0.11.0 approval.

Candidate `df11f5d` passes [full CI](https://github.com/simenkva/GIAOINT/actions/runs/44900247365):
serial/OpenMP native and Python suites, sanitizers, static analysis, conda
runtime compatibility and isolated source installs. Python 3.11–3.14 are
covered on Linux/macOS. The isolated local conda install also passes 681
tests. Black, Ruff, local documentation links, 44 documentation Python
fences and `git diff --check` pass. The optional libcint oracle was rerun
against that installed 0.11.0 package and agrees to 1.776e-14.

The [artifact build](https://github.com/simenkva/GIAOINT/actions/runs/44900250724)
builds and smoke-tests all 16 repaired wheels and the source archive.
Publication is skipped on this branch run. Feature-stage CI also passed for
[Cholesky](https://github.com/simenkva/GIAOINT/actions/runs/44891258829),
[gradient assembly](https://github.com/simenkva/GIAOINT/actions/runs/44892263308)
and [TD4](https://github.com/simenkva/GIAOINT/actions/runs/44893576225).

NMR shielding/spin-spin operators, Cholesky-vector derivatives and spherical
functions remain out of scope. No additional design decision is outstanding;
the publication decisions remain with the maintainer.

## Cholesky before/after measurements

Single runs on macOS arm64, CPython 3.13.3, NumPy 2.5.3/Accelerate,
AppleClang 21 and Homebrew libomp 23.1.1, with both BLAS thread limits set
to 1. The unique uncontracted aug-cc-pCVQZ basis has 152 Cartesian AOs
for each atom; B=(0,0,0.1). Baseline is `4e2bb57`; final selected backend
is candidate `df11f5d`. Serial counts alone are line-profiled in both sets.
These are developer-host observations, not pinned or repeated performance
guarantees; occasional small development checks overlapped the baseline.

### Ne, threshold 1e-08

Rank 1052, M/nao=6.921; residual 7.48128e-09.

| Threads | Before seconds | After seconds | Before scaling | After scaling | Improvement |
|---|---:|---:|---:|---:|---:|
| 1 | 438.320 | 34.336 | 1.00x | 1.00x | 12.77x |
| 4 | 131.977 | 12.265 | 3.32x | 2.80x | 10.76x |
| 8 | 78.718 | 8.436 | 5.57x | 4.07x | 9.33x |
| 12 | 72.013 | 7.831 | 6.09x | 4.38x | 9.20x |

Returned vectors: 370.87 MiB.

### Be, threshold 1e-08

Rank 1016, M/nao=6.684; residual 9.26082e-09.

| Threads | Before seconds | After seconds | Before scaling | After scaling | Improvement |
|---|---:|---:|---:|---:|---:|
| 1 | 384.086 | 31.722 | 1.00x | 1.00x | 12.11x |
| 4 | 115.391 | 11.127 | 3.33x | 2.85x | 10.37x |
| 8 | 68.475 | 7.633 | 5.61x | 4.16x | 8.97x |
| 12 | 64.043 | 6.985 | 6.00x | 4.54x | 9.17x |

Returned vectors: 358.18 MiB.

### Be, threshold 1e-06

Rank 815, M/nao=5.362; residual 9.00771e-07.

| Threads | Before seconds | After seconds | Before scaling | After scaling | Improvement |
|---|---:|---:|---:|---:|---:|
| 1 | 338.928 | 25.861 | 1.00x | 1.00x | 13.11x |
| 4 | 100.979 | 8.621 | 3.36x | 3.00x | 11.71x |
| 8 | 59.297 | 5.857 | 5.72x | 4.42x | 10.12x |
| 12 | 55.256 | 5.133 | 6.13x | 5.04x | 10.76x |

Returned vectors: 287.32 MiB.

All 12 final measurements match baseline vector SHA-256, pivot SHA-256, rank
and residual exactly. Raw hashes, build information, timings and the earlier
fully instrumented pilot are in
[the benchmark data](../benchmarks/results/vice_cholesky_macos_arm64.json).
Reproduce using [the driver](../benchmarks/cholesky_threads.py) with
`--atom Ne --threshold 1e-8 --threads 1 4 8 12 --profile-serial`
and the analogous Be thresholds.

Largest final process RSS is 936.48 MiB (0.915 GiB),
versus 7.95 GiB for a dense complex128 ERI tensor. RSS
is a cumulative process high-water mark and includes Python, basis loading,
NumPy, workspace and temporary factor growth/conversion. The algorithm
stores O(M*nao²) factors and up to eight columns plus bounded native tasks;
no full pair matrix or dense ERI tensor is generated. `max_bytes` continues
to cap returned arrays, not process RSS.

The unchanged projection reductions now form a substantial fraction of the
shorter threaded runtime: the instrumented Ne pilot spends 3.60 s of 7.57 s
in vector updates at 12 threads. This limits further scaling. Keeping their
existing BLAS reduction order preserves the requested old output bits; the
measured bottleneck removed here was unnecessary integral generation.
