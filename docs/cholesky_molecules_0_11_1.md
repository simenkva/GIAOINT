# Molecular Cholesky: 0.11.1 preparation

## Profile recorded before production changes

Published PyPI 0.11.0 wheel, macOS arm64, CPython 3.13.3, NumPy 2.5.5,
AppleClang 17 wheel, bundled libomp, 12 threads. OPENBLAS_NUM_THREADS and
VECLIB_MAXIMUM_THREADS are both 1. Water uses the supplied asymmetric bohr
geometry, cc-pVTZ (65 Cartesian AOs), tau=1e-8. Finite B=(.03,-.02,.05),
gauge origin=(.7,-.4,1.1). Host load averages are retained in raw data.

| Phase, seconds | Zero field | Finite field |
|---|---:|---:|
| Diagonal | 0.0282 | 0.0263 |
| Native columns | 42.7965 | 95.3560 |
| Cholesky update including residual | 0.3101 | 0.8728 |
| Python/NumPy setup, dispatch and overhead | 0.0871 | 0.1368 |
| Total | 43.2219 | 96.3920 |
| Dense eri() | 0.5667 | 0.9450 |

The profile wraps actual native calls and line-times the pivot update;
remaining wall time is overhead (including profiler request accounting).
No production code had changed when these two profiles were collected.

| Work | Zero field | Finite field |
|---|---:|---:|
| Diagonal shell quartets | 55 | 55 |
| Diagonal primitive integrals | 511,476 | 511,476 |
| Native column batches | 588 | 622 |
| Column shell-quartet visits | 32,340 | 62,200 |
| Distinct column shell quartets | 2,585 | 8,000 |
| Selected AO evaluations (including zero-field averaging) | 12,611,933 | 19,658,925 |
| Column primitive-kernel calls | 711,677,659 | 1,044,102,075 |
| Native column tasks | 176,522 | 333,682 |
| Dense canonical shell quartets | 2,575 | 2,575 |
| Dense primitive-kernel calls | 11,198,307 | 11,198,307 |

Counts follow the actual requested columns, shell primitive/contraction counts,
and native loop structure; they are deterministic, not sampled estimates.
Column counts use literal requested orientations, whereas dense quartets use
fourfold canonical classes.
The dense driver uses the same fourfold shell symmetry even at zero field,
not the real eightfold estimate in the request. Its primitive-first shell
kernel computes a Cartesian component once before distributing contractions.
The old selected kernel repeats it for each contraction row. The eight-column
cache is replaced whenever the global pivot changes shell pairs, wasting most
speculative columns in these molecules. It does **not** generate full shell
quartets: that hypothesis is disproved by the code and evaluation counts.

The dense field/zero ratio is 1.67x with an unchanged primitive count. The
old column field/zero ratio is 2.23x: 1.47x as many primitive calls and about
1.52x cost per call. The finite-field path evaluates ordered bra shell pairs
(100 versus 55),
uses complex factors in the update, and runs the complex six-coordinate
auxiliary recurrence instead of the collapsed real recurrence. Together
these explain the higher column count and cost. Neither column path screens
at screening_threshold=0. Single-center London phases cancel in pair densities,
allowing exact real dispatch. The uncontracted atom bases also lack the repeated
general-contraction work that dominates the molecular profile.

## Implementation

Primitive-first selected-column evaluation shares each component integral
across general contraction rows. Gaussian pair products are reused within a
shell-pair task. Each output retains its original pa/pb/pc/pd summation and
coefficient multiplication order. No other integral kernel changes.

Only contraction variants of the current Cartesian pivot component are
prefetched, up to 32 at a time. An LRU retains raw columns across shell-pair
changes, with at most min(1024,rank+8) cached rows. The uncontracted single-center path retains
its eight-candidate policy. Real factors retain the global largest-residual
pivot rule. Complex contracted factors select candidates from the most recent
Cartesian component block while their updated residual is at least 1% of the
global maximum, bounded by pivot_batch_size (default 32). Stopping still tests
the global largest residual, so the PSD error bound is unchanged. This changes
complex molecular pivots/vectors and may slightly change rank. Setting
pivot_batch_size=1 restores strict global pivoting and the old factors. NumPy
projection reductions per pivot remain unchanged. Factors, cache and selected-column
workspace are O(rank*nao^2) plus bounded native/column workspace; no dense
ERI or full ordered-pair matrix is formed. max_bytes remains an output cap.

Column screening remains absent; no new approximation or changed error bound
is introduced. Reproducible runs use benchmarks/cholesky_molecules.py; profile
with --profile; --dense times dense generation and --verify checks the full
reconstruction outside the timed decomposition. The complete paired timings
are in [the Cholesky tables](cholesky_eri.md).

The larger molecular case is nonplanar NH3/aug-cc-pVTZ (130 Cartesian AOs),
with N=(0,0,.2) and three H centers at radius 1.8, z=-.5 bohr, separated by
120 degrees. It adds a center and exceeds both water basis sizes. The optional
benzene example was replaced after an exploratory 12-thread old-backend run
had exceeded nine minutes; that interrupted run supplies no timing result.

## Final water profile

The source candidate is `e6c2d26`, built with AppleClang 21 and Homebrew
libomp 23.1.1 on the same Apple M2 Pro (12 cores, 32 GiB RAM), Python and
NumPy as the published-wheel baseline. Kernels outside Cholesky are unchanged.
The compiler/runtime difference is retained in the raw provenance. Timings
are single-run observations, not affinity-controlled performance guarantees.

| Phase, seconds | Zero before | Zero after | Field before | Field after |
|---|---:|---:|---:|---:|
| Diagonal | 0.0282 | 0.0234 | 0.0263 | 0.0286 |
| Native columns | 42.7965 | 0.5057 | 95.3560 | 1.2372 |
| Cholesky updates | 0.3101 | 0.1251 | 0.8728 | 0.5854 |
| Python/NumPy setup/overhead | 0.0871 | 0.0946 | 0.1368 | 0.1221 |
| Total profiled | 43.2219 | 0.7488 | 96.3920 | 1.9733 |
| Dense eri() | 0.5667 | 0.4502 | 0.9450 | 0.9849 |

| Column work | Zero before | Zero after | Field before | Field after |
|---|---:|---:|---:|---:|
| Native batches | 588 | 492 | 622 | 389 |
| Shell-quartet visits | 32,340 | 27,060 | 62,200 | 38,900 |
| Primitive-kernel calls | 711,677,659 | 9,644,326 | 1,044,102,075 | 14,132,394 |
| Native tasks | 176,522 | 27,201 | 333,682 | 39,029 |

The after work is 0.86x and 1.26x the dense primitive-kernel count (11,198,307),
rather than the original 64x/93x. NumPy updates now occupy a material fraction
of the much shorter complex runtime. Prefetching contraction variants requires
extra coefficient accumulation but shares the expensive primitive evaluations.
Primitive pair products are cached up to 4096 entries per side of a task;
longer contractions use an on-demand fallback, preserving bounded workspace.

Unscreened zero-field water factors/pivots remain bitwise equal to 0.11.0.
Default finite-field rank is 687 versus 668 (2.84% increase); its independently
reconstructed maximum ERI error is 9.99392506945e-9, below the achieved residual
9.99392511675e-9 and threshold 1e-8. Strict global pivoting takes 2.5595 s,
rank 668, and reproduces both the published vector and pivot SHA-256 hashes.
No user decision is required to preserve the old result: request
`pivot_batch_size=1`. Default batching changes only complex contracted factors;
real factors and uncontracted atom evaluation retain their original policies.

All 695 tests pass in the refreshed editable source installation (100.84 s),
including the independent
reference suite, molecular reconstruction/rank checks, explicit selected-column
comparison to full shell values, thread determinism and the long-contraction
pair-cache fallback. [Full CI at final source a34b5db](https://github.com/simenkva/GIAOINT/actions/runs/44917711684)
passes all 695 tests in the serial/OpenMP native/Python matrix, Linux sanitizers, static analysis,
conda and source installations. The [final artifact run](https://github.com/simenkva/GIAOINT/actions/runs/44917713935)
passes all 16 repaired wheels (CPython 3.11–3.14, macOS arm64/x86_64 and Linux
x86_64/aarch64) and the source archive, including installed smoke checks.
Publication was skipped. All 64 paired benchmark stages are complete and
validated. No tags or PyPI publication have been created.

Segmented contractions also use primitive-first shell-pair tasks, even when
there is only one contraction row. Their component group has one candidate,
so global pivoting is retained. The selected-column tests exercise both
segmented and general contractions at zero/finite field. All requested
benchmark bases have either multiple contraction rows when primitive counts
exceed one, or fully uncontracted shells; this dispatch extension leaves
their measured execution paths unchanged.

## Why the error bound is independent of pivot order

In exact arithmetic, for any selected positive pivot p the residual update is
`R' = R - R[:,p] R[p,:] / R[p,p]`. This is a Schur-complement update and
preserves positive semidefiniteness. Therefore
`|R[I,J]| <= sqrt(R[I,I]*R[J,J])`. The algorithm always stops on the **global**
maximum diagonal, including during local pivot batches. Once that maximum is
at most tau, every ERI error is at most tau, because the returned reconstruction uses
`V[mu,nu,lam,sig] = W_approx[(sig,lam),(mu,nu)]`, a permutation of
ordered-pair entries. The local 1% pivot
eligibility test is not integral screening or an accuracy threshold. Floating
point exhaustion retains the documented 64*eps*max(initial diagonal) floor.

For the atom benchmarks all orbitals share one origin-centered Gaussian center;
the phase cancels and Boys arguments are zero, making the existing workspace
caches particularly effective. That uncontracted path is retained.

Uncontracted multi-center bases also generate only their requested component
columns, avoiding the old speculative eight-column waste. Their contraction
component group has one entry, so the old global pivot order is retained.
The eight-candidate policy is reserved for uncontracted single-center bases;
work-count regression tests cover zero/finite-field uncontracted molecules.
This extension does not alter any requested benchmark case's execution path.

## Complete matrix and final installed-wheel checks

All eight zero/finite-field/system cases have measured before/after results at
1/4/8/12 threads, with rank, residual, process RSS, vector storage and load
averages. See [the full tables](cholesky_eri.md) and
[raw measurements](../benchmarks/results/cholesky_molecules_0_11_1_macos_arm64.json).
Within each version/policy all vector and pivot hashes, ranks and residuals
match across thread counts. Every real and atomic case also matches 0.11.0
bitwise. The largest complex rank increase is 2.84%. Peak candidate Cholesky
RSS across the main matrix is 1.30 GiB; no dense ERI was held during those
measurements. The optional dense comparison was generated afterwards.

At twelve threads the cc-pVTZ water timings are 43.295 -> 0.672 s (zero field)
and 93.469 -> 1.826 s (finite field), respectively 1.41x and 1.93x the candidate's
dense ERI times. At one thread they are 325.698 -> 3.511 s and
747.533 -> 9.325 s. The larger molecule is NH3/aug-cc-pVTZ, 130 Cartesian AOs;
its finite-field twelve-thread result is 216.905 -> 15.360 s, rank 1174 -> 1192.

The final CPython 3.13 arm64 wheel from the artifact run was independently
installed and checked using the same NumPy/BSE environment. It uses AppleClang
17 and a bundled runtime, matching the baseline's compiler family. Water at
zero/finite field, Ne and Na reproduce the source vector/pivot hashes, ranks
and residuals exactly. Both water reconstructions satisfy the error bound.
At the later, higher observed host load, wheel water takes 0.800/4.356 s versus
dense 1.591/3.376 s; these are separate checks, not replacements for the main
matrix. Full load/build provenance is retained in `wheel_checks` in the raw data.

A single Na/four-thread source measurement was 5.6% slower, so it was followed
by three paired wheel comparisons with alternating execution order. Median
0.11.0/candidate times are 64.359/63.178 s (candidate 1.84% faster); ranges are
61.179–65.579/61.560–74.967 s. The original observation is retained in the
main table. These repetitions show substantial timing variation and do not
establish an atom slowdown; the uncontracted atomic work and numerical outputs
are unchanged. No significance or controlled-host performance guarantee is
claimed from three repetitions.

The benchmark core was built at `76d8d21`, with the finalized Python numerical
policy from `e6c2d26`; later dispatch generalizations have identical execution
paths on these requested bases. Final source `a34b5db` and its installed wheel
validate the generalized paths. The refreshed project environment now loads
0.11.1, removing the stale editable-extension ambiguity during development.
