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
The dense driver uses the same fourfold shell symmetry even at zero field,
not the real eightfold estimate in the request. Its primitive-first shell
kernel computes a Cartesian component once before distributing contractions.
The old selected kernel repeats it for each contraction row. The eight-column
cache is replaced whenever the global pivot changes shell pairs, wasting most
speculative columns in these molecules. It does **not** generate full shell
quartets: that hypothesis is disproved by the code and evaluation counts.

The finite-field path evaluates ordered bra shell pairs (100 versus 55),
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
changes, with at most min(1024,rank+8) cached rows. The uncontracted path retains
its eight-candidate policy. Real factors retain the global largest-residual
pivot rule. Complex contracted factors select candidates from the most recent
Cartesian component block while their updated residual is at least 10% of the
global maximum, bounded by pivot_batch_size (default 32). Stopping still tests
the global largest residual, so the PSD error bound is unchanged. This changes
complex molecular pivots/vectors and may slightly change rank. Setting
pivot_batch_size=1 restores strict global pivoting and the old factors. NumPy
projection reductions per pivot remain unchanged. Factors, cache and selected-column
workspace are O(rank*nao^2) plus bounded native/column workspace; no dense
ERI or full ordered-pair matrix is formed. max_bytes remains an output cap.

Column screening remains absent; no new approximation or changed error bound
is introduced. Reproducible runs use benchmarks/cholesky_molecules.py; profile
with --profile and compare dense values with --dense (checking is outside the
timed decomposition). Final timings and validation follow after collection.

The larger molecular case is nonplanar NH3/aug-cc-pVTZ (130 Cartesian AOs),
with N=(0,0,.2) and three H centers at radius 1.8, z=-.5 bohr, separated by
120 degrees. It adds a center and exceeds both water basis sizes. The optional
benzene example was replaced after an exploratory 12-thread old-backend run
had exceeded nine minutes; that interrupted run supplies no timing result.
