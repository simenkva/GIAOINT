# Streamed pivoted Cholesky ERIs

`Molecule.cholesky_eri()` and `gi.cholesky_eri(basis, field=...)` return
`CholeskyERI(vectors, pivots, threshold, max_residual_diagonal)`. The factors
use the same Cartesian AO ordering, normalization, and complex128 dtype as
`eri()`:

```python
chol = mol.cholesky_eri(threshold=1e-8, max_bytes=None)
# (mu nu | lam sig) ≈ sum_P L[P,mu,nu] * conj(L[P,sig,lam])
L = chol.vectors
```

Since 0.9, `threads=None` selects the configured default; `threads=1` remains
explicitly serial. See [installation and thread policy](installation.md).
The historical measurements below explicitly used one thread. Current
1/4/8/12-thread Ne and Be comparisons are in the
[0.11 validation report](release_0_11_report.md).

The Hermitian PSD matrix being factored is
`W[(mu,nu),(lam,sig)] = (nu mu | lam sig)`, with **nao² ordered pairs**.
Internally the implementation stores conjugated Cholesky columns, so it
fetches `(MN|SL)` for pivot `(L,S)`. This is equivalent to Cholesky on W.
Diagonal elements come from `(MN|NM)`. The global residual maximum controls
stopping. The uncontracted single-center backend retains up to eight candidate
columns from one shell pair. The 0.11.1 molecular preparation shares primitive
component integrals across general contraction rows, generates up to 32 matching
contraction variants, and retains at most min(1024,rank+8) raw columns in an LRU.
The ERI kernel, primitive contraction order and per-pivot projection reductions
remain unchanged. Complex contracted factors use local pivot blocks (up to 32,
while each selected residual is at least 1% of the global maximum).
`pivot_batch_size=1` restores the old global pivot order; zero-field factors and
uncontracted atoms keep their original order. Stopping always uses the global
largest residual, independently of the pivot batching policy. Native work is
parallel across shell pairs or selected-entry tasks.
See [the molecular profile and benchmarks](cholesky_molecules_0_11_1.md).
The older full-shell-quartet extraction backend remains the historical
comparison baseline in the 0.11 report.

There is no full AO tensor or full pair matrix. Factor storage grows
geometrically and uses O(M*nao²) memory; growth and final complex128 conversion
can temporarily keep another factor allocation alive. `max_bytes` bounds the
returned vector and pivot arrays, rather than process RSS or workspace.
Screening never removes column entries: Schwarz bounds can only replace
small initial diagonal blocks by conservative upper bounds. Skipped diagonals
are measured at the end to report the actual residual maximum. Those bounds
currently require the existing backend's own diagonal computations, so this
screening mode is conservative rather than a major speed optimization.

At zero field, the implementation uses real arithmetic and enforces symmetric
vectors. For a single-center basis, the common London phase cancels exactly
in every density even at finite field, permitting the same real ERI backend.
For multicenter finite fields, individual factors need not be Hermitian;
finite-rank reconstructions satisfy the two ERI symmetries to within twice
the elementwise error bound. Numerical exhaustion recovers both to roundoff.
The stopping floor is `64*eps*max(initial diagonal)`, including for
`threshold=0`; the result reports the achieved residual rather than claiming
an unattainable zero error.

<!-- BEGIN 0.11.1 molecular benchmarks -->
## Molecular performance in 0.11.1

Apple M2 Pro, 12 physical cores, 32 GiB RAM; macOS arm64, CPython 3.13.3,
NumPy 2.5.5/Accelerate and BSE 0.12. BLAS thread limits are both 1.
Before: published 0.11.0 arm64 wheel (AppleClang 17, bundled OpenMP).
After: 0.11.1 source (AppleClang 21, Homebrew libomp 23.1.1).
Each before/after/count/case is a fresh process, measured sequentially.
These are single runs on a shared developer host, with no affinity control.
Load columns give starting one-minute averages; full starting/ending
1/5/15-minute averages, hashes and build information are in
[the raw data](../benchmarks/results/cholesky_molecules_0_11_1_macos_arm64.json).

Water geometry and B=(.03,-.02,.05), gauge origin=(.7,-.4,1.1) bohr follow
the downstream reproducer. NH3 has 130 Cartesian AOs and four nonplanar centers;
its geometry is recorded in [the profile report](cholesky_molecules_0_11_1.md).
Atoms use Bz=.1 and the three requested BSE uncontracting flags. Threshold
is 1e-8 throughout. Basis loading, hashing and optional dense validation are
outside Cholesky timing. Dense output allocation is included in dense timing.
RSS is measured before the optional dense call and includes Python/basis data,
workspace, temporary factor growth and conversion, not just returned vectors.

All before and all after results separately have identical factors, pivots,
ranks and residuals across 1/4/8/12 threads. Real and atom outputs also match
0.11.0 bitwise. Complex molecular pivot order changes; `pivot_batch_size=1`
restores the old global pivots and factors. Default rank changes stay below
3% in these cases, and every residual stays below the requested threshold.
The [profile and validation report](cholesky_molecules_0_11_1.md) gives the
before/after phase breakdown, work counts and reconstruction checks.

### H2O cc-pVTZ, zero field

65 Cartesian AOs; rank 656 → 656 (M/nao=10.092);
maximum residual 9.88591e-09. Returned vectors: 42.29 MiB.
Largest Cholesky process RSS, before/after: 116.66/170.20 MiB;
dense complex tensor estimate: 0.27 GiB.

| Threads | Before s | After s | Dense s | Before scaling | After scaling | Start load before/after |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 325.698 | 3.511 | 3.682 | 1.00x | 1.00x | 16.32/6.72 |
| 4 | 92.980 | 1.171 | 1.022 | 3.50x | 3.00x | 6.03/8.13 |
| 8 | 52.599 | 0.781 | 0.588 | 6.19x | 4.49x | 35.84/21.78 |
| 12 | 43.295 | 0.672 | 0.477 | 7.52x | 5.22x | 4.63/12.81 |

### H2O cc-pVTZ, finite field

65 Cartesian AOs; rank 668 → 687 (M/nao=10.569);
maximum residual 9.99393e-09. Returned vectors: 44.29 MiB.
Largest Cholesky process RSS, before/after: 141.08/224.83 MiB;
dense complex tensor estimate: 0.27 GiB.

| Threads | Before s | After s | Dense s | Before scaling | After scaling | Start load before/after |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 747.533 | 9.325 | 7.702 | 1.00x | 1.00x | 7.38/15.34 |
| 4 | 196.712 | 3.092 | 2.032 | 3.80x | 3.02x | 9.00/7.36 |
| 8 | 112.217 | 2.049 | 1.144 | 6.66x | 4.55x | 20.84/12.63 |
| 12 | 93.469 | 1.826 | 0.948 | 8.00x | 5.11x | 12.81/4.01 |

### H2O aug-cc-pVTZ, zero field

105 Cartesian AOs; rank 932 → 932 (M/nao=8.876);
maximum residual 9.99691e-09. Returned vectors: 156.79 MiB.
Largest Cholesky process RSS, before/after: 289.08/386.25 MiB;
dense complex tensor estimate: 1.81 GiB.

| Threads | Before s | After s | Dense s | Before scaling | After scaling | Start load before/after |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 340.307 | 8.558 | 19.381 | 1.00x | 1.00x | 13.86/14.89 |
| 4 | 91.404 | 3.592 | 5.503 | 3.72x | 2.38x | 7.49/6.43 |
| 8 | 52.477 | 3.193 | 3.631 | 6.48x | 2.68x | 12.33/12.24 |
| 12 | 42.297 | 2.835 | 3.245 | 8.05x | 3.02x | 4.01/10.74 |

### H2O aug-cc-pVTZ, finite field

105 Cartesian AOs; rank 953 → 967 (M/nao=9.210);
maximum residual 9.86882e-09. Returned vectors: 162.68 MiB.
Largest Cholesky process RSS, before/after: 379.91/414.14 MiB;
dense complex tensor estimate: 1.81 GiB.

| Threads | Before s | After s | Dense s | Before scaling | After scaling | Start load before/after |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 908.371 | 27.790 | 47.706 | 1.00x | 1.00x | 17.53/10.50 |
| 4 | 239.528 | 10.906 | 13.268 | 3.79x | 2.55x | 6.14/5.89 |
| 8 | 136.714 | 7.969 | 7.818 | 6.64x | 3.49x | 11.98/9.93 |
| 12 | 116.020 | 7.528 | 6.593 | 7.83x | 3.69x | 11.24/25.83 |

### NH3 aug-cc-pVTZ, zero field

130 Cartesian AOs; rank 1158 → 1158 (M/nao=8.908);
maximum residual 9.93792e-09. Returned vectors: 298.62 MiB.
Largest Cholesky process RSS, before/after: 502.97/693.38 MiB;
dense complex tensor estimate: 4.26 GiB.

| Threads | Before s | After s | Dense s | Before scaling | After scaling | Start load before/after |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 576.668 | 15.608 | 40.400 | 1.00x | 1.00x | 10.36/12.29 |
| 4 | 158.392 | 7.922 | 13.591 | 3.64x | 1.97x | 6.71/7.52 |
| 8 | 89.044 | 7.575 | 8.159 | 6.48x | 2.06x | 9.49/14.60 |
| 12 | 75.380 | 6.850 | 7.697 | 7.65x | 2.28x | 21.85/24.51 |

### NH3 aug-cc-pVTZ, finite field

130 Cartesian AOs; rank 1174 → 1192 (M/nao=9.169);
maximum residual 9.94562e-09. Returned vectors: 307.39 MiB.
Largest Cholesky process RSS, before/after: 644.61/733.27 MiB;
dense complex tensor estimate: 4.26 GiB.

| Threads | Before s | After s | Dense s | Before scaling | After scaling | Start load before/after |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1911.998 | 75.689 | 148.137 | 1.00x | 1.00x | 10.64/15.72 |
| 4 | 424.188 | 21.104 | 28.191 | 4.51x | 3.59x | 7.82/13.98 |
| 8 | 238.779 | 16.629 | 16.133 | 8.01x | 4.55x | 11.67/11.21 |
| 12 | 216.905 | 15.360 | 14.984 | 8.81x | 4.93x | 28.11/26.36 |

### Ne uncontracted aug-cc-pCVQZ

152 Cartesian AOs; rank 1052 → 1052 (M/nao=6.921);
maximum residual 7.48128e-09. Returned vectors: 370.87 MiB.
Largest Cholesky process RSS, before/after: 803.33/802.28 MiB;
dense complex tensor estimate: 7.95 GiB.

| Threads | Before s | After s | Dense s | Before scaling | After scaling | Start load before/after |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 44.362 | 43.539 | — | 1.00x | 1.00x | 7.23/6.26 |
| 4 | 12.226 | 12.509 | — | 3.63x | 3.48x | 11.45/14.06 |
| 8 | 8.685 | 8.841 | — | 5.11x | 4.92x | 8.76/8.40 |
| 12 | 7.707 | 7.705 | — | 5.76x | 5.65x | 27.09/24.72 |

### Na uncontracted aug-cc-pCVQZ

208 Cartesian AOs; rank 1275 → 1275 (M/nao=6.130);
maximum residual 9.67958e-09. Returned vectors: 841.70 MiB.
Largest Cholesky process RSS, before/after: 1332.28/1329.39 MiB;
dense complex tensor estimate: 27.89 GiB.

| Threads | Before s | After s | Dense s | Before scaling | After scaling | Start load before/after |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 105.696 | 105.125 | — | 1.00x | 1.00x | 9.13/10.73 |
| 4 | 32.772 | 34.593 | — | 3.23x | 3.04x | 15.37/12.09 |
| 8 | 24.347 | 24.762 | — | 4.34x | 4.25x | 8.00/6.09 |
| 12 | 28.057 | 23.265 | — | 3.77x | 4.52x | 23.78/47.97 |

<!-- END 0.11.1 molecular benchmarks -->

The original single-run Na/four-thread timing is retained above. Three
additional alternating-order comparisons using the final wheel have median
0.11.0/candidate times 64.359/63.178 s, with all hashes unchanged. See
[the repeat measurements and compiler/load provenance](cholesky_molecules_0_11_1.md#complete-matrix-and-final-installed-wheel-checks).

## Validation and small cases

`tests/python/test_cholesky.py` checks H2O STO-3G and 6-31G, an O atom
(STO-3G, an open-shell electronic configuration), and displaced primitive d
and f shells. Occupations are irrelevant to these AO integrals. Every case
is checked at zero field and B=(0.05,-0.03,0.2), with off-symmetry gauge origin
(0.37,-0.29,0.43) bohr, at thresholds 1e-4, 1e-6, 1e-8, 1e-10 and numerical
exhaustion. Tests compare to unscreened `eri()`, check both symmetry relations,
measure the residual diagonal independently, compare two gauge origins, and
compare zero-field factors with an independent dense real-arithmetic Cholesky.
A separate single-center d/f test checks phase cancellation against finite-field
shell integrals. No derivative or MO-transform API changes are involved.

The original Cholesky validation passed 438 Python tests, with only existing
BSE deprecation warnings. The historical 0.9 validation is recorded in
[its release report](release_0_9_report.md); the 0.10 candidate passes 621 tests
and the 0.11 candidate passes 681. See
[current validation](release_0_11_report.md).
The measurements below preserve the earlier serial benchmark results.
An isolated OpenMP build also checks the Cholesky suite, and finite-field
H2O 6-31G gives identical factors and pivots with one and two ERI threads.

Representative serial measurements on macOS 26.6.2 arm64, Python 3.13.3,
NumPy 2.5.3/Accelerate (one BLAS thread; wall time includes
initial diagonals and factorization, excluding full-reference ERIs):

| Case | Field | nao | M at 1e-6 / 1e-8 | M/nao at 1e-8 | Seconds at 1e-6 / 1e-8 | Vector bytes at 1e-8 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H2O STO-3G | zero | 7 | 28 / 28 | 4.00 | 0.041 / 0.040 | 21,952 |
| H2O STO-3G | finite | 7 | 36 / 39 | 5.57 | 0.103 / 0.107 | 30,576 |
| H2O 6-31G | zero | 13 | 77 / 84 | 6.46 | 0.168 / 0.181 | 227,136 |
| H2O 6-31G | finite | 13 | 91 / 114 | 8.77 | 0.395 / 0.508 | 308,256 |
| O STO-3G | either | 5 | 15 / 15 | 3.00 | 0.011 / 0.011 | 6,000 |
| Displaced d/f shells | zero | 16 | 85 / 85 | 5.31 | 0.423 / 0.421 | 348,160 |
| Displaced d/f shells | finite | 16 | 105 / 106 | 6.63 | 5.350 / 5.426 | 434,176 |

For the memory regression, 20 uncontracted one-center s shells need
2,560,000 bytes for dense ERIs. With a 640,000-byte output cap, Cholesky
succeeds at 1e-8 with M=13, 83,200 vector bytes, and approximately 193,000
peak traced allocation bytes. Dense `eri()` rejects the same cap. The test
also forbids calls to the dense backend. Traced allocations include NumPy
arrays and Python objects but exclude native C++ workspace; the Be benchmark
below reports process peak RSS, including native allocations.

## Be aug-cc-pCVQZ, uncontracted

The benchmark removes contractions and deduplicates primitive exponents per
angular momentum, giving 16s 10p 6d 4f 2g = 152 Cartesian AOs. B=(0,0,0.1).
The full complex128 ERI tensor would occupy 8,540,717,056 bytes (7.95 GiB).
Reproduce each threshold in a separate process:

```sh
GIAO_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python benchmarks/cholesky_eri.py 1e-6
GIAO_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python benchmarks/cholesky_eri.py 1e-8
```

The ERI build used for timing is serial (OpenMP disabled), with one BLAS
thread. These are workstation wall times rather than isolated performance
measurements; the two threshold runs overlapped in separate processes.

| Threshold | M | M/nao | Wall seconds | Vector MiB | Peak process RSS MiB | Max residual diagonal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-6 | 815 | 5.362 | 468.75 | 287.32 | 515.14 | 9.008e-7 |
| 1e-8 | 1016 | 6.684 | 533.01 | 358.18 | 630.56 | 9.261e-9 |

Peak RSS includes Python, BSE, NumPy, all native quartet workspace, factor
capacity growth, and conversion of internal real factors to complex128.
Both runs stayed below 8% of the dense tensor's 7.95 GiB allocation.
Raw measurements: [JSON](../benchmarks/results/cholesky_macos_arm64.json).
