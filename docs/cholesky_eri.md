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
Diagonal elements come from `(MN|NM)`. Each pivot is selected from the
global residual maximum. The uncontracted backend retains up to eight candidate
columns from one shell pair. The 0.11.1 molecular preparation shares primitive
component integrals across general contraction rows, generates up to 32 matching
contraction variants, and retains at most min(1024,rank+8) raw columns in an LRU.
The ERI kernel, primitive contraction order and per-pivot projection reductions
remain unchanged. Complex contracted factors use local pivot blocks (up to 32,
while each selected residual is at least 1% of the global maximum).
`pivot_batch_size=1` restores the old global pivot order; zero-field factors and
uncontracted atoms keep their original order. Stopping always uses the global
largest residual, independently of the pivot batching policy. Native work is parallel across shell pairs or selected-entry tasks.
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
