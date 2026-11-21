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
Before/after 152-AO Ne and Be measurements and hashes are recorded below after
final controlled runs. The serial Ne baseline is 438.320 s; vector updates take
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
(no copy). The prior full suite passed 662 tests, followed by the added screening
check; final CI includes all checks. Exact formulae are in [the API](api.md).
Measured errors and timings are appended after final collection.

## Item 5: TD4.0 operators

Commit `b7f8063` adds kinetic angular momentum about independent G with
pi0=p+B0 cross (r-G0)/2, and magnetic kinetic with independent operator C/G at
fixed London B0/G0. Existing moment powers give all second moments about G.
Matching operator and phase fields delegate to the old kinetic kernel bitwise.

18 focused checks cover independent reference values through g, fixed-phase
operator-field differences, Hermiticity, gauge consistency and origin sum
rules. The optional `tests/external/td4_pyscf.py` compares H2O/cc-pVDZ (25
Cartesian AOs) with libcint angular momentum, quadrupole and kinetic matrices
at B0=0: maximum discrepancy 1.776e-14. The full suite passed 680 tests before
the additional gauge-consistency check (681 total in final CI).

## Item 6: release preparation

Version 0.11.0 includes items 3–5 on top of the independently frozen 0.10.0
candidate. Wheel/source smoke tests exercise atom assembly, streamed density
contraction, TD4.0 operators and the derivative priorities. The existing
CPython 3.11–3.14 macOS/Linux architecture matrix and protected trusted
publication workflow are retained. Final CI/artifact links are added after
completion. Publishing requires a separate explicit v0.11.0 approval.

NMR shielding/spin-spin operators, Cholesky-vector derivatives and spherical
functions remain out of scope. No additional design decision is outstanding;
the publication decisions remain with the maintainer.
