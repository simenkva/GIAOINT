# Milestone 9 implementation plan: ERI performance

Status: planned, not started.

This document is the detailed implementation plan for Milestone 9. It exists
separately from `docs/implementation_plan.md` because it needs to carry
measured profiling data and a staged rollout that would otherwise clutter the
per-milestone summary. `docs/implementation_plan.md` links here rather than
repeating this content.

## 1. Motivation and measured baseline

Milestone 6 recorded ERI throughput but did not profile it. STATUS.md
documents the qualitative limitation ("ERIs remain correctness-first and
scalar"); this plan replaces that qualitative statement with a measured one
and a concrete scope.

### Method

A standalone microbenchmark was linked against the Milestone 8 release static
library (`build/presets/release/libgiao_integrals_core.a`) and profiled with
macOS `sample` (stack sampling, 1 ms interval) while repeatedly computing one
d-d-d-d shell quartet (2 primitives/shell, non-aligned centers, zero field).
Separately, `examples/performance_comparison.py` (added alongside this plan)
gives an end-to-end, basis-set-scale comparison against PySCF/libcint on the
same geometry and basis data.

### Finding 1: the cost is concentrated in one function

Of 4,880 top-of-stack samples, 4,695 (96%) landed inside the recursive
`auxiliary` lambda in `EriKernel::compute`
(`src/eri.cpp:242-319`), and another 167 (3.4%) in the
enclosing function. Combined, 99.6% of wall time is spent building the
six-index Hermite/Boys-derivative auxiliary tensor ("R-tensor"). Boys
evaluation, `gaussian_product`, and Hermite-coefficient construction are each
under 1%.

### Finding 2: per-quartet cost scales worse than the combinatorics alone predict

Single-core rate for the same shell geometry at increasing angular momentum
(2 primitives/shell, zero field):

| L | shell class | primitive x Cartesian combinations | measured rate |
|---|---|---|---|
| 0 | s-s-s-s | 16 | 214,691 quartets/s |
| 1 | p-p-p-p | 1,296 (81x more) | 730 quartets/s (294x slower) |
| 2 | d-d-d-d | 20,736 (16x more) | 5.6 quartets/s (130x slower) |

The call count grows 81x then 16x between steps, but the measured rate drops
294x then 130x. Each individual `primitive_eri` call is *also* getting more
expensive as L grows, not just more numerous — consistent with a recursion
tree whose node count grows faster than the requested tensor size, which is
the expected cost profile of a top-down memoized recursion versus an
iterative bottom-up fill.

### Finding 3: zero field does not currently cost less than finite field

Zero field vs. a representative finite field, same geometry: 214,691/s vs
190,318/s (s-s-s-s), 730/s vs 716/s (p-p-p-p), 5.60/s vs 5.49/s (d-d-d-d) —
a 6-11% gap, not the 2-4x a real-arithmetic implementation would give. The
recursion is `std::complex<double>` unconditionally; it never checks whether
the field is exactly zero, so it always pays full complex-multiply cost even
though every imaginary part is then structurally zero. This is an
opportunity, not evidence the opportunity is already captured.

### Conclusion

The bottleneck is the six-index auxiliary recursion's *architecture*
(top-down, memoized, complex-general, per-node array construction), not the
Boys function, not `gaussian_product`, and not contraction/normalization
bookkeeping. Optimization effort belongs there first.

## 2. Scope, in priority order

**A. Rewrite the R-tensor recursion from recursive+memoized to iterative,
bottom-up.** This is the primary item: it is where 99.6% of measured time
already is, and its scaling is the specific problem in Finding 2.

**B. Add a collapsed real-arithmetic fast path for exact zero field.**
`docs/algorithms.md` section 6 already documents why the six-index tensor
cannot collapse to a single three-index table *at finite field*: "the two
derivative triples cannot be collapsed using a sign and combined index." That
constraint does not apply at zero field. The standard real MD/Hermite-Coulomb
algorithm collapses the bra triple `(t,u,v)` and ket triple `(tau,phi,chi)`
into one combined-order table using the `(-1)^{tau+phi+chi}` sign identity,
which shrinks both the table and the recursion tree from a six-index to a
three-index structure — this is a structural win on top of, and likely larger
than, the real-vs-complex arithmetic saving. It applies exactly when
`field.B == (0,0,0)` exactly (gauge origin is then irrelevant).

**C. Hoist per-quartet-invariant `gaussian_product` calls out of the
per-Cartesian-component loop** in `compute_eri`
(`src/eri.cpp:466-497`). `gaussian_product(a, b, field)` depends only on
primitive exponents, centers, and field — not on which Cartesian component is
being evaluated — yet it is currently recomputed, including two `exp` calls
per call (one of them complex), for every one of up to `cartesian_count^4`
component combinations sharing the same primitive quartet. Profiling shows
this is under 1% of current wall time, so it is a correctness-neutral
cleanup to fold into the item A refactor, not a standalone priority.

**D. Screening and threading guidance.** Schwarz screening
(`screening_threshold`) and OpenMP (`threads=`) are implemented and measured
(Milestone 6) but opt-in. Once A-C land, per-quartet cost drops enough that
threading and screening become proportionally more valuable. This item is
documentation/guidance plus a decision on whether to change any Python
default; it does not touch the numerical kernel.

**Explicitly not in scope**, based on the measurements above:

- Replacing the Boys-function backend (the "exponential-sum" idea mentioned
  in `docs/algorithms.md` section 9 and `STATUS.md`). Boys evaluation is
  under 1% of current time; optimizing it first would not move the needle.
- Swapping in the benchmark-only direct Obara-Saika prototype as the
  production ERI backend. Milestone 6 found it "competitive but mixed across
  its small shell matrix," and it does not address the recursion-architecture
  problem identified here for the MD path either.
- The spherical-transform layer (`docs/spherical_transform_design.md`) is an
  independent, unrelated line of work.

## 3. Non-goal: parity with libcint

libcint is a mature, hand-tuned, real-arithmetic-only C library. This
milestone's target is a measured, order-of-magnitude improvement on the
architecture problem identified above, not parity with a decades-tuned real
engine. Complex/London generality at finite field will remain intrinsically
more expensive than a real-only engine; the goal is to stop paying
architectural overhead on top of that intrinsic cost.

## 4. Staged rollout

Each phase keeps the current recursive implementation as a correctness
oracle until its replacement is proven equivalent, per the project rule
"never use the optimized implementation as its sole validation reference."

### Phase 9.1 — Iterative general-field rewrite (no formula change)

Replace the recursive `auxiliary` lambda with an iterative fill of the same
six-index table using the same recurrence relations and the same complex
arithmetic, only changing control flow: fill by ascending combined degree
`t+u+v+tau+phi+chi` (each degree's entries depend only on the previous
degree's entries at the same and next Boys order `n`), using precomputed flat
strides instead of `auxiliary_index` calls and `std::array<size_t,7>`
construction per node.

*Tests and exit gate*: every existing ERI test passes unchanged (Milestone 5
80-digit Obara-Saika oracle, symmetry/exchange/conjugation tests, Hermiticity,
zero-field, gauge-origin, PySCF/libcint zero-field comparison, sanitizer
builds). Add a randomized property test asserting the iterative and (still
present) recursive implementations agree to machine precision across random
shell quartets through the currently tested angular momentum, then remove the
recursive path. Record before/after JSON-lines benchmarks on the development
machine for the existing shell classes plus s-s-s-s/p-p-p-p/d-d-d-d at the
geometry used in this plan's profiling.

### Phase 9.2 — Zero-field collapsed real fast path

Implement the collapsed single-index, real-arithmetic R-tensor recursion
(item B) as an additional path selected only when `field.B == (0,0,0)`
exactly. The general complex path from Phase 9.1 (itself already checked
against the Obara-Saika oracle) is the immediate cross-check for this new
path; it is not trusted as its own oracle.

*Tests and exit gate*: a dedicated equality test comparing the fast path
against the general path at zero field over randomized shells through the
currently tested angular momentum, plus the full existing zero-field test
suite (analytic cases, PySCF/libcint comparison, Hermiticity, symmetry). An
explicit boundary test confirms a tiny nonzero field still takes the general
path. Record before/after benchmarks; report the zero-field speedup
separately from the finite-field (Phase 9.1-only) speedup.

### Phase 9.3 — Cleanup

Hoist `gaussian_product` calls per item C. No behavior change; existing tests
must pass unchanged.

### Phase 9.4 — Screening and threading guidance

Decide, and document, whether the Python `eri()` default
(`screening_threshold=0.0`, `threads=1`) should change now that per-quartet
cost is lower, or whether the change stays opt-in with clearer documentation
and an updated `examples/performance_comparison.py` demonstration. Any
default change is a compatibility-policy decision (`docs/compatibility.md`)
and needs explicit maintainer sign-off, not just a benchmark improvement.

### Phase 9.5 — Re-benchmark and document

Re-run `benchmarks/giao_integral_benchmark`, `examples/performance_comparison.py`,
and the profiling method from Section 1 on the same development machine.
Update `STATUS.md`'s benchmark section, `benchmarks/results/`, and this
document's Section 1 with final measured numbers. Close the milestone in
`docs/implementation_plan.md` with the same "exit gate passed with..."
paragraph style used for Milestones 1-8.

## 5. Risks

| Risk | Failure mode | Planned control |
|---|---|---|
| Recursion-to-iteration reordering bug | Silently wrong ERIs for some angular-momentum/order combination not covered by existing tests | Keep the recursive path as a temporary oracle (Phase 9.1); add randomized iterative-vs-recursive equality tests before deletion; do not touch the recurrence relations themselves, only control flow |
| Zero-field fast-path activation is imprecise | Fast path silently used at near-zero-but-nonzero field, or general path silently used at exact zero losing the speedup | Activation predicate is exact floating-point equality to zero on `field.B`, tested explicitly at the boundary (Phase 9.2) |
| Changing a Python default (screening/threads) breaks existing caller expectations | Silent behavior change for existing integrations | Treat as a compatibility-policy decision (Phase 9.4); default stays unscreened/serial unless explicitly decided otherwise |
| Benchmark claims are not reproducible off this machine | Reported speedups mislead other environments | Follow `benchmarks/README.md` practice: archive raw JSONL with compiler/CPU/OS/thread count; report ratios only against the same machine |

## 6. Milestone 9 exit gate (summary)

- Phases 9.1-9.3 land with every existing correctness test green (Python and
  C++, release and ASan/UBSan builds) and no change to any documented
  formula, convention, or public API shape.
- Phase 9.2's fast path is exercised by the existing zero-field test paths
  (including the PySCF/libcint comparison) with no tolerance loosening.
- Measured before/after numbers, using the same profiling and benchmark
  methodology as Section 1, are recorded in `STATUS.md` and
  `benchmarks/results/`.
- Phase 9.4's default-change decision (if any) is recorded explicitly in
  `docs/compatibility.md` and `CHANGELOG.md`.
