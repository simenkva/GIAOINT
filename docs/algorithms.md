# Algorithm decisions

## 1. Decision summary

Use a McMurchie-Davidson (MD)/Hermite formulation for the first auditable
production path. Use direct complex-center polynomial moments in the Python
reference so the reference and production implementations do not share their
main recurrence. Hide recurrence details behind shell-level drivers, then add
an Obara-Saika/Head-Gordon-Pople backend only after MD supplies trusted values
and benchmarks identify a worthwhile target.

This is a correctness and architecture decision, not a claim that MD will be
the fastest final kernel.

## 2. Recurrence comparison

| Family | Fit to London Gaussians | Strengths | Main costs or risks | Decision |
|---|---|---|---|---|
| McMurchie-Davidson/Hermite | Direct. Standard field-free Hermite expansion coefficients survive; London dependence moves into Fourier-transformed Hermites and complex Coulomb auxiliaries. | Compact mathematics, arbitrary Cartesian angular momentum, natural moments and derivatives, direct precedent for finite-field London orbitals. | Six-index Coulomb auxiliary tables lose a zero-field dimensional reduction; transforms and temporary storage can cost more than direct Cartesian recurrences. | Initial reference-quality C++ backend and first Coulomb backend. |
| Obara-Saika (OS) | Valid after complex corrections to product-center terms and complex Boys seeds. | Builds Cartesian integrals directly, avoids an explicit Hermite-to-Cartesian transform, established performance path. | More recurrence branches to audit; field-dependent imaginary shifts make sign errors easy; contraction strategy needs care. | Prototype after the MD ERI oracle passes. |
| Head-Gordon-Pople (HGP) | Extends OS with recurrence placement that can move work outside contraction loops. | Strong candidate for contracted ERIs and derivatives; known field-free efficiency. | Requires a trusted OS base and recurrence-path planner; complex London symmetry reduces familiar shortcuts. | Candidate optimized backend in Milestone 6. |
| Rys quadrature | Formally extendable, but roots and weights become complex when the Boys argument is complex. | Efficient at higher angular momentum in conventional engines; separable Cartesian factors. | Complex quadrature-root generation, root ordering, and coalescence introduce a second difficult special-function problem. | Defer until complex-root conditioning is demonstrated. |
| Direct quadrature/Fourier integration | Useful as an independent low-order check. | Different numerical route, simple to reason about for special cases. | Too slow and difficult to bound for production four-center integrals. | Tests only. |

Tellgren, Soncini, and Helgaker derive both MD and OS London recurrences. Their
MD construction keeps the usual Hermite coefficients and evaluates the London
factor through Fourier transforms. That gives each equation a short path back
to the conventional zero-field implementation. It also supports the planned
property-operator and derivative milestones without separate low-angular-
momentum formulas.

## 3. Initial MD organization

The engine will separate four layers:

1. Build a shell pair: primitive-pair exponents, real product centers, Gaussian
   prefactors, pair wave vectors, and complex phase/damping factors.
2. Build one-dimensional field-free Hermite coefficients for each primitive
   pair and Cartesian component.
3. Apply an operator-specific auxiliary: a Fourier moment for overlap and
   local moments, a differential raising/lowering expression for kinetic and
   momentum, or a complex Boys/R-tensor for Coulomb operators.
4. Contract primitives and write a complete shell block in documented order.

The kernel must allocate its largest scratch tables once per engine or worker,
not once per primitive pair. Table extents derive from shell angular momenta;
there will be no s/p/d/f switch as the fundamental implementation.

For overlap, a Hermite term integrates as

\[
\widehat\Lambda_{tuv}(\mathbf q)
=\left(\frac\pi p\right)^{3/2}
(-iq_x)^t(-iq_y)^u(-iq_z)^v
e^{-i\mathbf q\cdot\mathbf P-\mathbf q^2/(4p)}.
\]

At \(\mathbf q=0\), all terms with a nonzero Hermite index vanish. This gives
a sharp zero-field regression check and avoids special formulas.

## 4. Complex Boys strategy

### 4.1 Contract

The numerical function is

\[
F_n(z)=\int_0^1t^{2n}e^{-zt^2}dt,
\qquad n\in\mathbb N_0,\ z\in\mathbb C.
\]

It must return all orders \(0\ldots n_{\max}\) in one call, because Coulomb
recurrences consume a sequence. The production contract uses `complex128`,
reports non-finite inputs, and defines accuracy against the larger of absolute
and relative tolerances. It must support negative real parts because London
complex centers can generate them.

### 4.2 Independent high-precision references

Milestone 4 will use at least two mpmath routes at 80 or more decimal digits:

- direct quadrature of the defining finite-interval integral; and
- the entire confluent-hypergeometric expression
  \(F_n(z)={}_1F_1(n+1/2;n+3/2;-z)/(2n+1)\).

The two references must agree before either certifies production results. Near
branch cuts, the incomplete-gamma formula serves only as a third diagnostic.

### 4.3 Production prototype regions

Prototype a piecewise algorithm, then freeze its regions from error maps rather
than from a guessed threshold:

- Near the origin, sum
  \[
  F_n(z)=\sum_{k=0}^{\infty}
  \frac{(-z)^k}{k!(2n+2k+1)}
  \]
  with a compensated sum and a truncation bound.
- For moderate complex arguments, independently implement the exponential-sum
  strategy of Beylkin and Sharma (2021), using the paper's equations as a
  reference and verifying generated constants rather than copying source code.
- For large arguments in safe sectors, test asymptotic seeds followed by the
  stable recurrence direction.
- For large negative real parts, evaluate scaled functions or a combined
  pair-prefactor/Boys auxiliary. Computing \(e^{-z}\) and \(F_n(z)\) separately
  can overflow before their product is formed.

The integration-by-parts recurrence is

\[
F_{n+1}(z)=\frac{(2n+1)F_n(z)-e^{-z}}{2z}.
\]

It suffers cancellation near \(z=0\). Upward and downward stability also
depends on \(z\) and the highest required order. The implementation will choose
a direction from validated regions and check the recurrence residual in debug
and test builds.

### 4.4 Domain discovery and acceptance gate

Before setting production regions, generate the complex \(T\) values reached
by randomized physically plausible shells:

- exponents from \(10^{-4}\) to \(10^6\);
- inter-center distances from coincident to 30 bohr;
- field magnitudes from zero through a documented strong-field ceiling;
- non-axis-aligned field and geometry vectors;
- angular momentum through the milestone's tested maximum.

Augment this reachable set with a rectangular adversarial grid, points on both
sides of the real axis, and neighborhoods of region boundaries. A candidate
passes only if it meets the stated tolerance against high precision, has no
discontinuity across dispatch boundaries, and avoids intermediate overflow
whenever the final scaled auxiliary is representable.

The 2008 finite-field implementation used three regions and noted that its
negative-real-part branch did not occur in its test systems. That observation
does not justify excluding the region here. The 2021 Beylkin-Sharma algorithm
covers real and complex arguments, including separate handling for negative
real parts, and is the leading production candidate.

## 5. One-electron operator reuse

Overlap, Cartesian moments, and derivatives will share one-dimensional
raising/lowering machinery. For example, multiplication by \(x-C_x\) raises a
ket polynomial and adds \(B_x-C_x\); differentiation of a ket London primitive
adds both ordinary Gaussian terms and \(-i\kappa_{B,x}\) times the primitive.
Kinetic energy applies the derivative machinery twice and contracts the
result. This avoids operator-specific s/p/d/f formulas.

Nuclear attraction uses the same shell-pair and Hermite data as overlap, with
complex Coulomb auxiliaries centered at each nucleus. The driver accumulates
the nuclear charge sum in a fixed order for reproducibility.

## 6. ERI tensor and symmetry handling

The first ERI path computes complete shell quartets without applying invalid
within-pair permutations. At finite field it may reuse only

\[
(ab|cd)=(cd|ab),\qquad (ab|cd)^*=(ba|dc).
\]

The quartet scheduler will encode those transformations by named operations
that state whether conjugation is required. It will not encode them as an
eightfold integer-index canonicalizer inherited from a real engine.

The MD Coulomb auxiliary at finite field depends on derivatives with respect
to both pair centers. Unlike the zero-field case, the two derivative triples
cannot be collapsed using a sign and combined index. Scratch-size estimates
and benchmarks must use the full tensor.

## 7. Screening policy

Milestone 5 initially runs unscreened. Screening enters only after a bound has
been proved for the modulus of London shell quartets. The real plane-wave
factor has unit modulus at the AO level, but complex-center rearrangements
contain damping and potentially growing auxiliaries; bounds must apply to the
original integral or a cancellation-safe equivalent.

A screening implementation must expose its threshold, preserve the unscreened
path, and pass monotone-convergence tests as the threshold tightens. The code
will distinguish rigorous upper bounds from heuristic estimates. Heuristics
may order work but may not discard quartets.

## 8. Optimization sequence

Optimization begins after reference agreement:

1. reuse scratch buffers and precompute shell-pair invariants;
2. choose contraction loop order from measurements;
3. reduce temporary table extents and improve contiguous access;
4. add shell-pair/quartet screening with a validated bound;
5. compare MD with OS/HGP for representative shell classes;
6. parallelize independent shell blocks with thread-local scratch;
7. consider SIMD or generated kernels only when profiles justify them.

Each change keeps a scalar correctness path and records before/after benchmark
data with compiler, CPU, field, basis, shell class, and checksum.

