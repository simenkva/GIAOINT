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
| Head-Gordon-Pople (HGP) | Extends OS with recurrence placement that can move work outside contraction loops. | Strong candidate for contracted ERIs and derivatives; known field-free efficiency. | Requires a trusted OS base and recurrence-path planner; complex London symmetry reduces familiar shortcuts. | The Milestone 6 direct-OS prototype was competitive but mixed; a contracted HGP prototype remains future work. |
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

### 3.1 Implemented one-dimensional recurrence

Milestone 2 represents the unnormalized one-dimensional product as

\[
(x-A)^a(x-B)^b e^{-\alpha(x-A)^2-\beta(x-B)^2}
=K_{AB}\sum_{t=0}^{a+b}E_t^{ab}\Lambda_t(x;p,P),
\]

with (E_0^{00}=1), (p=\alpha+\beta), and the ordinary real Gaussian
prefactor (K_{AB}) kept outside the table. Angular momentum is raised by

\[
E_t^{a+1,b}=\frac{1}{2p}E_{t-1}^{ab}
 +(P-A)E_t^{ab}+(t+1)E_{t+1}^{ab},
\]

\[
E_t^{a,b+1}=\frac{1}{2p}E_{t-1}^{ab}
 +(P-B)E_t^{ab}+(t+1)E_{t+1}^{ab}.
\]

Terms outside the current table extent are zero. The implementation contracts
the finished real table directly with powers of (-iq). It allocates the two
recurrence rows and component-normalization tables before entering primitive
pair loops and reuses them for the complete shell block.

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

Milestone 4 uses two mpmath routes at 80 or more decimal digits:

- direct quadrature of the defining finite-interval integral; and
- the entire confluent-hypergeometric expression
  \(F_n(z)={}_1F_1(n+1/2;n+3/2;-z)/(2n+1)\).

The two references must agree before either certifies production results. Near
branch cuts, the incomplete-gamma formula serves only as a third diagnostic.

### 4.3 Production regions

The accepted correctness-first implementation returns the complete sequence
in one call and dispatches as follows:

- Near the origin, sum
  \[
  F_n(z)=\sum_{k=0}^{\infty}
  \frac{(-z)^k}{k!(2n+2k+1)}
  \]
  with a compensated sum and a truncation bound.
- For \(|z|\le0.75\), use the compensated entire power series independently
  for every requested order.
- For \(|z|\le160\) outside the series disk, use adaptive composite
  Gauss--Legendre quadrature. A 16/32-point embedded comparison supplies the
  error diagnostic and drives subdivision.
- For \(\operatorname{Re}(z)\ge2n_{\max}+60\) with
  \(|\operatorname{Im}(z)|\le\operatorname{Re}(z)/2\), use the exponentially
  accurate positive-sector gamma asymptotic independently for every order.
- For large negative real parts, evaluate scaled functions or a combined
  pair-prefactor/Boys auxiliary. Computing \(e^{-z}\) and \(F_n(z)\) separately
  can overflow before their product is formed.

Arguments outside the direct disk and conservative positive asymptotic sector
raise `BoysNumericalError`; the implementation does not silently extrapolate
an unverified method. Orders above 32 are rejected. The Beylkin--Sharma
exponential-sum method remains a future performance alternative, not a
dependency of the accepted Milestone 4 correctness path.

The integration-by-parts recurrence is

\[
F_{n+1}(z)=\frac{(2n+1)F_n(z)-e^{-z}}{2z}.
\]

It suffers cancellation near \(z=0\). Upward and downward stability also
depends on \(z\) and the highest required order. The implementation will choose
a direction from validated regions and check the recurrence residual in debug
and test builds.

### 4.4 Domain discovery and acceptance gate

The committed reachable-domain map generates complex \(T\) values from:

- exponents from approximately 0.05 to 20;
- AO centers and nuclei with each coordinate in \([-2,2]\) bohr;
- field magnitudes from zero through 1 atomic unit;
- non-axis-aligned field and geometry vectors;
- angular momentum through the milestone's tested maximum.

The fixed-seed map compares five sampled orders through 32 for each of 120
physical arguments. It observed maximum absolute and relative errors of
\(2.28\times10^{-15}\) and \(1.06\times10^{-14}\), respectively, against
80-digit hypergeometric values. Separate adversarial tests cover both
half-planes, the 0.75 dispatch boundary, negative-real arguments through -120,
scaled/unscaled agreement, conjugation, and recurrence residuals. This is the
guaranteed Milestone 4 envelope; wider inputs may succeed only when they fall
in the explicitly checked asymptotic sector.

The 2008 finite-field implementation used three regions and noted that its
negative-real-part branch did not occur in its test systems. That observation
does not justify excluding the region here. The 2021 Beylkin-Sharma algorithm
covers real and complex arguments, including separate handling for negative
real parts, and is the leading production candidate.

## 5. One-electron operator reuse

Milestone 3 implements overlap, Cartesian moments, gradients, momentum, and
kinetic operators with shared angular-momentum shifts evaluated by the same MD
overlap kernel. Multiplication by \(x-C_x\) raises a ket polynomial and adds
\(B_x-C_x\); differentiation adds the ordinary lowering/raising terms and
\(-i\kappa_{B,x}\) times the primitive. Shifted terms retain the original
primitive normalization. This avoids operator-specific s/p/d/f formulas.

Canonical kinetic energy applies the first- and second-derivative identities
axis by axis. The physical magnetic kinetic kernel then composes canonical
kinetic energy, the paramagnetic
\(\tfrac12\boldsymbol{\mathcal B}\cdot\mathbf L_{\mathbf O}\) term from mixed
coordinate-momentum integrals, and the diamagnetic quadratic moment. The
production path uses MD overlaps for every shifted term; the Python reference
instead expands all polynomial factors directly about the complex center.

Nuclear attraction uses the same field-free shell-pair Hermite coefficients as
overlap. With \(X=P'_x-C_x\), its London Coulomb recurrence is

\[
R^n_{t+1,u,v}=-iq_xR^n_{tuv}+X R^{n+1}_{tuv}
              +tR^{n+1}_{t-1,u,v},
\]

with analogous y/z recurrences and seeds
\(R^n_{000}=e^{-i q\cdot P-q^2/(4p)}(-2p)^nF_n(T)\). When
\(\operatorname{Re}(T)<0\), the code instead forms \(e^T F_n(T)\) and uses
the simplified combined prefactor
\(e^{-p|P-C|^2-iq\cdot C}\), avoiding subtraction of large field-dependent
terms. The driver accumulates nuclei and primitives in input order and reuses
all recurrence storage after workspace warmup.

## 6. ERI tensor and symmetry handling

The first ERI path computes complete shell quartets without applying invalid
within-pair permutations. At finite field it may reuse only

\[
(ab|cd)=(cd|ab),\qquad (ab|cd)^*=(ba|dc).
\]

The quartet scheduler encodes that four-member orbit and reports whether the
selected representative requires conjugation. It does not use an eightfold
integer-index canonicalizer inherited from a real engine.

The MD Coulomb auxiliary at finite field depends on derivatives with respect
to both pair centers. Unlike the zero-field case, the two derivative triples
cannot be collapsed using a sign and combined index. Scratch-size estimates
and benchmarks must use the full tensor.

Milestone 5 implements the full six-index derivative recurrence with spherical
seed

\[
R^n_{000,000}=e^{-i\mathbf q_1\cdot\mathbf P-i\mathbf q_2\cdot\mathbf Q}
(-2\rho)^n F_n(T).
\]

The two pair London damping factors multiply this seed. For
\(\operatorname{Re}(T)<0\), production instead contracts scaled
\(e^T F_n(T)\) with the algebraically combined exponent

\[
-\rho|\mathbf P-\mathbf Q|^2
-\frac{|\mathbf q_1+\mathbf q_2|^2}{4(p+s)}
-i(\mathbf q_1+\mathbf q_2)\cdot
  \frac{p\mathbf P+s\mathbf Q}{p+s},
\]

so individually large factors are never formed. The correctness-first table
allows combined Cartesian order through 32 and caps an individual auxiliary
allocation at 4,000,000 complex entries. Larger requested tables fail
diagnostically rather than allocating without bound.

## 7. Screening policy

Milestone 6 implements the Coulomb-space Cauchy--Schwarz bound derived in the
mathematical specification. `EriSchwarzBounds` caches one symmetric factor per
shell pair from the positive self-pair integrals. A 64-epsilon multiplicative
margin and outward `nextafter` protect the floating representation of each
factor. A quartet is omitted only when `Q_ab * Q_cd < threshold`.

Threshold zero bypasses bound construction and is the exact Milestone 5 path.
The Python default remains zero. Tests verify the bound directly for complex
quartets and show monotonically decreasing tensor error as the threshold is
tightened. The threshold bounds each discarded AO integral; users must select
an application-level threshold appropriate to subsequent contractions.

## 8. Milestone 6 caching and parallel layout

An `EriWorkspace` now caches Boys sequences by complex argument and order,
field-free Hermite tables by pair geometry/angular key, and the three-axis
Hermite products used by the final contraction. Auxiliary readiness uses
generation tags instead of clearing the full six-index table for every
Cartesian integral. These caches change neither recurrence nor accumulation
order.

`evaluate_eri_shell_quartets` validates and screens work serially. With an
OpenMP-enabled build it then assigns independent shell blocks using static
scheduling and one workspace per thread. Completed blocks are delivered to
the consumer serially in input order. Consequently scheduling does not change
primitive reduction order, serial output is deterministic, and parallel output
is bitwise identical for the tested workload. Output-block storage is bounded
by the caller's batch size.

## 9. Optimization sequence

Optimization begins after reference agreement:

Milestone 6 completed scratch reuse, pair-invariant caches, contiguous
three-axis contraction coefficients, generation-tagged auxiliary storage,
Schwarz screening, an OS prototype comparison, and optional block-level
OpenMP.

Milestone 9 planning profiled the production path (stack sampling on a
standalone microbenchmark, `sample` on macOS) instead of guessing the next
target. The result: 99.6% of ERI wall time is inside the six-index
Hermite/Boys-derivative recursion in `EriKernel::compute`
(`src/eri.cpp:242-319`), and its per-quartet cost scales worse with angular
momentum than the primitive/Cartesian combinatorics alone predict (measured
rate drops 294x from s-s-s-s to p-p-p-p against an 81x growth in call count,
then 130x against a 16x growth from p-p-p-p to d-d-d-d). Boys evaluation,
`gaussian_product`, and Hermite-coefficient construction are each under 1% of
measured time. Zero field currently costs about the same as finite field
(6-11% faster, not the 2-4x a real-arithmetic path would give), because the
recursion is `std::complex<double>` unconditionally regardless of whether the
imaginary parts are structurally zero. A direct OS/HGP contracted backend
would not address this: it is a different recurrence for the same six-index
generality, not a change to the recursion's iteration architecture. See
`docs/eri_performance_plan.md` for the full measurement and the staged
rewrite it motivates (iterative bottom-up recursion, then a collapsed
real-arithmetic zero-field path using the sign-collapse identity this section
already rules out at finite field).

Each change keeps a scalar correctness path and records before/after benchmark
data with compiler, CPU, field, basis, shell class, and checksum.

## 10. Analytic first derivatives

Milestone 7 applies the center and field raising/lowering identities in the
mathematical specification to the already-audited primitive kernels. Shifted
terms are raw primitives carrying the original component's normalization;
they are never renormalized as members of a different angular shell. This one
response layer covers overlap, canonical kinetic, physical magnetic kinetic,
nuclear attraction, and ERIs without differentiating each MD recurrence.

Potential-center attraction derivatives use integration by parts and spatial
orbital gradients. The physical magnetic kinetic field derivative adds the
explicit angular-momentum and quadratic-moment response of its Hamiltonian.
Shell drivers contract derivative values with leading center/field axes, and
basis drivers scatter shell-center contributions into one matrix per shell and
Cartesian direction. Quartet derivatives remain shell-block APIs to preserve
the library's no-mandatory-\(O(N^4)\)-storage policy.
