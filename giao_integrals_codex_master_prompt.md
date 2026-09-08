# Codex Master Prompt: Fast AO Integral Library for Complex GIAO/London Gaussian Orbitals

## Role and objective

Act as a senior scientific-computing developer and quantum-chemistry
method developer. Design and implement a production-quality,
high-performance library for atomic-orbital (AO) integrals over
**complex gauge-including atomic orbitals (GIAOs), also called London
atomic orbitals**, suitable for electronic-structure calculations in
finite magnetic fields.

The library should have:

-   a **C++20 numerical core**;
-   **pybind11** Python bindings;
-   a clean, NumPy-oriented Python API;
-   CMake as the build system;
-   a small, mathematically transparent **pure-Python reference
    implementation** used for verification;
-   comprehensive unit, regression, property-based, and numerical tests;
-   benchmarks that make performance regressions visible.

Correctness comes before optimization. Do not optimize an unverified
recurrence or formula.

The final library should be fast enough to serve as an AO-integral
backend for realistic electronic-structure programs, while remaining
understandable and extensible for research.

------------------------------------------------------------------------

## 1. Work in milestones, not as one giant implementation

Do **not** attempt to implement the entire library immediately.

At the beginning:

1.  inspect the repository;
2.  identify any existing code, conventions, dependencies, tests, and
    build infrastructure;
3.  produce a concrete implementation plan;
4.  write down the mathematical conventions;
5.  identify unresolved mathematical or architectural choices;
6.  propose the repository layout;
7.  propose the test strategy;
8.  only then begin implementation.

For every major milestone:

1.  state what will be implemented;
2.  state the mathematical identities/formulas on which it relies;
3.  implement the smallest correct version;
4.  add tests;
5.  run the tests;
6.  compare against an independent reference where possible;
7.  only then optimize.

Do not silently change mathematical conventions during development.

Maintain a short `STATUS.md` containing:

-   completed functionality;
-   current limitations;
-   maximum tested angular momentum;
-   known numerical issues;
-   benchmark status;
-   next milestone.

------------------------------------------------------------------------

# 2. Mathematical specification first

Before implementing optimized kernels, create a document such as

`docs/mathematical_specification.md`

that defines all conventions unambiguously.

## 2.1 Ordinary Cartesian Gaussian primitives

Use Cartesian Gaussian primitives of the form

\[
g\_{`\mathbf `{=tex}a}(`\mathbf `{=tex}r;`\alpha`{=tex},`\mathbf `{=tex}A)
= N\_{`\mathbf `{=tex}a}(`\alpha`{=tex}) (x-A_x)\^{a_x} (y-A_y)\^{a_y}
(z-A_z)\^{a_z}
e^{-`\alpha`{=tex}\|`\mathbf `{=tex}r-`\mathbf `{=tex}A\|^2}, \]

where

\[ `\mathbf `{=tex}a=(a_x,a_y,a_z), `\qquad`{=tex}
a_x,a_y,a_z`\ge 0`{=tex}. \]

Document:

-   primitive normalization;
-   contracted normalization policy;
-   Cartesian ordering within shells;
-   shell ordering;
-   exponent and contraction-coefficient conventions.

Do not initially restrict the implementation to hard-coded s, p, d, or f
shells. The mathematical/API design must support arbitrary Cartesian
angular momentum, subject only to practical implementation limits.

## 2.2 GIAO/London phase

Define a GIAO centered at (`\mathbf `{=tex}A) by a convention of the
form

\[ `\chi`{=tex}*`\mu`{=tex}\^{`\mathbf `{=tex}B}(`\mathbf `{=tex}r) =
`\exp`{=tex}`\left[
-\frac{i}{2}
(\mathbf B\times\mathbf A)\cdot\mathbf r
\right]`{=tex}`\chi`{=tex}*`\mu`{=tex}(`\mathbf `{=tex}r), \]

in atomic units, unless a different convention is deliberately chosen.

The exact sign and factors depend on electromagnetic and charge
conventions. Therefore:

-   choose one convention;
-   derive it explicitly;
-   document it prominently;
-   use it consistently everywhere;
-   test it.

Do not copy formulas from references without reconciling their
conventions.

Introduce a phase/wave-vector representation where useful, e.g.

\[ `\mathbf `{=tex}k_A =
-`\frac12`{=tex},`\mathbf `{=tex}B`\times`{=tex}`\mathbf `{=tex}A. \]

When a bra function is complex-conjugated, ensure the phase changes sign
correctly.

## 2.3 Gauge origin

Design the representation so that a gauge origin can be specified
explicitly rather than implicitly assuming the Cartesian origin.

Document how

\[ `\mathbf `{=tex}A-`\mathbf `{=tex}O \]

enters the London phase for gauge origin (`\mathbf `{=tex}O).

Tests must verify expected gauge-origin/translational behavior of
physically appropriate quantities.

------------------------------------------------------------------------

# 3. Derive the complex Gaussian product algebra

This is a critical part of the project.

Before optimized implementation, derive how products such as

\[
`\chi`{=tex}\_`\mu`{=tex}\^{\*}(`\mathbf `{=tex}r)`\chi`{=tex}\_`\nu`{=tex}(`\mathbf `{=tex}r)
\]

reduce to ordinary Gaussian-product expressions multiplied by complex
exponential factors.

For two primitives centered at (`\mathbf `{=tex}A,`\mathbf `{=tex}B),
determine explicitly:

-   total exponent (p=`\alpha`{=tex}+`\beta`{=tex});
-   ordinary Gaussian product center;
-   London phase difference;
-   effective complex displacement/center, if that representation is
    used;
-   exponential prefactor;
-   polynomial/Hermite coefficients;
-   conjugation behavior.

Show algebraically how the finite-field formulas reduce to conventional
Gaussian-product formulas as

\[ `\mathbf `{=tex}B`\rightarrow 0`{=tex}. \]

Do not proceed to high-performance recurrence kernels until these
formulas have executable tests.

------------------------------------------------------------------------

# 4. Architecture

Use approximately the following layering, but adjust it if a better
structure emerges.

``` text
project/
├── CMakeLists.txt
├── pyproject.toml
├── README.md
├── STATUS.md
├── docs/
│   ├── mathematical_specification.md
│   ├── algorithms.md
│   └── api.md
├── include/
│   └── giao_integrals/
├── src/
├── python/
│   └── giao_integrals/
├── reference/
│   └── python/
├── tests/
│   ├── cpp/
│   └── python/
├── benchmarks/
└── examples/
```

Keep the mathematical layer distinct from:

-   shell/basis data structures;
-   integral drivers;
-   screening;
-   parallelism;
-   Python bindings.

Avoid exposing implementation-specific recurrence machinery
unnecessarily in the public API.

------------------------------------------------------------------------

# 5. Core data model

Design efficient representations for at least:

### Primitive Gaussian

Contains or references:

-   exponent;
-   contraction coefficient;
-   center;
-   angular momentum;
-   normalization information.

### Shell

Contains:

-   center;
-   total angular momentum;
-   primitive exponents;
-   contraction coefficients;
-   Cartesian component ordering;
-   normalization data.

Support segmented contractions initially if that simplifies
implementation, but design so general contractions can be added without
rewriting the engine.

### Basis set

Contains:

-   atoms/centers;
-   shells;
-   AO offsets;
-   number of Cartesian AOs.

### Magnetic field / gauge context

Represent:

-   magnetic field vector (`\mathbf `{=tex}B);
-   gauge origin (`\mathbf `{=tex}O);
-   any precomputed London wave vectors.

Prefer immutable or trivially copyable small objects in
performance-sensitive C++ code.

------------------------------------------------------------------------

# 6. Numerical foundation

Implement and test the numerical building blocks independently.

These should include, as needed:

-   factorials and double factorials;
-   Gaussian normalization;
-   Cartesian angular-momentum enumeration;
-   Gaussian product quantities;
-   Hermite Gaussian coefficients;
-   Boys functions;
-   complex-valued auxiliary functions;
-   recurrence tables;
-   stable small/large-argument treatments.

Avoid repeated dynamic allocation inside primitive or shell loops.

------------------------------------------------------------------------

# 7. Boys function and complex arguments

Finite-field/GIAO formulations may produce complex-valued arguments in
Coulomb-related auxiliary functions.

Treat this as a first-class numerical problem.

Develop:

-   a high-accuracy reference implementation, potentially using Python
    `mpmath`;
-   a production implementation for the required real/complex domain;
-   tests over small, moderate, large, nearly real, and genuinely
    complex arguments.

Investigate suitable algorithms rather than assuming the standard
real-positive Boys implementation is sufficient.

Test recurrence stability.

Use high-precision calculations to establish reference values.

Document numerical domains and error behavior.

------------------------------------------------------------------------

# 8. Pure-Python reference engine

Before or alongside optimized C++ kernels, create a deliberately simple
reference implementation.

It should prioritize:

-   clarity;
-   direct correspondence with the mathematical specification;
-   arbitrary-precision validation where useful;
-   ease of randomized testing.

It does **not** need to be fast.

Use it to generate reference values for primitive and small contracted
integrals.

Where practical, implement alternative direct/numerical checks for very
low angular momentum.

The optimized implementation must not be its own only oracle.

------------------------------------------------------------------------

# 9. One-electron integrals

Implement one-electron integrals incrementally.

Required baseline operators:

1.  overlap \[ S\_{`\mu`{=tex}`\nu`{=tex}} =
    `\langle`{=tex}`\chi`{=tex}*`\mu`{=tex}\|`\chi`{=tex}*`\nu`{=tex}`\rangle`{=tex};
    \]

2.  kinetic energy \[ T\_{`\mu`{=tex}`\nu`{=tex}} =
    `\left`{=tex}`\langle`{=tex}`\chi`{=tex}*`\mu`{=tex}`\left`{=tex}\|-`\frac12`{=tex}`\nabla`{=tex}\^2`\right`{=tex}\|`\chi`{=tex}*`\nu`{=tex}`\right`{=tex}`\rangle`{=tex};
    \]

3.  nuclear attraction \[ V\_{`\mu`{=tex}`\nu`{=tex}} = -`\sum`{=tex}*A
    Z_A `\left`{=tex}`\langle`{=tex} `\chi`{=tex}*`\mu`{=tex}
    `\left`{=tex}\| `\frac{1}{|\mathbf r-\mathbf R_A|}`{=tex}
    `\right`{=tex}\| `\chi`{=tex}\_`\nu`{=tex}
    `\right`{=tex}`\rangle`{=tex}. \]

Then add a general property-integral framework covering common
electronic-structure operators, including as appropriate:

-   (x,y,z);
-   Cartesian multipoles;
-   arbitrary low-order Cartesian moments;
-   momentum (-i`\nabla`{=tex});
-   gradients/derivative operators;
-   angular momentum (`\mathbf `{=tex}L);
-   electric-field operators;
-   magnetic operators needed in finite-field/GIAO work.

Avoid implementing every property as unrelated duplicated code. Identify
reusable raising/lowering or moment machinery.

------------------------------------------------------------------------

# 10. Two-electron electron-repulsion integrals

Implement full four-center Coulomb ERIs

\[ (`\mu`{=tex}`\nu`{=tex}\|`\lambda`{=tex}`\sigma`{=tex}) =
`\iint`{=tex} `\chi`{=tex}*`\mu`{=tex}\^*(1)`\chi`{=tex}*`\nu`{=tex}(1)
`\frac{1}{r_{12}}`{=tex}
`\chi`{=tex}*`\lambda`{=tex}\^*(2)`\chi`{=tex}*`\sigma`{=tex}(2) ,d1,d2.
\]

Support:

-   primitive ERIs;
-   contracted ERIs;
-   shell-quartet evaluation;
-   arbitrary Cartesian angular momentum within tested limits;
-   complex-valued results.

Derive the complex/GIAO ERI formula carefully.

Do not assume all conventional eightfold real-ERI permutation symmetries
survive unchanged.

Explicitly derive and test the valid permutation/conjugation relations.

At zero field, recover conventional real Gaussian ERIs to numerical
precision.

------------------------------------------------------------------------

# 11. Recurrence strategy

Evaluate established recurrence families rather than choosing one
blindly.

Candidates include:

-   McMurchie-Davidson;
-   Obara-Saika;
-   Head-Gordon/Pople-style approaches;
-   Rys quadrature where appropriate.

For the first correct implementation, prefer an approach that:

-   generalizes transparently to complex Gaussian quantities;
-   is easy to verify;
-   supports property integrals and derivatives;
-   does not create excessive code complexity.

A Hermite/McMurchie-Davidson-style reference or initial production
implementation is reasonable.

However, keep interfaces sufficiently separated that the underlying
recurrence engine can later be replaced or supplemented.

Document the choice and its tradeoffs in `docs/algorithms.md`.

------------------------------------------------------------------------

# 12. Derivatives and response properties

The architecture must anticipate derivatives from the beginning.

Eventually support at least:

-   first derivatives with respect to nuclear coordinates;
-   magnetic-field derivatives;
-   gauge-origin-sensitive intermediate derivatives where required;
-   mixed nuclear/magnetic derivatives if feasible.

Do not rely exclusively on finite differences in production.

Finite differences should, however, be used extensively to validate
analytic derivatives.

For each derivative type, derive contributions from both:

1.  the ordinary Gaussian dependence;
2.  the London/GIAO phase dependence.

This distinction is essential.

------------------------------------------------------------------------

# 13. Symmetry and complex algebra

Create explicit tests for complex Hermitian/conjugation identities.

For example, where appropriate,

\[ S\_{`\mu`{=tex}`\nu`{=tex}}=S\_{`\nu`{=tex}`\mu`{=tex}}\^\*. \]

Likewise determine the correct identities for kinetic, potential,
property, and ERI tensors.

Never replace a complex conjugation with an ordinary index permutation
unless mathematically justified.

At (`\mathbf `{=tex}B=0), test recovery of conventional real-valued
symmetry.

------------------------------------------------------------------------

# 14. Shell-based integral API

The C++ computational API should operate primarily on:

-   shell pairs;
-   shell quartets;
-   batches of shells.

Do not make Python call C++ once per AO integral.

Example conceptual interfaces:

``` cpp
compute_overlap(shell_a, shell_b, field_context, output);
compute_kinetic(shell_a, shell_b, field_context, output);
compute_nuclear_attraction(shell_a, shell_b, nuclei, field_context, output);
compute_eri(shell_a, shell_b, shell_c, shell_d, field_context, output);
```

The exact API may differ.

Output buffers should be contiguous and have clearly documented index
ordering.

Where feasible, allow caller-provided buffers to reduce allocations.

------------------------------------------------------------------------

# 15. Python API

Expose a convenient high-level API.

A target usage might resemble:

``` python
import numpy as np
import giao_integrals as gi

basis = gi.Basis(...)

field = gi.MagneticField(
    B=np.array([0.0, 0.0, 0.1]),
    gauge_origin=np.zeros(3),
)

S = gi.overlap(basis, field=field)
T = gi.kinetic(basis, field=field)
V = gi.nuclear_attraction(basis, atoms, field=field)
eri = gi.eri(basis, field=field)
```

Also expose shell-level APIs for expert use.

Return standard NumPy arrays with documented shapes, dtype, and
ordering.

Use `complex128` for the standard complex production API unless there is
a strong reason otherwise.

Avoid unnecessary Python-side copies.

Release the GIL around expensive C++ computation.

------------------------------------------------------------------------

# 16. Correctness test hierarchy

Use several independent levels of validation.

## Level 1: analytic special cases

Test:

-   s-s overlap;
-   simple kinetic integrals;
-   simple Coulomb cases;
-   coincident centers;
-   zero magnetic field;
-   simple field orientations.

## Level 2: Python reference implementation

Randomized comparisons across:

-   exponents;
-   centers;
-   angular momenta;
-   magnetic-field vectors;
-   gauge origins;
-   contraction coefficients.

## Level 3: conventional external engines at zero field

When licensing and availability permit, compare (`\mathbf `{=tex}B=0)
results against established packages such as Libint, libcint, or PySCF.

External packages are validation references, not code to copy.

## Level 4: high-precision arithmetic

Use `mpmath` or equivalent for difficult primitive cases and complex
auxiliary functions.

## Level 5: numerical quadrature

For low-angular-momentum one-electron cases, construct numerical
integration tests where practical.

## Level 6: finite differences

Validate analytic nuclear and magnetic derivatives against high-quality
finite differences.

------------------------------------------------------------------------

# 17. Physical and mathematical invariants

Tests should include:

### Zero-field limit

\[ `\lim`{=tex}\_{`\mathbf `{=tex}B`\to0`{=tex}}
I\^{`\mathrm{GIAO}`{=tex}} = I\^{`\mathrm{ordinary}`{=tex}}. \]

### Hermiticity

Verify appropriate one-electron matrices satisfy their expected
Hermitian relations.

### ERI conjugation/permutation identities

Derive and test exactly those identities valid for complex GIAOs.

### Gauge behavior

Test gauge-origin changes for quantities for which invariance/covariance
is expected.

Do not claim gauge-origin invariance for an intermediate quantity unless
the theory says it should be invariant.

### Translational behavior

Where appropriate, translate the molecular geometry and gauge origin
consistently and test the resulting transformation/invariance.

### Continuity

Check smooth behavior as:

-   (B`\to0`{=tex});
-   centers approach one another;
-   exponents become similar;
-   complex auxiliary-function arguments approach the real axis.

------------------------------------------------------------------------

# 18. Error tolerances

Define tolerances intentionally.

For each test category distinguish:

-   absolute error;
-   relative error;
-   expected scale;
-   near-zero behavior.

Do not use a single loose tolerance globally.

For well-conditioned double-precision primitive integrals, aim for
agreement near machine precision where realistic.

Track difficult numerical regions explicitly rather than hiding them
behind relaxed tests.

------------------------------------------------------------------------

# 19. Performance engineering

Only optimize after correctness is established.

Benchmark before and after each important optimization.

Focus on:

-   shell-pair and shell-quartet throughput;
-   primitive contraction loops;
-   recurrence construction;
-   memory layout;
-   temporary-buffer reuse;
-   Boys-function evaluation;
-   screening;
-   parallelism.

Avoid:

-   heap allocations in innermost loops;
-   virtual dispatch in hot kernels;
-   repeated recomputation of invariant quantities;
-   Python callbacks in numerical kernels.

Use compiler optimization and vectorization reports where useful.

------------------------------------------------------------------------

# 20. Screening

Implement mathematically justified screening for ERIs.

Start with conventional shell/primitive bounds where applicable, but
verify how complex London prefactors affect the bounds.

Do not assume a real-Gaussian screening inequality remains unchanged
without checking.

Document all screening approximations and thresholds.

Tests should verify that tightening the threshold converges to the
unscreened result.

------------------------------------------------------------------------

# 21. Parallelism

After establishing a reliable serial implementation:

-   add shared-memory CPU parallelism, preferably OpenMP or an equally
    lightweight strategy;
-   parallelize at shell-pair/shell-quartet or batch level;
-   avoid fine-grained synchronization;
-   use thread-local scratch buffers;
-   ensure deterministic or acceptably reproducible results where
    practical.

Benchmark scaling with thread count.

Do not make GPU implementation an initial requirement.

Keep the architecture compatible with future GPU/generated-kernel
backends, but first build an excellent CPU engine.

------------------------------------------------------------------------

# 22. Memory strategy

ERIs scale as (O(N\^4)), so do not assume storing the entire AO ERI
tensor is always appropriate.

Provide interfaces for:

-   shell-quartet callbacks or iteration;
-   batched ERI computation;
-   direct consumption by future J/K builders;
-   optional full tensor construction for small systems and testing.

Design toward eventual direct SCF usage.

------------------------------------------------------------------------

# 23. Potential future extensions

The architecture should not prevent later support for:

-   spherical/harmonic basis transformations;
-   spin-orbit operators;
-   relativistic one-electron operators;
-   density fitting / three-center integrals;
-   two-center Coulomb metric integrals;
-   J/K direct builds;
-   integral derivatives of higher order;
-   generated kernels;
-   SIMD-specialized kernels;
-   GPU backends.

Do not implement all of these initially.

------------------------------------------------------------------------

# 24. Spherical basis functions

Use Cartesian Gaussians as the primitive computational representation.

If spherical basis functions are added, implement them as well-defined
transformations from Cartesian shell integrals using documented real or
complex spherical-harmonic conventions.

Keep this transformation separate from the core Cartesian integral
engine.

------------------------------------------------------------------------

# 25. Dependencies

Keep the numerical core lightweight.

Reasonable dependencies include:

-   C++20 standard library;
-   pybind11;
-   CMake;
-   a testing framework such as Catch2 or GoogleTest;
-   Python NumPy;
-   pytest;
-   mpmath for reference tests.

Do not introduce a large dependency merely to avoid implementing a
small, central numerical kernel.

If an external library is proposed for production use, explain:

-   why;
-   licensing;
-   portability;
-   numerical consequences;
-   whether it supports the required complex domain.

------------------------------------------------------------------------

# 26. Coding standards

For C++:

-   use RAII;
-   prefer value semantics;
-   use `std::complex<double>` unless benchmarks demonstrate a
    compelling alternative;
-   use `std::span` where useful;
-   make ownership explicit;
-   avoid raw owning pointers;
-   avoid macros except where genuinely appropriate;
-   make hot-loop data layout obvious;
-   document nontrivial equations near their implementation;
-   keep headers and implementation reasonably separated.

For Python:

-   type annotate public APIs;
-   document array shapes and units;
-   use NumPy idioms;
-   avoid hidden global state.

Use atomic units internally unless explicitly documented otherwise.

------------------------------------------------------------------------

# 27. Documentation requirements

The documentation must explain enough mathematics that another
quantum-chemistry developer can audit the implementation.

At minimum document:

-   GIAO convention;
-   gauge-origin convention;
-   primitive normalization;
-   contraction convention;
-   Cartesian ordering;
-   Gaussian product theorem with London phases;
-   Boys/auxiliary-function definitions;
-   recurrence equations;
-   integral tensor ordering;
-   valid symmetry relations;
-   derivative conventions;
-   numerical limitations.

Whenever an equation is translated into code, make it possible to trace
the code back to the documented equation.

------------------------------------------------------------------------

# 28. Literature and external-code discipline

Use primary literature and reputable technical documentation to check
formulas.

When consulting existing open-source integral engines:

-   inspect them for algorithmic understanding only when licensing
    permits;
-   do not copy incompatible licensed code;
-   record relevant references;
-   independently derive GIAO-specific formulas;
-   distinguish conventional real-Gaussian algorithms from modifications
    needed for complex London orbitals.

Create `docs/references.md` with papers/books/software references and
notes on what each source was used to verify.

------------------------------------------------------------------------

# 29. Benchmark suite

Create reproducible benchmarks covering:

-   primitive overlap;
-   contracted shell-pair overlap;
-   kinetic;
-   nuclear attraction;
-   representative ERI shell quartets;
-   low and moderate angular momentum;
-   zero field;
-   finite field;
-   single-threaded performance;
-   multi-threaded performance.

Report quantities such as:

-   shell pairs/s;
-   shell quartets/s;
-   integrals/s;
-   wall time;
-   scaling versus threads.

Avoid misleading microbenchmarks that compiler optimization can
eliminate.

------------------------------------------------------------------------

# 30. Initial milestone sequence

Use approximately this sequence.

## Milestone 0 --- specification

Deliver:

-   repository assessment;
-   mathematical conventions;
-   architecture;
-   dependency proposal;
-   test strategy;
-   milestone plan.

No major optimized implementation yet.

## Milestone 1 --- Python mathematical reference

Implement:

-   primitive Cartesian Gaussians;
-   London phases;
-   normalization;
-   primitive overlap;
-   low-order tests;
-   high-precision/reference checks.

## Milestone 2 --- C++ overlap engine

Implement:

-   core data structures;
-   complex Gaussian product machinery;
-   arbitrary Cartesian overlap;
-   primitive contraction;
-   shell-pair driver;
-   Python binding.

Validate exhaustively against Milestone 1.

## Milestone 3 --- kinetic and moment/property integrals

Add:

-   kinetic;
-   coordinate moments;
-   momentum/gradient machinery;
-   reusable raising/lowering infrastructure.

## Milestone 4 --- nuclear attraction

Implement:

-   Boys/auxiliary functions;
-   complex-domain validation;
-   nuclear-attraction recurrences.

## Milestone 5 --- ERIs

Implement:

-   primitive ERIs;
-   shell quartets;
-   contractions;
-   complex symmetry handling;
-   zero-field external validation.

## Milestone 6 --- performance baseline

Add:

-   benchmark suite;
-   scratch-buffer reuse;
-   screening;
-   loop/layout improvements;
-   OpenMP.

## Milestone 7 --- derivatives

Add:

-   nuclear derivatives;
-   magnetic-field derivatives;
-   finite-difference validation.

## Milestone 8 --- production hardening

Add:

-   documentation;
-   examples;
-   packaging;
-   CI;
-   sanitizer runs;
-   edge-case testing;
-   performance-regression tracking.

------------------------------------------------------------------------

# 31. First concrete task

Start now with **Milestone 0 only**.

Do not immediately write hundreds of lines of implementation code.

Perform the following:

1.  inspect the current repository;
2.  summarize what already exists;
3.  propose the final repository layout;
4.  write the precise GIAO convention to be used;
5.  derive the bra-ket London phase structure;
6.  derive the complex Gaussian product representation for an overlap
    pair;
7.  identify how this generalizes to Coulomb integrals;
8.  evaluate MD/Hermite versus Obara-Saika versus other plausible
    recurrence approaches for this problem;
9.  propose the Boys-function strategy for complex arguments;
10. define the public C++ shell-level API;
11. define the Python API;
12. define the verification hierarchy;
13. identify the highest-risk mathematical/numerical issues;
14. produce a detailed milestone implementation plan.

For mathematical derivations, show enough intermediate algebra to expose
sign, conjugation, and factor-of-two errors.

At the end of Milestone 0, stop and present the plan and unresolved
decisions for review before implementing Milestone 1.

------------------------------------------------------------------------

# 32. Non-negotiable requirements

Throughout the project:

-   Never silently assume integrals are real.
-   Never use ordinary transpose where Hermitian conjugation is
    required.
-   Never assume conventional real-ERI permutation symmetry without
    deriving its complex counterpart.
-   Never optimize before a trustworthy reference test exists.
-   Never use the optimized implementation as its sole validation
    reference.
-   Never hide numerical failures by simply loosening tolerances.
-   Never hard-code only s/p/d/f formulas as the fundamental
    architecture.
-   Never perform one Python-to-C++ call per AO integral in production.
-   Never store the full (N\^4) ERI tensor as the only supported
    interface.
-   Never ignore derivatives of the London phase when differentiating
    GIAOs.
-   Never mix gauge, charge, phase, or unit conventions without
    documenting the conversion.
-   Always verify the (B=0) limit.
-   Always test complex conjugation/Hermiticity explicitly.
-   Always retain a readable mathematical specification alongside
    optimized code.

The goal is not merely to obtain numbers. The goal is a **fast,
auditable, numerically reliable integral library for complex GIAO
Gaussian basis functions that can eventually serve as infrastructure for
serious finite-magnetic-field electronic-structure research.**
