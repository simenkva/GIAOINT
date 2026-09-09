# Mathematical specification

Status: frozen for Milestones 1 and 2. Any later change requires a documented
compatibility decision, revised analytic tests, and an entry in `STATUS.md`.

## 1. Units, charge, and vector potential

The library uses atomic units. The electron charge is \(-1\), the canonical
momentum is \(\mathbf p=-i\nabla\), and a uniform magnetic field
\(\boldsymbol{\mathcal B}\) uses the symmetric-gauge vector potential

\[
\mathbf A_{\mathbf O}(\mathbf r)
=\frac12\boldsymbol{\mathcal B}\times(\mathbf r-\mathbf O),
\]

where \(\mathbf O\) is the gauge origin. The one-electron kinetic operator in
the magnetic Hamiltonian is \(\tfrac12(\mathbf p+\mathbf A_{\mathbf O})^2\).
The library also exposes the canonical field-free operator
\(-\tfrac12\nabla^2\), but its matrix over field-dependent GIAOs is a
gauge-dependent intermediate.

Vectors use right-handed Cartesian coordinates. A dot product between complex
vectors in analytic Gaussian formulas is bilinear,
\(\mathbf z\mathbin{\cdot}\mathbf z=\sum_i z_i^2\), with no conjugation.
Hermitian inner products will be labeled explicitly.

## 2. Cartesian primitive Gaussians

For \(\mathbf a=(a_x,a_y,a_z)\in\mathbb N_0^3\), \(\alpha>0\), and center
\(\mathbf A\), a normalized primitive is

\[
g_{\mathbf a}(\mathbf r;\alpha,\mathbf A)
=N_{\mathbf a}(\alpha)
\prod_{j\in\{x,y,z\}}(r_j-A_j)^{a_j}
\exp[-\alpha\lVert\mathbf r-\mathbf A\rVert^2],
\]

with

\[
N_{\mathbf a}(\alpha)
=\left(\frac{2\alpha}{\pi}\right)^{3/4}
\left[
\frac{(4\alpha)^{a_x+a_y+a_z}}
{(2a_x-1)!!(2a_y-1)!!(2a_z-1)!!}
\right]^{1/2},
\qquad (-1)!!=1.
\]

This normalization makes
\(\int_{\mathbb R^3}|g_{\mathbf a}(\mathbf r)|^2d\mathbf r=1\).
Exponents and centers are real. The API rejects non-positive or non-finite
exponents and non-finite coordinates.

## 3. Shells, contractions, and ordering

A Cartesian shell has one center, total angular momentum \(L\), a shared list
of primitive exponents, and one or more coefficient rows. The initial kernels
may accept one contraction row, but storage uses a two-dimensional
`(n_contraction, n_primitive)` layout from the start.

Input coefficients multiply normalized primitives. With raw coefficients
\(d_j\), contracted component \(\mathbf a\) is

\[
\phi_{c\mathbf a}(\mathbf r)
=C_{cL}\sum_j d_{cj}g_{\mathbf a}(\mathbf r;\alpha_j,\mathbf A).
\]

By default, \(C_{cL}>0\) normalizes each contraction at zero field. The value is
shared by all Cartesian components in a shell because the same-center overlap
between individually normalized primitives depends on \(L\), not on the
partition \((a_x,a_y,a_z)\). A caller may request `as_provided` coefficients,
which sets \(C_{cL}=1\). No normalization depends on the field or gauge origin.
Basis-set importers must convert source-specific coefficient conventions to
this one.

For any component \(\mathbf a\) with \(|\mathbf a|=L\), the default factor is

\[
C_{cL}=\left[
\sum_{ij}d_{ci}d_{cj}
\langle g_{\mathbf a}(\alpha_i,\mathbf A)
\mid g_{\mathbf a}(\alpha_j,\mathbf A)\rangle
\right]^{-1/2}.
\]

The bracket is real and evaluated at zero field. A zero or non-positive
contraction norm is invalid.

For fixed \(L\), Cartesian components are ordered by the loops

```text
for ax = L, L-1, ..., 0
  for ay = L-ax, L-ax-1, ..., 0
    az = L-ax-ay
```

Thus \(p=(x,y,z)\) and
\(d=(xx,xy,xz,yy,yz,zz)\). A shell stores contractions outermost, then
Cartesian components. Shell order in a basis is input order; AO offsets are
the prefix sums of shell AO counts.

## 4. Frozen London/GIAO convention

Define the real London wave vector

\[
\boldsymbol\kappa_{\mathbf A}
=\frac12\boldsymbol{\mathcal B}\times(\mathbf A-\mathbf O).
\]

The GIAO associated with an ordinary Gaussian at \(\mathbf A\) is

\[
\boxed{
\omega_{\mathbf a}^{\boldsymbol{\mathcal B}}
(\mathbf r;\alpha,\mathbf A,\mathbf O)
=\exp[-i\boldsymbol\kappa_{\mathbf A}\cdot\mathbf r]
g_{\mathbf a}(\mathbf r;\alpha,\mathbf A)
}.
\]

This sign matches an electron of charge \(-1\) with
\(\tfrac12(\mathbf p+\mathbf A)^2\), and matches Eq. (1) of Tellgren,
Soncini, and Helgaker (2008). The phase uses \(\mathbf r\), not
\(\mathbf r-\mathbf O\). Anchoring it at \(\mathbf O\) would introduce a
center-dependent constant phase and therefore a different AO representation.

At \(\boldsymbol{\mathcal B}=\mathbf0\), every \(\boldsymbol\kappa=0\) and the
GIAO equals its ordinary Gaussian exactly.

### 4.1 Gauge-origin change

For \(\mathbf O' = \mathbf O+\mathbf d\),

\[
\boldsymbol\kappa_{\mathbf A}'
=\boldsymbol\kappa_{\mathbf A}
-\frac12\boldsymbol{\mathcal B}\times\mathbf d,
\]

so every AO receives the same position-dependent gauge phase:

\[
\omega_{\mathbf A}'(\mathbf r)
=\exp\left[\frac{i}{2}
(\boldsymbol{\mathcal B}\times\mathbf d)\cdot\mathbf r\right]
\omega_{\mathbf A}(\mathbf r).
\]

Local scalar one-electron operators and Coulomb pair densities cancel this
common phase. The canonical kinetic operator does not. Tests of the physical
magnetic Hamiltonian must change both the basis phases and
\(\mathbf A_{\mathbf O}\).

## 5. Bra-ket London phase

Let the bra Gaussian be centered at \(\mathbf A\) and the ket Gaussian at
\(\mathbf B\). Complex conjugation reverses only the London phase because the
ordinary Cartesian Gaussian is real:

\[
\omega_{\mathbf a,A}^{*}(\mathbf r)
=e^{+i\boldsymbol\kappa_{\mathbf A}\cdot\mathbf r}g_{\mathbf a,A}(\mathbf r).
\]

Define the pair wave vector

\[
\mathbf q_{AB}
=\boldsymbol\kappa_{\mathbf B}-\boldsymbol\kappa_{\mathbf A}
=\frac12\boldsymbol{\mathcal B}\times(\mathbf B-\mathbf A).
\]

The gauge origin cancels in \(\mathbf q_{AB}\), and

\[
\boxed{
\omega_{\mathbf a,A}^{*}\omega_{\mathbf b,B}
=e^{-i\mathbf q_{AB}\cdot\mathbf r}
g_{\mathbf a,A}g_{\mathbf b,B}
}.
\]

Swapping bra and ket gives \(\mathbf q_{BA}=-\mathbf q_{AB}\), which is the
key sign behind Hermiticity.

## 6. Complex Gaussian product algebra

Suppress polynomial normalization factors first. Let

\[
p=\alpha+\beta,\quad
\mu=\frac{\alpha\beta}{p},\quad
\mathbf P=\frac{\alpha\mathbf A+\beta\mathbf B}{p}.
\]

The ordinary Gaussian product theorem gives

\[
-\alpha|\mathbf r-\mathbf A|^2-\beta|\mathbf r-\mathbf B|^2
=-p|\mathbf r-\mathbf P|^2-\mu|\mathbf A-\mathbf B|^2.
\]

Include the bra-ket London phase:

\[
-p|\mathbf r-\mathbf P|^2-i\mathbf q\cdot\mathbf r.
\]

Set

\[
\boxed{\mathbf P'=\mathbf P-\frac{i\mathbf q}{2p}}.
\]

Because

\[
-p(\mathbf r-\mathbf P')^2
=-p(\mathbf r-\mathbf P)^2
-i\mathbf q\cdot(\mathbf r-\mathbf P)
+\frac{\mathbf q^2}{4p},
\]

we obtain

\[
\boxed{
-p|\mathbf r-\mathbf P|^2-i\mathbf q\cdot\mathbf r
=-p(\mathbf r-\mathbf P')^2
-i\mathbf q\cdot\mathbf P-\frac{\mathbf q^2}{4p}
}.
\]

Here \(\mathbf q^2=\mathbf q\cdot\mathbf q\) is real and non-negative, while
\((\mathbf r-\mathbf P')^2\) uses the bilinear complex dot product. The full
pair distribution is therefore

\[
N_{\mathbf a}(\alpha)N_{\mathbf b}(\beta)
e^{-\mu|\mathbf A-\mathbf B|^2}
e^{-i\mathbf q\cdot\mathbf P-\mathbf q^2/(4p)}
\prod_j(r_j-A_j)^{a_j}(r_j-B_j)^{b_j}
e^{-p(\mathbf r-\mathbf P')^2}.
\]

The real factor \(e^{-\mathbf q^2/(4p)}\) damps separated pair wave vectors.
The phase \(e^{-i\mathbf q\cdot\mathbf P}\) must not be dropped; doing so
breaks translational covariance.

### 6.1 Direct polynomial moments for the reference engine

In one dimension,

\[
(x-A)^a(x-B)^b=\sum_{n=0}^{a+b}C_n(x-P')^n,
\]

where

\[
C_n=\sum_{u+v=n}
{a\choose u}{b\choose v}
(P'-A)^{a-u}(P'-B)^{b-v}.
\]

The integrand is entire, so the integration contour may be shifted from the
real axis to pass through \(P'\). Therefore

\[
\int_{-\infty}^{\infty}(x-P')^n e^{-p(x-P')^2}dx
=\begin{cases}
0,&n\text{ odd},\\
\Gamma(m+\tfrac12)p^{-(m+1/2)},&n=2m.
\end{cases}
\]

The three-dimensional overlap is the product of these one-dimensional sums
times the pair prefactor and primitive normalizations. This formula will be
the first Python oracle because it does not reuse the production Hermite
recurrence.

For two s primitives it reduces to

\[
\boxed{
S_{ss}=N_s(\alpha)N_s(\beta)
\left(\frac\pi p\right)^{3/2}
\exp\left[-\mu|\mathbf A-\mathbf B|^2
-i\mathbf q\cdot\mathbf P-\frac{\mathbf q^2}{4p}\right]
}.
\]

### 6.2 Hermite/London representation

The production MD representation keeps the real product center \(\mathbf P\).
Define ordinary Hermite Gaussians

\[
\Lambda_{tuv}(\mathbf r; p,\mathbf P)
=\partial_{P_x}^t\partial_{P_y}^u\partial_{P_z}^v
e^{-p|\mathbf r-\mathbf P|^2}.
\]

The usual field-free Hermite coefficients expand the polynomial Gaussian
product. The London Fourier factor enters through

\[
\int e^{-i\mathbf q\cdot\mathbf r}\Lambda_{tuv}(\mathbf r)d\mathbf r
=\left(\frac\pi p\right)^{3/2}
(-iq_x)^t(-iq_y)^u(-iq_z)^v
e^{-i\mathbf q\cdot\mathbf P-\mathbf q^2/(4p)}.
\]

The coefficients remain field-independent because \(\mathbf q_{AB}\) depends
on \(\mathbf B-\mathbf A\), not on the product center \(\mathbf P\), when
\(\mathbf P\) and \(\mathbf B-\mathbf A\) are treated as independent
coordinates. This separation is the main reason to start with MD/Hermite.

### 6.3 Conjugation and zero-field checks

On swapping \((\alpha,\mathbf a,\mathbf A)\) with
\((\beta,\mathbf b,\mathbf B)\), \(p,\mu,\mathbf P\) stay fixed and
\(\mathbf q\mapsto-\mathbf q\). Hence \(\mathbf P'\mapsto\mathbf P'^*\) and
the pair prefactor is conjugated. It follows that

\[
S_{AB}=S_{BA}^{*}.
\]

As \(\boldsymbol{\mathcal B}\to0\), \(\mathbf q\to0\),
\(\mathbf P'\to\mathbf P\), and every expression reduces continuously to the
ordinary real Gaussian product theorem.

## 7. Coulomb generalization

The baseline operator conventions are

\[
S_{ab}=\langle\omega_a\mid\omega_b\rangle,
\qquad
T_{ab}=\left\langle\omega_a\left|-\frac12\nabla^2\right|\omega_b\right\rangle,
\]

and, for nuclei with positive charges \(Z_C\),

\[
V_{ab}=-\sum_C Z_C
\left\langle\omega_a\left|\frac1{|\mathbf r-\mathbf C|}\right|\omega_b\right\rangle.
\]

Thus `nuclear_attraction` includes the electronic minus sign and the nuclear
charges. The electron-repulsion integral uses the positive kernel
\(r_{12}^{-1}\).

For pair \(AB\), use exponent \(p\), real center \(\mathbf P\), phase vector
\(\mathbf q_1\), and complex center
\(\mathbf P'=\mathbf P-i\mathbf q_1/(2p)\). For pair \(CD\), use exponent
\(s\), real center \(\mathbf Q\), phase vector \(\mathbf q_2\), and
\(\mathbf Q'=\mathbf Q-i\mathbf q_2/(2s)\). Define

\[
\rho=\frac{ps}{p+s},\qquad
T=\rho(\mathbf P'-\mathbf Q')\cdot(\mathbf P'-\mathbf Q').
\]

The Boys function is

\[
F_n(T)=\int_0^1 t^{2n}e^{-Tt^2}dt
=\frac{1}{2n+1}{}_1F_1\left(n+\frac12;n+\frac32;-T\right).
\]

It is entire in \(T\); the incomplete-gamma representation requires a
consistent branch choice and is not the defining production contract.

For one unnormalized s pair and one nucleus \(\mathbf C\), the attraction seed
before multiplication by \(-Z_C\) is

\[
\frac{2\pi}{p}
\exp[-\mu_{AB}|\mathbf A-\mathbf B|^2]
\exp\left[-i\mathbf q_1\cdot\mathbf P-\frac{\mathbf q_1^2}{4p}\right]
F_0\!\left(p(\mathbf P'-\mathbf C)^2\right).
\]

For unnormalized s functions, the four-center ERI seed is

\[
\begin{aligned}
(ss|ss)={}&\frac{2\pi^{5/2}}{ps\sqrt{p+s}}
\exp[-\mu_{AB}|\mathbf A-\mathbf B|^2
      -\mu_{CD}|\mathbf C-\mathbf D|^2]\\
&\times
\exp\left[-i\mathbf q_1\cdot\mathbf P-\frac{\mathbf q_1^2}{4p}
           -i\mathbf q_2\cdot\mathbf Q-\frac{\mathbf q_2^2}{4s}\right]
F_0(T).
\end{aligned}
\]

Primitive normalizations multiply this expression. Higher angular momentum
comes from Hermite derivatives with respect to both complex-shifted pair
centers. Nuclear attraction is the one-pair counterpart, with complex Boys
argument \(p(\mathbf P'-\mathbf C)^2\).

The formula shows why Coulomb kernels need a complex Boys implementation even
though exponents, coordinates, and the magnetic field are real. It also shows
why evaluating the pair exponentials and \(F_n(T)\) as unrelated floating-point
numbers can overflow or underflow. Production Coulomb code may need a scaled
auxiliary that combines them.

For nuclear attraction, define London Coulomb auxiliaries whose spherical
seeds already contain the pair phase,

\[
R^n_{000}=e^{-i\mathbf q\cdot\mathbf P-\mathbf q^2/(4p)}
           (-2p)^nF_n(T),\qquad
T=p(\mathbf P'-\mathbf C)^2.
\]

Differentiation with respect to the real product center gives

\[
R^n_{t+1,u,v}=-iq_xR^n_{tuv}
 +(P'_x-C_x)R^{n+1}_{tuv}
 +tR^{n+1}_{t-1,u,v},
\]

and cyclic y/z analogues. Contracting \(R^0_{tuv}\) with the ordinary,
field-independent Hermite coefficients yields the Cartesian primitive. This
is equivalent to differentiating the completed-square expression while
holding the pair separation, and therefore \(\mathbf q\), fixed.

For \(\operatorname{Re}(T)<0\), use \(G_n(T)=e^T F_n(T)\) and combine the
exponent as

\[
e^{-i\mathbf q\cdot\mathbf P-\mathbf q^2/(4p)-T}G_n(T).
\]

Its real exponent contains the cancellation before floating-point
exponentiation. Production evaluates the algebraically simplified form

\[
e^{-p|\mathbf P-\mathbf C|^2-i\mathbf q\cdot\mathbf C}G_n(T),
\]

so the \(\mathbf q^2/(4p)\) terms are never formed and subtracted as large
floating-point values.

At zero field, \(\mathbf P'\) and \(\mathbf Q'\) become real and
\(T=\rho|\mathbf P-\mathbf Q|^2\ge0\), recovering the conventional ERI seed.

## 8. Valid complex symmetries

For a real multiplicative one-electron operator \(f(\mathbf r)\), and for any
self-adjoint differential operator on the chosen domain,

\[
I_{ab}=I_{ba}^{*}.
\]

For the chemists' ERI convention

\[
(ab|cd)=\iint
\omega_a^*(1)\omega_b(1)r_{12}^{-1}
\omega_c^*(2)\omega_d(2)d1d2,
\]

electron exchange and complex conjugation give only

\[
\boxed{
(ab|cd)=(cd|ab),\qquad
(ab|cd)^*=(ba|dc)=(dc|ba).
}
\]

Swapping one member of a pair has no general symmetry. The conventional real
eightfold symmetry returns at zero field.

## 9. Translation covariance

Translate every center, nucleus, and gauge origin by the same vector
\(\mathbf t\), and compare after the integration variable is translated.
The wave vector \(\boldsymbol\kappa_A\) stays fixed, while each AO gains the
constant phase \(e^{-i\boldsymbol\kappa_A\cdot\mathbf t}\). Thus a local
one-electron matrix transforms as

\[
I'_{ab}=e^{-i\mathbf q_{AB}\cdot\mathbf t}I_{ab},
\]

and an ERI transforms as

\[
(ab|cd)'=e^{-i(\mathbf q_{AB}+\mathbf q_{CD})\cdot\mathbf t}(ab|cd).
\]

These are covariance relations for AO tensors, not elementwise invariance.
Observable contractions remain invariant when AO coefficients or density
matrices receive the matching diagonal rephasing.

## 10. One-electron moments and differential operators

Milestone 3 uses operator recurrences that retain the normalization of the
original ket primitive when its polynomial angular momentum is shifted. Write
\(I(\mathbf b+\mathbf n)\) for an overlap in which only the ket polynomial
powers have changed; it is not renormalized as a different primitive.

For a Cartesian moment about the explicit origin \(\mathbf C\),

\[
(r_j-C_j)^m(r_j-B_j)^{b_j}
=\sum_{s=0}^m {m\choose s}(B_j-C_j)^{m-s}
 (r_j-B_j)^{b_j+s}.
\]

Products over the three axes give arbitrary non-negative Cartesian moment
powers. In particular, the zero-order moment is exactly overlap.

### 10.1 Gradient and canonical momentum

Differentiating the ket London primitive gives both ordinary-Gaussian and
phase terms:

\[
\boxed{
\partial_j\omega_{\mathbf b,B}
=b_j\omega_{\mathbf b-\mathbf e_j,B}^{[N_{\mathbf b}]}
-2\beta\omega_{\mathbf b+\mathbf e_j,B}^{[N_{\mathbf b}]}
-i\kappa_{B,j}\omega_{\mathbf b,B}
}.
\]

The bracketed superscript emphasizes that every shifted term retains the
original \(N_{\mathbf b}(\beta)\). A negative-power term is zero. Canonical
momentum is \(p_j=-i\partial_j\), so its matrix is Hermitian while the gradient
matrix is anti-Hermitian.

### 10.2 Canonical kinetic energy

For one axis, the phase-free polynomial derivative is

\[
\partial_j^2 g_{\mathbf b}
=b_j(b_j-1)g_{\mathbf b-2\mathbf e_j}
-2\beta(2b_j+1)g_{\mathbf b}
+4\beta^2g_{\mathbf b+2\mathbf e_j}.
\]

Including the London phase,

\[
\partial_j^2\omega_{\mathbf b,B}
=e^{-i\boldsymbol\kappa_B\cdot\mathbf r}
\left[\partial_j^2g_{\mathbf b}
-2i\kappa_{B,j}\partial_jg_{\mathbf b}
-\kappa_{B,j}^2g_{\mathbf b}\right].
\]

The canonical kinetic integral applies
\(-\tfrac12\sum_j\partial_j^2\) to the ket. It is Hermitian but depends on the
gauge origin because the basis changes while the canonical operator does not.

### 10.3 Physical magnetic kinetic energy

For the frozen electron convention, the physical one-electron kinetic
operator is

\[
\boxed{
h_{\mathrm{mag}}(\mathbf O)
=\frac12(\mathbf p+\mathbf A_{\mathbf O})^2
=\frac{\mathbf p^2}{2}
+\frac12\boldsymbol{\mathcal B}\cdot\mathbf L_{\mathbf O}
+\frac18\left[
\mathcal B^2|\mathbf r-\mathbf O|^2
-(\boldsymbol{\mathcal B}\cdot(\mathbf r-\mathbf O))^2
\right]
},
\]

where
\(\mathbf L_{\mathbf O}=(\mathbf r-\mathbf O)\times\mathbf p\). The second
and third terms are the paramagnetic and diamagnetic contributions. They are
assembled from first coordinate-momentum products and second Cartesian
moments; no low-angular-momentum cases are hard-coded.

When \(\mathbf O'=\mathbf O+\mathbf d\), the common AO gauge transformation
and the transformed vector potential obey

\[
(\mathbf p+\mathbf A_{\mathbf O'})U_{\mathbf d}
=U_{\mathbf d}(\mathbf p+\mathbf A_{\mathbf O}),
\qquad
U_{\mathbf d}=e^{\frac i2(\boldsymbol{\mathcal B}\times\mathbf d)\cdot\mathbf r}.
\]

Consequently, the physical magnetic kinetic matrix over the matching GIAO
basis is gauge-origin independent. This is tested separately from the expected
gauge-origin dependence of the canonical kinetic matrix. At zero field,
\(h_{\mathrm{mag}}\) reduces exactly to canonical kinetic energy.

## 11. Nuclear and magnetic derivatives

Future nuclear derivatives act on both the ordinary Gaussian and the phase.
For a center \(\mathbf A\),

\[
\frac{\partial\boldsymbol\kappa_A}{\partial A_j}
=\frac12\boldsymbol{\mathcal B}\times\mathbf e_j,
\]

so differentiating \(\omega_A\) produces a polynomial-Gaussian derivative and
\(-i(\partial\boldsymbol\kappa_A/\partial A_j)\cdot\mathbf r\) times the
original orbital. Magnetic-field derivatives likewise act on every London
phase. Production derivatives may use raising/lowering relations, but their
tests must isolate both contributions.
