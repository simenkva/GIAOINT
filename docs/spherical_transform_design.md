# Spherical transformation design

The production kernels remain Cartesian. A future spherical interface will be
a separate linear-transformation layer over completed Cartesian shell blocks;
it will not enter primitive recurrences, Boys evaluation, screening bounds, or
workspace sizing.

For a shell of angular momentum \(L\), define \(C_L\) with shape
\((L+1)(L+2)/2 \times (2L+1)\). Each column is one spherical AO expanded
in the repository's Cartesian components: Cartesian expansion coefficients
are \(c_{\mathrm{cart}}=C_Lc_{\mathrm{sph}}\). A one-electron block
therefore transforms as

\[
  I^{\mathrm{sph}}_{AB} = C_A^\dagger I^{\mathrm{cart}}_{AB} C_B,
\]

and an ERI block receives one transform on each AO axis. Derivative axes stay
leading and are never transformed. General-contraction axes remain outside the
component axis, matching current AO ordering.

## Contract to freeze before implementation

- real versus complex spherical harmonics;
- phase convention and the order of \(m\) components;
- normalization of solid harmonics;
- whether transformation matrices are public data or implementation details;
- mixed Cartesian/spherical shell support and its serialized representation.

The initial candidate is normalized real solid harmonics with an explicitly
tabulated \(m\) order, but no convention is promised until values are checked
against an external engine. Matrices should be generated once per \(L\),
cached as immutable data, and tested for rank and metric consistency.

Tests must cover s/p identity cases, d/f tabulated coefficients, Hermiticity,
finite-field ERI symmetries, derivative-axis preservation, and equivalence
between transform-after-assembly and transform-per-shell workflows. Screening
continues to operate on Cartesian bounds unless a separate proof establishes a
tighter transformed bound.
