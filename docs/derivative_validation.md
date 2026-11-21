# Additional derivative integrals: validation and timings

Each priority is a separate releasable implementation step after PyPI 0.9.0.
No new PyPI version is published by this work.

## Priority 1: moment nuclear and moment-origin derivatives

Validated on 21 November 2026, macOS arm64, CPython 3.13 and AppleClang.
`tests/python/test_moment_derivatives.py` passes 35 checks, covering every
Cartesian power of total degree ≤2, a higher `(3,1,0)` moment, zero and finite
B=(0.17,-0.11,0.23) with off-centre gauge G=(0.2,-0.3,0.1), normalised primitive
functions through g, general contractions, same-shell accumulation, Hermiticity,
strict output buffers and serial/parallel equality. Two-step central differences
use h=4e-4 bohr and h/2 with Richardson extrapolation. The largest absolute
validation deviation is 2.699e-12. Independent pure-Python reference finite
differences also pass. Gaussian and London-phase terms are checked separately.

The optional `tests/external/moment_derivatives_pyscf.py` compares H2O/cc-pVDZ
(25 Cartesian AOs) to PySCF 2.8.0/libcint `int1e_irp_cart`, using identical bohr
coordinates, origin and Cartesian normalisation. Ket spatial derivatives have
the opposite sign to centre derivatives; bra contributions use transpose.
Maximum centre-derivative disagreement: 1.332e-15. Origin derivatives agree
with the exact `-overlap` dipole identity.

Translation is tested in the raw London AO convention: the derivative sum
including moment and gauge origins is `i*(k_mu-k_nu)*I`, and vanishes at B=0.
A literal zero sum at finite B is incompatible with these existing AO phases.
This convention is documented in [the API](api.md).

`benchmarks/property_derivatives.py --threads 1 4` times the x-moment nuclear
and origin derivatives for Ne/aug-cc-pVQZ: 105 Cartesian AOs, 10 shells,
B=(0.05,-0.03,0.2), G=(0.37,-0.29,0.43), O=(0.2,0.3,-0.1). BLAS is limited
to one thread. Basis construction and equality checks are excluded.

| Threads | Seconds | Output MiB | Equality |
|---:|---:|---:|---|
| 1 | 0.150322 | 5.552 | Reference |
| 4 | 0.040469 | 5.552 | Bitwise identical |

These are single local measurements, not portable speedup guarantees.

## Priority 2: remaining one-electron property derivatives

`tests/python/test_property_derivatives.py` passes 47 checks. All gradient
and momentum components are compared to two-step Richardson differences
(h=4e-4) at zero/finite field, with an off-centre gauge and primitive angular
momentum through g. General moment field derivatives, general contractions,
shell/basis consistency, strict buffers, canonical `momentum=-i*gradient`,
anti-/Hermiticity and translation covariance also pass. The independent Python
reference engine supplies value differences. Maximum absolute deviation across
these validation comparisons: 3.712e-12.

`tests/external/property_derivatives_pyscf.py` compares H2O/cc-pVDZ (25 Cartesian
AOs) to PySCF 2.8.0/libcint at B=0. Centre responses use `int1e_ipovlpip_cart`;
field responses use `int1e_irp_cart`, `int1e_rr_cart` and overlap, with the
pair-phase and ket spatial-phase terms constructed separately. All components
agree, maximum error 1.776e-15.

Same 105-AO basis/field/gauge/BLAS settings as Priority 1, x component and x
moment. Each four-thread output is bitwise identical to its serial reference:

| Derivative | 1 thread seconds | 4 threads seconds | Output MiB |
|---|---:|---:|---:|
| Gradient nuclear | 0.172335 | 0.044362 | 5.047 |
| Momentum nuclear | 0.172674 | 0.047202 | 5.047 |
| Moment magnetic | 0.099818 | 0.026322 | 0.505 |
| Gradient magnetic | 0.115853 | 0.030501 | 0.505 |
| Momentum magnetic | 0.116818 | 0.030771 | 0.505 |

## Priority 3: mixed nuclear/magnetic derivatives

49 mixed tests pass; the full Python suite has 590 passing tests. The tests
cover overlap, canonical/physical magnetic kinetic, attraction basis/potential
centres and ERI quartets, at zero and finite field with the same off-centre
gauge. Primitive angular momentum reaches g. Two-dimensional central
Richardson differences against value integrals use h=2e-3 in both coordinates;
independent differences of first derivatives use h=4e-4 in each differentiation
order. Maximum absolute deviation across all mixed validation comparisons:
2.494e-10. Tests also use independent pure-Python value kernels for all five
operators, check shell/basis assembly, same-shell accumulation, strict buffers,
ERI symmetries, Hermiticity and differentiated translation covariance.

`tests/external/mixed_derivatives_pyscf.py` checks H2O/cc-pVDZ (25 Cartesian AOs)
at B=0. The overlap cross coefficient and Gaussian moment-response terms are
built independently from libcint coordinate and `int1e_irp_cart` matrices.
The physical-minus-canonical kinetic mixed response, grouped by atom, agrees
with Richardson differences of the explicit angular-momentum operator built
from libcint `irp`. Maximum libcint/Richardson disagreement: 1.210e-12.

Same 105-AO finite-field benchmark as earlier priorities. The full mixed tensor
is timed, not a single axis. Shifted-pair magnetic responses are cached across
axes; no derivative is approximated numerically in production.

| Mixed derivative | 1 thread seconds | 4 threads seconds | Output MiB |
|---|---:|---:|---:|
| Overlap | 0.536072 | 0.146368 | 15.141 |
| Canonical kinetic | 5.826756 | 1.523662 | 15.141 |
| Physical magnetic kinetic | 39.092522 | 10.170701 | 15.141 |
| Attraction (basis + one Ne potential) | 18.249122 | 5.465344 | 16.655 |

Every four-thread result is bitwise identical to serial. These are single local
measurements with BLAS limited to one thread; the composed physical-kinetic
kernel is still substantially more expensive than overlap.

## Remaining priorities

Priority 4 adds streamed
ERI derivative batches and derivative-specific screening/performance work.
NMR shielding and spin–spin coupling require new nuclear magnetic operators;
they remain possible later milestones, alongside the separately planned TD4.0
work. Cholesky-vector derivatives remain out of scope.
