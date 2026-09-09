# Pure-Python reference engine

`giao_reference` implements overlap and Milestone 3/4 one-electron property
integrals over Cartesian London orbitals. The code mirrors the complex-center
derivation in the mathematical specification and favors direct formulas over
recurrence speed. It does not import the C++ extension.

The ordinary reference path uses only the Python standard library. The
`high_precision` module uses mpmath for 80-digit formula and direct real-axis
quadrature checks.

## Implemented scope

- immutable primitive, segmented-shell, and magnetic-field records;
- Cartesian normalization and x-major shell ordering;
- London wave vectors and complex Gaussian pair data;
- arbitrary-angular-momentum primitive overlap by polynomial moments;
- normalized segmented contractions and complete shell-pair blocks;
- high-precision direct-formula and independent quadrature oracles;
- arbitrary Cartesian moments, gradient, canonical momentum and kinetic
  energy, and physical magnetic kinetic energy;
- direct polynomial expansion for moments and real-space finite-difference
  quadrature checks for differential operators.
- dual 80+-digit complex Boys definitions using direct quadrature and the
  entire confluent-hypergeometric form;
- nuclear attraction through an independent high-precision Obara--Saika
  recurrence, distinct from the production McMurchie--Davidson path.

The reference package returns Python complex numbers and nested tuples. NumPy
arrays belong to the production API introduced with the C++ binding.

## Example

```python
from giao_reference import MagneticField, PrimitiveGaussian, primitive_overlap

bra = PrimitiveGaussian(0.7, (-0.4, 0.2, 0.1), (1, 0, 0))
ket = PrimitiveGaussian(1.3, (0.8, -0.5, 0.9), (0, 1, 0))
field = MagneticField(B=(0.0, 0.0, 0.2), gauge_origin=(0.0, 0.0, 0.0))

value = primitive_overlap(bra, ket, field)
```

## Test commands

```console
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/pytest -q
.venv/bin/ruff check reference/python tests/reference
.venv/bin/black --check reference/python tests/reference
```

`pyproject.toml` adds `reference/python` to pytest's import path. The reference
package deliberately remains outside the installed wheel; the unified
scikit-build-core package installs only the production `giao_integrals` API.
