# giao-integrals

`giao-integrals` is a C++20/pybind11 library for Cartesian
atomic-orbital integrals over complex gauge-including atomic orbitals (GIAOs),
also called London atomic orbitals.

Milestone 3 is complete. The repository contains the mathematical
specification, independent pure-Python overlap/property references, and a
production C++20 MD/Hermite engine with a NumPy API. Implemented operators are
overlap, Cartesian moments, gradient, canonical momentum, canonical kinetic
energy, and the full physical magnetic kinetic energy. Complex Boys functions
and nuclear attraction are the next milestone.

Install a development build and run the tests with:

```console
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install --no-build-isolation -e .
.venv/bin/pytest -q
```

The public Python data types are `PrimitiveGaussian`, `Shell`, `Basis`, and
`MagneticField`. Each operator has primitive, shell, and basis-level entry
points where applicable. Result arrays are C-contiguous `complex128`; shell
blocks use `(ao_a, ao_b)` ordering, and basis matrices use input shell order.

Start with:

- [Mathematical specification](docs/mathematical_specification.md)
- [Algorithm choices](docs/algorithms.md)
- [C++ and Python API](docs/api.md)
- [Verification strategy](docs/testing.md)
- [Implementation plan](docs/implementation_plan.md)
- [Repository assessment](docs/repository_assessment.md)
- [References](docs/references.md)
- [Current status](STATUS.md)
- [Pure-Python reference guide](reference/README.md)

The original project brief remains in
[`giao_integrals_codex_master_prompt.md`](giao_integrals_codex_master_prompt.md).
