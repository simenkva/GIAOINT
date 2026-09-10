# giao-integrals

`giao-integrals` is a C++20/pybind11 library for Cartesian
atomic-orbital integrals over complex gauge-including atomic orbitals (GIAOs),
also called London atomic orbitals.

Milestone 6 is complete. The repository contains the mathematical
specification, independent pure-Python overlap/property references, and a
production C++20 MD/Hermite engine with a NumPy API. Implemented operators are
overlap, Cartesian moments, gradient, canonical momentum, canonical kinetic
energy, the full physical magnetic kinetic energy, nuclear attraction, and
unscreened four-center electron repulsion. The complex Boys implementation
returns orders 0 through 32 with region and error diagnostics plus a
cancellation-safe scaled path.

Install a development build and run the tests with:

```console
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install --no-build-isolation -e .
.venv/bin/pytest -q
```

The public Python data types are `PrimitiveGaussian`, `Shell`, `Basis`,
`MagneticField`, and `Nucleus`. Each operator has primitive, shell, and
basis-level entry points where applicable. Result arrays are C-contiguous `complex128`; shell
blocks use `(ao_a, ao_b)` ordering, and basis matrices use input shell order.
ERIs default to packed canonical shell-quartet batches; a full `(nao,)*4`
tensor is opt-in and requires an explicit byte limit.
Optional, threshold-controlled complex Schwarz screening and OpenMP
shell-quartet parallelism are available without changing the unscreened
default.

Start with:

- [Mathematical specification](docs/mathematical_specification.md)
- [Algorithm choices](docs/algorithms.md)
- [C++ and Python API](docs/api.md)
- [Verification strategy](docs/testing.md)
- [Implementation plan](docs/implementation_plan.md)
- [Repository assessment](docs/repository_assessment.md)
- [References](docs/references.md)
- [Current status](STATUS.md)
- [Benchmark suite](benchmarks/README.md)
- [Pure-Python reference guide](reference/README.md)

The original project brief remains in
[`giao_integrals_codex_master_prompt.md`](giao_integrals_codex_master_prompt.md).
