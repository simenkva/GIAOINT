# giao-integrals

`giao-integrals` is a C++20/pybind11 library for Cartesian
atomic-orbital integrals over complex gauge-including atomic orbitals (GIAOs),
also called London atomic orbitals.

Milestone 8 is complete; Milestone 9 (ERI performance) is planned, see
`docs/eri_performance_plan.md`. The repository contains the mathematical
specification, independent pure-Python overlap/property references, and a
production C++20 MD/Hermite engine with a NumPy API. Implemented operators are
overlap, Cartesian moments, gradient, canonical momentum, canonical kinetic
energy, the full physical magnetic kinetic energy, nuclear attraction, and
unscreened four-center electron repulsion. The complex Boys implementation
returns orders 0 through 32 with region and error diagnostics plus a
cancellation-safe scaled path.

Install a development build and run the tests with Python 3.11--3.14:

```console
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install --no-build-isolation -e .
.venv/bin/pytest -q
```

With the project conda environment, replace the first line with
`conda activate pyscf` and use `python` in the remaining commands. The C++20
core can be built independently with `cmake --preset release`,
`cmake --build --preset release`, and `ctest --preset release`; the presets
require Ninja.

The public Python data types are `PrimitiveGaussian`, `Shell`, `Basis`,
`MagneticField`, and `Nucleus`. Each operator has primitive, shell, and
basis-level entry points where applicable. Result arrays are C-contiguous `complex128`; shell
blocks use `(ao_a, ao_b)` ordering, and basis matrices use input shell order.
ERIs default to packed canonical shell-quartet batches; a full `(nao,)*4`
tensor is opt-in and requires an explicit byte limit.
Optional, threshold-controlled complex Schwarz screening and OpenMP
shell-quartet parallelism are available without changing the unscreened
default.
Analytic first derivatives with respect to basis centers, attraction-potential
centers, and magnetic-field components are available for the baseline
Hamiltonian integrals and ERI shell quartets. Basis and potential-center
responses remain separate because the basis model does not assume atom-to-shell
ownership.

Start with:

- [Mathematical specification](docs/mathematical_specification.md)
- [Algorithm choices](docs/algorithms.md)
- [C++ and Python API](docs/api.md)
- [Verification strategy](docs/testing.md)
- [Implementation plan](docs/implementation_plan.md)
- [Repository assessment](docs/repository_assessment.md)
- [References](docs/references.md)
- [Current status](STATUS.md)
- [Compatibility policy](docs/compatibility.md)
- [Spherical transform design](docs/spherical_transform_design.md)
- [ERI performance plan](docs/eri_performance_plan.md)
- [Release checklist](docs/release_checklist.md)
- [Benchmark suite](benchmarks/README.md)
- [Pure-Python reference guide](reference/README.md)

Runnable examples are in [`examples/`](examples). The project is distributed
under the [BSD-3-Clause license](LICENSE).

The original project brief remains in
[`giao_integrals_codex_master_prompt.md`](giao_integrals_codex_master_prompt.md).
