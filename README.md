# GIAOINT

GIAOINT computes Cartesian atomic-orbital integrals over complex
gauge-including atomic orbitals (GIAOs), also called London atomic orbitals.
It targets finite-magnetic-field quantum chemistry: NMR shielding, magnetic
susceptibility, and other properties that need gauge-origin-independent
orbitals rather than the zero-field integrals most packages provide.

The C++20 core evaluates arbitrary-angular-momentum shells with a
McMurchie-Davidson/Hermite recurrence and a production complex Boys-function
implementation (orders 0 through 32, with region and error diagnostics). A
pybind11 layer exposes it to Python as `giao-integrals`, with NumPy arrays in
and out and the GIL released during evaluation.

## What's implemented

- Overlap, Cartesian moments, gradient, canonical momentum, canonical kinetic
  energy, and the full physical magnetic kinetic energy.
- Nuclear attraction and unscreened four-center electron repulsion (ERI).
- Analytic first derivatives of the baseline Hamiltonian integrals and ERI
  shell quartets, with respect to basis centers, attraction-potential
  centers, and magnetic-field components.
- Optional complex Schwarz screening and OpenMP shell-quartet parallelism
  for ERIs; both stay off by default, so the unscreened baseline stays
  fixed unless you opt in.
- A `Molecule` wrapper that loads an XYZ geometry plus a named basis
  (`cc-pVDZ`, `STO-3G`, ...) or a Gaussian94/NWChem/BSE-JSON file and returns
  ready-to-use matrices.

Milestone 9 (ERI performance) is complete; see
[`docs/eri_performance_plan.md`](docs/eri_performance_plan.md) and
[`STATUS.md`](STATUS.md) for what's done and what isn't yet.

## Install

```console
python -m pip install '.[molecule]'
```

The `molecule` extra adds Basis Set Exchange for named basis sets and
Gaussian94/NWChem parsing. Skip it if you only need the low-level `Basis`
API or BSE JSON input.

## Quick example

```python
import giao_integrals as gi

mol = gi.Molecule.from_xyz("water.xyz", basis="cc-pVDZ")
S = mol.overlap()
H = mol.core_hamiltonian()
eri = mol.eri()  # full Cartesian tensor, 512 MiB default output limit
```

XYZ coordinates default to ångström; every integral output uses atomic
units. Read the [user manual](docs/user_manual.md) for a full walkthrough,
or run [`examples/molecular_integrals.py`](examples/molecular_integrals.py)
directly.

## Conventions worth knowing up front

The public data types are `PrimitiveGaussian`, `Shell`, `Basis`,
`MagneticField`, and `Nucleus`. Each operator has primitive, shell, and
basis-level entry points where applicable. Results are C-contiguous
`complex128` arrays, even at zero field, because GIAOINT integrals stay
complex in general; shell blocks use `(ao_a, ao_b)` ordering, and basis
matrices follow input shell order. ERIs default to packed canonical
shell-quartet batches; request the full `(nao,)*4` tensor explicitly with a
byte limit.

## Development

```console
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install --no-build-isolation -e .
.venv/bin/pytest -q
```

Tests run on Python 3.11 through 3.14. With the project's conda
environment, run `conda activate pyscf` instead and drop `.venv/bin/` from
the remaining commands. Build and test the C++ core on its own with
`cmake --preset release`, `cmake --build --preset release`, and
`ctest --preset release` (these presets need Ninja).

## Documentation

- [User manual](docs/user_manual.md): installation, basic usage, and the
  `Molecule` interface.
- [Mathematical specification](docs/mathematical_specification.md)
- [Algorithm choices](docs/algorithms.md)
- [C++ and Python API](docs/api.md)
- [Molecule and basis-file wrapper](docs/molecule_api.md)
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

Runnable examples live in [`examples/`](examples). GIAOINT is distributed
under the [BSD-3-Clause license](LICENSE).

The original project brief remains in
[`giao_integrals_codex_master_prompt.md`](giao_integrals_codex_master_prompt.md).
