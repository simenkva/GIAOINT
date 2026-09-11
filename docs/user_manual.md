# GIAOINT user manual

GIAOINT (package name `giao-integrals`) computes Cartesian atomic-orbital
integrals over complex gauge-including atomic orbitals (GIAOs), also called
London atomic orbitals. This manual covers installation, the low-level array
API, and the `Molecule` wrapper for XYZ geometries and standard basis files.
For the full API reference, mathematical background, and algorithm notes, see
the [docs index](../README.md#documentation).

## 1. Installation

GIAOINT requires Python 3.11 through 3.14 and NumPy. The compiled core ships
as a prebuilt wheel dependency (`pybind11`) plus a C++20 extension module, so
installing from source needs a C++20 compiler and CMake.

### Standard install

```console
python -m pip install '.[molecule]'
```

The `molecule` extra pulls in Basis Set Exchange, which supplies named basis
sets (`cc-pVDZ`, `STO-3G`, and so on) and parsers for Gaussian94 and NWChem
basis files. Skip the extra if you only need the low-level `Basis`/`Shell`
API or BSE JSON basis input; those work without it.

### Development install

```console
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install --no-build-isolation -e .
.venv/bin/pytest -q
```

If you use the project's conda environment instead, run
`conda activate pyscf` and drop `.venv/bin/` from the remaining commands.

### Building the C++ core alone

```console
cmake --preset release
cmake --build --preset release
ctest --preset release
```

These presets require Ninja and build the C++ library and its test suite
without the Python bindings.

## 2. Basic usage

Import the package and build a basis directly when you already know your
shells, or use `Molecule` (Section 3) to load a geometry and a named basis
set. A minimal direct-basis example:

```python
import giao_integrals as gi

basis = gi.Basis([
    gi.Shell((0.0, 0.0, 0.0), 0, [5.0, 1.0], [0.35, 0.75]),  # s shell
    gi.Shell((0.0, 0.0, 0.0), 1, [1.2], [1.0]),               # p shell
])

overlap = gi.overlap(basis)
kinetic = gi.kinetic(basis)
```

`Shell` takes a center in bohr, an angular-momentum integer (0 for s, 1 for
p, 2 for d, ...), a list of exponents, and matching contraction
coefficients. Every result is a C-contiguous `complex128` NumPy array, even
at zero magnetic field, because GIAOINT integrals stay complex in general.

Add a magnetic field to any calculation with `MagneticField`:

```python
field = gi.MagneticField((0.0, 0.0, 0.02), gauge_origin=(0.1, -0.1, 0.0))
overlap = gi.overlap(basis, field=field)
kinetic = gi.magnetic_kinetic(basis, field=field)
```

`B` and the gauge origin are in atomic units. Every function in the package
accepts `field=None` (the default) for an exact zero field and zero gauge
origin.

Compute nuclear attraction and electron-repulsion integrals with `Nucleus`
objects and the `nuclear_attraction`/`eri` functions:

```python
nuclei = [
    gi.Nucleus(8.0, (0.0, 0.0, 0.0)),
    gi.Nucleus(1.0, (1.43, 1.11, 0.0)),
    gi.Nucleus(1.0, (-1.43, 1.11, 0.0)),
]
attraction = gi.nuclear_attraction(basis, nuclei, field=field)
eri = gi.eri(basis, field=field, storage="full", max_bytes=64 * 1024**2)
```

`gi.eri` streams shell-quartet blocks by default (`storage="blocks"`); pass
`storage="full"` and an explicit `max_bytes` to materialize the full
`(nao, nao, nao, nao)` tensor instead. See
[`examples/water_matrices.py`](../examples/water_matrices.py) and
[`examples/eri_batches.py`](../examples/eri_batches.py) for complete,
runnable versions of both patterns, and
[`docs/api.md`](api.md) for the full function reference, including
derivatives, screening, and threading options.

## 3. The molecule Python interface

For everyday work, skip manual shell construction and load a geometry and a
basis together through `Molecule`.

### Load a geometry and a basis

```python
import giao_integrals as gi

mol = gi.Molecule.from_xyz("water.xyz", basis="cc-pVDZ")

S = mol.overlap()
T = mol.kinetic()
V = mol.nuclear_attraction()
H = mol.core_hamiltonian()
eri = mol.eri()
```

Or request several matrices in one call:

```python
integrals = mol.integrals(eri=True)
S = integrals.overlap
H = integrals.core_hamiltonian
eri = integrals.eri  # None unless eri=True
```

`from_xyz` accepts a filename, a `pathlib.Path`, or XYZ text directly. It
reads exactly one XYZ frame (atom count, comment line, then one row per
atom), rejects malformed rows and unknown elements, and does not interpret
extended-XYZ metadata or periodic cells. Coordinates default to ångström;
pass `unit="bohr"` if they are already in atomic units. GIAOINT stores
geometry and reports every integral in atomic units regardless of the input
unit.

You can also build a molecule from Python coordinates directly, without a
file:

```python
mol = gi.Molecule(
    [("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
    basis="cc-pVDZ",
    unit="bohr",
)
```

### Basis input

| Input | Example | Needs the `molecule` extra? |
|---|---|---|
| Named BSE basis | `basis="cc-pVDZ"` | Yes |
| Gaussian94 file | `basis="custom.gbs"` | Yes |
| NWChem file | `basis="custom.nw"` | Yes |
| BSE JSON file | `basis="custom.json"` | No |
| BSE dictionary or JSON text | `basis=data` | No |

For a file extension `Molecule` doesn't recognize, name the format
explicitly with `basis_format="gaussian94"` or `"nwchem"`.

Output is always Cartesian, even when the source basis specifies spherical
functions: a d shell contributes six Cartesian components, an f shell ten.
`Molecule` only accepts all-electron Gaussian bases; it rejects effective
core potentials and non-Gaussian shells rather than silently dropping them.
Every element in the molecule needs basis data. Ghost centers, per-atom
basis overrides, charge/spin state, and an SCF solver sit outside this API's
scope.

### Magnetic fields

```python
field = gi.MagneticField((0.0, 0.0, 0.02), gauge_origin=(0.1, 0.0, 0.0))
mol = gi.Molecule.from_xyz("water.xyz", basis="cc-pVDZ", field=field)
H = mol.core_hamiltonian()
```

`Molecule` forwards this field to every integral method. `core_hamiltonian`
combines nuclear attraction with the physical magnetic kinetic energy; the
plain `kinetic()` method keeps the canonical operator `-1/2 ∇²` instead.
Neither includes spin coupling or nuclear repulsion.

### Large ERI tensors

`mol.eri()` allocates a full `(nao, nao, nao, nao)` tensor, guarded by a
default 512 MiB limit that raises `MemoryError` before allocation if
`16 * nao**4` would exceed it. Raise or lower that limit with
`mol.eri(max_bytes=...)`. For larger systems, stream canonical shell-quartet
blocks instead:

```python
for batch in mol.eri_batches(target_bytes=16 * 1024**2):
    print(batch.quartets, batch.values.shape)
```

`eri()`, `eri_batches()`, and `integrals(eri=True)` all accept
`screening_threshold=` for complex Schwarz screening and `threads=` for
OpenMP shell-quartet parallelism; both default off, so the unscreened,
serial baseline never changes underneath you.

### Dropping to the lower-level engine

`mol.basis`, `mol.nuclei`, and `mol.field` expose the same objects the
low-level functions in Section 2 take, so you can mix the two APIs freely:

```python
dS = gi.overlap_magnetic_derivatives(mol.basis, field=mol.field)
```

`mol.atoms` returns element symbols and coordinates in bohr; `mol.ao_count`
returns the Cartesian AO count; `mol.shell_atoms` returns each shell's
zero-based owner atom.

## 4. Where to go next

- [Runnable examples](../examples/): five scripts covering direct-basis
  matrices, batched ERIs, the molecule wrapper, and a PySCF cross-check.
- [`docs/api.md`](api.md): the complete C++ and Python function reference.
- [`docs/molecule_api.md`](molecule_api.md): the full `Molecule` reference
  this manual summarizes.
- [`docs/mathematical_specification.md`](mathematical_specification.md):
  conventions, normalization, and derivations.
- [`STATUS.md`](../STATUS.md): what's implemented and what isn't yet.
