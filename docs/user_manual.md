# GIAOINT user manual

GIAOINT (package name `giao-integrals`) computes Cartesian atomic-orbital
integrals over complex gauge-including atomic orbitals (GIAOs), also called
London atomic orbitals. This manual covers installation, the low-level array
API, and the `Molecule` wrapper for XYZ geometries and standard basis files.
For the full API reference, mathematical background, and algorithm notes, see
the [docs index](../README.md#documentation).

## 1. Installation

GIAOINT supports CPython 3.11–3.14 and depends on NumPy. Its compiled core is
a C++20 extension built with pybind11; pybind11 is a build dependency, not
a prebuilt GIAOINT core. Source installation requires a C++20 compiler; normal
pip build isolation installs the Python build dependencies and any required
CMake frontend. Version 0.9.0 publication is pending approval and release CI.
See [installation and threading](installation.md) for platform details.

### Install from a repository checkout

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
.venv/bin/python -m pip install -e .
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
without the Python bindings. OpenMP defaults to AUTO; on macOS, `brew install
libomp` enables automatic runtime discovery. Require OpenMP with
`-DGIAO_ENABLE_OPENMP=ON` or select serial with `OFF`. Verify Python installs
with `gi.build_info()`; editable installs require reinstalling after C++ edits.

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

`B` and the gauge origin are in atomic units. Low-level integral functions
accept `field=None` (the default) for exact zero field and zero gauge origin.
`Molecule` integral methods use the field stored on the molecule.

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

`mol.eri()` allocates a full `(nao, nao, nao, nao)` tensor, with no default
output cap (`max_bytes=None`). Set an explicit byte limit
with `mol.eri(max_bytes=...)` to raise `MemoryError` before an allocation
exceeding that limit. For larger systems, stream canonical shell-quartet
blocks instead:

```python
for batch in mol.eri_batches(target_bytes=16 * 1024**2):
    print(batch.quartets, batch.values.shape)
```

`eri()`, `eri_batches()`, and `integrals(eri=True)` all accept
`screening_threshold=` for complex Schwarz screening and `threads=` for
OpenMP shell-quartet parallelism. Screening remains off. Since 0.9,
`threads=None` uses the configured parallel default; `threads=1` stays serial.
See [installation and thread policy](installation.md).

For a low-rank representation, use streamed Cholesky without allocating the
full AO tensor:

```python
chol = mol.cholesky_eri(threshold=1e-8)
L = chol.vectors  # complex128, (M, nao, nao)
# ERI[mu,nu,lam,sig] ≈ sum_P L[P,mu,nu] * conj(L[P,sig,lam])
print(chol.pivots, chol.max_residual_diagonal)
```

`max_bytes` optionally bounds the vector and pivot output. The threshold is
on the largest residual diagonal of the Hermitian ordered-pair Coulomb matrix;
see [Cholesky conventions](cholesky_eri.md) for the numerical floor and error
bound. Avoid `chol.reconstruct()` without slices on large systems because that
helper creates a dense tensor.

### Timing parallel Cholesky for an atom

Run the [atom timing example](../examples/atom_cholesky_timing.py) from the
checkout. The default Ne/cc-pVDZ case is quick; explicit thread counts require
an OpenMP build when greater than one:

```console
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python examples/atom_cholesky_timing.py --threads 1 2 4
```

It reports one wall time per count, speedup relative to the first run, rank,
M/nao, vector MiB and the measured residual maximum. It checks factor and pivot
hashes for bitwise agreement and retains only one result at a time. No dense
ERI tensor is allocated. Small cases may be slower in parallel due to overhead.
For the larger 152-AO Ne case, which can take minutes per run:

```console
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python examples/atom_cholesky_timing.py --basis aug-cc-pCVQZ \
  --uncontracted --threads 1 4 8 12
```

Use `--atom`, `--threshold`, or `--field BX BY BZ` to change the inputs.
For one atom the common London phase cancels from pair densities, so the ERIs
are field-independent even when a finite field is supplied.

### Dropping to the lower-level engine

`mol.basis`, `mol.nuclei`, and `mol.field` expose the same objects the
low-level functions in Section 2 take, so you can mix the two APIs freely:

```python
dS = gi.overlap_magnetic_derivatives(mol.basis, field=mol.field)
```

`mol.atoms` returns element symbols and coordinates in bohr; `mol.ao_count`
returns the Cartesian AO count; `mol.shell_atoms` returns each shell's
zero-based owner atom.

### Moment derivatives for electric-field gradients

With a source build after 0.9.0, obtain basis-centre and moment-origin
responses separately:

```python
dR, dO = gi.moment_nuclear_derivatives(
    mol.basis, (1, 0, 0), origin=(0.0, 0.0, 0.0), field=mol.field,
)
```

`dR` has shape `(nshell, 3, nao, nao)` and includes London-phase centre
response. `dO` has shape `(3, nao, nao)`; for an x dipole its x component is
`-overlap`. These are literal derivatives of the unsigned coordinate moment,
not forces or electronic dipole derivatives. Multiply by the sign and electric
field coefficients used by your Hamiltonian. Sum `dR` over shells owned by each
atom using `mol.shell_atoms`. Moment-origin contributions remain separate.
General powers, strict `out=(shell_buffer, origin_buffer)`, and `threads=None`
are supported. See [the API conventions](api.md#moment-nuclear-derivatives-priority-1-after-090)
and [validation and timings](derivative_validation.md).

### Gradient, momentum and property field derivatives

With the post-0.9.0 source build, choose the property component separately
from the derivative axis:

```python
dR_px = gi.momentum_nuclear_derivatives(mol.basis, "x", field=mol.field)
dB_px = gi.momentum_magnetic_derivatives(mol.basis, "x", field=mol.field)
dB_x = gi.moment_magnetic_derivatives(mol.basis, (1, 0, 0), field=mol.field)
```

Replace `momentum` with `gradient` for ket spatial derivatives; canonical
momentum is `-i*gradient`. Nuclear shapes are `(nshell,3,nao,nao)` and field
shapes `(3,nao,nao)`. Field derivatives hold both origins fixed and include
London response under the spatial operator. These APIs accept `out=` and
`threads=None`; [validation](derivative_validation.md) includes independent
reference/libcint checks and a 105-AO timing table.

### Mixed nuclear/magnetic derivatives

```python
dRdB_S = gi.overlap_mixed_derivatives(mol.basis, field=mol.field)
dRdB_T = gi.magnetic_kinetic_mixed_derivatives(mol.basis, field=mol.field)
dRdB_V_shell, dRdB_V_nucleus = gi.nuclear_attraction_mixed_derivatives(
    mol.basis, mol.nuclei, field=mol.field,
)
```

The leading axes are `(centre, nuclear xyz, field xyz)`, followed by the AO
matrix. Basis and attraction-potential centres are independent and returned
separately. `kinetic_mixed_derivatives` supplies canonical kinetic response;
`magnetic_kinetic_mixed_derivatives` includes the explicit physical operator
field response. All hold the gauge origin fixed and return literal derivatives.
ERI quartet response is `eri_mixed_derivatives_shell(a,b,c,d,field=mol.field)`,
shape `(4,3,3,na,nb,nc,nd)`. See [API details](api.md) for output buffers,
threading, translation covariance and extra angular-order requirements.
These APIs require a source build after 0.9.0.

### Streaming ERI gradient blocks

```python
for batch in mol.eri_derivative_batches(target_bytes=8 * 1024**2, threads=None):
    for index, quartet in enumerate(batch.quartets):
        dERI = batch.block(index)  # (4,3,na,nb,nc,nd), a view
        atoms = [mol.shell_atoms[int(shell)] for shell in quartet]
        # Contract each centre slot into the owning atom's gradient here.
```

This post-0.9.0 interface replaces Python quartet-evaluation loops. It yields
canonical finite-field quartets; expand symmetry with matching centre-slot
permutations when your contraction requires all ordered quartets. Repeated
shell/atom slots contribute separately and must be added. Explicit `quartets=`
is also supported. The memory target counts derivative values, with at least
one complete quartet per batch, plus metadata and native thread workspaces.
`screening_threshold=0.0` preserves exact unscreened results. Positive thresholds
use derivative-specific Coulomb bounds including London terms; see
[the screening proof and conventions](api.md#safe-derivative-screening).
The optional density-to-J/K gradient contraction helper is not part of this API.

## 4. Where to go next

- [Runnable examples](../examples/): scripts covering direct-basis
  matrices, batched ERIs, the molecule wrapper, and a PySCF cross-check.
- [`docs/api.md`](api.md): the C++ and Python API overview and conventions.
- [`docs/molecule_api.md`](molecule_api.md): the full `Molecule` reference
  this manual summarizes.
- [`docs/mathematical_specification.md`](mathematical_specification.md):
  conventions, normalization, and derivations.
- [`STATUS.md`](../STATUS.md): what's implemented and what isn't yet.
