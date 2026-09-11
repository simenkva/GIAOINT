# Molecules and standard basis files

Use `Molecule` to load a geometry, attach a basis, and compute NumPy arrays.
Install the molecule extra once from this repository:

```console
python -m pip install '.[molecule]'
```

For an editable development installation, use `pip install -e '.[molecule]'`.
The extra installs Basis Set Exchange and its local basis database. Named
basis lookup and file parsing do not make network requests. The low-level
integral package and BSE JSON input still work without the extra.

## Start with an XYZ file

```python
import giao_integrals as gi

mol = gi.Molecule.from_xyz("water.xyz", basis="cc-pVDZ")

S = mol.overlap()
T = mol.kinetic()
V = mol.nuclear_attraction()
H = mol.core_hamiltonian()
eri = mol.eri()
```

Or compute them together:

```python
integrals = mol.integrals(eri=True)
S = integrals.overlap
H = integrals.core_hamiltonian
eri = integrals.eri
```

`mol.integrals()` omits ERIs by default; its `eri` attribute is `None`.
The other attributes are `overlap`, `kinetic`, `nuclear_attraction`, and
`core_hamiltonian`. Results use atomic units and C-contiguous `complex128`
arrays. One-electron matrices have shape `(nao, nao)`; full ERIs have shape
`(nao, nao, nao, nao)` in chemists' `(ab|cd)` ordering.

A runnable example and geometry are in
[`examples/molecular_integrals.py`](../examples/molecular_integrals.py) and
[`examples/water.xyz`](../examples/water.xyz).

## Geometry input and units

Pass a filename, a `pathlib.Path`, or multiline XYZ text:

```python
mol = gi.Molecule.from_xyz(
    """3
Water
O 0.00 0.00  0.00
H 0.00 0.00  0.96
H 0.93 0.00 -0.24
""",
    basis="STO-3G",
)
```

The reader accepts one XYZ frame: an atom count, a comment line (which may
be blank), and one element plus three coordinates per atom. Symbols are
case-insensitive; atomic numbers 1 through 118 are also accepted. Extra
frames, malformed rows, unknown elements, and nonfinite coordinates raise
errors. It does not interpret extended-XYZ metadata, periodic cells, or
additional atom columns.

XYZ coordinates default to ångström, following the usual
[XYZ convention](https://openbabel.org/docs/FileFormats/XYZ_cartesian_coordinates_format.html).
Use `unit="bohr"` for coordinates already in atomic units. The conversion
uses `1 bohr = 0.529177210544 angstrom`, from
[CODATA 2022](https://physics.nist.gov/cuu/pdf/wall_2022.pdf). Stored coordinates
and all outputs use bohr and atomic units; the XYZ comment does not override
`unit`.

For coordinates already in Python:

```python
mol = gi.Molecule(
    [("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
    basis="cc-pVDZ",
    unit="bohr",
)
```

## Basis input

| Input | Example | Optional extra needed? |
|---|---|---|
| Named BSE basis | `basis="cc-pVDZ"` | Yes |
| Gaussian94 file | `basis="custom.gbs"` or `"custom.g94"` | Yes |
| NWChem file | `basis="custom.nw"` or `"custom.nwchem"` | Yes |
| BSE JSON file | `basis="custom.json"` | No |
| BSE dictionary or JSON text | `basis=data` | No |

For another file extension, specify the format:

```python
mol = gi.Molecule.from_xyz(
    "water.xyz", basis="custom.txt", basis_format="gaussian94"
)
```

Pass formatted basis text with `basis_format="gaussian94"` or `"nwchem"`.
For JSON, supply a BSE object containing `elements`, keyed by atomic number,
with `electron_shells`, `angular_momentum`, `exponents`, and `coefficients`.
The wrapper delegates Gaussian94 and NWChem parsing to the
[Basis Set Exchange reader API](https://molssi-bse.github.io/basis_set_exchange/user_api.html).

The wrapper preserves general contractions and splits shared-exponent SP
shells into their angular components. Atom order follows the geometry. Within
each atom, shells have stable increasing angular-momentum order; contractions
and Cartesian components follow the existing library ordering. `mol.shell_atoms`
provides the zero-based owner atom for each shell.

**Output is always Cartesian**, including when a source basis specifies
spherical functions: a d shell contributes six Cartesian components, an f
shell ten. The wrapper uses the Gaussian exponents and contraction coefficients
from that source and normalizes each Cartesian AO with the core library's
convention. It does not provide a spherical transformation.

Only all-electron Gaussian orbital bases are supported. The wrapper rejects
ECP data and non-Gaussian shells instead of discarding unsupported operators.
Each element in the molecule must have basis data. Ghost centers, per-atom
basis overrides, charge/spin state, and an SCF solver are outside this API.
Electronic charge and spin do not enter these spatial AO integral definitions.

## Magnetic fields

```python
field = gi.MagneticField((0.0, 0.0, 0.02), gauge_origin=(0.1, 0.0, 0.0))
mol = gi.Molecule.from_xyz("water.xyz", basis="cc-pVDZ", field=field)
H = mol.core_hamiltonian()
```

`field` uses the existing `MagneticField` type: field components are in atomic
units and the gauge origin is in bohr, regardless of the geometry's input
unit. The wrapper forwards this field to every integral method.
`kinetic()` and the result's `kinetic` attribute retain the low-level
canonical operator `-1/2 ∇²`. `magnetic_kinetic()` evaluates the physical
orbital magnetic kinetic energy; `core_hamiltonian()` and the result's
`core_hamiltonian` use that operator plus nuclear attraction. Spin coupling
and nuclear repulsion are not included in the one-electron Hamiltonian.

## ERI size, screening, and threads

`mol.eri()` returns a full tensor with a default 512 MiB output limit.
It raises `MemoryError` before allocating the tensor if `16 * nao**4` exceeds
the limit. The limit covers the output tensor, not total process memory.
Change it with `mol.eri(max_bytes=...)` or
`mol.integrals(eri=True, max_eri_bytes=...)`.

For larger calculations, stream canonical shell-quartet blocks:

```python
for batch in mol.eri_batches(target_bytes=16 * 1024**2):
    print(batch.quartets, batch.values.shape)
```

`eri()`, `eri_batches()`, and `integrals(eri=True)` accept
`screening_threshold=...` and `threads=...` for ERI evaluation. Defaults remain
unscreened and serial; more than one thread requires an OpenMP build.

The low-level `gi.eri(basis, ...)` continues to default to streamed blocks
and requires an explicit byte limit for full output. The molecule wrapper's
bounded full-tensor convenience does not change that API.

## Access to the rest of the engine

`mol.basis`, `mol.nuclei`, and `mol.field` work with the existing property,
derivative, screening, and shell APIs. `mol.atoms` returns the element symbols
and coordinates in bohr; `mol.ao_count` returns the Cartesian AO count.
For example:

```python
dS = gi.overlap_magnetic_derivatives(mol.basis, field=mol.field)
```

The wrapper retains shell ownership for callers, but the low-level derivative
functions keep their existing shell-center and potential-center axes.
