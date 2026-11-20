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
- Optional complex Schwarz screening and automatic OpenMP shell-quartet
  parallelism for ERIs. Screening remains off; parallel and serial results
  are tested for bitwise equality.
- Streamed pivoted Cholesky ERIs with ordered-pair complex factors and a
  residual-diagonal stopping criterion, without a dense AO tensor.
- A `Molecule` wrapper that loads an XYZ geometry plus a named basis
  (`cc-pVDZ`, `STO-3G`, ...) or a Gaussian94/NWChem/BSE-JSON file and returns
  ready-to-use matrices.

Milestone 9 (ERI performance) is complete; see
[`docs/eri_performance_plan.md`](docs/eri_performance_plan.md) and
[`STATUS.md`](STATUS.md) for what's done and what isn't yet.

## Install

From a repository checkout:

```console
python -m pip install '.[molecule]'
```

The `molecule` extra adds Basis Set Exchange for named basis sets and
Gaussian94/NWChem parsing. Skip it if you only need the low-level `Basis`
API or BSE JSON input.

Source installs require Python 3.11–3.14 and a C++20 compiler (AppleClang,
GCC, or Clang). pip installs build dependencies in an isolated environment:

```console
python -m pip install .
python -m pip install -e .
```

On macOS, install Xcode Command Line Tools (`xcode-select --install`) and
`brew install libomp` for OpenMP. CMake defaults to `GIAO_ENABLE_OPENMP=AUTO`:
it finds Homebrew automatically or reports a serial fallback. Linux GCC
normally includes OpenMP support. Force a required parallel build with
`-Ccmake.define.GIAO_ENABLE_OPENMP=ON`, or serial with `OFF`.

```python
import giao_integrals as gi
print(gi.build_info())
gi.set_num_threads(4)  # process-wide default; threads=1 on a call stays serial
```

`threads=None` resolves `set_num_threads`, `GIAO_NUM_THREADS`, `OMP_NUM_THREADS`,
then physical cores. A serial-only build warns when a parallel default is
unavailable; explicit parallel requests raise. GIAO regions finish before
NumPy/BLAS operations begin. BLAS threads are independent; for controlled
benchmarks set `OPENBLAS_NUM_THREADS=1` or `VECLIB_MAXIMUM_THREADS=1` before
starting Python.

Editable installs pick up Python edits, but **do not automatically rebuild C++**.
Run `python -m pip install -e .` again after C++ changes.
On macOS conda, source builds prefer that Python environment's OpenMP runtime
to avoid loading a second Homebrew copy. See the
[conda runtime guidance](docs/installation.md#macos-conda-source-installs-and-error-15). Avoid
`--no-build-isolation` unless you installed `requirements-dev.txt` first.

Version 0.9.0 has prepared wheel/release workflows; PyPI publication awaits
maintainer approval. See [installation and threading](docs/installation.md)
for wheel platforms, bundled runtime conflicts, and release setup.

## Quick example

```python
import giao_integrals as gi

mol = gi.Molecule.from_xyz("water.xyz", basis="cc-pVDZ")
S = mol.overlap()
H = mol.core_hamiltonian()
eri = mol.eri(max_bytes=512 * 1024**2)  # optional full-tensor output limit
chol = mol.cholesky_eri(threshold=1e-8)  # streamed; no full ERI allocation
```

XYZ coordinates default to ångström; every integral output uses atomic
units. Read the [user manual](docs/user_manual.md) for a full walkthrough,
or run [`examples/molecular_integrals.py`](examples/molecular_integrals.py)
directly.

## Parallel atom Cholesky timing

From the repository root, install into your active Python environment, then run
the [atom timing example](examples/atom_cholesky_timing.py):

```console
python -m pip install '.[molecule]'
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python examples/atom_cholesky_timing.py --threads 1 2 4
```

The default is Ne/cc-pVDZ (15 Cartesian AOs), with B=(0, 0, 0.1) and a
Cholesky threshold of 1e-8. The script times streamed integral generation and
factorization without forming the full ERI tensor. It reports wall time,
speedup relative to the first run, rank M, M/nao, vector memory, and the largest
residual diagonal. It also checks that factors and pivots are bitwise identical
across thread counts. Basis loading and result checks are outside the timing.

For the larger uncontracted Ne/aug-cc-pCVQZ case (152 Cartesian AOs):

```console
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python examples/atom_cholesky_timing.py --basis aug-cc-pCVQZ \
  --uncontracted --threads 1 4 8 12
```

Large cases can take minutes per run; small cases may show little speedup
because of parallel overhead. Explicit thread counts above one require an
OpenMP build. For a single atom, the London phases cancel in each pair density,
so these ERIs are independent of the supplied field. Use `--help` for atom,
basis, field, threshold, and uncontracting options.

## Conventions worth knowing up front

The public data types are `PrimitiveGaussian`, `Shell`, `Basis`,
`MagneticField`, and `Nucleus`. Each operator has primitive, shell, and
basis-level entry points where applicable. Results are C-contiguous
`complex128` arrays, even at zero field, because GIAOINT integrals stay
complex in general; shell blocks use `(ao_a, ao_b)` ordering, and basis
matrices follow input shell order. ERIs default to packed canonical
shell-quartet batches; request the full `(nao,)*4` tensor explicitly with a
byte limit.

**Coordinate conversion and PySCF comparisons:** GIAOINT uses
1 bohr = 0.529177210544 Å from
[CODATA 2022](https://physics.nist.gov/cuu/pdf/wall_2022.pdf).
PySCF 2.8.0 uses 0.52917721092 Å, a relative difference of about
7.1 × 10⁻¹⁰. Converting the same ångström geometry separately can therefore
affect integral comparisons at tight tolerances. Convert the geometry once
and supply identical coordinates to both codes with `unit="bohr"`.

## Development

```console
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e .
.venv/bin/pytest -q
```

Tests run on Python 3.11 through 3.14. With the project's conda
environment, run `conda activate pyscf` instead and drop `.venv/bin/` from
the remaining commands. Build and test the C++ core on its own with
`cmake --preset release`, `cmake --build --preset release`, and
`ctest --preset release` (these presets need Ninja).

## Documentation

- [Installation and thread configuration](docs/installation.md)
- [0.9 validation and release gates](docs/release_0_9_report.md)
- [Cholesky conventions and benchmarks](docs/cholesky_eri.md)
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
under the [BSD-3-Clause license](LICENSE). Bundled OpenMP runtimes carry
[separate upstream notices](python/giao_integrals/_licenses/README.txt).

The original project brief remains in
[`giao_integrals_codex_master_prompt.md`](giao_integrals_codex_master_prompt.md).
