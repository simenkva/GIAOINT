# Installation, threading, and binary releases (0.9)

## Source and editable installs

Use Python 3.11–3.14 with a C++20 compiler, for example AppleClang 21 or GCC.
Normal `python -m pip install .` and `python -m pip install -e .` use build
isolation: pip installs scikit-build-core, pybind11, and any needed build
frontend tools. Runtime NumPy is installed as a dependency. Basis Set Exchange
is optional: `pip install '.[molecule]'` enables named basis loading.

On macOS install Xcode Command Line Tools (`xcode-select --install`). Install
OpenMP with `brew install libomp`. On macOS, AUTO first respects explicit
OpenMP library settings or `OpenMP_ROOT`. For a conda Python it next prefers
that interpreter's environment runtime and matching `omp.h`, before trying
`brew --prefix libomp`, `/opt/homebrew/opt/libomp` and `/usr/local/opt/libomp`.
The interpreter's `sys.prefix` takes precedence over a stale ambient
`CONDA_PREFIX`; native builds without Python use the active conda prefix.
On Linux GCC generally supplies OpenMP directly. Other compilers may need their
OpenMP runtime/development package.

CMake's `GIAO_ENABLE_OPENMP` is AUTO by default. AUTO enables discovered OpenMP
or prints a clear serial-fallback warning. ON fails if no runtime is found;
OFF deliberately builds serial. pip accepts these overrides:

```sh
python -m pip install . -Ccmake.define.GIAO_ENABLE_OPENMP=ON
python -m pip install . -Ccmake.define.GIAO_ENABLE_OPENMP=OFF
```

Source installs link the host runtime. They are not relocatable binary
artifacts; use a repaired wheel to distribute the extension to another host.
Editable installs reflect Python changes immediately but **do not rebuild C++
on import**. Re-run `pip install -e .` after native edits. This avoids depending
on a temporary build-isolation environment after it has been removed. If using
`--no-build-isolation`, first install the build requirements yourself, e.g.
`pip install -r requirements-dev.txt`.

## Runtime policy

```python
import giao_integrals as gi
print(gi.build_info())
gi.set_num_threads(4)
chol = mol.cholesky_eri()            # configured default
serial = mol.cholesky_eri(threads=1) # explicit serial
parallel = mol.eri(threads=2)
gi.set_num_threads(None)            # clear the process override
```

Python `eri`, `eri_batches`, `cholesky_eri`, the Molecule wrappers and
`integrals`, and native `_eri_full`/`_eri_batch` bindings accept `threads=None`.
Resolution occurs before native work, in this order:

1. Explicit `threads=n` on the call.
2. `gi.set_num_threads(n)` process override.
3. `GIAO_NUM_THREADS`.
4. `OMP_NUM_THREADS`.
5. Available physical cores (macOS sysctl; Linux topology within CPU affinity).

If topology cannot be queried, a conservative one-thread fallback is used.
Values must be positive integers up to 2147483647, not booleans or nested
OpenMP thread lists.
An invalid environment value raises a clear error. Explicit `threads=1` ignores
the default/environment. Explicit unsupported parallel calls or setters raise
`ValueError`. A serial build with a parallel default warns once and uses one
thread. Runtime OpenMP settings such as `OMP_THREAD_LIMIT` and `OMP_DYNAMIC`
can further constrain teams; use ordinary non-nested execution for benchmarks.
The C++ options struct keeps an explicit integer thread budget and serial
default; process/environment policy is implemented in the Python API.

`build_info()` reports version, compiler, OpenMP enabled status, loaded runtime
path, detected physical cores and effective default threads. The runtime path
uses the loaded symbol's image, so it distinguishes host libomp/libgomp from a
repaired wheel's bundled runtime.

Unlike 0.8, parallel execution is now the default when available. Every ERI
entry point is tested for **bitwise equality**, including screened results,
zero/finite fields and Cholesky pivots/vectors, at 1, 2 and maximum runtime
threads. Primitive contraction/reduction order is unchanged. Dynamic scheduling
assigns independent complete shell blocks to workers; results are consumed in
original order. Screening and integral conventions are unchanged.

OpenMP regions finish before Python calls NumPy/BLAS; there are no BLAS calls
inside GIAO OpenMP regions. GIAO's setter does not control BLAS. For repeatable
measurements set `VECLIB_MAXIMUM_THREADS=1` (Accelerate) or
`OPENBLAS_NUM_THREADS=1` before launching Python. Applications running multiple
processes should divide their CPU budget accordingly.

## Binary wheels and source archive

The release matrix prepares CPython 3.11–3.14 wheels for manylinux x86_64 and
aarch64, plus macOS arm64 and x86_64. macOS wheel runners and deployment baseline
are macOS 15; older macOS users must build from source with a suitable runtime.
Intel macOS is included using a native Intel runner, rather than cross-compiling
an untested slice. Windows and musllinux wheels are deliberately omitted for
0.9; native Windows source builds are not validated by this release matrix.
The package remains named `giao-integrals`, imported as `giao_integrals`.

Release wheel builds require OpenMP ON. auditwheel bundles libgomp on Linux;
delocate bundles libomp on macOS and rewrites load paths. License notices are
included under `giao_integrals/_licenses`. Each **repaired and installed** wheel
must report OpenMP, resolve a bundled runtime path, and pass ERI and Cholesky
serial/parallel smoke checks. An sdist is also built and installed in a clean
venv with isolated pip. The ordinary CI tests OFF and ON Python/C++ builds,
ASan/UBSan, AUTO source and editable installation, and missing-runtime fallback.
Local and completed cross-platform CI validation are recorded in the
[0.9 report](release_0_9_report.md).

A process that loads multiple independently bundled OpenMP runtimes (for
example another package's libomp alongside this wheel) may fail with
**OMP: Error #15**. Do not bypass the check with `KMP_DUPLICATE_LIB_OK`.
Use a compatible set of binary packages, or build GIAOINT from source against
the shared runtime used by the application. As a fallback, build with OpenMP
OFF. Inspect `build_info()` to see which runtime this extension actually uses.

### macOS conda source installs and Error #15

Conda NumPy/OpenBLAS can already load the environment's libomp. A GIAOINT
extension linked to Homebrew's separate copy then aborts when both initialize,
even if BLAS is limited to one thread. Thread-count settings do not select or
unload runtimes. AUTO source builds now prefer the target conda environment's
runtime and headers, and retain its library directory in the installed macOS
extension's rpath. If that runtime is present but its header is missing, AUTO
falls back to serial instead of selecting a conflicting Homebrew copy; ON
fails with instructions to install llvm-openmp.

From this checkout, inside the intended environment:

```sh
conda activate pyscf
python -m pip install --no-cache-dir --no-deps .
python -c "import giao_integrals as gi; print(gi.build_info())"
```

`--no-deps` leaves existing conda-managed NumPy/BSE packages untouched; install
those dependencies separately if missing. To pin a runtime explicitly, use:

```sh
python -m pip install --no-cache-dir --no-deps . \
  -Ccmake.define.GIAO_ENABLE_OPENMP=ON \
  -Ccmake.define.OpenMP_ROOT="$(python -c 'import sys; print(sys.prefix)')"
```

Restart Python after reinstalling, because an existing interpreter retains
already loaded dylibs. `openmp_runtime_path` should point into the same conda
environment, not Homebrew. Do not enable `KMP_DUPLICATE_LIB_OK` or replace
conda's library files/symlinks. Prebuilt wheels bundle a runtime and can still
conflict with a conda stack; use a source install to share its runtime.

## Publication approval

`.github/workflows/wheels.yml` builds artifacts on version tags and manual
runs. Publishing is a separate job using PyPI trusted publishing. It is disabled
unless repository variable `PYPI_APPROVED_TAG` exactly matches the current
version tag. Configure the GitHub `pypi` environment with required reviewers;
register PyPI's trusted publisher for owner `simenkva`, repository `GIAOINT`,
workflow `wheels.yml`, environment `pypi`, project `giao-integrals`.

Before approving `v0.9.0`, verify ownership/availability of that PyPI project
name, all target CI jobs, and the repaired artifacts. Set the approved-tag
variable only after explicit maintainer approval, then push or rerun the tag
workflow and approve the protected environment. New versions need a new exact
approved tag. No tag or PyPI publication is performed by source installation.

Workflow references: [cibuildwheel repair commands](https://cibuildwheel.pypa.io/en/stable/options/#repair-wheel-command),
[PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/),
[editable-install guidance](https://scikit-build-core.readthedocs.io/en/stable/configuration/editable.html).
