# 0.9.0 installation and threading validation

Prepared on 20 November 2026, macOS 26.6.2 arm64, AppleClang
21.0.0.21000401, CPython 3.13.3. Profiles use NumPy 2.5.3/Accelerate;
fresh installation environments resolved NumPy 2.5.5. Publication has **not** been performed.

## Installs and diagnostics

Verified locally in separate fresh virtual environments:

| Artifact/path | Result |
| --- | --- |
| Plain `pip install .` | Builds with isolation, finds Homebrew libomp automatically |
| Plain `pip install -e .` | Same; Python edits are live, native changes require reinstall |
| sdist, isolated pip install | Builds and passes installed ERI/Cholesky smoke test |
| delocate-repaired arm64 cp313 wheel | Installs and passes smoke; runtime is bundled in `.dylibs` |
| AUTO with discovery disabled | Installs serial; warns on an unavailable parallel default |
| ON with discovery disabled | Fails configuration with a clear missing-runtime error |
| Explicit threads=2 on serial build | Raises ValueError |

The unavailable-runtime case simulates missing discovery rather than removing
Homebrew from this Mac. The installed repaired wheel was built on macOS 26
and tagged accordingly; release CI separately targets macOS 15.

Actual source-install `gi.build_info()` outside the restricted tool sandbox:

```python
{'version': '0.9.0',
 'compiler': 'AppleClang 21.0.0.21000401',
 'openmp_enabled': True,
 'openmp_runtime_path': '/opt/homebrew/Cellar/libomp/23.1.1/lib/libomp.dylib',
 'default_threads': 12,
 'physical_cores': 12}
```

The repaired wheel instead reports its venv's
`site-packages/giao_integrals/.dylibs/libomp.dylib`. A simulated serial install
reports OpenMP False, runtime None and effective default 1, with a RuntimeWarning.
The tool sandbox denies sysctl, so benchmark metadata collected there reports
a conservative physical/default count of 1; all measured budgets below were
passed explicitly and are unaffected by that detection restriction.

## Tests and numerical equality

| Build | Full Python suite | Full C++ suite |
| --- | --- | --- |
| Serial OFF, Release | 458 passed | Passed |
| OpenMP AUTO/ON, Release | 458 passed | Passed |
| OpenMP with ASan/UBSan, -O1 + debug symbols | 458 passed | Passed |

`test_threading.py` covers each threaded ERI entry point, including Molecule
integrals, streamed batches, native bindings and Cholesky factors/pivots.
Results are compared with `assert_array_equal` at 1, 2 and the runtime maximum
(12 here), with and without screening and with zero/finite magnetic field.
The full suite also retains the reconstruction, gauge, normalization and memory
checks. A distant center ensures screened equality tests actually discard
quartets, while a close pair retains nonzero complex finite-field integrals.
The GIL test completion timeout was raised from 5 to 60 seconds to accommodate
sanitizer overhead in any build; its separate GIL-release assertion is unchanged.
Maximum-thread testing is explicit even inside the restricted sandbox.

## Completed remote CI and Linux sanitizer correction

[CI run 44716434345](https://github.com/simenkva/GIAOINT/actions/runs/44716434345)
passed all jobs on the sanitizer-fix branch at `710d425`, including the
Linux/macOS Python 3.11–3.14 OFF/ON matrix, source installs, native C++ tests,
static analysis, and Linux ASan/UBSan Python and C++ checks.

The original Linux sanitizer run exited during an expected C++ exception.
GCC's ASan runtime needs libstdc++ present at startup to resolve its exception
interceptor. CI now pins CXX=g++, preloads that driver's libasan followed by
libstdc++, exercises the exception path before the full suite, and leaves
native diagnostics uncaptured. ASan and UBSan halt on reported errors; leak
detection remains disabled for the Python host as before. This is a sanitizer
bootstrap correction; no numerical kernel or integral convention changed.

## macOS conda runtime compatibility correction

A source install in the local `pyscf` conda environment reproduced Error #15:
NumPy 2.2.6/OpenMP OpenBLAS loaded conda's llvm-openmp 20.1.3 while GIAOINT
loaded Homebrew libomp 23.1.1. Limiting BLAS threads does not prevent both
libraries from loading. The source-build configuration now prefers the target
Python's conda runtime and headers, with an installed rpath for @rpath libraries.

A fresh isolated pip source rebuild (with dependencies unchanged) now reports
`/opt/anaconda3/envs/pyscf/lib/libomp.dylib`. Both import-order regression checks
load exactly one runtime and produce identical serial/parallel factors and
pivots. The user's Ne/cc-pVDZ example succeeds at 1/2/4 threads, and the full
conda Python suite passes 459 tests; the conda-linked C++ suite also passes.
The ordinary pip/Accelerate build retains Homebrew selection and passes 52
focused Cholesky/thread/example tests, and a repaired wheel still passes its
installed smoke test with a bundled runtime. Missing conda headers, strict ON,
explicit root overrides, and OFF configuration paths have been checked.
A dedicated conda/OpenMP OpenBLAS job has been added to CI; see
[conda setup and recovery](installation.md#macos-conda-source-installs-and-error-15).

## Ne Cholesky profile and timings

Basis Set Exchange aug-cc-pCVQZ for Ne, with uncontract_general,
uncontract_segmented and uncontract_spdf all True: 152 Cartesian AOs.
B=(0,0,0.1), tau=1e-8, M=1052. Accelerate BLAS is limited to one thread.
No full AO ERI or pair matrix is constructed.

A cProfile run, with timers around the existing update blocks, gives:

| Threads | Before/static seconds | After/dynamic seconds | Vector update before/after | Residual update before/after |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 664.72 | 636.94 | 4.29 / 4.40 | 0.087 / 0.075 |
| 12 | 154.27 | 106.86 | 3.93 / 4.07 | 0.072 / 0.073 |

At 12 threads, `_eri_batch` accounts for 141.75/154.27 seconds before the
change (91.9%) and 94.10/106.86 afterward. The vector and residual updates
combined account for only 2.6% before the change. Moving those updates to C++
would at best recover a few percent and risks changing the BLAS reduction
path, so the conditional native-update rewrite is **not warranted by this
profile**. Instead, dynamic assignment balances heterogeneous shell quartets;
each block still has its original primitive reduction order and output slot.
The measured profiled 12-thread improvement is about 31% in elapsed time.

Unprofiled runs of the new reproducible benchmark:

| Threads | Seconds |
| ---: | ---: |
| 4 | 181.09 |
| 8 | 112.43 |
| 12 | 97.36 |

The 4/8/12-thread vector SHA256 is
`eb74eb520061bda5ba56268ace9a90f3a4068931733aac5bb7773fb2a6139527`,
and pivot SHA256 is
`c63ca69401b67b0a23490d3bac7c727086c747b69a39a7d10c4436b58953d853`.
The measured residual maximum is 7.4813e-9. Measurements overlapped other
validation/build jobs on this workstation and are not isolated performance
claims. In particular the serial wall times differ from the user's supplied
424 s baseline; those supplied VICE figures must not be mixed into a controlled
speedup comparison. Re-run on a quiet machine using:

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/cholesky_threads.py --threads 1 4 8 12 --profile
```

Raw [benchmark records](../benchmarks/results/threading_0_9_macos_arm64.json)
include the measured hashes and build metadata.

## Distribution decisions and remaining release gates

The prepared workflow builds CPython 3.11–3.14 manylinux_2_28 x86_64/aarch64,
macOS arm64 and Intel x86_64 (macOS 15 baseline). Windows/musllinux wheels are
omitted for 0.9. Runtime repair, license notices, installed smoke tests and
sdist installation are required CI gates. No running Docker daemon is available
on this Mac, so those platforms were validated remotely rather than locally.
[Wheel run 44715246114](https://github.com/simenkva/GIAOINT/actions/runs/44715246114)
passed the sdist job and all four wheel jobs at commit `45ab1a2`, covering
CPython 3.11–3.14 on the configured Linux and macOS architectures. The publish
job was skipped. This verifies runtime repair and installed smoke tests without
publishing or creating a release tag.

The distribution name stays `giao-integrals`; confirm PyPI ownership before
release. Configure the trusted publisher and protected pypi environment as in
[installation](installation.md). Publication requires explicit maintainer
approval of the exact `v0.9.0` tag and successful CI. Neither a tag nor a PyPI
upload has been performed during this task.
