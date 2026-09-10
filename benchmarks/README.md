# Benchmarks

Configure a release build with `-DGIAO_BUILD_BENCHMARKS=ON`. The executables
write one JSON object per line so results can be archived or compared without
parsing presentation-oriented text.

- `giao_integral_benchmark` covers overlap, canonical kinetic energy, nuclear
  attraction, and ERI shell blocks at zero and finite field for low and
  moderate angular momentum. `--quick` shortens calibration.
- `giao_eri_backend_benchmark` compares the production MD primitive with a
  benchmark-only direct Cartesian OS prototype and checks their values.
- `giao_eri_parallel_benchmark` measures complete canonical p-shell batches at
  1, 2, 4, and 8 threads. It is built only with `GIAO_ENABLE_OPENMP=ON`.
- `giao_overlap_benchmark` preserves the original Milestone 2 f--f overlap
  baseline.

Checksums are mandatory output: they keep the integral work observable and
also expose accidental numerical changes. Run performance comparisons on an
otherwise idle, frequency-stable machine; the checked-in reports are
informational and are not portable CI thresholds.

On AppleClang with the conda `llvm-openmp` package, an explicit configuration
is needed:

```console
cmake -S . -B build-openmp \
  -DGIAO_BUILD_BENCHMARKS=ON -DGIAO_ENABLE_OPENMP=ON \
  '-DOpenMP_CXX_FLAGS=-Xpreprocessor -fopenmp' \
  -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY="$CONDA_PREFIX/lib/libomp.dylib" \
  -DOpenMP_CXX_INCLUDE_DIR="$CONDA_PREFIX/include"
```

Other compilers use CMake's normal `FindOpenMP` discovery.

