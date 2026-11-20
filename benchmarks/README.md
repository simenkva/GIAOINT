# Benchmarks

Configure a release build with `-DGIAO_BUILD_BENCHMARKS=ON`. The executables
write one JSON object per line so results can be archived or compared without
parsing presentation-oriented text. The native benchmark preset disables Python
bindings and requires Ninja:

```console
cmake --preset benchmarks
cmake --build --preset benchmarks
build/presets/benchmarks/giao_integral_benchmark --quick
```

- `giao_integral_benchmark` covers overlap, canonical kinetic energy, nuclear
  attraction, and ERI shell blocks at zero and finite field for low and
  moderate angular momentum. `--quick` shortens calibration.
- `giao_eri_backend_benchmark` compares the production MD primitive with a
  benchmark-only direct Cartesian OS prototype and checks their values.
- `giao_eri_parallel_benchmark` measures complete canonical p-shell batches at
  1, 2, 4, and 8 threads. It is built when OpenMP is enabled, whether by AUTO
  detection or explicit ON.
- `giao_overlap_benchmark` preserves the original Milestone 2 f--f overlap
  baseline.

Checksums are mandatory output: they keep the integral work observable and
also expose accidental numerical changes. Run performance comparisons on an
otherwise idle, frequency-stable machine; the checked-in reports are
informational and are not portable CI thresholds.

Compare two runs with:

```console
python benchmarks/compare_results.py baseline.jsonl current.jsonl
```

The default is deliberately non-gating. On a controlled, dedicated runner,
`--minimum-ratio 0.90` makes a throughput loss greater than 10% fail. The
comparator fails on missing or malformed cases; new cases are reported without
failing. Archive the raw JSONL alongside the compiler, CPU, operating system,
thread count, and build options rather than treating cross-machine ratios as
meaningful.

CMake defaults to AUTO detection, including Homebrew libomp on macOS. Use
ON to require OpenMP and OFF for a serial build. For a conda `llvm-openmp`
runtime that is not found automatically, an explicit configuration is:

```console
cmake -S . -B build-openmp \
  -DCMAKE_BUILD_TYPE=Release -DGIAO_BUILD_PYTHON=OFF \
  -DGIAO_BUILD_BENCHMARKS=ON -DGIAO_ENABLE_OPENMP=ON \
  '-DOpenMP_CXX_FLAGS=-Xpreprocessor -fopenmp' \
  -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY="$CONDA_PREFIX/lib/libomp.dylib" \
  -DOpenMP_CXX_INCLUDE_DIR="$CONDA_PREFIX/include"
```

Other compilers use CMake's normal `FindOpenMP` discovery.

## Milestone 9 ERI measurements

`giao_integral_benchmark --profile` runs ssss, pppp, and dddd blocks with
two primitives per shell at four distinct centers. The source fixes the
centers, exponents, coefficients, and finite field for reproducibility.
`--sample-zero` and `--sample-finite` keep the dddd case running long enough
for a stack sampler. For example, in a second terminal on macOS:

```console
sample <benchmark-pid> 5 1 -file profile.txt
```

Use the same benchmark source against the before and after libraries. Run
timings separately from compilation, tests, and profiling. The
`results/m9_macos_arm64.md` report links the archived JSONL and records the
compiler, machine, source baseline, and commands.

For basis-scale timings, run:

```console
python examples/performance_comparison.py
python examples/performance_comparison.py --basis cc-pVDZ --threads 4 --screening-threshold 1e-10
```

The second command requires an OpenMP build. The example includes Schwarz
bound construction in each timed `eri()` call. Screening may cost more than
it saves for small or dense bases. Choose a threshold by checking the error
in your target observable; it bounds omitted AO integrals, not an accumulated
energy error. Python `eri_batches()` shares bound setup across its chunks;
C++ callers can reuse `EriSchwarzBounds` for the same basis and field.
Benchmark thread counts for your workload and avoid oversubscribing an outer
parallel calculation. The performance-comparison CLI explicitly defaults to
one thread and no screening; the library default since 0.9 is `threads=None`
with screening disabled. See [thread configuration](../docs/installation.md).

## Streamed Cholesky benchmarks

`cholesky_eri.py` measures Be aug-cc-pCVQZ rank, time and peak RSS. To
reproduce the serial budget used in [the Be report](../docs/cholesky_eri.md):

```console
GIAO_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python benchmarks/cholesky_eri.py 1e-8
```

`cholesky_threads.py` measures the BSE-uncontracted Ne case at explicit thread
counts and emits vector/pivot SHA256 hashes. Its optional `--profile` measures
update lines and adds profiling overhead:

```console
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmarks/cholesky_threads.py --threads 1 4 8 12
```

The [0.9 report](../docs/release_0_9_report.md) separates profiled/unprofiled
measurements and records the overlapping validation jobs. Do not treat those
workstation samples as isolated or portable performance guarantees.
