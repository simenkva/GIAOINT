# Milestone 9 benchmark report: macOS arm64

Measured 2026-09-11 on an Apple M2 Pro MacBook Pro (12 cores: 8 performance,
4 efficiency; 32 GB), macOS 26.6.2 (25G83), AppleClang 21.0.0
(`clang-2100.1.1.101`), CMake 4.1.2, Ninja, Release (`-O3 -DNDEBUG`).
The serial measurements use one thread with screening disabled. The baseline
is commit `2d5b3b13c85fb1258788d698bff551bebbde4b93`; the optimized source is
the unreleased Milestone 9 implementation. Both report library version 0.8.0.
See [m9_environment.json](m9_environment.json) for source hashes and options.

The final before/after runs use the same harness, machine, and build flags.
They ran sequentially, separate from compilation, tests, and profiling.
CPU frequency and core placement were not pinned. Treat the results as local
engineering measurements, not portable CI thresholds.

## Shell throughput

The `--profile` matrix fixes four distinct non-aligned centers in
`benchmarks/integral_benchmark.cpp`, with two primitives per shell. It makes
the planning experiment reproducible; the planning notes did not preserve
the exact original geometry. The final speedups below use the newly measured
baseline at this checked-in geometry.

| Quartet | Field | Before blocks/s | After blocks/s | Speedup |
|---|---|---:|---:|---:|
| s-s-s-s | zero | 165,744.54 | 164,034.74 | 0.99x |
| p-p-p-p | zero | 713.43 | 3,314.60 | 4.65x |
| d-d-d-d | zero | 5.47 | 89.37 | 16.34x |
| s-s-s-s | finite | 127,260.61 | 128,631.37 | 1.01x |
| p-p-p-p | finite | 703.80 | 2,133.18 | 3.03x |
| d-d-d-d | finite | 5.48 | 23.24 | 4.24x |

Raw records: [before](m9_profile_before.jsonl), [after](m9_profile_after.jsonl).
The full operator harness also records
[before](m9_before.jsonl) and [after](m9_after.jsonl). Its pppp speedups are
4.70x at zero field and 3.04x at finite field. The ssss and unchanged
one-electron cases vary by about 3% or less. The dddd zero-field result meets
the order-of-magnitude target; the finite-field result is a smaller 4.24x.

Intermediate [iterative-only](m9_profile_iterative.jsonl) and
[real-path-before-pair-hoisting](m9_profile_zero_path.jsonl) runs document the
staged implementation. They reached about 22.7 dddd blocks/s at finite field
and 82.5 at zero field, respectively. These earlier single runs are diagnostic;
use the final matched runs for speedup claims.

## Basis-scale example

The updated `examples/performance_comparison.py` ran against both extensions,
using the same conda Python 3.13.3, pybind11 2.13.6, PySCF 2.8.0, BSE data,
and compiler. The example requests one thread for both engines. This PySCF
installation has no OpenMP support and reports that its thread setter has no
effect. Each ERI timing is the best of three calls after one warm-up. Full
results: [before](m9_water_before.txt), [after](m9_water_after.txt).

| Water basis | Cartesian AOs | Before ERI ms | After ERI ms | Speedup | PySCF after ms |
|---|---:|---:|---:|---:|---:|
| STO-3G | 7 | 28.132 | 14.178 | 1.98x | 2.241 |
| 6-31G | 13 | 106.509 | 66.120 | 1.61x | 9.788 |
| cc-pVDZ | 25 | 691.658 | 238.796 | 2.90x | 24.715 |

The basis-scale gains are smaller than those for a pure dddd workload.
Libcint remains faster on these zero-field calculations.

## Profiling and remaining costs

macOS `sample` collected five seconds per workload at 1 ms intervals while
`--sample-zero` or `--sample-finite` repeated the dddd quartet. The archived
[m9_profiles.json](m9_profiles.json) retains the total sample count and the
sampler's collapsed top-of-stack counts (symbols with fewer than five samples
are omitted).

| Workload | Total samples | Main top-of-stack counts |
|---|---:|---|
| Before, zero | 4,165 | recursive auxiliary 3,905 (93.8%); enclosing kernel 168 (4.0%) |
| After, zero | 4,174 | kernel 3,003 (71.9%); Hermite coefficients 282 (6.8%) |
| After, finite | 4,172 | kernel 3,878 (93.0%); Hermite coefficients 66 (1.6%) |

The optimized kernel has no recursive auxiliary frames. The general kernel
still dominates finite-field time. Its symbol includes recurrence filling,
contraction, and cache bookkeeping; stack sampling alone does not separate
those inlined costs. At zero field, Hermite construction and primitive setup
now account for a larger fraction of the reduced runtime.

## OpenMP and screening

The [OpenMP run](m9_parallel_after.jsonl) uses the existing 76-quartet finite-
field p-shell workload, three iterations, with conda's `libomp.dylib` and
`-Xpreprocessor -fopenmp`. Times at 1/2/4/8 threads were
0.11325/0.05688/0.03269/0.01931 seconds: 1.00/1.99/3.46/5.86x scaling.
Checksums match at all four thread counts. These short runs do not establish
an optimal thread count for other bases.

The defaults stay `screening_threshold=0.0`, `threads=1`. The performance
example accepts explicit screening and threading options and includes bound
setup in each timed call. Its screened STO-3G smoke run also completed. See
[benchmark guidance](../README.md) for choosing these options.

## Correctness

- 376 Python tests passed, including the unchanged high-precision OS oracle,
  derivative, symmetry, gauge, screening, and hardening tests.
- 192 frozen recursive values cover 96 randomized quartets through g functions.
  The finite-field outputs match bit for bit on this machine; the largest
  zero-field absolute difference is `1.734723475976807e-18`.
- 128 randomized C++ real-versus-general checks cover zero-field dispatch,
  signed zero, arbitrary gauge origins, tiny/subnormal fields, and workspace
  reuse. Warmed real and complex shell calls allocate no heap memory.
- Release, ASan/UBSan, and OpenMP C++ tests passed; Ruff and Black passed.
- The optimized extension passed the standalone PySCF/libcint comparison with
  maximum ERI error `1.704e-14`, using the existing tolerances.
- The [benchmark-only OS cross-check](m9_backend_after.jsonl) also passed all
  six cases. No production backend, Boys domain, or public convention changed.

## Reproduction

```console
cmake --preset benchmarks
cmake --build --preset benchmarks
build/presets/benchmarks/giao_integral_benchmark > current.jsonl
build/presets/benchmarks/giao_integral_benchmark --profile > current-profile.jsonl
python benchmarks/compare_results.py baseline.jsonl current.jsonl
python examples/performance_comparison.py
```

Build the same harness against the baseline library for the `before` files.
The profile mode was added in this milestone, so use its source with the
baseline's headers and library. For sampling, start `--sample-zero` or
`--sample-finite` and run `sample <pid> 5 1 -file profile.txt` separately.
