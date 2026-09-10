# Milestone 6 benchmark report: macOS arm64

Environment: macOS 26.6.2 arm64, AppleClang 21.0.0, CMake 4.1.2, release
build. Results are single runs of the adaptive benchmark harness and should be
read as an engineering baseline, not as portable acceptance thresholds.

## Serial operator baseline

The complete JSON schema records case, operator, field, shell class, primitive
shape, threads, iterations, time, blocks/s, integrals/s, and a complex
checksum. Representative final rates are:

| Operator/case | Zero field | Finite field |
|---|---:|---:|
| overlap s--p, 2x2 primitives | 1.143M blocks/s | 1.110M blocks/s |
| overlap d--d, 3x3 primitives | 38.74k blocks/s | 38.06k blocks/s |
| kinetic d--d, 3x3 primitives | 1.889k blocks/s | 1.749k blocks/s |
| attraction d--d, 3x3 primitives | 1.698k blocks/s | 1.304k blocks/s |
| ERI s--s--s--s, 2x2x2x2 primitives | 209.1k blocks/s | 178.2k blocks/s |
| ERI p--p--p--p, 2x2x2x2 primitives | 706.6 blocks/s | 696.9 blocks/s |

Caching unchanged Boys sequences and pair Hermite products, plus generation
tags for auxiliary scratch, changed p--p--p--p throughput from 580.4 to 706.6
blocks/s at zero field (+21.7%) and from 536.4 to 696.9 blocks/s at finite
field (+29.9%). The s-shell ERI and one-electron cases stayed within about 1%
of baseline, as expected because their kernels were not targeted.

## MD versus OS prototype

The benchmark-only OS vertical recurrence agreed with MD to at most
`5.6e-17` in all six tested zero/finite-field ssss, pppp, and dpps primitive
cases. Relative OS time divided by MD time was:

| Primitive case | Zero field | Finite field |
|---|---:|---:|
| ssss | 1.07 | 1.79 |
| pppp | 0.85 | 0.89 |
| dpps | 0.81 | 0.98 |

This small prototype suggests direct OS/HGP deserves a contracted-shell
prototype, but it does not justify replacing the audited MD backend yet: the
result depends on angular class, the OS prototype lacks contraction and
screening integration, and no wider numerical-domain study has been done.

## OpenMP scaling

The finite-field workload contains 76 canonical p--p--p--p shell quartets,
6,156 Cartesian integrals per iteration, and 2x2x2x2 primitive contractions.
Callbacks remain serial and input ordered after parallel block computation.

| Threads | Time for 3 iterations | Speedup | Checksum |
|---:|---:|---:|---:|
| 1 | 0.3288 s | 1.00 | 457.405938662 + 0.865155452244i |
| 2 | 0.1783 s | 1.84 | same |
| 4 | 0.0904 s | 3.64 | same |
| 8 | 0.0508 s | 6.48 | same |

Each OpenMP worker owns its recurrence workspace. Serial and parallel blocks
are bitwise equal in the test workload, and callback order is deterministic.
The warmed scalar shell kernels perform no heap allocations; parallel batch
evaluation allocates output blocks before entering the OpenMP region and does
not allocate in primitive or recurrence inner loops after each thread's first
workspace growth.
