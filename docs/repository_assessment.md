# Repository assessment and proposed layout

This file preserves the Milestone 0 starting assessment and design proposal.
The implemented layout has since been kept deliberately flatter; see
`README.md`, `STATUS.md`, and the installed headers for the current project.

## Assessment on 2026-09-08

The repository contained one file when Milestone 0 began:

- `giao_integrals_codex_master_prompt.md`, 1,075 lines, containing the project
  brief.

No Git repository, source code, package metadata, tests, continuous-integration
configuration, build files, dependency lock files, license, or coding
conventions existed. Milestone 0 therefore has no compatibility constraints
beyond the project brief. Before Milestone 1, the maintainer should initialize
Git and choose a project license. The proposed code will remain original and
will use the cited literature only to verify equations and algorithm choices.

## Proposed final layout

```text
giao-integrals/
├── CMakeLists.txt
├── CMakePresets.json
├── pyproject.toml
├── README.md
├── STATUS.md
├── LICENSE
├── cmake/
│   └── CompilerWarnings.cmake
├── docs/
│   ├── repository_assessment.md
│   ├── mathematical_specification.md
│   ├── algorithms.md
│   ├── api.md
│   ├── testing.md
│   ├── implementation_plan.md
│   └── references.md
├── include/giao_integrals/
│   ├── types.hpp
│   ├── shell.hpp
│   ├── basis.hpp
│   ├── field.hpp
│   ├── operators.hpp
│   ├── engine.hpp
│   └── version.hpp
├── src/
│   ├── math/
│   │   ├── normalization.cpp
│   │   ├── cartesian.cpp
│   │   ├── gaussian_product.cpp
│   │   ├── hermite.cpp
│   │   └── boys.cpp
│   ├── one_electron/
│   │   ├── overlap.cpp
│   │   ├── kinetic.cpp
│   │   ├── moments.cpp
│   │   └── nuclear_attraction.cpp
│   ├── two_electron/
│   │   ├── eri.cpp
│   │   ├── screening.cpp
│   │   └── quartet_driver.cpp
│   ├── drivers/
│   │   ├── basis_one_electron.cpp
│   │   └── eri_batches.cpp
│   └── python/bindings.cpp
├── python/giao_integrals/
│   ├── __init__.py
│   ├── _api.py
│   └── py.typed
├── reference/python/giao_reference/
│   ├── __init__.py
│   ├── gaussian.py
│   ├── overlap.py
│   ├── boys.py
│   ├── one_electron.py
│   └── eri.py
├── tests/
│   ├── cpp/
│   ├── python/
│   ├── reference/
│   ├── data/
│   └── external/
├── benchmarks/
│   ├── cpp/
│   ├── python/
│   └── baselines/
└── examples/
    ├── water_overlap.py
    └── shell_quartet_stream.py
```

The public headers contain data types and shell/basis drivers. Recurrence
tables and operator-specific kernels stay under `src/`. The Python reference
package does not import the C++ extension, which prevents it from becoming a
circular oracle. Full ERI construction remains an optional small-system helper;
the main route emits shell-quartet batches to a consumer.

## Dependency proposal

The numerical core depends only on C++20. Python packaging uses CMake,
pybind11, and `scikit-build-core`; the runtime Python dependency is NumPy.
Development dependencies are Catch2 3, pytest, Hypothesis, mpmath, and Ruff.
Optional zero-field validation may use PySCF/libcint, but the library will not
link against either. OpenMP remains an optional build feature introduced after
the serial kernels pass their numerical gates.

Catch2 and pybind11 should come from installed packages in release builds.
Developer presets may use CMake `FetchContent` with pinned versions, but offline
builds must have a documented package-manager path. No external special-function
library enters the production ABI until the complex Boys prototype is compared
against an independent implementation and its license is recorded.

## Build and quality targets

The initial CMake targets will be:

- `giao_integrals_core`, a position-independent C++ library;
- `_giao_integrals`, the pybind11 extension;
- `giao_integrals_cpp_tests`, enabled by `GIAO_BUILD_TESTS`;
- benchmark executables, enabled by `GIAO_BUILD_BENCHMARKS`;
- optional OpenMP through `GIAO_ENABLE_OPENMP`.

CI should cover Linux and macOS, GCC and Clang, Python 3.11 through 3.14, and
Debug/Release builds. Separate jobs will run AddressSanitizer plus
UndefinedBehaviorSanitizer, Python tests against an installed wheel, and a
non-failing benchmark report. Milestone 6 established stable benchmark
executables, a JSON-lines schema, and checked-in reference reports. The reports
remain non-gating until repeated dedicated-runner measurements justify
noise-aware thresholds.
