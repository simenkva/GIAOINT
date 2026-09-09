# Public API

The API works at shell or shell-batch granularity. Recurrence tables, primitive
pair loops, and complex-center machinery remain private.

The primitive, shell, basis, magnetic-field, overlap, moment, gradient,
momentum, canonical-kinetic, and magnetic-kinetic portions documented below
are implemented through Milestone 3. Coulomb, ERI, and batch-consumer
interfaces remain planned for their stated later milestones.

## 1. C++ value types

The installed C++ declarations are in `giao_integrals/types.hpp` and
`giao_integrals/overlap.hpp`. The following condensed declarations describe
the value-type interface.

```cpp
namespace giao {

using Complex = std::complex<double>;

struct Vec3 {
    double x;
    double y;
    double z;
};

struct CartesianExponent {
    std::uint16_t x;
    std::uint16_t y;
    std::uint16_t z;
};

struct PrimitiveGaussian {
    double exponent;
    double coefficient;
    Vec3 center;
    CartesianExponent angular;
    bool normalized;
};

enum class ContractionNormalization {
    normalize,
    as_provided,
};

class Shell {
public:
    Shell(Vec3 center,
          std::uint16_t angular_momentum,
          std::vector<double> exponents,
          std::vector<double> coefficients,
          std::size_t contraction_count,
          ContractionNormalization normalization =
              ContractionNormalization::normalize);

    [[nodiscard]] Vec3 center() const noexcept;
    [[nodiscard]] std::uint16_t angular_momentum() const noexcept;
    [[nodiscard]] std::size_t primitive_count() const noexcept;
    [[nodiscard]] std::size_t contraction_count() const noexcept;
    [[nodiscard]] std::size_t cartesian_count() const noexcept;
    [[nodiscard]] std::size_t ao_count() const noexcept;
};

class Basis {
public:
    explicit Basis(std::vector<Shell> shells);
    [[nodiscard]] std::span<const Shell> shells() const noexcept;
    [[nodiscard]] std::span<const std::size_t> ao_offsets() const noexcept;
    [[nodiscard]] std::size_t ao_count() const noexcept;
};

struct MagneticField {
    Vec3 B;
    Vec3 gauge_origin;

    [[nodiscard]] Vec3 london_wave_vector(Vec3 center) const noexcept;
};

struct Nucleus {
    double charge;
    Vec3 center;
};

}  // namespace giao
```

Constructors own their data and validate it once. Hot kernels receive immutable
views and a reusable workspace. `MagneticField` remains a small value type;
the basis driver may cache one London wave vector per shell for a field.

Coefficients form a row-major matrix with shape
`(contraction_count, primitive_count)`. Milestones 1 and 2 may validate that
`contraction_count == 1`, but the data layout will not need replacement for
general contractions. `PrimitiveGaussian` supports reference tests and expert
primitive calls; production basis drivers use shells to avoid duplicating
centers, angular momentum, and exponents for each Cartesian component.

## 2. Shell-level C++ functions

```cpp
namespace giao {

class IntegralWorkspace;

[[nodiscard]] std::size_t shell_pair_size(
    const Shell& a, const Shell& b) noexcept;

[[nodiscard]] std::size_t shell_quartet_size(
    const Shell& a, const Shell& b,
    const Shell& c, const Shell& d) noexcept;

void compute_overlap(
    const Shell& a, const Shell& b,
    const MagneticField& field,
    std::span<Complex> output,
    IntegralWorkspace& workspace);

void compute_kinetic(
    const Shell& a, const Shell& b,
    const MagneticField& field,
    std::span<Complex> output,
    IntegralWorkspace& workspace);

void compute_nuclear_attraction(
    const Shell& a, const Shell& b,
    std::span<const Nucleus> nuclei,
    const MagneticField& field,
    std::span<Complex> output,
    IntegralWorkspace& workspace);

void compute_eri(
    const Shell& a, const Shell& b,
    const Shell& c, const Shell& d,
    const MagneticField& field,
    std::span<Complex> output,
    IntegralWorkspace& workspace);

}  // namespace giao
```

All outputs use `std::complex<double>` even at zero field. A caller supplies an
exactly sized contiguous buffer. Functions reject incorrect sizes before
entering primitive loops. The workspace owns temporary tables and grows only
when a requested shell class exceeds its capacity.

For shell AO indices \(i_a,i_b,i_c,i_d\), pair data are C-order with flat index

```text
ia * nao_b + ib
```

and quartets use

```text
((ia * nao_b + ib) * nao_c + ic) * nao_d + id
```

Within a shell, the contraction index is outer and the Cartesian component is
inner, using the ordering in the mathematical specification.

### 2.1 Property-operator extension

Milestone 3 provides typed shell-level functions over the same pair buffer:

```cpp
enum class Axis : std::uint8_t { x, y, z };

struct CartesianMoment {
    Vec3 origin;
    CartesianExponent powers;
};

void compute_moment(
    const Shell& a, const Shell& b,
    const CartesianMoment& moment,
    const MagneticField& field,
    std::span<Complex> output,
    IntegralWorkspace& workspace);

void compute_gradient(
    const Shell& a, const Shell& b, Axis component,
    const MagneticField& field,
    std::span<Complex> output,
    IntegralWorkspace& workspace);

void compute_momentum(
    const Shell& a, const Shell& b, Axis component,
    const MagneticField& field,
    std::span<Complex> output,
    IntegralWorkspace& workspace);

void compute_kinetic(
    const Shell& a, const Shell& b,
    const MagneticField& field,
    std::span<Complex> output,
    IntegralWorkspace& workspace);

void compute_magnetic_kinetic(
    const Shell& a, const Shell& b,
    const MagneticField& field,
    std::span<Complex> output,
    IntegralWorkspace& workspace);
```

Named descriptors keep origins and component choices explicit. Their kernels
reuse multiplication and derivative recurrences. The compute_kinetic function
is the canonical operator \(-\tfrac12\nabla^2\) over field-dependent GIAOs;
compute_magnetic_kinetic is the physical
\(\tfrac12(\mathbf p+\mathbf A_{\mathbf O})^2\) combination.

## 3. Basis drivers and ERI consumption

```cpp
namespace giao {

struct ShellQuartetIndex {
    std::uint32_t a;
    std::uint32_t b;
    std::uint32_t c;
    std::uint32_t d;
};

struct ShellQuartetBlockView {
    ShellQuartetIndex shells;
    std::array<std::size_t, 4> shape;
    std::span<const Complex> values;
};

using QuartetConsumer = void (*)(
    const ShellQuartetBlockView&, void* user_data);

void compute_overlap_matrix(
    const Basis& basis, const MagneticField& field,
    std::span<Complex> output);

void compute_moment_matrix(
    const Basis& basis, const CartesianMoment& moment,
    const MagneticField& field, std::span<Complex> output);

void compute_gradient_matrix(
    const Basis& basis, Axis component,
    const MagneticField& field, std::span<Complex> output);

void compute_momentum_matrix(
    const Basis& basis, Axis component,
    const MagneticField& field, std::span<Complex> output);

void compute_kinetic_matrix(
    const Basis& basis, const MagneticField& field,
    std::span<Complex> output);

void compute_magnetic_kinetic_matrix(
    const Basis& basis, const MagneticField& field,
    std::span<Complex> output);

void for_each_eri_shell_quartet(
    const Basis& basis,
    std::span<const ShellQuartetIndex> quartets,
    const MagneticField& field,
    QuartetConsumer consumer,
    void* user_data);

}  // namespace giao
```

The C callback occurs once per shell block, outside primitive and recurrence
loops. A templated convenience wrapper can preserve type safety without adding
virtual dispatch to kernels. Later direct J/K builders can consume blocks in
C++ without materializing \(N^4\) data.

The scheduler must state which finite-field symmetries it uses. Its default
conservative mode computes requested shell quartets exactly as indexed. A
symmetry-aware mode may use pair exchange and simultaneous within-pair reversal
with conjugation, never the real eightfold canonicalization.

## 4. Python data model

```python
from __future__ import annotations

import numpy as np
import numpy.typing as npt

Real3 = npt.NDArray[np.float64]

class Shell:
    def __init__(
        self,
        center: npt.ArrayLike,
        angular_momentum: int,
        exponents: npt.ArrayLike,
        coefficients: npt.ArrayLike,
        *,
        normalize: bool = True,
    ) -> None: ...

class Basis:
    def __init__(self, shells: list[Shell]) -> None: ...

class MagneticField:
    def __init__(
        self,
        B: npt.ArrayLike,
        gauge_origin: npt.ArrayLike = (0.0, 0.0, 0.0),
    ) -> None: ...

class Nucleus:
    def __init__(self, charge: float, center: npt.ArrayLike) -> None: ...
```

A one-dimensional coefficient array denotes one contraction. A two-dimensional
array has shape `(n_contraction, n_primitive)`. Constructors copy and validate
small metadata arrays; integral results are ordinary NumPy arrays.

## 5. Python integral functions

```python
def overlap(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    out: npt.NDArray[np.complex128] | None = None,
) -> npt.NDArray[np.complex128]: ...

def kinetic(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    out: npt.NDArray[np.complex128] | None = None,
) -> npt.NDArray[np.complex128]: ...

def nuclear_attraction(
    basis: Basis,
    nuclei: list[Nucleus],
    *,
    field: MagneticField | None = None,
    out: npt.NDArray[np.complex128] | None = None,
) -> npt.NDArray[np.complex128]: ...

def overlap_shell(
    a: Shell,
    b: Shell,
    *,
    field: MagneticField | None = None,
    out: npt.NDArray[np.complex128] | None = None,
) -> npt.NDArray[np.complex128]: ...

def eri_shell(
    a: Shell,
    b: Shell,
    c: Shell,
    d: Shell,
    *,
    field: MagneticField | None = None,
    out: npt.NDArray[np.complex128] | None = None,
) -> npt.NDArray[np.complex128]: ...

def moment(
    basis: Basis,
    powers: tuple[int, int, int],
    *,
    origin: npt.ArrayLike = (0.0, 0.0, 0.0),
    field: MagneticField | None = None,
    out: npt.NDArray[np.complex128] | None = None,
) -> npt.NDArray[np.complex128]: ...

def momentum(
    basis: Basis,
    component: str,
    *,
    field: MagneticField | None = None,
    out: npt.NDArray[np.complex128] | None = None,
) -> npt.NDArray[np.complex128]: ...

def gradient(
    basis: Basis,
    component: str,
    *,
    field: MagneticField | None = None,
    out: npt.NDArray[np.complex128] | None = None,
) -> npt.NDArray[np.complex128]: ...

def magnetic_kinetic(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    out: npt.NDArray[np.complex128] | None = None,
) -> npt.NDArray[np.complex128]: ...
```

`field=None` means an exactly zero field and zero gauge origin. Basis-level
one-electron functions return C-contiguous `(nao, nao)` arrays. Shell functions
return `(nao_a, nao_b)` or `(nao_a, nao_b, nao_c, nao_d)` arrays. `out` must
have exact shape, `complex128` dtype, C contiguity, and writeability; no hidden
copy repairs an incompatible output.

Expensive calls release the GIL. The binding makes one C++ call per matrix,
shell block, or block batch, never per AO integral.

## 6. Full and streamed ERIs in Python

```python
def eri(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    storage: str = "blocks",
    max_bytes: int | None = None,
): ...

def eri_batches(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    quartets: npt.ArrayLike | None = None,
    target_bytes: int = 64 * 1024**2,
): ...
```

`storage="full"` returns a C-contiguous complex tensor with shape
`(nao, nao, nao, nao)` only after checking the requested allocation against
`max_bytes`. The default `storage="blocks"` returns a batch iterator. Each
batch contains quartet indices, block offsets and one packed complex buffer,
which limits Python crossings and supports heterogeneous shell shapes.

The long-term production path is a C++ consumer interface for direct J/K and
post-Hartree-Fock contractions. Python callbacks are an expert convenience,
not the inner-loop architecture.

## 7. Errors and reproducibility

Invalid model data raise `std::invalid_argument` in C++ and `ValueError` in
Python. Size mismatches raise `std::length_error` and `ValueError`. Numerical
special-function failures raise a typed C++ exception carrying the order,
argument, algorithm region, and error estimate; Python maps it to
`NumericalError`.

Serial drivers accumulate primitives and nuclei in input order. Parallel
drivers document their reduction order and offer a deterministic mode when
bitwise repeatability matters. Public functions never read mutable global
field or screening state.
