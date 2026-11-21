# Public API

For XYZ geometries and named, Gaussian94, NWChem, or BSE JSON bases, start with
[`Molecule`](molecule_api.md). It provides matrix and ERI methods and exposes
its `basis`, `nuclei`, and `field` for the lower-level functions below.
The molecule reader accepts ångström input by default and converts to bohr;
the lower-level API below takes atomic units throughout.

The API works at shell or shell-batch granularity. Recurrence tables, primitive
pair loops, and complex-center machinery remain private.

This overview describes the published 0.9.0 API and subsequent derivative additions,
including streamed Cholesky and configured Python thread defaults. Signatures below are condensed;
installed headers and Python docstrings give individual overload details.
Version 0.9.0 is published; derivative additions are prepared for 0.10.0 and
require a source checkout until that release is approved and published.
See [release validation](release_0_9_report.md).

All positions, Gaussian exponents, moment origins, magnetic fields, energies,
and derivatives use atomic units. Shell coefficients multiply normalized
primitives; disabling shell normalization disables only the contraction factor.
Cartesian components are ordered by descending x power and then descending y
power; contraction index is outermost. Model objects own copies of their input
data. Results manage their array-storage lifetimes; `EriBatch.block()` returns
a view into its packed buffer. With `out=`, that exact array is filled and
returned.

| Quantity | Unit |
|---|---|
| centers and moment origins | bohr \(a_0\) |
| Gaussian exponents | \(a_0^{-2}\) |
| magnetic field | atomic magnetic-field unit |
| overlap | dimensionless |
| moment with powers `(px, py, pz)` | \(a_0^{px+py+pz}\) |
| kinetic, magnetic kinetic, attraction, and ERI | hartree \(E_h\) |
| center or potential-center derivative | value unit per \(a_0\) |
| magnetic derivative | value unit per atomic magnetic-field unit |

## 1. C++ value types

The installed C++ declarations are in `giao_integrals/types.hpp`,
`giao_integrals/overlap.hpp`, `giao_integrals/boys.hpp`,
`giao_integrals/nuclear.hpp`, `giao_integrals/property.hpp`,
`giao_integrals/eri.hpp`, and
`giao_integrals/derivatives.hpp`. The following
condensed declarations describe the value-type interface.

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
          std::size_t contraction_count = 1,
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
`(contraction_count, primitive_count)`. Production shells support multiple
contraction rows; only the pure-Python reference shell is segmented. `PrimitiveGaussian` supports reference tests and expert
primitive calls; production basis drivers use shells to avoid duplicating
centers, angular momentum, and exponents for each Cartesian component.

## 2. Shell-level C++ functions

```cpp
namespace giao {

class IntegralWorkspace;
class NuclearAttractionWorkspace;
class EriWorkspace;

[[nodiscard]] std::size_t shell_pair_size(
    const Shell& a, const Shell& b) noexcept;

[[nodiscard]] std::size_t shell_quartet_size(
    const Shell& a, const Shell& b,
    const Shell& c, const Shell& d);

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
    NuclearAttractionWorkspace& workspace);

void compute_eri(
    const Shell& a, const Shell& b,
    const Shell& c, const Shell& d,
    const MagneticField& field,
    std::span<Complex> output,
    EriWorkspace& workspace);

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

### 2.2 Complex Boys API

The installed `giao_integrals/boys.hpp` exposes sequence evaluation without
requiring the nuclear-attraction driver:

```cpp
enum class BoysScaling { unscaled, exp_z };
enum class BoysRegion {
    power_series,
    adaptive_quadrature,
    scaled_quadrature,
    positive_asymptotic,
};

struct BoysDiagnostics {
    BoysRegion region;
    BoysScaling scaling;
    double estimated_absolute_error;
    std::size_t quadrature_segments;
};

BoysDiagnostics compute_boys(
    Complex argument, std::span<Complex> values,
    BoysScaling scaling = BoysScaling::unscaled);
```

The span length selects orders zero through `values.size() - 1`. An empty span,
non-finite argument, or order above 32 is invalid. A finite but unsupported
argument or a failed convergence check raises `BoysNumericalError`.

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

struct EriEvaluationOptions {
    double screening_threshold{};
    std::size_t thread_count{1U};
};

struct EriStatistics {
    std::size_t requested_quartets;
    std::size_t computed_quartets;
    std::size_t screened_quartets;
};

class EriSchwarzBounds {
public:
    EriSchwarzBounds(const Basis&, const MagneticField& = {});
    double operator()(std::size_t a, std::size_t b) const;
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

void compute_nuclear_attraction_matrix(
    const Basis& basis, std::span<const Nucleus> nuclei,
    const MagneticField& field, std::span<Complex> output);

void for_each_eri_shell_quartet(
    const Basis& basis,
    std::span<const ShellQuartetIndex> quartets,
    const MagneticField& field,
    QuartetConsumer consumer,
    void* user_data);

EriStatistics evaluate_eri_shell_quartets(
    const Basis& basis,
    std::span<const ShellQuartetIndex> quartets,
    const MagneticField& field,
    const EriEvaluationOptions& options,
    const EriSchwarzBounds* bounds,
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

Milestone 5 exposes `canonicalize_shell_quartet` for that exact four-member
orbit. `compute_eri_tensor` and the default Python iterator schedule one
canonical representative and expand only those identities. Explicit quartet
lists passed to the C++ consumer or Python iterator are evaluated exactly as
requested.

The original `for_each_eri_shell_quartet` is the serial, unscreened
compatibility entry point. `evaluate_eri_shell_quartets` adds explicit
screening and thread options and returns counts. Positive screening requires
bounds built for the same basis and field. `thread_count > 1` requires a build
with OpenMP enabled (AUTO discovery or explicit ON); `openmp_enabled()` and
`openmp_max_threads()` report runtime capability.

## 4. Python data model

```python
from __future__ import annotations

from collections.abc import Iterable, Sequence

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
        B: npt.ArrayLike = (0.0, 0.0, 0.0),
        gauge_origin: npt.ArrayLike = (0.0, 0.0, 0.0),
    ) -> None: ...

class Nucleus:
    def __init__(self, charge: float, center: npt.ArrayLike) -> None: ...
```

`Nucleus` requires a finite positive charge. The attraction functions include
both that charge and the electronic minus sign.

The special-function API is:

```python
def boys(
    argument: complex, maximum_order: int, *, scaled: bool = False
) -> npt.NDArray[np.complex128]: ...

def boys_with_diagnostics(
    argument: complex, maximum_order: int, *, scaled: bool = False
) -> tuple[npt.NDArray[np.complex128], dict[str, object]]: ...
```

`scaled=True` returns \(e^zF_n(z)\). Diagnostics report the selected
`BoysRegion`, an absolute error estimate, and the number of composite
quadrature segments. Orders 0--32 are supported; unsupported complex sectors
raise `BoysNumericalError`.

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

def nuclear_attraction_shell(
    a: Shell,
    b: Shell,
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
    target_bytes: int = 64 * 1024**2,
    screening_threshold: float = 0.0,
    threads: int | None = None,
): ...

def eri_batches(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    quartets: Iterable[Sequence[int]] | None = None,
    target_bytes: int = 64 * 1024**2,
    screening_threshold: float = 0.0,
    threads: int | None = None,
): ...

def eri_schwarz_bounds(
    basis: Basis, *, field: MagneticField | None = None
) -> npt.NDArray[np.float64]: ...
```

`storage="full"` returns a C-contiguous complex tensor with shape
`(nao, nao, nao, nao)` only after checking the requested allocation against
`max_bytes`, which must be an explicit non-negative integer for this low-level
full-output call. By contrast, `Molecule.eri(max_bytes=None)` allows uncapped
full output. The default low-level `storage="blocks"` returns a batch iterator. Each
batch contains quartet indices, block offsets and one packed complex buffer,
which limits Python crossings and supports heterogeneous shell shapes.
`EriBatch.block(i)` returns a shaped view of its packed buffer. With no
explicit quartet list, the iterator emits canonical representatives under
pair exchange and conjugate double reversal; it never assumes a one-pair swap.
Every requested quartet is evaluated: Milestone 5 applies no screening.
Milestone 6 retains that behavior at the default `screening_threshold=0.0`.
Positive thresholds use cached complex Schwarz factors. Batches expose
`requested_count` and `screened_count`; their packed quartet arrays contain
only computed blocks. `threads` applies within each batch. Its default is
`None`; `set_num_threads(n)` overrides environment/core defaults, and
`build_info()` reports the effective budget and loaded OpenMP runtime.
See [thread configuration](installation.md).

The long-term production path is a C++ consumer interface for direct J/K and
post-Hartree-Fock contractions. Python callbacks are an expert convenience,
not the inner-loop architecture.

### 6.1 Streamed Cholesky

```python
def cholesky_eri(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    threshold: float = 1e-8,
    screening_threshold: float = 0.0,
    threads: int | None = None,
    max_bytes: int | None = None,
) -> CholeskyERI: ...

class CholeskyERI:
    vectors: npt.NDArray[np.complex128]  # (M, nao, nao)
    pivots: npt.NDArray[np.int64]        # (M, 2), ordered AO index pairs
    threshold: float
    max_residual_diagonal: float

    def reconstruct(
        self, slices: tuple[slice, slice, slice, slice] | None = None
    ) -> npt.NDArray[np.complex128]: ...
```

`Molecule.cholesky_eri()` exposes the same keyword arguments except `field`,
which comes from the molecule. The exact factor convention is
`ERI[mu,nu,lam,sig] ≈ sum_P L[P,mu,nu] * conj(L[P,sig,lam])`.
The Hermitian matrix being factored is
`W[(mu,nu),(lam,sig)] = ERI[nu,mu,lam,sig]`, using all ordered AO pairs.
Neither the full AO tensor nor a full pair matrix is formed.

The largest residual diagonal bounds every ERI error in exact arithmetic.
Roundoff imposes a stopping floor of `64*eps*max(initial diagonal)`; the result
reports the achieved residual, including when `threshold=0` requests numerical
exhaustion. `max_bytes=None` means no cap; otherwise it bounds vectors plus
pivot indices, not process memory. `reconstruct()` without slices allocates a
full tensor; four slices limit output to that block. See
[Cholesky details](cholesky_eri.md) and the
[mathematical convention](mathematical_specification.md#82-ordered-pair-cholesky).

### 6.2 Build and thread diagnostics

```python
def build_info() -> dict: ...
def set_num_threads(n: int | None) -> None: ...
```

`build_info()` returns `version`, `compiler`, `openmp_enabled`,
`openmp_runtime_path` (None in serial builds), `default_threads`, and
`physical_cores`. `set_num_threads(None)` clears the process override.
Explicit `threads=n` takes precedence over that override, then defaults use
GIAO_NUM_THREADS, OMP_NUM_THREADS, and physical cores. Explicit unsupported
parallel requests raise; unsupported parallel defaults warn once and use one
thread. One-electron and derivative APIs remain serial. See
[installation and runtime policy](installation.md) for complete behavior.

## 7. Analytic derivative API

The installed `giao_integrals/derivatives.hpp` header exposes analytic first
derivatives for overlap, canonical kinetic energy, physical magnetic kinetic
energy, nuclear attraction, and four-center ERIs. Primitive functions return
fixed `std::array` values. Shell and basis functions accept caller-owned
contiguous spans.

Python uses leading derivative dimensions:

| Function family | Shape |
|---|---|
| `primitive_*_center_derivatives` | `(2, 3)` for one-electron integrals; `(4, 3)` for ERIs |
| `primitive_*_magnetic_derivatives` | `(3,)` |
| `*_center_derivatives_shell` | `(2, 3, nao_a, nao_b)` |
| `eri_center_derivatives_shell` | `(4, 3, nao_a, nao_b, nao_c, nao_d)` |
| `*_magnetic_derivatives_shell` | `(3, ...)` |
| `*_nuclear_derivatives` | `(nshell, 3, nao, nao)` |
| `*_magnetic_derivatives` | `(3, nao, nao)` |

The center index follows the input function order and the direction index is
`x, y, z`. `nuclear_attraction_nucleus_derivatives_shell` has shape
`(nnucleus, 3, nao_a, nao_b)`. At basis level,
`nuclear_attraction_nuclear_derivatives` returns a pair: shell-center response
with shape `(nshell, 3, nao, nao)` and potential-center response with shape
`(nnucleus, 3, nao, nao)`. This separation is deliberate because `Basis` has
no atom-to-shell ownership map. The caller sums the applicable entries to form
a molecular atom derivative.

All shell and basis derivative functions support strict `out=` buffers and
release the GIL. ERI derivatives are shell-quartet only; the API does not
allocate an automatic derivative of the full fourth-rank AO tensor. Derivatives of Cholesky vectors are not exposed
in 0.9.

## 8. Errors and reproducibility

Invalid model data raise `std::invalid_argument` in C++ and `ValueError` in
Python. Size mismatches raise `std::length_error` and `ValueError`. Numerical
special-function domain or convergence failures raise `BoysNumericalError` in
both C++ and Python. Successful calls expose their region and estimated error
through `BoysDiagnostics` or `boys_with_diagnostics`.

Serial drivers accumulate primitives and nuclei in input order. Parallel ERI
drivers assign complete blocks dynamically but retain the same per-block
reduction order and input-ordered output. Bitwise equality is tested across
thread counts; there is no separate deterministic-mode switch. Field and
screening choices are per call. The Python default thread budget uses the
process override/environment policy described above; BLAS settings are separate.

The compatibility boundary and staged warning mechanism are specified in
[`compatibility.md`](compatibility.md). In particular, changes to units,
normalization, ordering, shapes, or finite-field symmetries count as API
changes rather than ordinary numerical refinements.

## Moment nuclear derivatives (Priority 1, after 0.9.0)

```python
shell_response, origin_response = gi.moment_nuclear_derivatives(
    basis, (1, 0, 0), origin=(0.0, 0.0, 0.0), field=None,
    out=None, threads=None,
)
```

This differentiates `moment(basis, powers, origin=origin, field=field)` literally,
with no force or electronic-charge sign. The tuple has shapes
`(nshell, 3, nao, nao)` and `(3, nao, nao)`, respectively. Each shell centre is
an independent coordinate; if bra and ket belong to the same shell, both
responses are accumulated. Sum shell entries with `Molecule.shell_atoms` to
obtain atom contributions. `origin_response` differentiates the moment origin,
not the magnetic gauge origin, and is returned separately.

All non-negative Cartesian powers supported by the value kernel are accepted,
including monopoles, dipoles, quadrupoles and higher moments. The exact origin
identity is `dI/dO_k = -powers[k] * I(powers-e_k)` (zero if that power is zero).
Basis derivatives include ordinary Gaussian and London-phase centre response,
at fixed field, gauge origin and moment origin. The operator has no explicit
field dependence and no electronic-charge factor.

`out=(shell_array, origin_array)` fills and returns the exact buffers; either
entry may be `None`. Buffers must be writable, C-contiguous `complex128`, have
exact shapes and not overlap. `threads=None` follows the configured thread
policy, including serial-build fallback; explicit unsupported parallel counts
raise. Shell-pair tasks are parallel, with unchanged primitive summation order.
Native computations release the GIL.

Related primitive/shell entry points:

- `primitive_moment_center_derivatives(a, b, powers, *, origin, field)` → `(2,3)`.
- `primitive_moment_origin_derivatives(a, b, powers, *, origin, field)` → `(3,)`.
- `moment_center_derivatives_shell(a, b, powers, *, origin, field, out)` → `(2,3,na,nb)`.
- `moment_origin_derivatives_shell(a, b, powers, *, origin, field, out)` → `(3,na,nb)`.

The installed C++ header `derivatives.hpp` exposes corresponding primitive,
shell and basis functions; C++ thread counts default to one.

**Translation convention:** raw London AO matrices acquire a phase under
translation. With `k_mu = B × (R_mu-G)/2`, translating all basis centres, the
moment origin and gauge origin together gives
`sum_s dI/dR_s,k + dI/dO_k + dI/dG_k = i*(k_mu-k_nu)[k]*I_mu,nu`.
The gauge-origin derivative of this multiplicative moment matrix is zero.
At zero field the sum is zero; at finite field it is the covariant identity
above. A literal zero sum for raw finite-field matrices would contradict the
existing value and derivative conventions.

## Other property derivatives (Priority 2, after 0.9.0)

```python
dR_gradient = gi.gradient_nuclear_derivatives(basis, "x", field=field, threads=None)
dR_momentum = gi.momentum_nuclear_derivatives(basis, "x", field=field, threads=None)
dB_gradient = gi.gradient_magnetic_derivatives(basis, "x", field=field, threads=None)
dB_momentum = gi.momentum_magnetic_derivatives(basis, "x", field=field, threads=None)
dB_moment = gi.moment_magnetic_derivatives(
    basis, (1, 0, 0), origin=(0.0, 0.0, 0.0), field=field, threads=None,
)
```

| Function | Property argument | Returned shape | Held fixed |
|---|---|---|---|
| `moment_magnetic_derivatives` | `powers`, optional `origin` | `(3,nao,nao)` | basis centres, moment origin, gauge origin |
| `gradient_nuclear_derivatives` | `component` (`x`, `y`, `z`) | `(nshell,3,nao,nao)` | B and gauge origin |
| `momentum_nuclear_derivatives` | `component` (`x`, `y`, `z`) | `(nshell,3,nao,nao)` | B and gauge origin |
| `gradient_magnetic_derivatives` | `component` (`x`, `y`, `z`) | `(3,nao,nao)` | basis centres and gauge origin |
| `momentum_magnetic_derivatives` | `component` (`x`, `y`, `z`) | `(3,nao,nao)` | basis centres and gauge origin |

Nuclear responses include Gaussian and London centre terms; magnetic responses
include the London phase and, for gradient/momentum, its ket wave-vector response
under the spatial derivative. None of these operators has explicit B dependence.
Gradient and canonical momentum have no moment-origin argument. A moment takes
`powers`, not a `component` argument.

Each function also accepts a strict `out=` buffer and returns that exact buffer.
The property component is `"x"`, `"y"` or `"z"`; it is independent of the
leading derivative axis. Nuclear outputs are `(nshell,3,nao,nao)`, magnetic
outputs `(3,nao,nao)`. No electronic-charge or force sign is applied.
Gradient is the ket spatial derivative; momentum is canonical `-i*gradient`.
They do not include a vector-potential term. Field derivatives hold the gauge
origin (and moment origin) fixed, include the complete London response, and
have no additional explicit operator B dependence. In particular, differentiating
the ket gradient includes the derivative of its London wave vector; retaining
only the pair-phase derivative would be incomplete.

Primitive variants are `primitive_{gradient,momentum}_center_derivatives`,
`primitive_{gradient,momentum}_magnetic_derivatives`, and
`primitive_moment_magnetic_derivatives`, with the same property argument.
Shell variants replace `primitive_` by an empty prefix and append `_shell`;
they support `out=` and have the same leading derivative dimensions as the
existing shell APIs. All native computation releases the GIL. Basis matrix
assembly uses the same configured shell-pair threading policy as Priority 1.

## Mixed nuclear/magnetic derivatives (Priority 3, after 0.9.0)

For `overlap`, `kinetic` and `magnetic_kinetic`:

```python
mixed = gi.overlap_mixed_derivatives(basis, field=field, out=None, threads=None)
```

The result is `(nshell,3,3,nao,nao)` with axes
`(shell centre, nuclear xyz, field xyz, bra AO, ket AO)`. It is literally
`d²I/(dR dB)`, with fixed gauge origin and no hidden force sign. Shell functions
are `*_mixed_derivatives_shell(a,b,*,field,out)`, with leading `(2,3,3)`;
primitive functions are `primitive_*_mixed_derivatives(a,b,*,field)`, shape
`(2,3,3)`. Gaussian, London centre/field, and the cross derivative of the London
centre coefficient are all included. Physical `magnetic_kinetic` additionally
includes the explicit B dependence of `1/2*(p+A)^2`; canonical `kinetic` does not
include the vector-potential operator.

```python
shell_mixed, potential_mixed = gi.nuclear_attraction_mixed_derivatives(
    basis, nuclei, field=field, out=None, threads=None,
)
```

The shapes are `(nshell,3,3,nao,nao)` and `(nnucleus,3,3,nao,nao)`.
The Coulomb potential has no explicit B dependence. Potential centres remain
independent of basis centres; assemble atom derivatives using ownership, as
with first derivatives. `out=(shell_out,nucleus_out)` supports separately
allocated strict buffers; either may be `None`, and overlap is rejected.
`nuclear_attraction_mixed_derivatives_shell` and
`primitive_nuclear_attraction_mixed_derivatives` give the basis-centre parts;
`nuclear_attraction_nucleus_mixed_derivatives_shell` and
`primitive_nuclear_attraction_nucleus_mixed_derivatives` give the separate
potential-centre parts, with leading `(nnucleus,3,3)`.

```python
quartet_mixed = gi.eri_mixed_derivatives_shell(a, b, c, d, field=field, out=None)
```

Shape: `(4,3,3,na,nb,nc,nd)`, centre slots in quartet order, nuclear axis then
field axis. `primitive_eri_mixed_derivatives` returns `(4,3,3)`. These include
London response of all four functions, obey pair exchange and simultaneous
conjugate reversal with the corresponding centre-slot permutation, and retain
the value kernel's angular/workspace admission limits (including the extra
shifted powers needed by differentiation).

All native paths release the GIL; basis matrices use configured shell-pair
threading. Shell/primitive functions evaluate one block serially. Strict
`out=` arrays are returned unchanged. Differentiating the translation identity
in the moment section adds both `i*d(k_mu-k_nu)/dB*I` and
`i*(k_mu-k_nu)*dI/dB`; even at B=0, raw mixed AO matrices need not have a zero
translation sum. Tests check this covariant identity, including potential
and gauge-centre contributions where applicable.

## Streamed ERI centre derivatives (Priority 4, after 0.9.0)

```python
for batch in gi.eri_derivative_batches(
    basis, field=field, quartets=None, target_bytes=64 * 1024**2,
    screening_threshold=0.0, threads=None,
):
    for index, quartet in enumerate(batch.quartets):
        derivative = batch.block(index)
```

`Molecule.eri_derivative_batches` forwards the same options with its basis and
field. `EriDerivativeBatch` has the same `quartets`, `shapes`, `offsets`, `values`,
`requested_count`, `screened_count` interface as `EriBatch`. Shapes have six
columns, and `block(index)` is a view shaped `(4,3,na,nb,nc,nd)`.
The first axis is the quartet's four *slots*, not unique atoms or shells.
Repeated shell slots must be added when assembling a shell or atom derivative;
use `mol.shell_atoms` to map the quartet's shell indices to atoms.

Without explicit quartets, emit canonical representatives under exactly the
finite-field ERI symmetries. Pair exchange and simultaneous conjugate reversal
also permute the leading centre-slot axis: permutations `(2,3,0,1)` and
`(1,0,3,2)`, respectively, together with the same AO-axis permutations.
Explicit quartets are evaluated in input order, including duplicates.

`target_bytes` bounds packed complex derivative values in a batch. A single
oversized quartet is emitted alone. Small metadata arrays, per-thread native
workspaces and optional `O(nshell²)` screening bounds are additional. Batches
are evaluated directly into their NumPy packed buffers without a second full
output copy. Release batches after contraction to keep consumer memory bounded;
collecting all batches in a list retains all their data. There is no automatic
full `nao⁴` derivative allocation.

Unscreened outputs preserve existing primitive summation order and are bitwise
identical at different thread counts. Quartets are native parallel tasks; GIL
release and the configured `threads=None` policy match the other new APIs.

### Safe derivative screening

Screening is disabled by default. Ordinary ERI Schwarz factors alone do not
bound derivatives. For a pair density `rho_ab`, define the Coulomb norm
`||rho||² = (rho*|rho)`. A differentiated density is a finite linear combination
`d rho_ab = sum_t c_t rho_t` of shifted primitive pairs: Gaussian lowering and
raising terms, plus London terms proportional to `r_k = (r_k-R_k)+R_k`.
The implementation computes
`D_ab >= max_(AO pair,slot,axis) sum_t |c_t| ||rho_t||`, including absolute
contracted coefficients, and ordinary pair bounds `Q_ab`. The triangle and
Coulomb Cauchy–Schwarz inequalities give the quartet-component bound
`max(D_ab*Q_cd, Q_ab*D_cd)` for all twelve local derivatives. Self-norms use
`(ba|ab)`, preserving complex ordered-pair conventions. Floating-point padding
is added to the non-negative bounds.

A quartet is omitted only if this derivative-specific bound is below
`screening_threshold`. The guarantee applies to each omitted local component,
not a subsequent sum over slots, atoms or quartets. Bound preparation can be
expensive and is conservative for strongly contracted or oscillatory densities.
Screening has the same shifted-integral admission limits as the derivative
kernels. Mixed-derivative screening is not provided by this API.

The installed C++ `EriDerivativeSchwarzBounds` exposes `quartet_bound()` and
basis/field matching. `compute_eri_derivative_blocks` fills caller-owned packed
buffers with explicit cumulative element offsets and native quartet threading.

## VICE gradient assembly (0.11 preparation)

`Molecule` supplies `overlap_nuclear_derivatives`, `kinetic_nuclear_derivatives`,
`magnetic_kinetic_nuclear_derivatives`, `nuclear_attraction_nuclear_derivatives`
and `moment_nuclear_derivatives(powers, *, origin=...)`. Non-moment calls return
`(natom,3,nao,nao)`. The attraction call includes both shell-centre terms and
the potential-centre term for the owning atom. The moment call returns
`(atom_response, origin_response)`, with origin shape `(3,nao,nao)`; that origin
is independent and is **not assigned to any atom**. All accept strict `out=`
(moment: a tuple of buffers) and configured `threads=None`.

The corresponding `*_mixed_derivatives` methods return `(natom,3,3,nao,nao)`;
axes are atom, nuclear xyz, field xyz, AO axes. Moment origin response has
shape `(3,3,nao,nao)` and remains separate. `gi.moment_mixed_derivatives` exposes
the lower-level shell layout `(nshell,3,3,nao,nao)` plus the same separate
origin response. Gauge origin is fixed; London and, for physical kinetic,
explicit operator field response are included. No charge or force signs.

```python
gradient = mol.eri_derivative_contraction(
    coulomb_pairs=[(P, Q)], exchange_pairs=[(R, S)], gamma=None,
    target_bytes=8*1024**2, screening_threshold=0.0, threads=None,
)
```

The free function is `gi.eri_derivative_contraction(basis, shell_atoms, natom,
*, field, coulomb_pairs, exchange_pairs, gamma, target_bytes,
screening_threshold, threads, out)`. Each density contribution is added to a
complex128 `(natom,3)` result, with exactly these indices:

- Coulomb: `sum d(mu nu|lam sig)/dR[A,r] * P[nu,mu] * Q[sig,lam]`.
- Exchange: `sum d(mu nu|lam sig)/dR[A,r] * P[nu,lam] * Q[sig,mu]`.
- Dense: `sum d(mu nu|lam sig)/dR[A,r] * gamma[mu,nu,lam,sig]`.

There is no implicit conjugation of densities, spin/exchange/force sign or
factor of 1/2. Non-Hermitian complex densities are accepted; the result is not
forced real. Matrices P/Q are converted to contiguous complex128; gamma must
already be C-contiguous float64 or complex128 and is read without copying.
Strict `out=` is returned exactly and must not overlap density inputs.

The native contraction expands canonical images `(0,1,2,3)`, `(2,3,0,1)`,
`(1,0,3,2)` and `(3,2,1,0)`; only the derivative is conjugated for the last
two. Both AO axes and centre slots follow the permutation. Equal **shell**
images are counted once; all AO entries within a block are still included.
No real eight-fold assumption is made, including at B=0. Repeated slots add
into their owning atom. Each quartet is contracted in a fixed scalar order,
then twelve local partial sums per quartet are accumulated in canonical/slot
order, giving bitwise thread- and batch-boundary equality.

Memory is a bounded packed derivative batch plus twelve complex sums per
quartet, native recurrence workspace and input densities. No full AO derivative
tensor or extra dense gamma copy is formed. Screening is off by default and
uses the existing per-component bound; it does not bound the final density
contraction error. Native computation releases the GIL.

## TD4.0: independent operator field and origin (0.11 preparation)

```python
LKx = gi.kinetic_angular_momentum(basis, "x", origin=G, field=phase_field,
                               out=None, threads=None)
TC = gi.magnetic_kinetic_operator(basis, field=phase_field,
                                operator_field=C, origin=G, out=None, threads=None)
```

`phase_field` is `MagneticField(B0,G0)` and determines only the London phases.
`kinetic_angular_momentum` returns the selected component of
`<(r-G) cross pi0>`, where `pi0=p+A0`, `p=-i*gradient` and
`A0=B0 cross (r-G0)/2`. The operator origin G is independent of G0 and defaults
to zero. Both the canonical angular momentum and vector-potential terms are
included. It is not merely `(r-G) cross p` when B0 is nonzero.

`magnetic_kinetic_operator` returns `<1/2*(p+Aop)^2>` with
`Aop=C cross (r-G)/2`. C is a three-vector; it never changes the London phases.
Defaults C=B0 and G=G0 delegate to the existing physical kinetic kernel,
reproducing `magnetic_kinetic` bitwise. C=0 is canonical kinetic. Explicit C/G
are independent of B0/G0. `origin=None` means G0 for this function, whereas
the angular-momentum function's default origin is zero.

With `u=r-G`, canonical `L(G)=u cross p` and `Q_ij(G)=<u_i u_j>`, the variant is
`T(C,G)=Tcanonical + 1/2 sum_i C_i L_i(G) + 1/8 sum_ij (|C|² delta_ij-C_i C_j) Q_ij(G)`.
Holding phases fixed, its derivative with respect to C at C=B0,G=G0 is
`1/2*LK(G0)`. This is an **operator-field** derivative, not a derivative of the
phases. Neither field nor origin argument modifies the caller's phase object.

Second moments about an independent G are already supplied exactly by `moment`:

```python
Qij = gi.moment(basis, tuple(int(k == i) + int(k == j) for k in range(3)),
                origin=G, field=phase_field)
```

The six independent powers are `(2,0,0)`, `(1,1,0)`, `(1,0,1)`, `(0,2,0)`,
`(0,1,1)`, `(0,0,2)`. There is no tie between the moment origin and G0.
Useful sum rules at fixed phases are `LK(G+d)=LK(G)-d cross <pi0>` and
`Q_ij(G+d)=Q_ij(G)-d_i <u_j>-d_j <u_i>+d_i*d_j*S`.

Both new matrix functions return complex128 `(nao,nao)` with strict `out=`, GIL
release and configured shell-pair threading. Primitive versions have prefix
`primitive_` and return a complex scalar; shell versions have suffix `_shell`
and return `(na,nb)` with strict `out=`. Cartesian AO order and normalisation
are unchanged. `Molecule` offers matching methods using its fixed phase field.
No new nuclear-magnetic, spin or spherical operators are introduced.
