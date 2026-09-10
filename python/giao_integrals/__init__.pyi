from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

Real3 = Sequence[float] | npt.NDArray[np.float64]
Angular = Sequence[int]
ComplexArray = npt.NDArray[np.complex128]

__version__: str

class BoysNumericalError(RuntimeError): ...

class BoysRegion(Enum):
    POWER_SERIES: BoysRegion
    ADAPTIVE_QUADRATURE: BoysRegion
    SCALED_QUADRATURE: BoysRegion
    POSITIVE_ASYMPTOTIC: BoysRegion

class CartesianExponent:
    def __init__(self, values: Angular) -> None: ...
    @property
    def x(self) -> int: ...
    @property
    def y(self) -> int: ...
    @property
    def z(self) -> int: ...
    @property
    def total(self) -> int: ...
    def __iter__(self): ...

class MagneticField:
    def __init__(
        self,
        B: Real3 = (0.0, 0.0, 0.0),
        gauge_origin: Real3 = (0.0, 0.0, 0.0),
    ) -> None: ...
    @property
    def B(self) -> tuple[float, float, float]: ...
    @property
    def gauge_origin(self) -> tuple[float, float, float]: ...
    def london_wave_vector(self, center: Real3) -> tuple[float, float, float]: ...

class Nucleus:
    def __init__(self, charge: float, center: Real3) -> None: ...
    charge: float
    @property
    def center(self) -> tuple[float, float, float]: ...

class PrimitiveGaussian:
    def __init__(
        self,
        exponent: float,
        center: Real3,
        angular: Angular = (0, 0, 0),
        coefficient: float = 1.0,
        normalized: bool = True,
    ) -> None: ...
    exponent: float
    coefficient: float
    normalized: bool
    @property
    def center(self) -> tuple[float, float, float]: ...
    @property
    def angular(self) -> tuple[int, int, int]: ...

class Shell:
    def __init__(
        self,
        center: Real3,
        angular_momentum: int,
        exponents: npt.ArrayLike,
        coefficients: npt.ArrayLike,
        *,
        normalize: bool = True,
    ) -> None: ...
    @property
    def center(self) -> tuple[float, float, float]: ...
    @property
    def angular_momentum(self) -> int: ...
    @property
    def primitive_count(self) -> int: ...
    @property
    def contraction_count(self) -> int: ...
    @property
    def cartesian_count(self) -> int: ...
    @property
    def ao_count(self) -> int: ...
    @property
    def components(self) -> tuple[tuple[int, int, int], ...]: ...

class Basis:
    def __init__(self, shells: Sequence[Shell]) -> None: ...
    @property
    def shells(self) -> list[Shell]: ...
    @property
    def ao_offsets(self) -> tuple[int, ...]: ...
    @property
    def ao_count(self) -> int: ...

@dataclass(frozen=True)
class EriBatch:
    quartets: npt.NDArray[np.uint32]
    shapes: npt.NDArray[np.int64]
    offsets: npt.NDArray[np.int64]
    values: ComplexArray
    requested_count: int
    screened_count: int
    def block(self, index: int) -> ComplexArray: ...

def primitive_eri(
    a: PrimitiveGaussian,
    b: PrimitiveGaussian,
    c: PrimitiveGaussian,
    d: PrimitiveGaussian,
    *,
    field: MagneticField | None = None,
) -> complex: ...
def eri_shell(
    a: Shell,
    b: Shell,
    c: Shell,
    d: Shell,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def eri_batches(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    quartets: Iterable[Sequence[int]] | None = None,
    target_bytes: int = 64 * 1024 * 1024,
    screening_threshold: float = 0.0,
    threads: int = 1,
) -> Iterator[EriBatch]: ...
def eri(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    storage: str = "blocks",
    max_bytes: int | None = None,
    target_bytes: int = 64 * 1024 * 1024,
    screening_threshold: float = 0.0,
    threads: int = 1,
) -> Iterator[EriBatch] | ComplexArray: ...
def eri_schwarz_bounds(
    basis: Basis, *, field: MagneticField | None = None
) -> npt.NDArray[np.float64]: ...
def openmp_enabled() -> bool: ...
def openmp_max_threads() -> int: ...
def cartesian_components(
    total_angular_momentum: int,
) -> tuple[tuple[int, int, int], ...]: ...
def primitive_normalization(exponent: float, angular: Angular) -> float: ...
def boys(
    argument: complex, maximum_order: int, *, scaled: bool = False
) -> ComplexArray: ...
def boys_with_diagnostics(
    argument: complex, maximum_order: int, *, scaled: bool = False
) -> tuple[ComplexArray, dict[str, object]]: ...
def primitive_overlap(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    *,
    field: MagneticField | None = None,
) -> complex: ...
def primitive_moment(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    powers: Angular,
    *,
    origin: Real3 = (0.0, 0.0, 0.0),
    field: MagneticField | None = None,
) -> complex: ...
def primitive_gradient(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    component: str,
    *,
    field: MagneticField | None = None,
) -> complex: ...
def primitive_momentum(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    component: str,
    *,
    field: MagneticField | None = None,
) -> complex: ...
def primitive_kinetic(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    *,
    field: MagneticField | None = None,
) -> complex: ...
def primitive_magnetic_kinetic(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    *,
    field: MagneticField | None = None,
) -> complex: ...
def primitive_nuclear_attraction(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    nuclei: Sequence[Nucleus],
    *,
    field: MagneticField | None = None,
) -> complex: ...
def overlap_shell(
    a: Shell,
    b: Shell,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def moment_shell(
    a: Shell,
    b: Shell,
    powers: Angular,
    *,
    origin: Real3 = (0.0, 0.0, 0.0),
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def gradient_shell(
    a: Shell,
    b: Shell,
    component: str,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def momentum_shell(
    a: Shell,
    b: Shell,
    component: str,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def kinetic_shell(
    a: Shell,
    b: Shell,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def magnetic_kinetic_shell(
    a: Shell,
    b: Shell,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def nuclear_attraction_shell(
    a: Shell,
    b: Shell,
    nuclei: Sequence[Nucleus],
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def overlap(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def moment(
    basis: Basis,
    powers: Angular,
    *,
    origin: Real3 = (0.0, 0.0, 0.0),
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def gradient(
    basis: Basis,
    component: str,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def momentum(
    basis: Basis,
    component: str,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def kinetic(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def magnetic_kinetic(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def nuclear_attraction(
    basis: Basis,
    nuclei: Sequence[Nucleus],
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def primitive_overlap_center_derivatives(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    *,
    field: MagneticField | None = None,
) -> ComplexArray: ...
def primitive_overlap_magnetic_derivatives(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    *,
    field: MagneticField | None = None,
) -> ComplexArray: ...
def primitive_kinetic_center_derivatives(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    *,
    field: MagneticField | None = None,
) -> ComplexArray: ...
def primitive_kinetic_magnetic_derivatives(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    *,
    field: MagneticField | None = None,
) -> ComplexArray: ...
def primitive_magnetic_kinetic_center_derivatives(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    *,
    field: MagneticField | None = None,
) -> ComplexArray: ...
def primitive_magnetic_kinetic_magnetic_derivatives(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    *,
    field: MagneticField | None = None,
) -> ComplexArray: ...
def primitive_nuclear_attraction_center_derivatives(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    nuclei: Sequence[Nucleus],
    *,
    field: MagneticField | None = None,
) -> ComplexArray: ...
def primitive_nuclear_attraction_nucleus_derivatives(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    nuclei: Sequence[Nucleus],
    *,
    field: MagneticField | None = None,
) -> ComplexArray: ...
def primitive_nuclear_attraction_magnetic_derivatives(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    nuclei: Sequence[Nucleus],
    *,
    field: MagneticField | None = None,
) -> ComplexArray: ...
def primitive_eri_center_derivatives(
    a: PrimitiveGaussian,
    b: PrimitiveGaussian,
    c: PrimitiveGaussian,
    d: PrimitiveGaussian,
    *,
    field: MagneticField | None = None,
) -> ComplexArray: ...
def primitive_eri_magnetic_derivatives(
    a: PrimitiveGaussian,
    b: PrimitiveGaussian,
    c: PrimitiveGaussian,
    d: PrimitiveGaussian,
    *,
    field: MagneticField | None = None,
) -> ComplexArray: ...
def overlap_center_derivatives_shell(
    a: Shell,
    b: Shell,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def overlap_magnetic_derivatives_shell(
    a: Shell,
    b: Shell,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def kinetic_center_derivatives_shell(
    a: Shell,
    b: Shell,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def kinetic_magnetic_derivatives_shell(
    a: Shell,
    b: Shell,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def magnetic_kinetic_center_derivatives_shell(
    a: Shell,
    b: Shell,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def magnetic_kinetic_magnetic_derivatives_shell(
    a: Shell,
    b: Shell,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def nuclear_attraction_center_derivatives_shell(
    a: Shell,
    b: Shell,
    nuclei: Sequence[Nucleus],
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def nuclear_attraction_nucleus_derivatives_shell(
    a: Shell,
    b: Shell,
    nuclei: Sequence[Nucleus],
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def nuclear_attraction_magnetic_derivatives_shell(
    a: Shell,
    b: Shell,
    nuclei: Sequence[Nucleus],
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def eri_center_derivatives_shell(
    a: Shell,
    b: Shell,
    c: Shell,
    d: Shell,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def eri_magnetic_derivatives_shell(
    a: Shell,
    b: Shell,
    c: Shell,
    d: Shell,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def overlap_nuclear_derivatives(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def overlap_magnetic_derivatives(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def kinetic_nuclear_derivatives(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def kinetic_magnetic_derivatives(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def magnetic_kinetic_nuclear_derivatives(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def magnetic_kinetic_magnetic_derivatives(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
def nuclear_attraction_nuclear_derivatives(
    basis: Basis,
    nuclei: Sequence[Nucleus],
    *,
    field: MagneticField | None = None,
    shell_out: ComplexArray | None = None,
    nucleus_out: ComplexArray | None = None,
) -> tuple[ComplexArray, ComplexArray]: ...
def nuclear_attraction_magnetic_derivatives(
    basis: Basis,
    nuclei: Sequence[Nucleus],
    *,
    field: MagneticField | None = None,
    out: ComplexArray | None = None,
) -> ComplexArray: ...
