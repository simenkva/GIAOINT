from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Real3 = Sequence[float] | npt.NDArray[np.float64]
Angular = Sequence[int]
ComplexArray = npt.NDArray[np.complex128]

__version__: str

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

def cartesian_components(
    total_angular_momentum: int,
) -> tuple[tuple[int, int, int], ...]: ...
def primitive_normalization(exponent: float, angular: Angular) -> float: ...
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
