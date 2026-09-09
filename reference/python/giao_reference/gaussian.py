"""Gaussian, shell, and magnetic-field data for the reference engine."""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

Vector3 = tuple[float, float, float]
ComplexVector3 = tuple[complex, complex, complex]
AngularMomentum = tuple[int, int, int]


def _vector3(values: Sequence[float], name: str) -> Vector3:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain three real numbers") from error

    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} values must be finite")
    return result  # type: ignore[return-value]


def _angular_momentum(values: Sequence[int]) -> AngularMomentum:
    values = tuple(values)
    if len(values) != 3:
        raise ValueError("angular momentum must contain exactly three values")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        raise ValueError("angular momentum values must be integers")

    result = tuple(int(value) for value in values)
    if any(value < 0 for value in result):
        raise ValueError("angular momentum values must be non-negative")
    return result  # type: ignore[return-value]


def _dot(left: Sequence[complex], right: Sequence[complex]) -> complex:
    """Return the bilinear Cartesian dot product without conjugation."""

    return sum(a * b for a, b in zip(left, right, strict=True))


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def double_factorial(value: int) -> int:
    """Return ``value!!`` using the convention ``(-1)!! = 0!! = 1``."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("double factorial requires an integer")
    value = int(value)
    if value < -1:
        raise ValueError("double factorial is defined here only for values >= -1")

    result = 1
    for factor in range(value, 0, -2):
        result *= factor
    return result


def primitive_normalization(exponent: float, angular: Sequence[int]) -> float:
    """Return the L2 normalization of a Cartesian primitive Gaussian."""

    exponent = float(exponent)
    angular = _angular_momentum(angular)
    if not math.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("Gaussian exponent must be finite and positive")

    total_angular_momentum = sum(angular)
    denominator = math.prod(double_factorial(2 * value - 1) for value in angular)
    radial = (2.0 * exponent / math.pi) ** 0.75
    cartesian = math.sqrt((4.0 * exponent) ** total_angular_momentum / denominator)
    return radial * cartesian


def cartesian_components(total_angular_momentum: int) -> tuple[AngularMomentum, ...]:
    """Enumerate a Cartesian shell in the documented x-major ordering."""

    if (
        isinstance(total_angular_momentum, bool)
        or not isinstance(total_angular_momentum, Integral)
        or total_angular_momentum < 0
    ):
        raise ValueError("total angular momentum must be a non-negative integer")

    components = []
    for angular_x in range(int(total_angular_momentum), -1, -1):
        remaining = int(total_angular_momentum) - angular_x
        for angular_y in range(remaining, -1, -1):
            angular_z = remaining - angular_y
            components.append((angular_x, angular_y, angular_z))
    return tuple(components)


@dataclass(frozen=True, slots=True)
class MagneticField:
    """Uniform magnetic field and gauge origin, both in atomic units."""

    B: Vector3 = (0.0, 0.0, 0.0)
    gauge_origin: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "B", _vector3(self.B, "B"))
        object.__setattr__(
            self,
            "gauge_origin",
            _vector3(self.gauge_origin, "gauge_origin"),
        )

    def london_wave_vector(self, center: Sequence[float]) -> Vector3:
        """Return ``0.5 * B x (center - gauge_origin)``."""

        center = _vector3(center, "center")
        displacement = tuple(
            center[index] - self.gauge_origin[index] for index in range(3)
        )
        cross_product = _cross(self.B, displacement)  # type: ignore[arg-type]
        return tuple(0.5 * value for value in cross_product)  # type: ignore[return-value]


ZERO_FIELD = MagneticField()


@dataclass(frozen=True, slots=True)
class Nucleus:
    """A positive point nuclear charge in atomic units."""

    charge: float
    center: Vector3

    def __post_init__(self) -> None:
        charge = float(self.charge)
        if not math.isfinite(charge) or charge <= 0.0:
            raise ValueError("nuclear charge must be finite and positive")
        object.__setattr__(self, "charge", charge)
        object.__setattr__(self, "center", _vector3(self.center, "nuclear center"))


@dataclass(frozen=True, slots=True)
class PrimitiveGaussian:
    """One Cartesian primitive, including a scalar prefactor."""

    exponent: float
    center: Vector3
    angular: AngularMomentum = (0, 0, 0)
    coefficient: float = 1.0
    normalized: bool = True

    def __post_init__(self) -> None:
        exponent = float(self.exponent)
        coefficient = float(self.coefficient)
        if not math.isfinite(exponent) or exponent <= 0.0:
            raise ValueError("Gaussian exponent must be finite and positive")
        if not math.isfinite(coefficient):
            raise ValueError("primitive coefficient must be finite")
        if not isinstance(self.normalized, bool):
            raise ValueError("normalized must be a bool")

        object.__setattr__(self, "exponent", exponent)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "center", _vector3(self.center, "center"))
        object.__setattr__(self, "angular", _angular_momentum(self.angular))

    @property
    def normalization(self) -> float:
        """Return the primitive normalization selected by ``normalized``."""

        if not self.normalized:
            return 1.0
        return primitive_normalization(self.exponent, self.angular)


@dataclass(frozen=True, slots=True)
class Shell:
    """One segmented Cartesian contraction with a shared center and angular L."""

    center: Vector3
    angular_momentum: int
    exponents: tuple[float, ...]
    coefficients: tuple[float, ...]
    normalize: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.angular_momentum, bool)
            or not isinstance(self.angular_momentum, Integral)
            or self.angular_momentum < 0
        ):
            raise ValueError("shell angular momentum must be a non-negative integer")

        exponents = tuple(float(value) for value in self.exponents)
        coefficients = tuple(float(value) for value in self.coefficients)
        if not exponents:
            raise ValueError("a shell requires at least one primitive")
        if len(exponents) != len(coefficients):
            raise ValueError("exponents and coefficients must have equal lengths")
        if any(not math.isfinite(value) or value <= 0.0 for value in exponents):
            raise ValueError("shell exponents must be finite and positive")
        if any(not math.isfinite(value) for value in coefficients):
            raise ValueError("shell coefficients must be finite")
        if not any(value != 0.0 for value in coefficients):
            raise ValueError("a shell contraction cannot have all-zero coefficients")
        if not isinstance(self.normalize, bool):
            raise ValueError("normalize must be a bool")

        object.__setattr__(self, "center", _vector3(self.center, "center"))
        object.__setattr__(self, "angular_momentum", int(self.angular_momentum))
        object.__setattr__(self, "exponents", exponents)
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def components(self) -> tuple[AngularMomentum, ...]:
        """Return this shell's Cartesian components in public AO order."""

        return cartesian_components(self.angular_momentum)

    @property
    def ao_count(self) -> int:
        """Return the number of Cartesian AOs in this segmented shell."""

        return len(self.components)


def gaussian_value(
    primitive: PrimitiveGaussian,
    position: Sequence[float],
    field: MagneticField = ZERO_FIELD,
) -> complex:
    """Evaluate a Cartesian London primitive at one real-space position."""

    position = _vector3(position, "position")
    displacement = tuple(
        position[index] - primitive.center[index] for index in range(3)
    )
    polynomial = math.prod(
        displacement[index] ** primitive.angular[index] for index in range(3)
    )
    radius_squared = sum(value * value for value in displacement)
    ordinary = (
        primitive.coefficient
        * primitive.normalization
        * polynomial
        * math.exp(-primitive.exponent * radius_squared)
    )
    wave_vector = field.london_wave_vector(primitive.center)
    phase = cmath.exp(-1j * _dot(wave_vector, position))
    return ordinary * phase
