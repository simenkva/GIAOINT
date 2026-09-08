"""Direct overlap formulas for Cartesian London Gaussian functions."""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from dataclasses import dataclass

from .gaussian import (
    ZERO_FIELD,
    ComplexVector3,
    MagneticField,
    PrimitiveGaussian,
    Shell,
    Vector3,
    _angular_momentum,
    _dot,
)


@dataclass(frozen=True, slots=True)
class GaussianPair:
    """Scalar and vector data from the London Gaussian product theorem."""

    exponent: float
    reduced_exponent: float
    product_center: Vector3
    pair_wave_vector: Vector3
    complex_center: ComplexVector3
    ordinary_prefactor: float
    london_prefactor: complex

    @property
    def prefactor(self) -> complex:
        """Return the ordinary Gaussian and London prefactors together."""

        return self.ordinary_prefactor * self.london_prefactor


def gaussian_product(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    field: MagneticField = ZERO_FIELD,
) -> GaussianPair:
    """Build the product data for ``conj(bra) * ket``.

    The pair wave vector is ``kappa_ket - kappa_bra`` and the complex center is
    ``P - 1j * q / (2 p)``. These signs follow the frozen GIAO convention.
    """

    exponent = bra.exponent + ket.exponent
    reduced_exponent = bra.exponent * ket.exponent / exponent
    product_center = tuple(
        (bra.exponent * bra.center[index] + ket.exponent * ket.center[index]) / exponent
        for index in range(3)
    )

    bra_wave_vector = field.london_wave_vector(bra.center)
    ket_wave_vector = field.london_wave_vector(ket.center)
    pair_wave_vector = tuple(
        ket_wave_vector[index] - bra_wave_vector[index] for index in range(3)
    )
    complex_center = tuple(
        product_center[index] - 0.5j * pair_wave_vector[index] / exponent
        for index in range(3)
    )

    separation_squared = sum(
        (bra.center[index] - ket.center[index]) ** 2 for index in range(3)
    )
    wave_vector_squared = sum(value * value for value in pair_wave_vector)
    ordinary_prefactor = math.exp(-reduced_exponent * separation_squared)
    london_prefactor = cmath.exp(
        -1j * _dot(pair_wave_vector, product_center)
        - wave_vector_squared / (4.0 * exponent)
    )

    return GaussianPair(
        exponent=exponent,
        reduced_exponent=reduced_exponent,
        product_center=product_center,  # type: ignore[arg-type]
        pair_wave_vector=pair_wave_vector,  # type: ignore[arg-type]
        complex_center=complex_center,  # type: ignore[arg-type]
        ordinary_prefactor=ordinary_prefactor,
        london_prefactor=london_prefactor,
    )


def gaussian_moment(order: int, exponent: float) -> float:
    """Integrate ``u**order * exp(-exponent * u**2)`` over the real line."""

    if order < 0:
        raise ValueError("Gaussian moment order must be non-negative")
    if order % 2:
        return 0.0
    half_order = order // 2
    return math.gamma(half_order + 0.5) / exponent ** (half_order + 0.5)


def polynomial_coefficients(
    angular_bra: int,
    angular_ket: int,
    center_bra: float,
    center_ket: float,
    complex_center: complex,
) -> tuple[complex, ...]:
    """Expand two Cartesian factors around one complex product center."""

    coefficients = [0.0j] * (angular_bra + angular_ket + 1)
    for power_bra in range(angular_bra + 1):
        for power_ket in range(angular_ket + 1):
            total_power = power_bra + power_ket
            coefficients[total_power] += (
                math.comb(angular_bra, power_bra)
                * math.comb(angular_ket, power_ket)
                * (complex_center - center_bra) ** (angular_bra - power_bra)
                * (complex_center - center_ket) ** (angular_ket - power_ket)
            )
    return tuple(coefficients)


def _one_dimensional_overlap(
    angular_bra: int,
    angular_ket: int,
    center_bra: float,
    center_ket: float,
    complex_center: complex,
    exponent: float,
) -> complex:
    coefficients = polynomial_coefficients(
        angular_bra,
        angular_ket,
        center_bra,
        center_ket,
        complex_center,
    )
    return sum(
        coefficient * gaussian_moment(order, exponent)
        for order, coefficient in enumerate(coefficients)
        if order % 2 == 0
    )


def primitive_overlap(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    field: MagneticField = ZERO_FIELD,
) -> complex:
    """Return ``<bra|ket>`` by direct complex-center polynomial moments."""

    pair = gaussian_product(bra, ket, field)
    cartesian_integral = 1.0 + 0.0j
    for axis in range(3):
        cartesian_integral *= _one_dimensional_overlap(
            bra.angular[axis],
            ket.angular[axis],
            bra.center[axis],
            ket.center[axis],
            pair.complex_center[axis],
            pair.exponent,
        )

    return (
        bra.coefficient
        * ket.coefficient
        * bra.normalization
        * ket.normalization
        * pair.prefactor
        * cartesian_integral
    )


def ss_overlap(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    field: MagneticField = ZERO_FIELD,
) -> complex:
    """Return the analytic overlap for two s primitives."""

    if bra.angular != (0, 0, 0) or ket.angular != (0, 0, 0):
        raise ValueError("ss_overlap requires two s primitives")

    pair = gaussian_product(bra, ket, field)
    return (
        bra.coefficient
        * ket.coefficient
        * bra.normalization
        * ket.normalization
        * (math.pi / pair.exponent) ** 1.5
        * pair.prefactor
    )


def contraction_normalization(shell: Shell) -> float:
    """Return the zero-field normalization of one segmented contraction."""

    if not shell.normalize:
        return 1.0

    component = (shell.angular_momentum, 0, 0)
    norm_squared = 0.0j
    for exponent_bra, coefficient_bra in zip(
        shell.exponents, shell.coefficients, strict=True
    ):
        bra = PrimitiveGaussian(
            exponent=exponent_bra,
            center=shell.center,
            angular=component,
            coefficient=coefficient_bra,
        )
        for exponent_ket, coefficient_ket in zip(
            shell.exponents, shell.coefficients, strict=True
        ):
            ket = PrimitiveGaussian(
                exponent=exponent_ket,
                center=shell.center,
                angular=component,
                coefficient=coefficient_ket,
            )
            norm_squared += primitive_overlap(bra, ket)

    scale = max(1.0, abs(norm_squared.real))
    if abs(norm_squared.imag) > 1.0e-13 * scale:
        raise ArithmeticError("contraction norm acquired an unexpected imaginary part")
    if not math.isfinite(norm_squared.real) or norm_squared.real <= 0.0:
        raise ValueError("contraction has a non-positive or non-finite norm")
    return 1.0 / math.sqrt(norm_squared.real)


def contracted_component_overlap(
    shell_bra: Shell,
    angular_bra: Sequence[int],
    shell_ket: Shell,
    angular_ket: Sequence[int],
    field: MagneticField = ZERO_FIELD,
) -> complex:
    """Return one contracted Cartesian overlap from two segmented shells."""

    angular_bra = _angular_momentum(angular_bra)
    angular_ket = _angular_momentum(angular_ket)
    if sum(angular_bra) != shell_bra.angular_momentum:
        raise ValueError("bra component does not belong to its shell")
    if sum(angular_ket) != shell_ket.angular_momentum:
        raise ValueError("ket component does not belong to its shell")

    normalization_bra = contraction_normalization(shell_bra)
    normalization_ket = contraction_normalization(shell_ket)
    result = 0.0j
    for exponent_bra, coefficient_bra in zip(
        shell_bra.exponents, shell_bra.coefficients, strict=True
    ):
        bra = PrimitiveGaussian(
            exponent=exponent_bra,
            center=shell_bra.center,
            angular=angular_bra,
            coefficient=normalization_bra * coefficient_bra,
        )
        for exponent_ket, coefficient_ket in zip(
            shell_ket.exponents, shell_ket.coefficients, strict=True
        ):
            ket = PrimitiveGaussian(
                exponent=exponent_ket,
                center=shell_ket.center,
                angular=angular_ket,
                coefficient=normalization_ket * coefficient_ket,
            )
            result += primitive_overlap(bra, ket, field)
    return result


def shell_overlap(
    shell_bra: Shell,
    shell_ket: Shell,
    field: MagneticField = ZERO_FIELD,
) -> tuple[tuple[complex, ...], ...]:
    """Return a complete shell-pair block in the documented AO ordering."""

    return tuple(
        tuple(
            contracted_component_overlap(
                shell_bra,
                angular_bra,
                shell_ket,
                angular_ket,
                field,
            )
            for angular_ket in shell_ket.components
        )
        for angular_bra in shell_bra.components
    )
