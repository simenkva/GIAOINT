"""Reference one-electron properties over Cartesian London Gaussians."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from .gaussian import ZERO_FIELD, MagneticField, PrimitiveGaussian, Shell
from .overlap import contraction_normalization, gaussian_moment, gaussian_product


def _axis_index(component: str) -> int:
    if component not in {"x", "y", "z"}:
        raise ValueError("component must be 'x', 'y', or 'z'")
    return {"x": 0, "y": 1, "z": 2}[component]


def _powers(values: Sequence[int]) -> tuple[int, int, int]:
    values = tuple(values)
    if len(values) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ValueError("moment powers must contain three non-negative integers")
    return values  # type: ignore[return-value]


def _origin(values: Sequence[float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in values)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("moment origin must contain three finite values")
    return values  # type: ignore[return-value]


def _raw(primitive: PrimitiveGaussian) -> PrimitiveGaussian:
    return PrimitiveGaussian(
        primitive.exponent,
        primitive.center,
        primitive.angular,
        primitive.coefficient * primitive.normalization,
        normalized=False,
    )


def _shifted(angular: tuple[int, int, int], axis: int, amount: int):
    result = list(angular)
    result[axis] += amount
    if result[axis] < 0:
        raise ValueError("shifted angular momentum became negative")
    return tuple(result)


def _raw_moment(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    powers: tuple[int, int, int],
    origin: tuple[float, float, float],
    field: MagneticField,
) -> complex:
    """Evaluate a moment by direct three-polynomial complex-center expansion."""

    pair = gaussian_product(bra, ket, field)
    cartesian_integral = 1.0 + 0.0j
    for axis in range(3):
        angular_bra = bra.angular[axis]
        angular_ket = ket.angular[axis]
        moment_power = powers[axis]
        coefficients = [0.0j] * (angular_bra + angular_ket + moment_power + 1)
        for power_bra in range(angular_bra + 1):
            for power_ket in range(angular_ket + 1):
                for power_moment in range(moment_power + 1):
                    total_power = power_bra + power_ket + power_moment
                    coefficients[total_power] += (
                        math.comb(angular_bra, power_bra)
                        * math.comb(angular_ket, power_ket)
                        * math.comb(moment_power, power_moment)
                        * (pair.complex_center[axis] - bra.center[axis])
                        ** (angular_bra - power_bra)
                        * (pair.complex_center[axis] - ket.center[axis])
                        ** (angular_ket - power_ket)
                        * (pair.complex_center[axis] - origin[axis])
                        ** (moment_power - power_moment)
                    )
        cartesian_integral *= sum(
            coefficient * gaussian_moment(order, pair.exponent)
            for order, coefficient in enumerate(coefficients)
            if order % 2 == 0
        )
    return bra.coefficient * ket.coefficient * pair.prefactor * cartesian_integral


def primitive_moment(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    powers: Sequence[int],
    *,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    field: MagneticField = ZERO_FIELD,
) -> complex:
    """Return ``<bra|prod_j (r_j-origin_j)^powers_j|ket>``."""

    return _raw_moment(_raw(bra), _raw(ket), _powers(powers), _origin(origin), field)


def _raw_overlap_with_angular(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    angular: tuple[int, int, int],
    field: MagneticField,
) -> complex:
    shifted_ket = PrimitiveGaussian(
        ket.exponent,
        ket.center,
        angular,
        ket.coefficient,
        normalized=False,
    )
    return _raw_moment(bra, shifted_ket, (0, 0, 0), (0.0, 0.0, 0.0), field)


def _raw_gradient(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    axis: int,
    field: MagneticField,
) -> complex:
    angular = ket.angular[axis]
    result = 0.0j
    if angular:
        result += angular * _raw_overlap_with_angular(
            bra, ket, _shifted(ket.angular, axis, -1), field
        )
    result -= (
        2.0
        * ket.exponent
        * _raw_overlap_with_angular(bra, ket, _shifted(ket.angular, axis, 1), field)
    )
    wave_vector = field.london_wave_vector(ket.center)
    result -= (
        1j * wave_vector[axis] * _raw_overlap_with_angular(bra, ket, ket.angular, field)
    )
    return result


def primitive_gradient(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    component: str,
    field: MagneticField = ZERO_FIELD,
) -> complex:
    """Return the matrix element of a Cartesian derivative acting on the ket."""

    return _raw_gradient(_raw(bra), _raw(ket), _axis_index(component), field)


def primitive_momentum(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    component: str,
    field: MagneticField = ZERO_FIELD,
) -> complex:
    """Return the matrix element of canonical momentum ``-1j*gradient``."""

    return -1j * primitive_gradient(bra, ket, component, field)


def primitive_kinetic(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    field: MagneticField = ZERO_FIELD,
) -> complex:
    """Return canonical kinetic energy ``<-1/2 nabla^2>``."""

    raw_bra = _raw(bra)
    raw_ket = _raw(ket)
    wave_vector = field.london_wave_vector(raw_ket.center)
    result = 0.0j
    for axis in range(3):
        angular = raw_ket.angular[axis]
        if angular >= 2:
            result -= (
                0.5
                * angular
                * (angular - 1)
                * _raw_overlap_with_angular(
                    raw_bra, raw_ket, _shifted(raw_ket.angular, axis, -2), field
                )
            )
        base = _raw_overlap_with_angular(raw_bra, raw_ket, raw_ket.angular, field)
        result += (
            raw_ket.exponent * (2 * angular + 1) + 0.5 * wave_vector[axis] ** 2
        ) * base
        result -= (
            2.0
            * raw_ket.exponent**2
            * _raw_overlap_with_angular(
                raw_bra, raw_ket, _shifted(raw_ket.angular, axis, 2), field
            )
        )
        ordinary_gradient = 0.0j
        if angular:
            ordinary_gradient += angular * _raw_overlap_with_angular(
                raw_bra, raw_ket, _shifted(raw_ket.angular, axis, -1), field
            )
        ordinary_gradient -= (
            2.0
            * raw_ket.exponent
            * _raw_overlap_with_angular(
                raw_bra, raw_ket, _shifted(raw_ket.angular, axis, 1), field
            )
        )
        result += 1j * wave_vector[axis] * ordinary_gradient
    return result


def _raw_momentum_moment(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    coordinate_axis: int,
    momentum_axis: int,
    field: MagneticField,
) -> complex:
    powers = [0, 0, 0]
    powers[coordinate_axis] = 1
    powers_tuple = tuple(powers)
    angular = ket.angular[momentum_axis]
    derivative_moment = 0.0j
    if angular:
        lower = PrimitiveGaussian(
            ket.exponent,
            ket.center,
            _shifted(ket.angular, momentum_axis, -1),
            ket.coefficient,
            normalized=False,
        )
        derivative_moment += angular * _raw_moment(
            bra, lower, powers_tuple, field.gauge_origin, field
        )
    upper = PrimitiveGaussian(
        ket.exponent,
        ket.center,
        _shifted(ket.angular, momentum_axis, 1),
        ket.coefficient,
        normalized=False,
    )
    derivative_moment -= (
        2.0
        * ket.exponent
        * _raw_moment(bra, upper, powers_tuple, field.gauge_origin, field)
    )
    base_moment = _raw_moment(bra, ket, powers_tuple, field.gauge_origin, field)
    wave_vector = field.london_wave_vector(ket.center)
    return -1j * derivative_moment - wave_vector[momentum_axis] * base_moment


def primitive_magnetic_kinetic(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    field: MagneticField = ZERO_FIELD,
) -> complex:
    """Return ``1/2 (p + A_O)^2`` for the frozen electron convention."""

    raw_bra = _raw(bra)
    raw_ket = _raw(ket)
    result = primitive_kinetic(raw_bra, raw_ket, field)
    magnetic_field = field.B
    result += (
        0.5
        * magnetic_field[0]
        * (
            _raw_momentum_moment(raw_bra, raw_ket, 1, 2, field)
            - _raw_momentum_moment(raw_bra, raw_ket, 2, 1, field)
        )
    )
    result += (
        0.5
        * magnetic_field[1]
        * (
            _raw_momentum_moment(raw_bra, raw_ket, 2, 0, field)
            - _raw_momentum_moment(raw_bra, raw_ket, 0, 2, field)
        )
    )
    result += (
        0.5
        * magnetic_field[2]
        * (
            _raw_momentum_moment(raw_bra, raw_ket, 0, 1, field)
            - _raw_momentum_moment(raw_bra, raw_ket, 1, 0, field)
        )
    )

    field_squared = sum(value * value for value in magnetic_field)
    radius_squared = 0.0j
    projected_squared = 0.0j
    for axis in range(3):
        powers = [0, 0, 0]
        powers[axis] = 2
        moment = _raw_moment(raw_bra, raw_ket, tuple(powers), field.gauge_origin, field)
        radius_squared += moment
        projected_squared += magnetic_field[axis] ** 2 * moment
    for first in range(3):
        for second in range(first + 1, 3):
            powers = [0, 0, 0]
            powers[first] = powers[second] = 1
            projected_squared += (
                2.0
                * magnetic_field[first]
                * magnetic_field[second]
                * _raw_moment(
                    raw_bra, raw_ket, tuple(powers), field.gauge_origin, field
                )
            )
    return result + 0.125 * (field_squared * radius_squared - projected_squared)


PrimitiveOperator = Callable[[PrimitiveGaussian, PrimitiveGaussian], complex]


def _shell_property(
    shell_bra: Shell,
    shell_ket: Shell,
    operator: PrimitiveOperator,
) -> tuple[tuple[complex, ...], ...]:
    normalization_bra = contraction_normalization(shell_bra)
    normalization_ket = contraction_normalization(shell_ket)
    result = []
    for angular_bra in shell_bra.components:
        row = []
        for angular_ket in shell_ket.components:
            value = 0.0j
            for exponent_bra, coefficient_bra in zip(
                shell_bra.exponents, shell_bra.coefficients, strict=True
            ):
                bra = PrimitiveGaussian(
                    exponent_bra,
                    shell_bra.center,
                    angular_bra,
                    normalization_bra * coefficient_bra,
                )
                for exponent_ket, coefficient_ket in zip(
                    shell_ket.exponents, shell_ket.coefficients, strict=True
                ):
                    ket = PrimitiveGaussian(
                        exponent_ket,
                        shell_ket.center,
                        angular_ket,
                        normalization_ket * coefficient_ket,
                    )
                    value += operator(bra, ket)
            row.append(value)
        result.append(tuple(row))
    return tuple(result)


def shell_moment(
    shell_bra: Shell,
    shell_ket: Shell,
    powers: Sequence[int],
    *,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    field: MagneticField = ZERO_FIELD,
):
    return _shell_property(
        shell_bra,
        shell_ket,
        lambda bra, ket: primitive_moment(bra, ket, powers, origin=origin, field=field),
    )


def shell_gradient(
    shell_bra: Shell,
    shell_ket: Shell,
    component: str,
    field: MagneticField = ZERO_FIELD,
):
    return _shell_property(
        shell_bra,
        shell_ket,
        lambda bra, ket: primitive_gradient(bra, ket, component, field),
    )


def shell_momentum(
    shell_bra: Shell,
    shell_ket: Shell,
    component: str,
    field: MagneticField = ZERO_FIELD,
):
    return _shell_property(
        shell_bra,
        shell_ket,
        lambda bra, ket: primitive_momentum(bra, ket, component, field),
    )


def shell_kinetic(
    shell_bra: Shell,
    shell_ket: Shell,
    field: MagneticField = ZERO_FIELD,
):
    return _shell_property(
        shell_bra,
        shell_ket,
        lambda bra, ket: primitive_kinetic(bra, ket, field),
    )


def shell_magnetic_kinetic(
    shell_bra: Shell,
    shell_ket: Shell,
    field: MagneticField = ZERO_FIELD,
):
    return _shell_property(
        shell_bra,
        shell_ket,
        lambda bra, ket: primitive_magnetic_kinetic(bra, ket, field),
    )
