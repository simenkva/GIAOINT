"""High-precision overlap checks based on mpmath.

These functions are test oracles. The main reference package does not import
this module, so ordinary double-precision use has no mpmath dependency.
"""

from __future__ import annotations

import math

import mpmath as mp

from .gaussian import ZERO_FIELD, MagneticField, PrimitiveGaussian, double_factorial


def _mpf(value: float) -> mp.mpf:
    return mp.mpf(str(value))


def _wave_vector(
    field: MagneticField, center: tuple[float, float, float]
) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    magnetic_field = tuple(_mpf(value) for value in field.B)
    displacement = tuple(
        _mpf(center[index]) - _mpf(field.gauge_origin[index]) for index in range(3)
    )
    cross_product = (
        magnetic_field[1] * displacement[2] - magnetic_field[2] * displacement[1],
        magnetic_field[2] * displacement[0] - magnetic_field[0] * displacement[2],
        magnetic_field[0] * displacement[1] - magnetic_field[1] * displacement[0],
    )
    return tuple(value / 2 for value in cross_product)


def _normalization(primitive: PrimitiveGaussian) -> mp.mpf:
    if not primitive.normalized:
        return mp.mpf(1)

    exponent = _mpf(primitive.exponent)
    total_angular_momentum = sum(primitive.angular)
    denominator = math.prod(
        double_factorial(2 * value - 1) for value in primitive.angular
    )
    return (2 * exponent / mp.pi) ** mp.mpf("0.75") * mp.sqrt(
        (4 * exponent) ** total_angular_momentum / denominator
    )


def primitive_overlap_formula(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    field: MagneticField = ZERO_FIELD,
    *,
    dps: int = 80,
) -> mp.mpc:
    """Evaluate the direct complex-center formula with arbitrary precision."""

    if dps < 30:
        raise ValueError("high-precision checks require at least 30 decimal digits")

    with mp.workdps(dps):
        alpha = _mpf(bra.exponent)
        beta = _mpf(ket.exponent)
        exponent = alpha + beta
        reduced_exponent = alpha * beta / exponent
        center_bra = tuple(_mpf(value) for value in bra.center)
        center_ket = tuple(_mpf(value) for value in ket.center)
        product_center = tuple(
            (alpha * center_bra[index] + beta * center_ket[index]) / exponent
            for index in range(3)
        )

        wave_vector_bra = _wave_vector(field, bra.center)
        wave_vector_ket = _wave_vector(field, ket.center)
        pair_wave_vector = tuple(
            wave_vector_ket[index] - wave_vector_bra[index] for index in range(3)
        )
        complex_center = tuple(
            product_center[index] - mp.j * pair_wave_vector[index] / (2 * exponent)
            for index in range(3)
        )

        separation_squared = sum(
            (center_bra[index] - center_ket[index]) ** 2 for index in range(3)
        )
        wave_vector_squared = sum(value * value for value in pair_wave_vector)
        phase_argument = -mp.j * sum(
            pair_wave_vector[index] * product_center[index] for index in range(3)
        )
        prefactor = mp.exp(
            -reduced_exponent * separation_squared
            + phase_argument
            - wave_vector_squared / (4 * exponent)
        )

        cartesian_integral = mp.mpc(1)
        for axis in range(3):
            coefficients = [mp.mpc(0)] * (bra.angular[axis] + ket.angular[axis] + 1)
            for power_bra in range(bra.angular[axis] + 1):
                for power_ket in range(ket.angular[axis] + 1):
                    total_power = power_bra + power_ket
                    coefficients[total_power] += (
                        math.comb(bra.angular[axis], power_bra)
                        * math.comb(ket.angular[axis], power_ket)
                        * (complex_center[axis] - center_bra[axis])
                        ** (bra.angular[axis] - power_bra)
                        * (complex_center[axis] - center_ket[axis])
                        ** (ket.angular[axis] - power_ket)
                    )

            axis_integral = mp.mpc(0)
            for order in range(0, len(coefficients), 2):
                half_order = order // 2
                moment = mp.gamma(half_order + mp.mpf("0.5")) / exponent ** (
                    half_order + mp.mpf("0.5")
                )
                axis_integral += coefficients[order] * moment
            cartesian_integral *= axis_integral

        return (
            _mpf(bra.coefficient)
            * _mpf(ket.coefficient)
            * _normalization(bra)
            * _normalization(ket)
            * prefactor
            * cartesian_integral
        )


def primitive_overlap_quadrature(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    field: MagneticField = ZERO_FIELD,
    *,
    dps: int = 80,
) -> mp.mpc:
    """Integrate the original bra-ket product along each real axis."""

    if dps < 30:
        raise ValueError("high-precision checks require at least 30 decimal digits")

    with mp.workdps(dps):
        alpha = _mpf(bra.exponent)
        beta = _mpf(ket.exponent)
        center_bra = tuple(_mpf(value) for value in bra.center)
        center_ket = tuple(_mpf(value) for value in ket.center)
        wave_vector_bra = _wave_vector(field, bra.center)
        wave_vector_ket = _wave_vector(field, ket.center)
        pair_wave_vector = tuple(
            wave_vector_ket[index] - wave_vector_bra[index] for index in range(3)
        )

        result = mp.mpc(1)
        for axis in range(3):
            angular_bra = bra.angular[axis]
            angular_ket = ket.angular[axis]
            coordinate_bra = center_bra[axis]
            coordinate_ket = center_ket[axis]
            phase_wave_vector = pair_wave_vector[axis]

            result *= _quadrature_axis(
                alpha,
                beta,
                angular_bra,
                angular_ket,
                coordinate_bra,
                coordinate_ket,
                phase_wave_vector,
            )

        return (
            _mpf(bra.coefficient)
            * _mpf(ket.coefficient)
            * _normalization(bra)
            * _normalization(ket)
            * result
        )


def _quadrature_axis(
    alpha: mp.mpf,
    beta: mp.mpf,
    angular_bra: int,
    angular_ket: int,
    coordinate_bra: mp.mpf,
    coordinate_ket: mp.mpf,
    phase_wave_vector: mp.mpf,
) -> mp.mpc:
    def integrand(coordinate: mp.mpf) -> mp.mpc:
        return (
            (coordinate - coordinate_bra) ** angular_bra
            * (coordinate - coordinate_ket) ** angular_ket
            * mp.exp(
                -alpha * (coordinate - coordinate_bra) ** 2
                - beta * (coordinate - coordinate_ket) ** 2
                - mp.j * phase_wave_vector * coordinate
            )
        )

    return mp.quad(integrand, [-mp.inf, mp.inf])
