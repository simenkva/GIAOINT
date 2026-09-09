"""Independent Obara--Saika nuclear-attraction reference integrals."""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import cache

import mpmath as mp

from .gaussian import (
    ZERO_FIELD,
    MagneticField,
    Nucleus,
    PrimitiveGaussian,
    Shell,
    _angular_momentum,
    double_factorial,
)
from .overlap import contraction_normalization


def _normalization(primitive: PrimitiveGaussian):
    if not primitive.normalized:
        return mp.mpf(1)
    exponent = mp.mpf(primitive.exponent)
    denominator = math.prod(
        double_factorial(2 * angular - 1) for angular in primitive.angular
    )
    return (2 * exponent / mp.pi) ** mp.mpf("0.75") * mp.sqrt(
        (4 * exponent) ** sum(primitive.angular) / denominator
    )


def primitive_nuclear_attraction(
    bra: PrimitiveGaussian,
    ket: PrimitiveGaussian,
    nuclei: Sequence[Nucleus],
    field: MagneticField = ZERO_FIELD,
    *,
    dps: int = 80,
):
    """Return ``<bra|-sum_C Z_C/r_C|ket>`` by Obara--Saika VRR."""

    with mp.workdps(dps):
        alpha = mp.mpf(bra.exponent)
        beta = mp.mpf(ket.exponent)
        p = alpha + beta
        reduced_exponent = alpha * beta / p
        center_a = tuple(mp.mpf(value) for value in bra.center)
        center_b = tuple(mp.mpf(value) for value in ket.center)
        product_center = tuple(
            (alpha * center_a[axis] + beta * center_b[axis]) / p for axis in range(3)
        )
        magnetic_field = tuple(mp.mpf(value) for value in field.B)
        separation = tuple(center_b[axis] - center_a[axis] for axis in range(3))
        pair_wave_vector = (
            (magnetic_field[1] * separation[2] - magnetic_field[2] * separation[1]) / 2,
            (magnetic_field[2] * separation[0] - magnetic_field[0] * separation[2]) / 2,
            (magnetic_field[0] * separation[1] - magnetic_field[1] * separation[0]) / 2,
        )
        complex_center = tuple(
            product_center[axis] - mp.j * pair_wave_vector[axis] / (2 * p)
            for axis in range(3)
        )
        ordinary_prefactor = mp.exp(
            -reduced_exponent * sum(value * value for value in separation)
        )
        london_prefactor = mp.exp(
            -mp.j
            * sum(pair_wave_vector[axis] * product_center[axis] for axis in range(3))
            - sum(value * value for value in pair_wave_vector) / (4 * p)
        )
        common = (
            mp.mpf(bra.coefficient)
            * mp.mpf(ket.coefficient)
            * _normalization(bra)
            * _normalization(ket)
            * ordinary_prefactor
            * london_prefactor
            * (2 * mp.pi / p)
        )
        angular_bra = tuple(int(value) for value in bra.angular)
        angular_ket = tuple(int(value) for value in ket.angular)
        result = mp.mpc(0)

        for nucleus in nuclei:
            center = tuple(mp.mpf(value) for value in nucleus.center)
            pc = tuple(complex_center[axis] - center[axis] for axis in range(3))
            pa = tuple(complex_center[axis] - center_a[axis] for axis in range(3))
            pb = tuple(complex_center[axis] - center_b[axis] for axis in range(3))
            argument = p * sum(value * value for value in pc)

            @cache
            def integral(
                a: tuple[int, int, int],
                b: tuple[int, int, int],
                m: int,
                *,
                argument=argument,
                pa=pa,
                pb=pb,
                pc=pc,
            ):
                if sum(a) == 0 and sum(b) == 0:
                    return mp.hyp1f1(
                        m + mp.mpf("0.5"), m + mp.mpf("1.5"), -argument
                    ) / (2 * m + 1)

                for axis in range(3):
                    if a[axis] > 0:
                        parent = list(a)
                        parent[axis] -= 1
                        parent_tuple = tuple(parent)
                        value = pa[axis] * integral(parent_tuple, b, m)
                        value -= pc[axis] * integral(parent_tuple, b, m + 1)
                        if parent[axis] > 0:
                            lower = list(parent)
                            lower[axis] -= 1
                            value += (
                                parent[axis]
                                / (2 * p)
                                * (
                                    integral(tuple(lower), b, m)
                                    - integral(tuple(lower), b, m + 1)
                                )
                            )
                        if b[axis] > 0:
                            lower_b = list(b)
                            lower_b[axis] -= 1
                            value += (
                                b[axis]
                                / (2 * p)
                                * (
                                    integral(parent_tuple, tuple(lower_b), m)
                                    - integral(parent_tuple, tuple(lower_b), m + 1)
                                )
                            )
                        return value

                for axis in range(3):
                    if b[axis] > 0:
                        parent = list(b)
                        parent[axis] -= 1
                        parent_tuple = tuple(parent)
                        value = pb[axis] * integral(a, parent_tuple, m)
                        value -= pc[axis] * integral(a, parent_tuple, m + 1)
                        if parent[axis] > 0:
                            lower = list(parent)
                            lower[axis] -= 1
                            value += (
                                parent[axis]
                                / (2 * p)
                                * (
                                    integral(a, tuple(lower), m)
                                    - integral(a, tuple(lower), m + 1)
                                )
                            )
                        if a[axis] > 0:
                            lower_a = list(a)
                            lower_a[axis] -= 1
                            value += (
                                a[axis]
                                / (2 * p)
                                * (
                                    integral(tuple(lower_a), parent_tuple, m)
                                    - integral(tuple(lower_a), parent_tuple, m + 1)
                                )
                            )
                        return value
                raise AssertionError("unreachable angular-momentum state")

            result -= (
                mp.mpf(nucleus.charge) * common * integral(angular_bra, angular_ket, 0)
            )
        return result


def contracted_component_nuclear_attraction(
    shell_bra: Shell,
    angular_bra: Sequence[int],
    shell_ket: Shell,
    angular_ket: Sequence[int],
    nuclei: Sequence[Nucleus],
    field: MagneticField = ZERO_FIELD,
    *,
    dps: int = 60,
):
    """Return one segmented contracted nuclear-attraction integral."""

    angular_bra = _angular_momentum(angular_bra)
    angular_ket = _angular_momentum(angular_ket)
    normalization_bra = contraction_normalization(shell_bra)
    normalization_ket = contraction_normalization(shell_ket)
    result = mp.mpc(0)
    for exponent_bra, coefficient_bra in zip(
        shell_bra.exponents, shell_bra.coefficients, strict=True
    ):
        for exponent_ket, coefficient_ket in zip(
            shell_ket.exponents, shell_ket.coefficients, strict=True
        ):
            result += primitive_nuclear_attraction(
                PrimitiveGaussian(
                    exponent_bra,
                    shell_bra.center,
                    angular_bra,
                    normalization_bra * coefficient_bra,
                ),
                PrimitiveGaussian(
                    exponent_ket,
                    shell_ket.center,
                    angular_ket,
                    normalization_ket * coefficient_ket,
                ),
                nuclei,
                field,
                dps=dps,
            )
    return result


def shell_nuclear_attraction(
    shell_bra: Shell,
    shell_ket: Shell,
    nuclei: Sequence[Nucleus],
    field: MagneticField = ZERO_FIELD,
    *,
    dps: int = 60,
):
    """Return a complete segmented shell-pair attraction block."""

    return tuple(
        tuple(
            contracted_component_nuclear_attraction(
                shell_bra,
                angular_bra,
                shell_ket,
                angular_ket,
                nuclei,
                field,
                dps=dps,
            )
            for angular_ket in shell_ket.components
        )
        for angular_bra in shell_bra.components
    )
