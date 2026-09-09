"""Independent high-precision Obara--Saika ERI reference."""

from __future__ import annotations

from functools import cache

import mpmath as mp

from .gaussian import ZERO_FIELD, MagneticField, PrimitiveGaussian
from .nuclear import _normalization


def primitive_eri(
    a: PrimitiveGaussian,
    b: PrimitiveGaussian,
    c: PrimitiveGaussian,
    d: PrimitiveGaussian,
    field: MagneticField = ZERO_FIELD,
    *,
    dps: int = 80,
):
    """Return ``(ab|cd)`` using Cartesian Obara--Saika vertical recurrence."""

    with mp.workdps(dps):
        primitives = (a, b, c, d)
        exponents = tuple(mp.mpf(value.exponent) for value in primitives)
        centers = tuple(
            tuple(mp.mpf(value) for value in primitive.center)
            for primitive in primitives
        )
        magnetic_field = tuple(mp.mpf(value) for value in field.B)

        def pair_data(left: int, right: int):
            alpha = exponents[left]
            beta = exponents[right]
            exponent = alpha + beta
            separation = tuple(
                centers[right][axis] - centers[left][axis] for axis in range(3)
            )
            product_center = tuple(
                (alpha * centers[left][axis] + beta * centers[right][axis]) / exponent
                for axis in range(3)
            )
            wave = (
                (magnetic_field[1] * separation[2] - magnetic_field[2] * separation[1])
                / 2,
                (magnetic_field[2] * separation[0] - magnetic_field[0] * separation[2])
                / 2,
                (magnetic_field[0] * separation[1] - magnetic_field[1] * separation[0])
                / 2,
            )
            complex_center = tuple(
                product_center[axis] - mp.j * wave[axis] / (2 * exponent)
                for axis in range(3)
            )
            ordinary = mp.exp(
                -(alpha * beta / exponent) * sum(value * value for value in separation)
            )
            london = mp.exp(
                -mp.j * sum(wave[axis] * product_center[axis] for axis in range(3))
                - sum(value * value for value in wave) / (4 * exponent)
            )
            return exponent, product_center, complex_center, ordinary, london

        p, _, complex_p, ordinary_ab, london_ab = pair_data(0, 1)
        q, _, complex_q, ordinary_cd, london_cd = pair_data(2, 3)
        rho = p * q / (p + q)
        displacement = tuple(complex_p[axis] - complex_q[axis] for axis in range(3))
        weighted_center = tuple(
            (p * complex_p[axis] + q * complex_q[axis]) / (p + q) for axis in range(3)
        )
        argument = rho * sum(value * value for value in displacement)
        common = (
            2
            * mp.pi ** mp.mpf("2.5")
            / (p * q * mp.sqrt(p + q))
            * ordinary_ab
            * ordinary_cd
            * london_ab
            * london_cd
        )
        for primitive in primitives:
            common *= mp.mpf(primitive.coefficient) * _normalization(primitive)

        angular = tuple(tuple(int(value) for value in x.angular) for x in primitives)

        def lower(value: tuple[int, int, int], axis: int):
            result = list(value)
            result[axis] -= 1
            return tuple(result)

        @cache
        def integral(
            angular_a: tuple[int, int, int],
            angular_b: tuple[int, int, int],
            angular_c: tuple[int, int, int],
            angular_d: tuple[int, int, int],
            order: int,
        ):
            if not any(angular_a + angular_b + angular_c + angular_d):
                return mp.hyp1f1(
                    order + mp.mpf("0.5"),
                    order + mp.mpf("1.5"),
                    -argument,
                ) / (2 * order + 1)

            values = (angular_a, angular_b, angular_c, angular_d)
            for center_index, value in enumerate(values):
                for axis in range(3):
                    if value[axis] == 0:
                        continue
                    parent = list(values)
                    parent[center_index] = lower(value, axis)
                    parent_tuple = tuple(parent)
                    if center_index < 2:
                        center_term = complex_p[axis] - centers[center_index][axis]
                        w_term = weighted_center[axis] - complex_p[axis]
                        pair_exponent = p
                        same_pair = (0, 1)
                        other_pair = (2, 3)
                    else:
                        center_term = complex_q[axis] - centers[center_index][axis]
                        w_term = weighted_center[axis] - complex_q[axis]
                        pair_exponent = q
                        same_pair = (2, 3)
                        other_pair = (0, 1)

                    result = center_term * integral(*parent_tuple, order)
                    result += w_term * integral(*parent_tuple, order + 1)
                    for index in same_pair:
                        coefficient = parent_tuple[index][axis]
                        if coefficient:
                            reduced = list(parent_tuple)
                            reduced[index] = lower(reduced[index], axis)
                            result += (
                                coefficient
                                / (2 * pair_exponent)
                                * (
                                    integral(*tuple(reduced), order)
                                    - rho
                                    / pair_exponent
                                    * integral(*tuple(reduced), order + 1)
                                )
                            )
                    for index in other_pair:
                        coefficient = parent_tuple[index][axis]
                        if coefficient:
                            reduced = list(parent_tuple)
                            reduced[index] = lower(reduced[index], axis)
                            result += (
                                coefficient
                                / (2 * (p + q))
                                * integral(*tuple(reduced), order + 1)
                            )
                    return result
            raise AssertionError("unreachable angular-momentum state")

        return common * integral(*angular, 0)
