from __future__ import annotations

from itertools import combinations_with_replacement

import giao_integrals as gi
import giao_reference as ref
import numpy as np
import pytest

CENTERS = ((-0.3, 0.2, 0.5), (0.6, -0.4, 0.1), (0.2, 0.7, -0.6), (-0.5, -0.1, 0.4))
EXPONENTS = (0.7, 1.2, 0.9, 1.5)
FIELD = (0.7, -0.4, 0.3)
GAUGE_ORIGIN = (0.1, 0.2, -0.5)


def primitives(angular, *, normalized=False):
    production = [
        gi.PrimitiveGaussian(
            EXPONENTS[index], CENTERS[index], angular[index], normalized=normalized
        )
        for index in range(4)
    ]
    reference = [
        ref.PrimitiveGaussian(
            EXPONENTS[index], CENTERS[index], angular[index], normalized=normalized
        )
        for index in range(4)
    ]
    return production, reference


def compare_reference(angular, *, field=FIELD, atol=2.0e-12):
    production, reference = primitives(angular)
    actual = gi.primitive_eri(*production, field=gi.MagneticField(field, GAUGE_ORIGIN))
    expected = complex(
        ref.primitive_eri(
            *reference,
            field=ref.MagneticField(field, GAUGE_ORIGIN),
            dps=80,
        )
    )
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=2.0e-11)


def test_finite_field_ssss_formula_against_high_precision_reference():
    compare_reference(((0, 0, 0),) * 4, atol=2.0e-13)


def test_cartesian_recurrence_exhaustive_through_combined_degree_two():
    # This is the documented exhaustive M5 core: every distribution of zero,
    # one, or two Cartesian powers over all 12 center/axis positions (91 ERIs).
    angular_cases = []
    for degree in range(3):
        for positions in combinations_with_replacement(range(12), degree):
            powers = [0] * 12
            for position in positions:
                powers[position] += 1
            angular_cases.append(
                tuple(tuple(powers[3 * i : 3 * i + 3]) for i in range(4))
            )
    for angular in angular_cases:
        compare_reference(angular)


@pytest.mark.parametrize(
    "angular",
    [
        ((4, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)),
        ((2, 1, 1), (0, 2, 1), (1, 0, 2), (1, 1, 0)),
        ((0, 0, 3), (2, 0, 0), (0, 2, 0), (0, 0, 1)),
    ],
)
def test_selected_higher_cartesian_cases(angular):
    compare_reference(angular, atol=8.0e-12)


def test_exact_finite_field_symmetries_and_invalid_one_pair_swap():
    angular = ((2, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1))
    values, _ = primitives(angular)
    field = gi.MagneticField(FIELD, GAUGE_ORIGIN)
    a, b, c, d = values
    abcd = gi.primitive_eri(a, b, c, d, field=field)
    np.testing.assert_allclose(
        abcd, gi.primitive_eri(c, d, a, b, field=field), atol=2.0e-12
    )
    np.testing.assert_allclose(
        abcd.conjugate(),
        gi.primitive_eri(b, a, d, c, field=field),
        atol=2.0e-12,
    )
    assert abs(abcd - gi.primitive_eri(b, a, c, d, field=field)) > 1.0e-6


def test_scaled_negative_real_boys_seed_against_reference():
    centers = ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))
    angular = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0))
    prod = [
        gi.PrimitiveGaussian(0.8, center, power, normalized=False)
        for center, power in zip(centers, angular, strict=True)
    ]
    oracle = [
        ref.PrimitiveGaussian(0.8, center, power, normalized=False)
        for center, power in zip(centers, angular, strict=True)
    ]
    actual = gi.primitive_eri(*prod, field=gi.MagneticField((0.0, 0.0, 4.0)))
    expected = complex(
        ref.primitive_eri(*oracle, field=ref.MagneticField((0.0, 0.0, 4.0)), dps=100)
    )
    assert np.isfinite(actual)
    np.testing.assert_allclose(actual, expected, atol=2.0e-12, rtol=3.0e-11)
