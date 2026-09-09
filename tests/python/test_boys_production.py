import cmath

import giao_integrals as gi
import numpy as np
import pytest
from giao_reference import boys_hypergeometric


def assert_mixed_close(actual, expected, *, atol=2.0e-14, rtol=5.0e-13):
    assert abs(actual - expected) <= atol + rtol * abs(expected)


@pytest.mark.parametrize(
    "argument",
    [
        0.0,
        1.0e-14 + 1.0e-14j,
        0.749999,
        0.750001,
        1.0 + 4.0j,
        20.0 - 35.0j,
        -1.0 + 2.0j,
        -30.0 - 12.0j,
        -100.0 + 20.0j,
        123.999,
        124.001,
        130.0 + 10.0j,
        159.999j,
        1000.0 - 25.0j,
    ],
)
@pytest.mark.parametrize("maximum_order", [0, 3, 12, 32])
def test_complex_plane_map_against_90_digit_reference(argument, maximum_order):
    values = gi.boys(argument, maximum_order)
    for order, actual in enumerate(values):
        expected = complex(boys_hypergeometric(order, argument, dps=90))
        assert_mixed_close(actual, expected, atol=3.0e-14, rtol=8.0e-13)


@pytest.mark.parametrize("argument", [0.2 + 0.4j, 3.0 - 7.0j, -12.0 + 6.0j])
def test_recurrence_residual_and_conjugation(argument):
    values = gi.boys(argument, 20)
    conjugated = gi.boys(argument.conjugate(), 20)
    np.testing.assert_allclose(conjugated, values.conj(), atol=2e-14, rtol=4e-13)
    exponential = cmath.exp(-argument)
    for order in range(20):
        residual = (2 * order + 1) * values[order] - exponential
        residual -= 2 * argument * values[order + 1]
        scale = abs((2 * order + 1) * values[order]) + abs(exponential)
        assert abs(residual) <= 3.0e-13 * max(1.0, scale)


@pytest.mark.parametrize("argument", [-0.3 + 0.2j, -20.0 + 5.0j, -120.0 - 8.0j])
def test_scaled_and_unscaled_agree(argument):
    unscaled = gi.boys(argument, 16)
    scaled = gi.boys(argument, 16, scaled=True)
    np.testing.assert_allclose(
        scaled, cmath.exp(argument) * unscaled, atol=2e-14, rtol=8e-13
    )


def test_dispatch_diagnostics_cover_regions():
    _, small = gi.boys_with_diagnostics(0.2 + 0.1j, 4)
    _, moderate = gi.boys_with_diagnostics(3.0 + 2.0j, 4)
    _, negative = gi.boys_with_diagnostics(-3.0 + 2.0j, 4)
    _, large = gi.boys_with_diagnostics(1000.0 + 2.0j, 4)
    assert small["region"] == gi.BoysRegion.POWER_SERIES
    assert moderate["region"] == gi.BoysRegion.ADAPTIVE_QUADRATURE
    assert negative["region"] == gi.BoysRegion.SCALED_QUADRATURE
    assert large["region"] == gi.BoysRegion.POSITIVE_ASYMPTOTIC
    assert moderate["quadrature_segments"] >= 1
    assert moderate["estimated_absolute_error"] >= 0.0


def test_outside_verified_sector_fails_diagnostically():
    with pytest.raises(gi.BoysNumericalError, match="outside the verified"):
        gi.boys(10.0 + 500.0j, 4)
    with pytest.raises(gi.BoysNumericalError, match="outside the verified"):
        gi.boys(160.001j, 4)
    with pytest.raises(ValueError, match="through 32"):
        gi.boys(1.0, 33)
