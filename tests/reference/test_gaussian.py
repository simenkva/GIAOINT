import math
from dataclasses import FrozenInstanceError

import pytest
from giao_reference import (
    MagneticField,
    PrimitiveGaussian,
    Shell,
    cartesian_components,
    double_factorial,
    gaussian_value,
    primitive_normalization,
    primitive_overlap,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-1, 1), (0, 1), (1, 1), (2, 2), (5, 15), (8, 384)],
)
def test_double_factorial(value, expected):
    assert double_factorial(value) == expected


@pytest.mark.parametrize("value", [-3, 1.5, True])
def test_double_factorial_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        double_factorial(value)


def test_cartesian_component_ordering():
    assert cartesian_components(0) == ((0, 0, 0),)
    assert cartesian_components(1) == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert cartesian_components(2) == (
        (2, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 2, 0),
        (0, 1, 1),
        (0, 0, 2),
    )


@pytest.mark.parametrize("angular_momentum", range(11))
def test_cartesian_component_count_and_partition(angular_momentum):
    components = cartesian_components(angular_momentum)
    assert len(components) == (angular_momentum + 1) * (angular_momentum + 2) // 2
    assert len(set(components)) == len(components)
    assert all(sum(component) == angular_momentum for component in components)


@pytest.mark.parametrize("bad_value", [-1, 1.2, True])
def test_cartesian_components_reject_invalid_values(bad_value):
    with pytest.raises(ValueError):
        cartesian_components(bad_value)


def test_known_s_normalization():
    exponent = 0.73
    expected = (2.0 * exponent / math.pi) ** 0.75
    assert primitive_normalization(exponent, (0, 0, 0)) == pytest.approx(expected)


def test_all_components_through_l6_are_normalized():
    for total_angular_momentum in range(7):
        for angular in cartesian_components(total_angular_momentum):
            primitive = PrimitiveGaussian(0.83, (0.2, -0.4, 0.7), angular)
            assert primitive_overlap(primitive, primitive) == pytest.approx(
                1.0, abs=3.0e-14, rel=3.0e-14
            )


def test_london_wave_vector_includes_gauge_origin():
    field = MagneticField(B=(0.2, -0.4, 0.7), gauge_origin=(-0.3, 0.1, 0.5))
    center = (0.9, -0.2, 1.0)
    assert field.london_wave_vector(center) == pytest.approx((0.005, 0.37, 0.21))


def test_gaussian_value_uses_frozen_phase_sign():
    primitive = PrimitiveGaussian(
        exponent=0.5,
        center=(1.0, 0.0, 0.0),
        normalized=False,
    )
    field = MagneticField(B=(0.0, 0.0, 2.0))
    value = gaussian_value(primitive, (1.0, 1.0, 0.0), field)
    assert value == pytest.approx(
        0.6065306597126334 * complex(0.5403023058681398, -0.8414709848078965)
    )


def test_value_objects_are_frozen():
    primitive = PrimitiveGaussian(1.0, (0.0, 0.0, 0.0))
    field = MagneticField()
    shell = Shell((0.0, 0.0, 0.0), 0, (1.0,), (1.0,))
    with pytest.raises(FrozenInstanceError):
        primitive.exponent = 2.0
    with pytest.raises(FrozenInstanceError):
        field.B = (1.0, 0.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        shell.angular_momentum = 1


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: PrimitiveGaussian(0.0, (0.0, 0.0, 0.0)),
        lambda: PrimitiveGaussian(1.0, (0.0, 0.0)),
        lambda: PrimitiveGaussian(1.0, (0.0, 0.0, 0.0), (-1, 0, 0)),
        lambda: MagneticField(B=(0.0, float("nan"), 0.0)),
        lambda: Shell((0.0, 0.0, 0.0), -1, (1.0,), (1.0,)),
        lambda: Shell((0.0, 0.0, 0.0), 0, (), ()),
        lambda: Shell((0.0, 0.0, 0.0), 0, (1.0,), (1.0, 2.0)),
        lambda: Shell((0.0, 0.0, 0.0), 0, (1.0,), (0.0,)),
    ],
)
def test_data_models_reject_invalid_inputs(constructor):
    with pytest.raises(ValueError):
        constructor()
