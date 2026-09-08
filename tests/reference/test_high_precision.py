import ast
import pathlib

import mpmath as mp
import pytest
from giao_reference import MagneticField, PrimitiveGaussian, primitive_overlap
from giao_reference.high_precision import (
    primitive_overlap_formula,
    primitive_overlap_quadrature,
)
from hypothesis import given, settings
from hypothesis import strategies as st


def assert_matches_mpmath(actual, expected, *, atol=5.0e-13, rtol=1.0e-12):
    expected_complex = complex(expected)
    assert abs(actual - expected_complex) <= atol + rtol * abs(expected_complex)


@pytest.mark.parametrize(
    ("bra", "ket", "field"),
    [
        (
            PrimitiveGaussian(0.7, (-0.4, 0.2, 0.1)),
            PrimitiveGaussian(1.3, (0.8, -0.5, 0.9)),
            MagneticField(B=(0.3, -0.2, 0.5)),
        ),
        (
            PrimitiveGaussian(0.7, (-0.4, 0.2, 0.1), (2, 1, 0)),
            PrimitiveGaussian(1.3, (0.8, -0.5, 0.9), (1, 0, 2)),
            MagneticField(B=(0.3, -0.2, 0.5), gauge_origin=(4.0, -1.0, 3.0)),
        ),
        (
            PrimitiveGaussian(0.22, (-1.1, 0.3, 0.7), (4, 0, 0)),
            PrimitiveGaussian(2.8, (0.2, -0.8, -0.4), (0, 3, 1)),
            MagneticField(B=(-0.6, 0.4, 0.9)),
        ),
        (
            PrimitiveGaussian(1.5, (0.2, 0.2, 0.2), (1, 1, 1)),
            PrimitiveGaussian(1.5, (0.2, 0.2, 0.2), (1, 1, 1)),
            MagneticField(B=(2.0, -1.0, 0.4)),
        ),
    ],
)
def test_formula_matches_direct_real_axis_quadrature(bra, ket, field):
    formula = primitive_overlap_formula(bra, ket, field, dps=80)
    quadrature = primitive_overlap_quadrature(bra, ket, field, dps=80)
    with mp.workdps(80):
        assert abs(formula - quadrature) < mp.mpf("1.0e-60")


@st.composite
def angular_components(draw):
    total = draw(st.integers(0, 6))
    angular_x = draw(st.integers(0, total))
    angular_y = draw(st.integers(0, total - angular_x))
    return angular_x, angular_y, total - angular_x - angular_y


@given(
    exponent_bra=st.floats(0.1, 5.0, allow_nan=False, allow_infinity=False),
    exponent_ket=st.floats(0.1, 5.0, allow_nan=False, allow_infinity=False),
    center_values=st.lists(
        st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
        min_size=9,
        max_size=9,
    ),
    angular_bra=angular_components(),
    angular_ket=angular_components(),
)
@settings(max_examples=60, deadline=None)
def test_randomized_double_matches_80_digit_formula(
    exponent_bra,
    exponent_ket,
    center_values,
    angular_bra,
    angular_ket,
):
    bra = PrimitiveGaussian(exponent_bra, tuple(center_values[:3]), angular_bra)
    ket = PrimitiveGaussian(exponent_ket, tuple(center_values[3:6]), angular_ket)
    field = MagneticField(tuple(center_values[6:9]))
    high_precision = primitive_overlap_formula(bra, ket, field, dps=80)
    assert_matches_mpmath(primitive_overlap(bra, ket, field), high_precision)


def test_reference_package_does_not_import_production_extension():
    package_root = pathlib.Path(__file__).parents[2] / "reference/python/giao_reference"
    forbidden_roots = {"giao_integrals", "_giao_integrals", "pybind11"}
    for source_path in package_root.glob("*.py"):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not (
                    {alias.name.split(".")[0] for alias in node.names} & forbidden_roots
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden_roots
