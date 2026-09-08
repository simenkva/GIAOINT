import cmath
import itertools

import pytest
from giao_reference import (
    MagneticField,
    PrimitiveGaussian,
    Shell,
    cartesian_components,
    contracted_component_overlap,
    gaussian_product,
    primitive_overlap,
    shell_overlap,
    ss_overlap,
)
from giao_reference.overlap import contraction_normalization
from hypothesis import given, settings
from hypothesis import strategies as st


def assert_complex_close(actual, expected, *, atol=5.0e-13, rtol=5.0e-13):
    assert abs(actual - expected) <= atol + rtol * abs(expected)


@st.composite
def angular_components(draw, maximum_l=6):
    total = draw(st.integers(min_value=0, max_value=maximum_l))
    angular_x = draw(st.integers(min_value=0, max_value=total))
    angular_y = draw(st.integers(min_value=0, max_value=total - angular_x))
    return angular_x, angular_y, total - angular_x - angular_y


finite_float = st.floats(
    min_value=-2.0,
    max_value=2.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
positive_exponent = st.floats(
    min_value=0.15,
    max_value=4.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
vector = st.tuples(finite_float, finite_float, finite_float)


def test_gaussian_product_known_values_and_signs():
    bra = PrimitiveGaussian(0.7, (-0.4, 0.2, 0.8))
    ket = PrimitiveGaussian(1.3, (0.8, -0.5, 0.1))
    field = MagneticField(B=(0.3, -0.2, 0.5), gauge_origin=(9.0, -4.0, 2.0))
    pair = gaussian_product(bra, ket, field)

    expected_q = tuple(
        0.5 * value
        for value in (
            0.49,
            0.81,
            0.03,
        )
    )
    assert pair.exponent == pytest.approx(2.0)
    assert pair.reduced_exponent == pytest.approx(0.455)
    assert pair.product_center == pytest.approx((0.38, -0.255, 0.345))
    assert pair.pair_wave_vector == pytest.approx(expected_q)
    assert pair.complex_center == pytest.approx(
        tuple(pair.product_center[i] - 0.25j * expected_q[i] for i in range(3))
    )


@given(
    exponent_bra=positive_exponent,
    exponent_ket=positive_exponent,
    center_bra=vector,
    center_ket=vector,
    magnetic_field=vector,
)
@settings(max_examples=80, deadline=None)
def test_general_ss_formula_matches_closed_form(
    exponent_bra,
    exponent_ket,
    center_bra,
    center_ket,
    magnetic_field,
):
    bra = PrimitiveGaussian(exponent_bra, center_bra)
    ket = PrimitiveGaussian(exponent_ket, center_ket)
    field = MagneticField(magnetic_field)
    assert_complex_close(
        primitive_overlap(bra, ket, field), ss_overlap(bra, ket, field)
    )


@given(
    exponent_bra=positive_exponent,
    exponent_ket=positive_exponent,
    center_bra=vector,
    center_ket=vector,
    angular_bra=angular_components(),
    angular_ket=angular_components(),
    magnetic_field=vector,
    gauge_origin=vector,
)
@settings(max_examples=120, deadline=None)
def test_randomized_hermiticity_through_l6(
    exponent_bra,
    exponent_ket,
    center_bra,
    center_ket,
    angular_bra,
    angular_ket,
    magnetic_field,
    gauge_origin,
):
    bra = PrimitiveGaussian(exponent_bra, center_bra, angular_bra)
    ket = PrimitiveGaussian(exponent_ket, center_ket, angular_ket)
    field = MagneticField(magnetic_field, gauge_origin)
    forward = primitive_overlap(bra, ket, field)
    reverse = primitive_overlap(ket, bra, field)
    assert_complex_close(forward, reverse.conjugate(), atol=2.0e-12, rtol=2.0e-12)


def test_exhaustive_component_hermiticity_through_l4():
    components = tuple(
        itertools.chain.from_iterable(cartesian_components(value) for value in range(5))
    )
    field = MagneticField(B=(0.19, -0.31, 0.27), gauge_origin=(0.4, -0.2, 0.1))
    for angular_bra in components:
        bra = PrimitiveGaussian(0.91, (-0.3, 0.2, 0.5), angular_bra)
        for angular_ket in components:
            ket = PrimitiveGaussian(1.17, (0.6, -0.4, 0.1), angular_ket)
            forward = primitive_overlap(bra, ket, field)
            reverse = primitive_overlap(ket, bra, field)
            assert_complex_close(
                forward,
                reverse.conjugate(),
                atol=3.0e-12,
                rtol=3.0e-12,
            )


@given(
    center_bra=vector,
    center_ket=vector,
    angular_bra=angular_components(),
    angular_ket=angular_components(),
    magnetic_field=vector,
    first_origin=vector,
    second_origin=vector,
)
@settings(max_examples=80, deadline=None)
def test_overlap_is_independent_of_gauge_origin(
    center_bra,
    center_ket,
    angular_bra,
    angular_ket,
    magnetic_field,
    first_origin,
    second_origin,
):
    bra = PrimitiveGaussian(0.67, center_bra, angular_bra)
    ket = PrimitiveGaussian(1.29, center_ket, angular_ket)
    first = primitive_overlap(bra, ket, MagneticField(magnetic_field, first_origin))
    second = primitive_overlap(bra, ket, MagneticField(magnetic_field, second_origin))
    assert_complex_close(first, second, atol=2.0e-13, rtol=2.0e-13)


@given(
    center_bra=vector,
    center_ket=vector,
    angular_bra=angular_components(maximum_l=4),
    angular_ket=angular_components(maximum_l=4),
    magnetic_field=vector,
    gauge_origin=vector,
    translation=vector,
)
@settings(max_examples=100, deadline=None)
def test_common_translation_has_expected_ao_rephasing(
    center_bra,
    center_ket,
    angular_bra,
    angular_ket,
    magnetic_field,
    gauge_origin,
    translation,
):
    bra = PrimitiveGaussian(0.79, center_bra, angular_bra)
    ket = PrimitiveGaussian(1.41, center_ket, angular_ket)
    field = MagneticField(magnetic_field, gauge_origin)
    original = primitive_overlap(bra, ket, field)
    pair_wave_vector = gaussian_product(bra, ket, field).pair_wave_vector

    translated_bra = PrimitiveGaussian(
        bra.exponent,
        tuple(center_bra[i] + translation[i] for i in range(3)),
        angular_bra,
    )
    translated_ket = PrimitiveGaussian(
        ket.exponent,
        tuple(center_ket[i] + translation[i] for i in range(3)),
        angular_ket,
    )
    translated_field = MagneticField(
        magnetic_field,
        tuple(gauge_origin[i] + translation[i] for i in range(3)),
    )
    translated = primitive_overlap(translated_bra, translated_ket, translated_field)
    phase = cmath.exp(-1j * sum(pair_wave_vector[i] * translation[i] for i in range(3)))
    assert_complex_close(translated, phase * original, atol=3.0e-12, rtol=3.0e-12)


def test_zero_field_is_real_and_is_the_continuous_limit():
    bra = PrimitiveGaussian(0.48, (-0.7, 0.1, 0.2), (3, 1, 0))
    ket = PrimitiveGaussian(1.62, (0.6, -0.3, 0.9), (1, 2, 2))
    zero_field = primitive_overlap(bra, ket)
    assert zero_field.imag == pytest.approx(0.0, abs=1.0e-15)

    previous_error = None
    for magnitude in (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5):
        finite_field = primitive_overlap(
            bra,
            ket,
            MagneticField(B=(0.3 * magnitude, -0.7 * magnitude, magnitude)),
        )
        error = abs(finite_field - zero_field)
        if previous_error is not None:
            assert error < previous_error
        previous_error = error


def test_segmented_shell_normalization_and_block_ordering():
    shell = Shell(
        center=(0.1, -0.3, 0.2),
        angular_momentum=3,
        exponents=(3.2, 0.9, 0.24),
        coefficients=(0.18, -0.42, 0.73),
    )
    block = shell_overlap(shell, shell, MagneticField(B=(0.2, 0.1, -0.3)))
    assert len(block) == shell.ao_count == 10
    assert all(len(row) == shell.ao_count for row in block)
    for row in range(shell.ao_count):
        assert block[row][row] == pytest.approx(1.0, abs=2.0e-13, rel=2.0e-13)
        for column in range(shell.ao_count):
            assert_complex_close(block[row][column], block[column][row].conjugate())


def test_contraction_normalization_can_be_disabled():
    normalized = Shell((0.0, 0.0, 0.0), 0, (1.0, 0.3), (0.4, 0.8))
    as_provided = Shell(
        (0.0, 0.0, 0.0),
        0,
        (1.0, 0.3),
        (0.4, 0.8),
        normalize=False,
    )
    assert contraction_normalization(normalized) != pytest.approx(1.0)
    assert contraction_normalization(as_provided) == 1.0
    assert contracted_component_overlap(
        normalized, (0, 0, 0), normalized, (0, 0, 0)
    ) == pytest.approx(1.0)
    assert contracted_component_overlap(
        as_provided, (0, 0, 0), as_provided, (0, 0, 0)
    ) != pytest.approx(1.0)


def test_component_must_belong_to_shell():
    shell = Shell((0.0, 0.0, 0.0), 1, (1.0,), (1.0,))
    with pytest.raises(ValueError):
        contracted_component_overlap(shell, (0, 0, 0), shell, (1, 0, 0))
