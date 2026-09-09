import itertools
import random

import giao_integrals as gi
import numpy as np
import pytest
from giao_reference import MagneticField as ReferenceField
from giao_reference import PrimitiveGaussian as ReferencePrimitive
from giao_reference import Shell as ReferenceShell
from giao_reference import cartesian_components as reference_components
from giao_reference import primitive_overlap as reference_primitive_overlap
from giao_reference import shell_overlap as reference_shell_overlap
from giao_reference.high_precision import primitive_overlap_formula


def assert_complex_close(actual, expected, *, atol=5.0e-14, rtol=1.0e-12):
    assert abs(actual - expected) <= atol + rtol * abs(expected)


def test_exhaustive_cartesian_components_through_l4():
    components = tuple(
        itertools.chain.from_iterable(reference_components(value) for value in range(5))
    )
    field_values = (0.19, -0.31, 0.27)
    origin = (0.4, -0.2, 0.1)
    field = gi.MagneticField(field_values, origin)
    reference_field = ReferenceField(field_values, origin)
    for angular_bra in components:
        bra = gi.PrimitiveGaussian(0.91, (-0.3, 0.2, 0.5), angular_bra)
        reference_bra = ReferencePrimitive(0.91, (-0.3, 0.2, 0.5), angular_bra)
        for angular_ket in components:
            ket = gi.PrimitiveGaussian(1.17, (0.6, -0.4, 0.1), angular_ket)
            reference_ket = ReferencePrimitive(1.17, (0.6, -0.4, 0.1), angular_ket)
            assert_complex_close(
                gi.primitive_overlap(bra, ket, field=field),
                reference_primitive_overlap(
                    reference_bra, reference_ket, reference_field
                ),
            )


def test_randomized_primitive_comparisons_through_l6():
    generator = random.Random(1940521)
    for _ in range(300):
        exponent_bra = 10 ** generator.uniform(-0.8, 0.6)
        exponent_ket = 10 ** generator.uniform(-0.8, 0.6)
        center_bra = tuple(generator.uniform(-2.0, 2.0) for _ in range(3))
        center_ket = tuple(generator.uniform(-2.0, 2.0) for _ in range(3))
        field_values = tuple(generator.uniform(-1.0, 1.0) for _ in range(3))
        origin = tuple(generator.uniform(-2.0, 2.0) for _ in range(3))
        angular_bra = generator.choice(reference_components(generator.randrange(7)))
        angular_ket = generator.choice(reference_components(generator.randrange(7)))
        coefficient_bra = generator.uniform(-1.5, 1.5)
        coefficient_ket = generator.uniform(-1.5, 1.5)

        bra = gi.PrimitiveGaussian(
            exponent_bra,
            center_bra,
            angular_bra,
            coefficient_bra,
        )
        ket = gi.PrimitiveGaussian(
            exponent_ket,
            center_ket,
            angular_ket,
            coefficient_ket,
        )
        reference_bra = ReferencePrimitive(
            exponent_bra,
            center_bra,
            angular_bra,
            coefficient_bra,
        )
        reference_ket = ReferencePrimitive(
            exponent_ket,
            center_ket,
            angular_ket,
            coefficient_ket,
        )
        assert_complex_close(
            gi.primitive_overlap(
                bra, ket, field=gi.MagneticField(field_values, origin)
            ),
            reference_primitive_overlap(
                reference_bra, reference_ket, ReferenceField(field_values, origin)
            ),
            atol=2.0e-13,
            rtol=2.0e-11,
        )


def test_randomized_cpp_matches_80_digit_reference_through_l6():
    generator = random.Random(806417)
    for _ in range(60):
        exponent_bra = 10 ** generator.uniform(-0.8, 0.6)
        exponent_ket = 10 ** generator.uniform(-0.8, 0.6)
        center_bra = tuple(generator.uniform(-1.5, 1.5) for _ in range(3))
        center_ket = tuple(generator.uniform(-1.5, 1.5) for _ in range(3))
        field_values = tuple(generator.uniform(-0.8, 0.8) for _ in range(3))
        origin = tuple(generator.uniform(-1.5, 1.5) for _ in range(3))
        angular_bra = generator.choice(reference_components(generator.randrange(7)))
        angular_ket = generator.choice(reference_components(generator.randrange(7)))
        reference_bra = ReferencePrimitive(exponent_bra, center_bra, angular_bra)
        reference_ket = ReferencePrimitive(exponent_ket, center_ket, angular_ket)
        reference_field = ReferenceField(field_values, origin)
        actual = gi.primitive_overlap(
            gi.PrimitiveGaussian(exponent_bra, center_bra, angular_bra),
            gi.PrimitiveGaussian(exponent_ket, center_ket, angular_ket),
            field=gi.MagneticField(field_values, origin),
        )
        expected = complex(
            primitive_overlap_formula(
                reference_bra, reference_ket, reference_field, dps=80
            )
        )
        assert_complex_close(actual, expected, atol=2.0e-13, rtol=2.0e-11)


@pytest.mark.parametrize("angular_a,angular_b", [(0, 0), (1, 2), (3, 4), (5, 2)])
def test_segmented_shell_blocks_match_reference(angular_a, angular_b):
    center_a = (-0.3, 0.2, 0.5)
    center_b = (0.6, -0.4, 0.1)
    exponents_a = (2.4, 0.7, 0.18)
    exponents_b = (1.9, 0.44)
    coefficients_a = (0.13, -0.47, 0.81)
    coefficients_b = (-0.22, 0.93)
    field_values = (0.23, -0.11, 0.37)
    origin = (-0.4, 0.8, 0.2)

    actual = gi.overlap_shell(
        gi.Shell(center_a, angular_a, exponents_a, coefficients_a),
        gi.Shell(center_b, angular_b, exponents_b, coefficients_b),
        field=gi.MagneticField(field_values, origin),
    )
    expected = np.asarray(
        reference_shell_overlap(
            ReferenceShell(center_a, angular_a, exponents_a, coefficients_a),
            ReferenceShell(center_b, angular_b, exponents_b, coefficients_b),
            ReferenceField(field_values, origin),
        )
    )
    np.testing.assert_allclose(actual, expected, atol=2.0e-13, rtol=2.0e-12)


def test_general_contraction_rows_match_segmented_reference_blocks():
    center_a = (-0.3, 0.2, 0.5)
    center_b = (0.6, -0.4, 0.1)
    exponents_a = (2.4, 0.7, 0.18)
    exponents_b = (1.9, 0.44)
    coefficients_a = ((0.13, -0.47, 0.81), (0.52, 0.31, -0.28))
    coefficients_b = ((-0.22, 0.93), (0.71, 0.19))
    field_values = (0.23, -0.11, 0.37)
    origin = (-0.4, 0.8, 0.2)

    actual = gi.overlap_shell(
        gi.Shell(center_a, 2, exponents_a, coefficients_a),
        gi.Shell(center_b, 1, exponents_b, coefficients_b),
        field=gi.MagneticField(field_values, origin),
    )
    expected = np.empty_like(actual)
    count_a = len(reference_components(2))
    count_b = len(reference_components(1))
    for contraction_a, row_a in enumerate(coefficients_a):
        for contraction_b, row_b in enumerate(coefficients_b):
            block = reference_shell_overlap(
                ReferenceShell(center_a, 2, exponents_a, row_a),
                ReferenceShell(center_b, 1, exponents_b, row_b),
                ReferenceField(field_values, origin),
            )
            expected[
                contraction_a * count_a : (contraction_a + 1) * count_a,
                contraction_b * count_b : (contraction_b + 1) * count_b,
            ] = block
    np.testing.assert_allclose(actual, expected, atol=2.0e-13, rtol=2.0e-12)


def test_contraction_normalization_can_be_disabled():
    center = (0.1, -0.3, 0.2)
    exponents = (3.2, 0.9, 0.24)
    coefficients = (0.18, -0.42, 0.73)
    actual = gi.overlap_shell(
        gi.Shell(center, 2, exponents, coefficients, normalize=False),
        gi.Shell(center, 2, exponents, coefficients, normalize=False),
    )
    expected = np.asarray(
        reference_shell_overlap(
            ReferenceShell(center, 2, exponents, coefficients, normalize=False),
            ReferenceShell(center, 2, exponents, coefficients, normalize=False),
        )
    )
    np.testing.assert_allclose(actual, expected, atol=2.0e-13, rtol=2.0e-12)
    assert not np.allclose(np.diag(actual), 1.0)


def test_zero_field_is_real_and_finite_field_is_hermitian():
    shells = [
        gi.Shell((-0.3, 0.2, 0.5), 0, [2.4, 0.7], [0.2, 0.8]),
        gi.Shell((0.6, -0.4, 0.1), 2, [1.9, 0.44], [-0.22, 0.93]),
        gi.Shell((-0.8, 0.1, 0.7), 4, [0.83], [1.0]),
    ]
    basis = gi.Basis(shells)
    zero = gi.overlap(basis)
    assert np.max(np.abs(zero.imag)) < 2.0e-15
    finite = gi.overlap(
        basis, field=gi.MagneticField((0.23, -0.11, 0.37), (-0.4, 0.8, 0.2))
    )
    np.testing.assert_allclose(finite, finite.conj().T, atol=3.0e-13, rtol=3.0e-13)


def test_gauge_origin_independence_and_translation_rephasing():
    a = gi.PrimitiveGaussian(0.79, (-0.6, 0.2, 0.7), (3, 1, 0))
    b = gi.PrimitiveGaussian(1.41, (0.5, -0.4, 0.3), (1, 2, 2))
    field_values = (0.3, -0.7, 1.0)
    first = gi.primitive_overlap(
        a, b, field=gi.MagneticField(field_values, (0.1, -0.2, 0.4))
    )
    second = gi.primitive_overlap(
        a, b, field=gi.MagneticField(field_values, (-0.7, 0.6, 0.3))
    )
    assert_complex_close(first, second, atol=2.0e-13, rtol=2.0e-13)

    translation = (0.8, -0.5, 0.2)
    translated_a = gi.PrimitiveGaussian(
        a.exponent,
        tuple(a.center[index] + translation[index] for index in range(3)),
        a.angular,
    )
    translated_b = gi.PrimitiveGaussian(
        b.exponent,
        tuple(b.center[index] + translation[index] for index in range(3)),
        b.angular,
    )
    origin = (0.1, -0.2, 0.4)
    translated = gi.primitive_overlap(
        translated_a,
        translated_b,
        field=gi.MagneticField(
            field_values,
            tuple(origin[index] + translation[index] for index in range(3)),
        ),
    )
    kappa_a = np.asarray(
        gi.MagneticField(field_values, origin).london_wave_vector(a.center)
    )
    kappa_b = np.asarray(
        gi.MagneticField(field_values, origin).london_wave_vector(b.center)
    )
    phase = np.exp(-1j * np.dot(kappa_b - kappa_a, translation))
    assert_complex_close(translated, phase * first, atol=5.0e-13, rtol=5.0e-13)
