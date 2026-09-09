import cmath
import random

import giao_integrals as gi
import numpy as np
import pytest
from giao_reference import MagneticField as ReferenceField
from giao_reference import PrimitiveGaussian as ReferencePrimitive
from giao_reference import Shell as ReferenceShell
from giao_reference import gaussian_value
from giao_reference import primitive_gradient as reference_gradient
from giao_reference import primitive_kinetic as reference_kinetic
from giao_reference import (
    primitive_magnetic_kinetic as reference_magnetic_kinetic,
)
from giao_reference import primitive_moment as reference_moment
from giao_reference import primitive_momentum as reference_momentum
from giao_reference import shell_gradient as reference_shell_gradient
from giao_reference import shell_kinetic as reference_shell_kinetic
from giao_reference import (
    shell_magnetic_kinetic as reference_shell_magnetic_kinetic,
)
from giao_reference import shell_moment as reference_shell_moment
from giao_reference import shell_momentum as reference_shell_momentum


def assert_complex_close(actual, expected, *, atol=2.0e-13, rtol=2.0e-11):
    assert abs(actual - expected) <= atol + rtol * abs(expected)


def random_component(generator, maximum_l=6):
    total = generator.randrange(maximum_l + 1)
    x = generator.randrange(total + 1)
    y = generator.randrange(total - x + 1)
    return x, y, total - x - y


def make_primitive_pair(generator):
    exponent_bra = 10 ** generator.uniform(-0.7, 0.6)
    exponent_ket = 10 ** generator.uniform(-0.7, 0.6)
    center_bra = tuple(generator.uniform(-1.5, 1.5) for _ in range(3))
    center_ket = tuple(generator.uniform(-1.5, 1.5) for _ in range(3))
    angular_bra = random_component(generator)
    angular_ket = random_component(generator)
    production = (
        gi.PrimitiveGaussian(exponent_bra, center_bra, angular_bra),
        gi.PrimitiveGaussian(exponent_ket, center_ket, angular_ket),
    )
    reference = (
        ReferencePrimitive(exponent_bra, center_bra, angular_bra),
        ReferencePrimitive(exponent_ket, center_ket, angular_ket),
    )
    return production, reference


def test_randomized_primitive_properties_against_direct_reference_through_l6():
    generator = random.Random(3197781)
    for _ in range(100):
        (bra, ket), (reference_bra, reference_ket) = make_primitive_pair(generator)
        field_values = tuple(generator.uniform(-0.8, 0.8) for _ in range(3))
        gauge_origin = tuple(generator.uniform(-1.2, 1.2) for _ in range(3))
        field = gi.MagneticField(field_values, gauge_origin)
        reference_field = ReferenceField(field_values, gauge_origin)
        powers = tuple(generator.randrange(3) for _ in range(3))
        origin = tuple(generator.uniform(-1.0, 1.0) for _ in range(3))
        component = generator.choice("xyz")

        assert_complex_close(
            gi.primitive_moment(bra, ket, powers, origin=origin, field=field),
            reference_moment(
                reference_bra,
                reference_ket,
                powers,
                origin=origin,
                field=reference_field,
            ),
        )
        assert_complex_close(
            gi.primitive_gradient(bra, ket, component, field=field),
            reference_gradient(
                reference_bra, reference_ket, component, reference_field
            ),
        )
        assert_complex_close(
            gi.primitive_momentum(bra, ket, component, field=field),
            reference_momentum(
                reference_bra, reference_ket, component, reference_field
            ),
        )
        assert_complex_close(
            gi.primitive_kinetic(bra, ket, field=field),
            reference_kinetic(reference_bra, reference_ket, reference_field),
            atol=5.0e-13,
        )


def test_randomized_magnetic_kinetic_against_reference():
    generator = random.Random(119944)
    for _ in range(30):
        (bra, ket), (reference_bra, reference_ket) = make_primitive_pair(generator)
        field_values = tuple(generator.uniform(-0.6, 0.6) for _ in range(3))
        gauge_origin = tuple(generator.uniform(-1.0, 1.0) for _ in range(3))
        actual = gi.primitive_magnetic_kinetic(
            bra, ket, field=gi.MagneticField(field_values, gauge_origin)
        )
        expected = reference_magnetic_kinetic(
            reference_bra,
            reference_ket,
            ReferenceField(field_values, gauge_origin),
        )
        assert_complex_close(actual, expected, atol=1.0e-12, rtol=5.0e-11)


@pytest.mark.parametrize(
    ("production", "reference", "arguments"),
    [
        (gi.moment_shell, reference_shell_moment, ((2, 1, 0),)),
        (gi.gradient_shell, reference_shell_gradient, ("y",)),
        (gi.momentum_shell, reference_shell_momentum, ("z",)),
        (gi.kinetic_shell, reference_shell_kinetic, ()),
        (
            gi.magnetic_kinetic_shell,
            reference_shell_magnetic_kinetic,
            (),
        ),
    ],
)
def test_contracted_shell_properties_match_reference(production, reference, arguments):
    center_a = (-0.3, 0.2, 0.5)
    center_b = (0.6, -0.4, 0.1)
    exponents_a = (2.4, 0.7, 0.18)
    exponents_b = (1.9, 0.44)
    coefficients_a = (0.13, -0.47, 0.81)
    coefficients_b = (-0.22, 0.93)
    field_values = (0.23, -0.11, 0.37)
    gauge_origin = (-0.4, 0.8, 0.2)
    extra = {"origin": (0.3, -0.2, 0.1)} if production is gi.moment_shell else {}
    actual = production(
        gi.Shell(center_a, 3, exponents_a, coefficients_a),
        gi.Shell(center_b, 2, exponents_b, coefficients_b),
        *arguments,
        field=gi.MagneticField(field_values, gauge_origin),
        **extra,
    )
    expected = reference(
        ReferenceShell(center_a, 3, exponents_a, coefficients_a),
        ReferenceShell(center_b, 2, exponents_b, coefficients_b),
        *arguments,
        field=ReferenceField(field_values, gauge_origin),
        **extra,
    )
    np.testing.assert_allclose(actual, expected, atol=1.0e-12, rtol=2.0e-11)


def test_general_contraction_property_ordering_matches_segmented_reference():
    center_a = (-0.3, 0.2, 0.5)
    center_b = (0.6, -0.4, 0.1)
    exponents_a = (2.4, 0.7)
    exponents_b = (1.9, 0.44)
    coefficients_a = ((0.13, 0.81), (0.52, -0.28))
    coefficients_b = ((-0.22, 0.93), (0.71, 0.19))
    field_values = (0.23, -0.11, 0.37)
    gauge_origin = (-0.4, 0.8, 0.2)
    actual = gi.kinetic_shell(
        gi.Shell(center_a, 2, exponents_a, coefficients_a),
        gi.Shell(center_b, 1, exponents_b, coefficients_b),
        field=gi.MagneticField(field_values, gauge_origin),
    )
    expected = np.empty_like(actual)
    count_a = 6
    count_b = 3
    for contraction_a, row_a in enumerate(coefficients_a):
        for contraction_b, row_b in enumerate(coefficients_b):
            block = reference_shell_kinetic(
                ReferenceShell(center_a, 2, exponents_a, row_a),
                ReferenceShell(center_b, 1, exponents_b, row_b),
                ReferenceField(field_values, gauge_origin),
            )
            expected[
                contraction_a * count_a : (contraction_a + 1) * count_a,
                contraction_b * count_b : (contraction_b + 1) * count_b,
            ] = block
    np.testing.assert_allclose(actual, expected, atol=8.0e-13, rtol=2.0e-11)


def test_analytic_normalized_s_cases():
    exponent = 0.8
    center = (0.4, -0.3, 0.7)
    primitive = gi.PrimitiveGaussian(exponent, center)
    assert gi.primitive_moment(
        primitive, primitive, (2, 0, 0), origin=center
    ) == pytest.approx(1.0 / (4.0 * exponent), abs=5.0e-14)
    assert gi.primitive_kinetic(primitive, primitive) == pytest.approx(
        1.5 * exponent, abs=5.0e-14
    )

    field = gi.MagneticField((0.2, -0.4, 0.7), (-0.3, 0.1, 0.5))
    wave_vector = field.london_wave_vector(center)
    for axis, component in enumerate("xyz"):
        assert gi.primitive_gradient(
            primitive, primitive, component, field=field
        ) == pytest.approx(-1j * wave_vector[axis], abs=8.0e-14)
        assert gi.primitive_momentum(
            primitive, primitive, component, field=field
        ) == pytest.approx(-wave_vector[axis], abs=8.0e-14)
    field_squared = sum(value * value for value in field.B)
    assert gi.primitive_magnetic_kinetic(
        primitive, primitive, field=field
    ) == pytest.approx(1.5 * exponent + field_squared / (16.0 * exponent), abs=2.0e-13)


def test_displaced_ss_zero_field_kinetic_closed_form():
    alpha = 0.7
    beta = 1.3
    center_a = np.asarray((-0.4, 0.2, 0.1))
    center_b = np.asarray((0.8, -0.5, 0.9))
    bra = gi.PrimitiveGaussian(alpha, center_a)
    ket = gi.PrimitiveGaussian(beta, center_b)
    reduced = alpha * beta / (alpha + beta)
    overlap = gi.primitive_overlap(bra, ket)
    expected = reduced * (3.0 - 2.0 * reduced * np.sum((center_a - center_b) ** 2))
    assert gi.primitive_kinetic(bra, ket) == pytest.approx(
        expected * overlap, abs=5.0e-14, rel=2.0e-13
    )


def test_operator_symmetries_and_zero_field_reduction():
    basis = gi.Basis(
        [
            gi.Shell((-0.3, 0.2, 0.5), 0, [2.4, 0.7], [0.2, 0.8]),
            gi.Shell((0.6, -0.4, 0.1), 2, [1.9, 0.44], [-0.22, 0.93]),
            gi.Shell((-0.8, 0.1, 0.7), 3, [0.83], [1.0]),
        ]
    )
    field = gi.MagneticField((0.23, -0.11, 0.37), (-0.4, 0.8, 0.2))
    for powers in ((1, 0, 0), (0, 2, 1)):
        matrix = gi.moment(basis, powers, origin=(0.2, -0.1, 0.3), field=field)
        np.testing.assert_allclose(matrix, matrix.conj().T, atol=8.0e-13, rtol=8.0e-13)
    for component in "xyz":
        gradient = gi.gradient(basis, component, field=field)
        momentum = gi.momentum(basis, component, field=field)
        np.testing.assert_allclose(
            gradient, -gradient.conj().T, atol=8.0e-13, rtol=8.0e-13
        )
        np.testing.assert_allclose(
            momentum, momentum.conj().T, atol=8.0e-13, rtol=8.0e-13
        )
    kinetic = gi.kinetic(basis, field=field)
    magnetic = gi.magnetic_kinetic(basis, field=field)
    np.testing.assert_allclose(kinetic, kinetic.conj().T, atol=2.0e-12, rtol=2.0e-12)
    np.testing.assert_allclose(magnetic, magnetic.conj().T, atol=3.0e-12, rtol=3.0e-12)
    np.testing.assert_allclose(
        gi.magnetic_kinetic(basis), gi.kinetic(basis), atol=2.0e-13, rtol=2.0e-13
    )


def test_physical_magnetic_kinetic_is_gauge_origin_independent():
    basis = gi.Basis(
        [
            gi.Shell((-0.3, 0.2, 0.5), 1, [1.4, 0.38], [0.3, 0.8]),
            gi.Shell((0.6, -0.4, 0.1), 2, [0.9], [1.0]),
        ]
    )
    field_values = (0.31, -0.27, 0.18)
    first = gi.MagneticField(field_values, (-0.4, 0.8, 0.2))
    second = gi.MagneticField(field_values, (0.7, -0.3, 0.9))
    physical_first = gi.magnetic_kinetic(basis, field=first)
    physical_second = gi.magnetic_kinetic(basis, field=second)
    np.testing.assert_allclose(
        physical_first, physical_second, atol=2.0e-12, rtol=2.0e-12
    )
    assert (
        np.max(np.abs(gi.kinetic(basis, field=first) - gi.kinetic(basis, field=second)))
        > 1.0e-4
    )


def test_coordinate_origin_shift_and_translation_rephasing():
    a = gi.PrimitiveGaussian(0.79, (-0.6, 0.2, 0.7), (2, 1, 0))
    b = gi.PrimitiveGaussian(1.41, (0.5, -0.4, 0.3), (1, 0, 2))
    field_values = (0.3, -0.7, 1.0)
    gauge_origin = (0.1, -0.2, 0.4)
    field = gi.MagneticField(field_values, gauge_origin)
    first_origin = (0.2, -0.1, 0.3)
    displacement = 0.37
    overlap = gi.primitive_overlap(a, b, field=field)
    first = gi.primitive_moment(a, b, (1, 0, 0), origin=first_origin, field=field)
    shifted_origin = (first_origin[0] + displacement, *first_origin[1:])
    second = gi.primitive_moment(a, b, (1, 0, 0), origin=shifted_origin, field=field)
    assert_complex_close(second, first - displacement * overlap)

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
    translated_field = gi.MagneticField(
        field_values,
        tuple(gauge_origin[index] + translation[index] for index in range(3)),
    )
    translated_origin = tuple(
        first_origin[index] + translation[index] for index in range(3)
    )
    translated = gi.primitive_moment(
        translated_a,
        translated_b,
        (1, 0, 0),
        origin=translated_origin,
        field=translated_field,
    )
    kappa_a = np.asarray(field.london_wave_vector(a.center))
    kappa_b = np.asarray(field.london_wave_vector(b.center))
    phase = cmath.exp(-1j * np.dot(kappa_b - kappa_a, translation))
    assert_complex_close(translated, phase * first, atol=8.0e-13, rtol=8.0e-13)
    original_magnetic = gi.primitive_magnetic_kinetic(a, b, field=field)
    translated_magnetic = gi.primitive_magnetic_kinetic(
        translated_a, translated_b, field=translated_field
    )
    assert_complex_close(
        translated_magnetic,
        phase * original_magnetic,
        atol=2.0e-12,
        rtol=2.0e-12,
    )


def numerical_kinetic_quadrature(bra, ket, field, *, magnetic, order=14, step=8e-5):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    exponent = bra.exponent + ket.exponent
    product_center = (
        bra.exponent * np.asarray(bra.center) + ket.exponent * np.asarray(ket.center)
    ) / exponent
    result = 0.0j
    unit = np.eye(3)
    magnetic_field = np.asarray(field.B)
    gauge_origin = np.asarray(field.gauge_origin)
    for ix, x in enumerate(nodes):
        for iy, y in enumerate(nodes):
            for iz, z in enumerate(nodes):
                scaled = np.asarray((x, y, z))
                position = product_center + scaled / np.sqrt(exponent)
                ket_value = gaussian_value(ket, position, field)
                gradient = np.empty(3, dtype=np.complex128)
                laplacian = 0.0j
                for axis in range(3):
                    plus = gaussian_value(ket, position + step * unit[axis], field)
                    minus = gaussian_value(ket, position - step * unit[axis], field)
                    gradient[axis] = (plus - minus) / (2.0 * step)
                    laplacian += (plus - 2.0 * ket_value + minus) / step**2
                operated = -0.5 * laplacian
                if magnetic:
                    vector_potential = 0.5 * np.cross(
                        magnetic_field, position - gauge_origin
                    )
                    operated += np.dot(vector_potential, -1j * gradient)
                    operated += (
                        0.5 * np.dot(vector_potential, vector_potential) * ket_value
                    )
                value = gaussian_value(bra, position, field).conjugate() * operated
                result += (
                    weights[ix]
                    * weights[iy]
                    * weights[iz]
                    * np.exp(np.dot(scaled, scaled))
                    * value
                )
    return result / exponent**1.5


@pytest.mark.parametrize("magnetic", [False, True])
def test_kinetic_against_real_space_finite_difference_quadrature(magnetic):
    bra = ReferencePrimitive(0.73, (-0.4, 0.2, 0.1), (1, 0, 0))
    ket = ReferencePrimitive(1.21, (0.6, -0.5, 0.7), (0, 1, 0))
    field = ReferenceField((0.23, -0.17, 0.31), (0.2, -0.4, 0.1))
    actual = (
        reference_magnetic_kinetic(bra, ket, field)
        if magnetic
        else reference_kinetic(bra, ket, field)
    )
    errors = []
    numerical = None
    for step in (8.0e-4, 4.0e-4, 2.0e-4):
        numerical = numerical_kinetic_quadrature(
            bra, ket, field, magnetic=magnetic, step=step
        )
        errors.append(abs(actual - numerical))
    assert errors[1] < 0.3 * errors[0]
    assert errors[2] < 0.3 * errors[1]
    assert numerical is not None
    assert_complex_close(actual, numerical, atol=1.0e-8, rtol=1.0e-8)
