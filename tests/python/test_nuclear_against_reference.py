import random

import giao_integrals as gi
import numpy as np
from giao_reference import MagneticField as ReferenceField
from giao_reference import Nucleus as ReferenceNucleus
from giao_reference import PrimitiveGaussian as ReferencePrimitive
from giao_reference import Shell as ReferenceShell
from giao_reference import cartesian_components
from giao_reference import primitive_nuclear_attraction as reference_primitive
from giao_reference import shell_nuclear_attraction as reference_shell


def assert_complex_close(actual, expected, *, atol=3.0e-13, rtol=3.0e-11):
    assert abs(actual - expected) <= atol + rtol * abs(expected)


def test_analytic_finite_field_ss_seed_and_multiple_nuclei():
    bra = gi.PrimitiveGaussian(0.7, (-0.4, 0.2, 0.8))
    ket = gi.PrimitiveGaussian(1.3, (0.8, -0.5, 0.1))
    field_values = (0.3, -0.2, 0.5)
    origin = (0.2, -0.1, 0.4)
    nuclei = [gi.Nucleus(2.0, (0.1, -0.2, 0.3)), gi.Nucleus(1.0, (1.2, 0.3, -0.4))]
    expected = reference_primitive(
        ReferencePrimitive(0.7, (-0.4, 0.2, 0.8)),
        ReferencePrimitive(1.3, (0.8, -0.5, 0.1)),
        [
            ReferenceNucleus(2.0, (0.1, -0.2, 0.3)),
            ReferenceNucleus(1.0, (1.2, 0.3, -0.4)),
        ],
        ReferenceField(field_values, origin),
        dps=90,
    )
    actual = gi.primitive_nuclear_attraction(
        bra, ket, nuclei, field=gi.MagneticField(field_values, origin)
    )
    assert_complex_close(actual, complex(expected), atol=8e-14, rtol=5e-13)


def test_randomized_md_matches_independent_obara_saika_through_l4():
    generator = random.Random(421997)
    for _ in range(36):
        exponent_a = 10 ** generator.uniform(-0.6, 0.6)
        exponent_b = 10 ** generator.uniform(-0.6, 0.6)
        center_a = tuple(generator.uniform(-1.4, 1.4) for _ in range(3))
        center_b = tuple(generator.uniform(-1.4, 1.4) for _ in range(3))
        angular_a = generator.choice(cartesian_components(generator.randrange(5)))
        angular_b = generator.choice(cartesian_components(generator.randrange(5)))
        field_values = tuple(generator.uniform(-1.5, 1.5) for _ in range(3))
        origin = tuple(generator.uniform(-1.0, 1.0) for _ in range(3))
        nuclear_data = [
            (
                generator.uniform(0.2, 4.0),
                tuple(generator.uniform(-1.5, 1.5) for _ in range(3)),
            )
            for _ in range(2)
        ]
        actual = gi.primitive_nuclear_attraction(
            gi.PrimitiveGaussian(exponent_a, center_a, angular_a),
            gi.PrimitiveGaussian(exponent_b, center_b, angular_b),
            [gi.Nucleus(charge, center) for charge, center in nuclear_data],
            field=gi.MagneticField(field_values, origin),
        )
        expected = reference_primitive(
            ReferencePrimitive(exponent_a, center_a, angular_a),
            ReferencePrimitive(exponent_b, center_b, angular_b),
            [ReferenceNucleus(charge, center) for charge, center in nuclear_data],
            ReferenceField(field_values, origin),
            dps=75,
        )
        assert_complex_close(actual, complex(expected))


def test_strong_field_negative_real_boys_argument_regression():
    bra = gi.PrimitiveGaussian(0.4, (-1.0, 0.0, 0.0), (2, 1, 0))
    ket = gi.PrimitiveGaussian(0.6, (1.0, 0.0, 0.0), (1, 0, 2))
    nucleus = gi.Nucleus(3.0, (0.2, 0.1, -0.2))
    field = gi.MagneticField((0.0, 0.0, 12.0), (0.3, -0.2, 0.1))
    actual = gi.primitive_nuclear_attraction(bra, ket, [nucleus], field=field)
    expected = reference_primitive(
        ReferencePrimitive(0.4, (-1.0, 0.0, 0.0), (2, 1, 0)),
        ReferencePrimitive(0.6, (1.0, 0.0, 0.0), (1, 0, 2)),
        [ReferenceNucleus(3.0, (0.2, 0.1, -0.2))],
        ReferenceField((0.0, 0.0, 12.0), (0.3, -0.2, 0.1)),
        dps=100,
    )
    assert_complex_close(actual, complex(expected), atol=2e-12, rtol=2e-10)


def test_contracted_shell_block_and_basis_driver():
    shell_a = gi.Shell((-0.3, 0.2, 0.5), 1, [2.4, 0.7], [0.2, 0.8])
    shell_b = gi.Shell((0.6, -0.4, 0.1), 2, [1.9, 0.44], [-0.22, 0.93])
    nuclei = [gi.Nucleus(2.0, (0.0, 0.1, -0.2))]
    field_values = (0.23, -0.11, 0.37)
    origin = (-0.4, 0.8, 0.2)
    actual = gi.nuclear_attraction_shell(
        shell_a,
        shell_b,
        nuclei,
        field=gi.MagneticField(field_values, origin),
    )
    expected = np.asarray(
        reference_shell(
            ReferenceShell((-0.3, 0.2, 0.5), 1, (2.4, 0.7), (0.2, 0.8)),
            ReferenceShell((0.6, -0.4, 0.1), 2, (1.9, 0.44), (-0.22, 0.93)),
            [ReferenceNucleus(2.0, (0.0, 0.1, -0.2))],
            ReferenceField(field_values, origin),
        ),
        dtype=np.complex128,
    )
    np.testing.assert_allclose(actual, expected, atol=8e-13, rtol=2e-11)

    basis = gi.Basis([shell_a, shell_b])
    matrix = gi.nuclear_attraction(
        basis, nuclei, field=gi.MagneticField(field_values, origin)
    )
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=2e-12, rtol=2e-12)
    np.testing.assert_array_equal(matrix[:3, 3:], actual)


def test_zero_field_reality_gauge_origin_independence_and_conjugation():
    bra = gi.PrimitiveGaussian(0.8, (-0.7, 0.3, 0.5), (3, 0, 1))
    ket = gi.PrimitiveGaussian(1.2, (0.4, -0.6, 0.2), (1, 2, 0))
    nuclei = [gi.Nucleus(1.0, (0.0, 0.0, 0.0)), gi.Nucleus(4.0, (0.6, 0.1, -0.3))]
    zero = gi.primitive_nuclear_attraction(bra, ket, nuclei)
    assert abs(zero.imag) < 3e-15

    field_values = (0.3, -0.5, 0.7)
    first = gi.primitive_nuclear_attraction(
        bra,
        ket,
        nuclei,
        field=gi.MagneticField(field_values, (0.2, -0.1, 0.4)),
    )
    shifted_origin = gi.primitive_nuclear_attraction(
        bra,
        ket,
        nuclei,
        field=gi.MagneticField(field_values, (-1.2, 0.8, 0.1)),
    )
    reversed_value = gi.primitive_nuclear_attraction(
        ket,
        bra,
        nuclei,
        field=gi.MagneticField(field_values, (0.2, -0.1, 0.4)),
    )
    assert_complex_close(first, shifted_origin, atol=2e-13, rtol=8e-13)
    assert_complex_close(first, reversed_value.conjugate(), atol=2e-13, rtol=8e-13)
