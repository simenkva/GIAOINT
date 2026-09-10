import giao_integrals as gi
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


def test_empty_basis_has_well_defined_shapes():
    basis = gi.Basis([])
    assert gi.overlap(basis).shape == (0, 0)
    assert gi.kinetic(basis).shape == (0, 0)
    assert gi.nuclear_attraction(basis, []).shape == (0, 0)
    assert gi.overlap_nuclear_derivatives(basis).shape == (0, 3, 0, 0)
    assert gi.overlap_magnetic_derivatives(basis).shape == (3, 0, 0)
    assert list(gi.eri_batches(basis)) == []
    assert gi.eri(basis, storage="full", max_bytes=0).shape == (0, 0, 0, 0)


def test_empty_nucleus_sequence_is_zero_with_explicit_derivative_shapes():
    shell = gi.Shell((0.1, -0.2, 0.3), 1, [0.8], [1.0])
    basis = gi.Basis([shell])
    np.testing.assert_array_equal(gi.nuclear_attraction(basis, []), 0.0)
    shell_response, nucleus_response = gi.nuclear_attraction_nuclear_derivatives(
        basis, []
    )
    np.testing.assert_array_equal(shell_response, 0.0)
    assert shell_response.shape == (1, 3, 3, 3)
    assert nucleus_response.shape == (0, 3, 3, 3)


def test_coulomb_order_limit_is_diagnostic_for_values_and_derivatives():
    high = gi.PrimitiveGaussian(0.8, (0.0, 0.0, 0.0), (33, 0, 0))
    edge = gi.PrimitiveGaussian(0.8, (0.0, 0.0, 0.0), (32, 0, 0))
    s = gi.PrimitiveGaussian(1.1, (0.2, 0.1, -0.3))
    nucleus = [gi.Nucleus(1.0, (0.1, -0.2, 0.4))]
    with pytest.raises(ValueError, match="combined angular order through 32"):
        gi.primitive_nuclear_attraction(high, s, nucleus)
    with pytest.raises(ValueError, match="combined angular order through 32"):
        gi.primitive_nuclear_attraction_center_derivatives(edge, s, nucleus)
    with pytest.raises(ValueError, match="combined angular order through 32"):
        gi.primitive_eri(high, s, s, s)


@settings(max_examples=16, deadline=None)
@given(
    exponent_a=st.floats(0.2, 3.0, allow_nan=False, allow_infinity=False),
    exponent_b=st.floats(0.2, 3.0, allow_nan=False, allow_infinity=False),
    coordinates=st.lists(
        st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
        min_size=12,
        max_size=12,
    ),
)
def test_random_finite_field_overlap_and_response_conjugation(
    exponent_a, exponent_b, coordinates
):
    a = gi.PrimitiveGaussian(exponent_a, coordinates[0:3], (1, 0, 0))
    b = gi.PrimitiveGaussian(exponent_b, coordinates[3:6], (0, 1, 0))
    field = gi.MagneticField(coordinates[6:9], coordinates[9:12])

    np.testing.assert_allclose(
        gi.primitive_overlap(a, b, field=field),
        gi.primitive_overlap(b, a, field=field).conjugate(),
        atol=2.0e-13,
        rtol=2.0e-12,
    )
    response = gi.primitive_overlap_magnetic_derivatives(a, b, field=field)
    reversed_response = gi.primitive_overlap_magnetic_derivatives(b, a, field=field)
    np.testing.assert_allclose(
        response, reversed_response.conjugate(), atol=3.0e-13, rtol=3.0e-12
    )
