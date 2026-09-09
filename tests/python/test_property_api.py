import giao_integrals as gi
import numpy as np
import pytest


def make_basis():
    return gi.Basis(
        [
            gi.Shell((0.0, 0.0, 0.0), 0, [2.1, 0.55], [0.24, 0.81]),
            gi.Shell((0.7, -0.2, 0.3), 1, [1.6, 0.4], [0.4, 0.7]),
            gi.Shell((-0.5, 0.8, -0.1), 2, [1.1], [1.0]),
        ]
    )


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (gi.moment, ((1, 0, 2),)),
        (gi.gradient, ("x",)),
        (gi.momentum, ("y",)),
        (gi.kinetic, ()),
        (gi.magnetic_kinetic, ()),
    ],
)
def test_basis_property_shape_dtype_and_out(function, arguments):
    basis = make_basis()
    field = gi.MagneticField((0.2, -0.3, 0.1), (0.4, -0.1, 0.2))
    result = function(basis, *arguments, field=field)
    assert result.shape == (10, 10)
    assert result.dtype == np.complex128
    assert result.flags.c_contiguous

    output = np.empty_like(result)
    returned = function(basis, *arguments, field=field, out=output)
    assert returned is output
    np.testing.assert_array_equal(output, result)


def test_shell_and_basis_property_blocks_agree():
    basis = make_basis()
    shells = basis.shells
    field = gi.MagneticField((0.2, -0.3, 0.1), (0.4, -0.1, 0.2))
    cases = [
        (gi.moment, gi.moment_shell, ((2, 1, 0),), {"origin": (0.2, 0.1, -0.4)}),
        (gi.gradient, gi.gradient_shell, ("z",), {}),
        (gi.momentum, gi.momentum_shell, ("x",), {}),
        (gi.kinetic, gi.kinetic_shell, (), {}),
        (gi.magnetic_kinetic, gi.magnetic_kinetic_shell, (), {}),
    ]
    for basis_function, shell_function, arguments, keywords in cases:
        matrix = basis_function(basis, *arguments, field=field, **keywords)
        for shell_a, offset_a in zip(shells, basis.ao_offsets, strict=True):
            for shell_b, offset_b in zip(shells, basis.ao_offsets, strict=True):
                block = shell_function(
                    shell_a, shell_b, *arguments, field=field, **keywords
                )
                np.testing.assert_array_equal(
                    matrix[
                        offset_a : offset_a + shell_a.ao_count,
                        offset_b : offset_b + shell_b.ao_count,
                    ],
                    block,
                )


def test_zero_order_moment_and_momentum_gradient_identity():
    basis = make_basis()
    field = gi.MagneticField((0.2, -0.3, 0.1), (0.4, -0.1, 0.2))
    np.testing.assert_allclose(
        gi.moment(basis, (0, 0, 0), origin=(4.0, -3.0, 2.0), field=field),
        gi.overlap(basis, field=field),
        atol=2.0e-14,
        rtol=2.0e-13,
    )
    for component in "xyz":
        np.testing.assert_allclose(
            gi.momentum(basis, component, field=field),
            -1j * gi.gradient(basis, component, field=field),
            atol=2.0e-14,
            rtol=2.0e-13,
        )


@pytest.mark.parametrize(
    ("function", "arguments", "match"),
    [
        (gi.moment, ((-1, 0, 0),), "non-negative"),
        (gi.moment, ((1, 0),), "three"),
        (gi.moment, ((1, 0, 0),), "finite"),
        (gi.gradient, ("q",), "'x', 'y', or 'z'"),
        (gi.momentum, ("X",), "'x', 'y', or 'z'"),
    ],
)
def test_property_argument_validation(function, arguments, match):
    basis = make_basis()
    keywords = {"origin": (0.0, np.inf, 0.0)} if match == "finite" else {}
    with pytest.raises(ValueError, match=match):
        function(basis, *arguments, **keywords)


def test_property_out_validation():
    basis = make_basis()
    with pytest.raises(ValueError, match="complex128"):
        gi.kinetic(basis, out=np.empty((10, 10), dtype=np.float64))
    with pytest.raises(ValueError, match="shape"):
        gi.moment(basis, (1, 0, 0), out=np.empty((3, 3), dtype=np.complex128))
