"""Independent operator origins/fields at fixed London phases."""

# Loop closures are consumed synchronously.
# ruff: noqa: B023
import giao_integrals as gi
import giao_reference as ref
import numpy as np
import pytest
from test_moment_derivatives import FIELDS, ORIGIN, richardson


def basis():
    return gi.Basis(
        [
            gi.Shell((0.1, -0.2, 0.3), 1, [0.8, 1.2], [[0.7, -0.2], [0.1, 0.8]]),
            gi.Shell((-0.3, 0.4, 0.2), 2, [0.6], [1.0]),
        ]
    )


def moments(basis, origin, field):
    first = np.array(
        [
            gi.moment(
                basis, tuple(int(k == i) for k in range(3)), origin=origin, field=field
            )
            for i in range(3)
        ]
    )
    second = np.array(
        [
            [
                gi.moment(
                    basis,
                    tuple(int(k == i) + int(k == j) for k in range(3)),
                    origin=origin,
                    field=field,
                )
                for j in range(3)
            ]
            for i in range(3)
        ]
    )
    return first, second


@pytest.mark.parametrize("field", FIELDS)
def test_operator_field_differences_hold_london_phases_fixed(field):
    b = basis()
    g0 = field.gauge_origin
    # At C=B0, G=G0: dT/dC_i = 1/2 * [(r-G0) cross pi0]_i.
    for axis, component in enumerate("xyz"):
        expected = 0.5 * gi.kinetic_angular_momentum(
            b, component, origin=g0, field=field, threads=1
        )
        numerical = richardson(
            lambda step: gi.magnetic_kinetic_operator(
                b,
                operator_field=np.array(field.B) + step * np.eye(3)[axis],
                origin=g0,
                field=field,
                threads=1,
            )
        )
        np.testing.assert_allclose(numerical, expected, atol=3e-10, rtol=3e-9)
    # Arbitrary C and G, with a different phase gauge G0.
    first, second = moments(b, ORIGIN, field)
    delta = ORIGIN - np.asarray(g0)
    c = np.array([-0.12, 0.08, 0.19])
    for axis, component in enumerate("xyz"):
        physical = gi.kinetic_angular_momentum(
            b, component, origin=ORIGIN, field=field, threads=1
        )
        potential = 0.5 * (
            field.B[axis]
            * (
                np.trace(second, axis1=0, axis2=1)
                + np.einsum("i,imn->mn", delta, first)
            )
            - np.einsum("i,imn->mn", field.B, second[axis])
            - delta[axis] * np.einsum("i,imn->mn", field.B, first)
        )
        canonical = physical - potential
        expected = 0.5 * canonical + 0.25 * (
            c[axis] * np.trace(second, axis1=0, axis2=1)
            - np.einsum("i,imn->mn", c, second[axis])
        )
        numerical = richardson(
            lambda step: gi.magnetic_kinetic_operator(
                b,
                operator_field=c + step * np.eye(3)[axis],
                origin=ORIGIN,
                field=field,
                threads=1,
            )
        )
        np.testing.assert_allclose(numerical, expected, atol=3e-10, rtol=3e-9)


@pytest.mark.parametrize("field", FIELDS)
def test_hermiticity_origin_sum_rules_and_old_limits(field):
    b = basis()
    shift = np.array([0.13, -0.21, 0.17])
    first, second = moments(b, ORIGIN, field)
    translated_first, translated_second = moments(b, ORIGIN + shift, field)
    overlap = gi.overlap(b, field=field)
    np.testing.assert_allclose(second, second.swapaxes(-1, -2).conj(), atol=3e-13)
    for i in range(3):
        for j in range(3):
            expected = (
                second[i, j]
                - shift[i] * first[j]
                - shift[j] * first[i]
                + shift[i] * shift[j] * overlap
            )
            np.testing.assert_allclose(translated_second[i, j], expected, atol=3e-13)
    pi = np.array([gi.momentum(b, component, field=field) for component in "xyz"])
    phase_first, _ = moments(b, field.gauge_origin, field)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        pi[i] += 0.5 * (field.B[j] * phase_first[k] - field.B[k] * phase_first[j])
    for i, component in enumerate("xyz"):
        value = gi.kinetic_angular_momentum(
            b, component, origin=ORIGIN, field=field, threads=1
        )
        moved = gi.kinetic_angular_momentum(
            b, component, origin=ORIGIN + shift, field=field, threads=1
        )
        j, k = (i + 1) % 3, (i + 2) % 3
        np.testing.assert_allclose(
            moved, value - shift[j] * pi[k] + shift[k] * pi[j], atol=3e-13
        )
        np.testing.assert_allclose(value, value.T.conj(), atol=3e-13)
        out = np.empty_like(value)
        assert (
            gi.kinetic_angular_momentum(
                b,
                component,
                origin=ORIGIN,
                field=field,
                out=out,
                threads=2 if gi.openmp_enabled() else 1,
            )
            is out
        )
        np.testing.assert_array_equal(out, value)
    old = gi.magnetic_kinetic(b, field=field)
    np.testing.assert_array_equal(
        gi.magnetic_kinetic_operator(b, field=field, threads=1), old
    )
    np.testing.assert_array_equal(
        gi.magnetic_kinetic_operator(
            b, field=field, operator_field=(0, 0, 0), origin=ORIGIN, threads=1
        ),
        gi.kinetic(b, field=field),
    )
    value = gi.magnetic_kinetic_operator(
        b, field=field, operator_field=(0.03, -0.08, 0.12), origin=ORIGIN, threads=1
    )
    np.testing.assert_allclose(value, value.T.conj(), atol=3e-13)
    c = np.array([0.03, -0.08, 0.12])
    pi_operator = np.array(
        [gi.momentum(b, component, field=field) for component in "xyz"]
    )
    for axis in range(3):
        j, k = (axis + 1) % 3, (axis + 2) % 3
        pi_operator[axis] += 0.5 * (c[j] * first[k] - c[k] * first[j])
    constant = 0.5 * np.cross(c, shift)
    expected = (
        value
        - np.einsum("i,imn->mn", constant, pi_operator)
        + 0.5 * np.dot(constant, constant) * overlap
    )
    moved = gi.magnetic_kinetic_operator(
        b, field=field, operator_field=c, origin=ORIGIN + shift, threads=1
    )
    np.testing.assert_allclose(moved, expected, atol=3e-13)
    out = np.empty_like(value)
    assert (
        gi.magnetic_kinetic_operator(
            b,
            field=field,
            operator_field=(0.03, -0.08, 0.12),
            origin=ORIGIN,
            out=out,
            threads=2 if gi.openmp_enabled() else 1,
        )
        is out
    )
    np.testing.assert_array_equal(out, value)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("angular_momentum", range(5))
def test_primitive_and_shell_against_independent_reference(field, angular_momentum):
    a = gi.PrimitiveGaussian(
        0.8, (0.1, -0.2, 0.3), (angular_momentum, 0, 0), coefficient=0.7
    )
    b = gi.PrimitiveGaussian(0.6, (-0.3, 0.4, 0.2), (0, 1, 0), coefficient=-0.4)
    ra = ref.PrimitiveGaussian(0.8, a.center, a.angular, coefficient=0.7)
    rb = ref.PrimitiveGaussian(0.6, b.center, b.angular, coefficient=-0.4)
    rf = ref.MagneticField(field.B, field.gauge_origin)
    for component in "xyz":
        actual = gi.primitive_kinetic_angular_momentum(
            a, b, component, origin=ORIGIN, field=field
        )
        expected = ref.primitive_kinetic_angular_momentum(
            ra, rb, component, origin=ORIGIN, field=rf
        )
        np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=3e-12)
    actual = gi.primitive_magnetic_kinetic_operator(
        a, b, operator_field=(-0.12, 0.08, 0.19), origin=ORIGIN, field=field
    )
    expected = ref.primitive_magnetic_kinetic_operator(
        ra, rb, operator_field=(-0.12, 0.08, 0.19), origin=ORIGIN, field=rf
    )
    np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=3e-12)
    sa = gi.Shell(a.center, angular_momentum, [0.8], [1.0])
    sb = gi.Shell(b.center, 1, [0.6], [1.0])
    for component in "xyz":
        shell = gi.kinetic_angular_momentum_shell(
            sa, sb, component, origin=ORIGIN, field=field
        )
        pa = gi.PrimitiveGaussian(0.8, a.center, a.angular)
        pb = gi.PrimitiveGaussian(0.6, b.center, b.angular)
        np.testing.assert_allclose(
            shell[0, 1],
            gi.primitive_kinetic_angular_momentum(
                pa, pb, component, origin=ORIGIN, field=field
            ),
            atol=2e-14,
        )


def test_td4_validation_and_molecule_wrappers():
    mol = gi.Molecule(
        [("H", (0, 0, 0)), ("H", (0.1, 0.2, 1.4))],
        "STO-3G",
        unit="bohr",
        field=FIELDS[1],
    )
    np.testing.assert_array_equal(
        mol.kinetic_angular_momentum("x", origin=ORIGIN, threads=1),
        gi.kinetic_angular_momentum(
            mol.basis, "x", origin=ORIGIN, field=mol.field, threads=1
        ),
    )
    np.testing.assert_array_equal(
        mol.magnetic_kinetic_operator(
            operator_field=(0.1, 0, 0), origin=ORIGIN, threads=1
        ),
        gi.magnetic_kinetic_operator(
            mol.basis,
            operator_field=(0.1, 0, 0),
            origin=ORIGIN,
            field=mol.field,
            threads=1,
        ),
    )
    for kwargs in [
        {"component": "bad"},
        {"component": "x", "origin": (float("nan"), 0, 0)},
    ]:
        with pytest.raises(ValueError):
            gi.kinetic_angular_momentum(mol.basis, **kwargs)
    with pytest.raises(ValueError):
        gi.magnetic_kinetic_operator(mol.basis, operator_field=(float("inf"), 0, 0))
    with pytest.raises(ValueError):
        gi.magnetic_kinetic_operator(mol.basis, out=np.empty((2, 2)))
    with pytest.raises(ValueError):
        gi.kinetic_angular_momentum(gi.Basis([]), "x", origin=(float("nan"), 0, 0))
    assert gi.kinetic_angular_momentum(gi.Basis([]), "x", threads=1).shape == (0, 0)


@pytest.mark.parametrize("field", FIELDS)
def test_td4_shell_blocks_match_basis_and_out(field):
    b = basis()
    a, c = b.shells
    na = a.ao_count
    for component in "xyz":
        expected = gi.kinetic_angular_momentum(
            b, component, origin=ORIGIN, field=field, threads=1
        )[:na, na:]
        out = np.empty((a.ao_count, c.ao_count), complex)
        assert (
            gi.kinetic_angular_momentum_shell(
                a, c, component, origin=ORIGIN, field=field, out=out
            )
            is out
        )
        np.testing.assert_array_equal(out, expected)
    expected = gi.magnetic_kinetic_operator(
        b, operator_field=(0.12, -0.08, 0.03), origin=ORIGIN, field=field, threads=1
    )[:na, na:]
    out = np.empty((a.ao_count, c.ao_count), complex)
    assert (
        gi.magnetic_kinetic_operator_shell(
            a,
            c,
            operator_field=(0.12, -0.08, 0.03),
            origin=ORIGIN,
            field=field,
            out=out,
        )
        is out
    )
    np.testing.assert_array_equal(out, expected)


def test_kinetic_angular_operator_uses_phase_gauge_consistently():
    b = basis()
    field = FIELDS[1]
    shifted = gi.MagneticField(
        field.B, np.array(field.gauge_origin) + (0.21, -0.17, 0.13)
    )
    for component in "xyz":
        expected = gi.kinetic_angular_momentum(
            b, component, origin=ORIGIN, field=field, threads=1
        )
        actual = gi.kinetic_angular_momentum(
            b, component, origin=ORIGIN, field=shifted, threads=1
        )
        np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=3e-13)
