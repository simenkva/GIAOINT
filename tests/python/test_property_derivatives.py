"""Centre/field responses for all canonical gradient and momentum components."""

# ruff: noqa: B023
import giao_integrals as gi
import numpy as np
import pytest
from giao_reference import MagneticField as ReferenceField
from giao_reference import PrimitiveGaussian as ReferencePrimitive
from giao_reference import primitive_gradient as reference_gradient
from giao_reference import primitive_moment as reference_moment
from giao_reference import primitive_momentum as reference_momentum
from test_moment_derivatives import FIELDS, ORIGIN, POWERS, richardson, shells


@pytest.mark.parametrize("operator", ["gradient", "momentum"])
@pytest.mark.parametrize("component", ["x", "y", "z"])
@pytest.mark.parametrize("field", FIELDS)
def test_property_basis_derivatives(operator, component, field):
    centers = np.array([[0.1, -0.2, 0.3], [-0.4, 0.2, 0.1]])
    basis = gi.Basis(shells(centers))
    value = getattr(gi, operator)
    nuclear_fn = getattr(gi, operator + "_nuclear_derivatives")
    field_fn = getattr(gi, operator + "_magnetic_derivatives")
    nuclear = nuclear_fn(basis, component, field=field, threads=1)
    magnetic = field_fn(basis, component, field=field, threads=1)
    for center in range(2):
        for axis in range(3):

            def displaced(step):
                changed = centers.copy()
                changed[center, axis] += step
                return value(gi.Basis(shells(changed)), component, field=field)

            np.testing.assert_allclose(
                nuclear[center, axis], richardson(displaced), atol=3e-10, rtol=3e-10
            )
    for axis in range(3):
        numerical = richardson(
            lambda step: value(
                basis,
                component,
                field=gi.MagneticField(
                    np.array(field.B) + step * np.eye(3)[axis], field.gauge_origin
                ),
            )
        )
        np.testing.assert_allclose(magnetic[axis], numerical, atol=3e-10, rtol=3e-10)
        gauge = richardson(
            lambda step: value(
                basis,
                component,
                field=gi.MagneticField(
                    field.B, np.array(field.gauge_origin) + step * np.eye(3)[axis]
                ),
            )
        )
        wave = np.repeat([np.cross(field.B, c) / 2 for c in centers], [6, 6], axis=0)
        expected = (
            1j
            * (wave[:, axis, None] - wave[None, :, axis])
            * value(basis, component, field=field)
        )
        np.testing.assert_allclose(
            nuclear[:, axis].sum(axis=0) + gauge, expected, atol=3e-10, rtol=3e-10
        )
    sign = -1 if operator == "gradient" else 1
    for result in [nuclear, magnetic]:
        np.testing.assert_allclose(
            result, sign * result.swapaxes(-1, -2).conj(), atol=3e-13
        )
    if gi.openmp_enabled():
        for function, reference in [(nuclear_fn, nuclear), (field_fn, magnetic)]:
            out = np.empty_like(reference)
            assert function(basis, component, field=field, out=out, threads=2) is out
            np.testing.assert_array_equal(out, reference)
    empty = field_fn(gi.Basis([]), component, threads=1)
    assert empty.shape == (3, 0, 0)


@pytest.mark.parametrize("powers", POWERS + [(3, 1, 0)])
@pytest.mark.parametrize("field", FIELDS)
def test_moment_field_derivatives(powers, field):
    basis = gi.Basis(shells(np.array([[0.1, -0.2, 0.3], [-0.4, 0.2, 0.1]])))
    actual = gi.moment_magnetic_derivatives(
        basis, powers, origin=ORIGIN, field=field, threads=1
    )
    for axis in range(3):
        expected = richardson(
            lambda step: gi.moment(
                basis,
                powers,
                origin=ORIGIN,
                field=gi.MagneticField(
                    np.array(field.B) + step * np.eye(3)[axis], field.gauge_origin
                ),
            )
        )
        np.testing.assert_allclose(actual[axis], expected, atol=3e-10, rtol=3e-10)
    np.testing.assert_allclose(actual, actual.swapaxes(-1, -2).conj(), atol=3e-13)
    out = np.empty_like(actual)
    assert (
        gi.moment_magnetic_derivatives(
            basis,
            powers,
            origin=ORIGIN,
            field=field,
            out=out,
            threads=2 if gi.openmp_enabled() else 1,
        )
        is out
    )
    np.testing.assert_array_equal(out, actual)


@pytest.mark.parametrize("angular_momentum", range(5))
@pytest.mark.parametrize("field", FIELDS)
def test_primitive_properties_through_g_against_reference(angular_momentum, field):
    centers = np.array([[0.1, -0.2, 0.3], [-0.3, 0.4, 0.2]])
    angular = [(angular_momentum, 0, 0), (0, 1, 0)]
    exponents = [0.8, 0.6]
    primitives = [
        gi.PrimitiveGaussian(e, c, a)
        for e, c, a in zip(exponents, centers, angular, strict=True)
    ]
    rf = ReferenceField(field.B, field.gauge_origin)
    for operator, reference in [
        ("gradient", reference_gradient),
        ("momentum", reference_momentum),
        ("moment", reference_moment),
    ]:
        components = [(1, 1, 0)] if operator == "moment" else ["x", "y", "z"]
        for component in components:
            extra = {"origin": ORIGIN} if operator == "moment" else {}
            actual = getattr(gi, "primitive_" + operator + "_magnetic_derivatives")(
                *primitives, component, field=field, **extra
            )
            for axis in range(3):

                def changed(step):
                    refp = [
                        ReferencePrimitive(e, c, a)
                        for e, c, a in zip(exponents, centers, angular, strict=True)
                    ]
                    ref_field = ReferenceField(
                        np.array(field.B) + step * np.eye(3)[axis], field.gauge_origin
                    )
                    return reference(*refp, component, field=ref_field, **extra)

                np.testing.assert_allclose(
                    actual[axis], richardson(changed), atol=3e-10, rtol=3e-10
                )
            if operator == "moment":
                continue
            actual = getattr(gi, "primitive_" + operator + "_center_derivatives")(
                *primitives, component, field=field
            )
            for center in range(2):
                for axis in range(3):

                    def changed(step):
                        c = centers.copy()
                        c[center, axis] += step
                        refp = [
                            ReferencePrimitive(e, r, a)
                            for e, r, a in zip(exponents, c, angular, strict=True)
                        ]
                        return reference(*refp, component, field=rf)

                    np.testing.assert_allclose(
                        actual[center, axis],
                        richardson(changed),
                        atol=3e-10,
                        rtol=3e-10,
                    )


@pytest.mark.parametrize("component", ["x", "y", "z"])
def test_shell_assembly_and_momentum_relation(component):
    a, b = shells([[0.1, -0.2, 0.3], [-0.4, 0.2, 0.1]])
    basis = gi.Basis([a, b])
    for kind in ["center", "magnetic"]:
        gradient_fn = getattr(gi, "gradient_" + kind + "_derivatives_shell")
        momentum_fn = getattr(gi, "momentum_" + kind + "_derivatives_shell")
        gradient = gradient_fn(a, b, component, field=FIELDS[1])
        out = np.empty_like(gradient)
        assert gradient_fn(a, b, component, field=FIELDS[1], out=out) is out
        np.testing.assert_array_equal(out, gradient)
        np.testing.assert_array_equal(
            momentum_fn(a, b, component, field=FIELDS[1]), -1j * gradient
        )
        if kind == "center":
            matrices = gi.gradient_nuclear_derivatives(
                basis, component, field=FIELDS[1], threads=1
            )
            np.testing.assert_array_equal(matrices[:, :, 0:6, 6:], gradient)
        else:
            matrices = gi.gradient_magnetic_derivatives(
                basis, component, field=FIELDS[1], threads=1
            )
            np.testing.assert_array_equal(matrices[:, 0:6, 6:], gradient)
    for fn in [gi.gradient_nuclear_derivatives, gi.momentum_magnetic_derivatives]:
        with pytest.raises(ValueError):
            fn(basis, "bad")
        with pytest.raises((TypeError, ValueError)):
            fn(basis, component, out=np.empty(1))
        with pytest.raises(ValueError):
            fn(basis, component, threads=0)
