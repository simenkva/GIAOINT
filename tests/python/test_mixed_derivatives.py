"""Mixed derivatives: two-dimensional Richardson differences and both orders."""

# ruff: noqa: B023
import giao_integrals as gi
import numpy as np
import pytest
from test_moment_derivatives import FIELDS, richardson

NUCLEI = [gi.Nucleus(1.3, (0.4, -0.3, 0.2)), gi.Nucleus(0.7, (-0.2, 0.5, -0.4))]


def mixed_difference(value, step=2e-3):
    def central(h):
        return (value(h, h) - value(h, -h) - value(-h, h) + value(-h, -h)) / (4 * h * h)

    return (4 * central(step / 2) - central(step)) / 3


def field_shift(field, axis, step):
    return gi.MagneticField(
        np.array(field.B) + step * np.eye(3)[axis], field.gauge_origin
    )


@pytest.mark.parametrize(
    "operator", ["overlap", "kinetic", "magnetic_kinetic", "nuclear_attraction"]
)
@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("angular_momentum", [0, 4])
def test_primitive_mixed_against_values_and_both_orders(
    operator, field, angular_momentum
):
    centers = np.array([[0.1, -0.2, 0.3], [-0.3, 0.4, 0.2]])
    angular = [(angular_momentum, 0, 0), (0, 1, 0)]

    def primitives(changed):
        return [
            gi.PrimitiveGaussian(e, c, a)
            for e, c, a in zip([0.8, 0.6], changed, angular, strict=True)
        ]

    extra = [NUCLEI] if operator == "nuclear_attraction" else []
    actual = getattr(gi, "primitive_" + operator + "_mixed_derivatives")(
        *primitives(centers), *extra, field=field
    )
    for center in range(2):
        for r in range(3):
            for b in range(3):

                def value(dr, db):
                    changed = centers.copy()
                    changed[center, r] += dr
                    return getattr(gi, "primitive_" + operator)(
                        *primitives(changed), *extra, field=field_shift(field, b, db)
                    )

                np.testing.assert_allclose(
                    actual[center, r, b], mixed_difference(value), atol=8e-9, rtol=8e-8
                )
                numerical = richardson(
                    lambda step: getattr(
                        gi, "primitive_" + operator + "_center_derivatives"
                    )(*primitives(centers), *extra, field=field_shift(field, b, step))[
                        center, r
                    ]
                )
                np.testing.assert_allclose(
                    actual[center, r, b], numerical, atol=4e-10, rtol=4e-9
                )

                def reversed_order(step):
                    changed = centers.copy()
                    changed[center, r] += step
                    return getattr(
                        gi, "primitive_" + operator + "_magnetic_derivatives"
                    )(*primitives(changed), *extra, field=field)[b]

                np.testing.assert_allclose(
                    actual[center, r, b],
                    richardson(reversed_order),
                    atol=4e-10,
                    rtol=4e-9,
                )


@pytest.mark.parametrize("field", FIELDS)
def test_attraction_potential_mixed(field):
    a = gi.PrimitiveGaussian(0.8, (0.1, -0.2, 0.3), (1, 0, 0))
    b = gi.PrimitiveGaussian(0.6, (-0.3, 0.4, 0.2), (0, 1, 0))
    actual = gi.primitive_nuclear_attraction_nucleus_mixed_derivatives(
        a, b, NUCLEI, field=field
    )
    for nucleus in range(2):
        for r in range(3):
            for axis in range(3):

                def value(dr, db):
                    changed = list(NUCLEI)
                    changed[nucleus] = gi.Nucleus(
                        NUCLEI[nucleus].charge,
                        np.array(NUCLEI[nucleus].center) + dr * np.eye(3)[r],
                    )
                    return gi.primitive_nuclear_attraction(
                        a, b, changed, field=field_shift(field, axis, db)
                    )

                np.testing.assert_allclose(
                    actual[nucleus, r, axis],
                    mixed_difference(value),
                    atol=8e-9,
                    rtol=8e-8,
                )
                numerical = richardson(
                    lambda step: gi.primitive_nuclear_attraction_nucleus_derivatives(
                        a, b, NUCLEI, field=field_shift(field, axis, step)
                    )[nucleus, r]
                )
                np.testing.assert_allclose(
                    actual[nucleus, r, axis], numerical, atol=4e-10, rtol=4e-9
                )
    empty = gi.primitive_nuclear_attraction_nucleus_mixed_derivatives(
        a, b, [], field=field
    )
    assert empty.shape == (0, 3, 3)


@pytest.mark.parametrize(
    "operator", ["overlap", "kinetic", "magnetic_kinetic", "nuclear_attraction"]
)
@pytest.mark.parametrize("field", FIELDS)
def test_mixed_shell_and_basis(operator, field):
    centers = np.array([[0.1, -0.2, 0.3], [-0.3, 0.4, 0.2]])

    def shells(c):
        return [
            gi.Shell(c[0], 0, [0.7, 1.2], [0.6, -0.2]),
            gi.Shell(c[1], 1, [0.9], [1.0]),
        ]

    a, b = shells(centers)
    basis = gi.Basis([a, b])
    extra = [NUCLEI] if operator == "nuclear_attraction" else []
    result = getattr(gi, operator + "_mixed_derivatives")(
        basis, *extra, field=field, threads=1
    )
    if operator == "nuclear_attraction":
        actual, potential = result
        pot_shell = gi.nuclear_attraction_nucleus_mixed_derivatives_shell(
            a, b, NUCLEI, field=field
        )
        np.testing.assert_array_equal(potential[:, :, :, 0:1, 1:], pot_shell)
    else:
        actual = result
    shell = getattr(gi, operator + "_mixed_derivatives_shell")(
        a, b, *extra, field=field
    )
    np.testing.assert_array_equal(actual[:, :, :, 0:1, 1:], shell)
    # Same-shell contributions must add both slots, not overwrite one another.
    aa = getattr(gi, operator + "_mixed_derivatives_shell")(a, a, *extra, field=field)
    np.testing.assert_allclose(actual[0, :, :, 0:1, 0:1], aa.sum(axis=0), atol=2e-13)
    np.testing.assert_allclose(actual, actual.swapaxes(-1, -2).conj(), atol=2e-12)
    for r in range(3):
        for axis in range(3):
            if operator == "nuclear_attraction":
                derivative = gi.nuclear_attraction_nuclear_derivatives

                def field_response(step):
                    return derivative(
                        basis, NUCLEI, field=field_shift(field, axis, step)
                    )[0][0, r]

            else:
                derivative = getattr(gi, operator + "_nuclear_derivatives")

                def field_response(step):
                    return derivative(basis, field=field_shift(field, axis, step))[0, r]

            np.testing.assert_allclose(
                actual[0, r, axis], richardson(field_response), atol=4e-10, rtol=4e-9
            )
    count = 2 if gi.openmp_enabled() else 1
    if operator == "nuclear_attraction":
        out = tuple(np.empty_like(x) for x in result)
        repeated = gi.nuclear_attraction_mixed_derivatives(
            basis, NUCLEI, field=field, out=out, threads=count
        )
        for value, buffer, reference in zip(repeated, out, result, strict=True):
            assert value is buffer
            np.testing.assert_array_equal(value, reference)
    else:
        out = np.empty_like(result)
        assert (
            getattr(gi, operator + "_mixed_derivatives")(
                basis, field=field, out=out, threads=count
            )
            is out
        )
        np.testing.assert_array_equal(out, result)
    shell_out = np.empty_like(shell)
    assert (
        getattr(gi, operator + "_mixed_derivatives_shell")(
            a, b, *extra, field=field, out=shell_out
        )
        is shell_out
    )
    np.testing.assert_array_equal(shell_out, shell)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("angular_momentum", [0, 4])
def test_eri_mixed_primitive_shell_and_symmetries(field, angular_momentum):
    centers = np.array(
        [[0.1, -0.2, 0.3], [-0.3, 0.4, 0.2], [0.3, 0.2, -0.4], [-0.2, -0.1, 0.4]]
    )
    angular = [(angular_momentum, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
    exponents = [0.8, 0.6, 0.7, 0.9]

    def primitives(c):
        return [
            gi.PrimitiveGaussian(e, r, a)
            for e, r, a in zip(exponents, c, angular, strict=True)
        ]

    p = primitives(centers)
    actual = gi.primitive_eri_mixed_derivatives(*p, field=field)
    for center in range(4):
        for r in range(3):
            for b in range(3):

                def value(dr, db):
                    c = centers.copy()
                    c[center, r] += dr
                    return gi.primitive_eri(
                        *primitives(c), field=field_shift(field, b, db)
                    )

                np.testing.assert_allclose(
                    actual[center, r, b], mixed_difference(value), atol=8e-9, rtol=8e-8
                )
                numerical = richardson(
                    lambda step: gi.primitive_eri_center_derivatives(
                        *p, field=field_shift(field, b, step)
                    )[center, r]
                )
                np.testing.assert_allclose(
                    actual[center, r, b], numerical, atol=4e-10, rtol=4e-9
                )
    exchanged = gi.primitive_eri_mixed_derivatives(p[2], p[3], p[0], p[1], field=field)
    reversed_pair = gi.primitive_eri_mixed_derivatives(
        p[1], p[0], p[3], p[2], field=field
    )
    np.testing.assert_allclose(actual, exchanged[[2, 3, 0, 1]], atol=4e-12)
    np.testing.assert_allclose(actual, reversed_pair[[1, 0, 3, 2]].conj(), atol=4e-12)
    shells = [
        gi.Shell(c, angular, [e], [1.0])
        for c, angular, e in zip(
            centers, [angular_momentum, 0, 0, 0], exponents, strict=True
        )
    ]
    shell = gi.eri_mixed_derivatives_shell(*shells, field=field)
    np.testing.assert_allclose(shell[:, :, :, 0, 0, 0, 0], actual, atol=3e-13)
    out = np.empty_like(shell)
    assert gi.eri_mixed_derivatives_shell(*shells, field=field, out=out) is out
    np.testing.assert_array_equal(out, shell)


def test_empty_and_bad_mixed_outputs():
    basis = gi.Basis([])
    assert gi.overlap_mixed_derivatives(basis, threads=1).shape == (0, 3, 3, 0, 0)
    centers, potentials = gi.nuclear_attraction_mixed_derivatives(basis, [], threads=1)
    assert centers.shape == potentials.shape == (0, 3, 3, 0, 0)
    with pytest.raises((TypeError, ValueError)):
        gi.overlap_mixed_derivatives(basis, out=np.empty(1))
    with pytest.raises(ValueError):
        gi.overlap_mixed_derivatives(basis, threads=0)


@pytest.mark.parametrize(
    "operator", ["overlap", "kinetic", "magnetic_kinetic", "nuclear_attraction"]
)
@pytest.mark.parametrize("field", FIELDS)
def test_mixed_translation_covariance(operator, field):
    shells = [
        gi.Shell((0.1, -0.2, 0.3), 0, [0.8], [1.0]),
        gi.Shell((-0.3, 0.4, 0.2), 1, [0.6], [1.0]),
    ]
    basis = gi.Basis(shells)
    extra = [NUCLEI] if operator == "nuclear_attraction" else []
    result = getattr(gi, operator + "_mixed_derivatives")(
        basis, *extra, field=field, threads=1
    )
    nuclear = result[0] if operator == "nuclear_attraction" else result
    potential = result[1] if operator == "nuclear_attraction" else 0
    total = nuclear.sum(axis=0) + (
        potential.sum(axis=0) if isinstance(potential, np.ndarray) else 0
    )
    centers = np.repeat([s.center for s in shells], [1, 3], axis=0)
    delta = centers[:, None, :] - centers[None, :, :]
    k = np.cross(field.B, delta) / 2
    value = getattr(gi, operator)(basis, *extra, field=field)
    magnetic = getattr(gi, operator + "_magnetic_derivatives")(
        basis, *extra, field=field
    )
    for r in range(3):
        gauge = richardson(
            lambda step: getattr(gi, operator + "_magnetic_derivatives")(
                basis,
                *extra,
                field=gi.MagneticField(
                    field.B, np.array(field.gauge_origin) + step * np.eye(3)[r]
                ),
            )
        )
        for b in range(3):
            dk = np.cross(np.eye(3)[b], delta) / 2
            expected = 1j * dk[:, :, r] * value + 1j * k[:, :, r] * magnetic[b]
            np.testing.assert_allclose(
                total[r, b] + gauge[b], expected, atol=5e-10, rtol=5e-9
            )


@pytest.mark.parametrize(
    "operator", ["overlap", "kinetic", "magnetic_kinetic", "nuclear_attraction", "eri"]
)
@pytest.mark.parametrize("field", FIELDS)
def test_mixed_against_independent_python_values(operator, field):
    import giao_reference as ref

    centers = np.array(
        [[0.1, -0.2, 0.3], [-0.3, 0.4, 0.2], [0.3, 0.2, -0.4], [-0.2, -0.1, 0.4]]
    )
    angular = [(1, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 0)]
    exponents = [0.8, 0.6, 0.7, 0.9]
    count = 4 if operator == "eri" else 2
    p = [
        gi.PrimitiveGaussian(exponents[k], centers[k], angular[k]) for k in range(count)
    ]
    extra = [NUCLEI] if operator == "nuclear_attraction" else []
    actual = getattr(gi, "primitive_" + operator + "_mixed_derivatives")(
        *p, *extra, field=field
    )

    def value(dr, db):
        c = centers.copy()
        c[0, 1] += dr
        p = [
            ref.PrimitiveGaussian(exponents[k], c[k], angular[k]) for k in range(count)
        ]
        extra = (
            [[ref.Nucleus(x.charge, x.center) for x in NUCLEI]]
            if operator == "nuclear_attraction"
            else []
        )
        f = ref.MagneticField(np.array(field.B) + db * np.eye(3)[2], field.gauge_origin)
        return complex(getattr(ref, "primitive_" + operator)(*p, *extra, field=f))

    np.testing.assert_allclose(
        actual[0, 1, 2], mixed_difference(value), atol=8e-9, rtol=8e-8
    )
