"""Analytic moment derivatives against values, not another derivative formula."""

# Loop-local closures are evaluated synchronously by richardson.
# ruff: noqa: B023

import giao_integrals as gi
import numpy as np
import pytest
from giao_reference import MagneticField as ReferenceField
from giao_reference import PrimitiveGaussian as ReferencePrimitive
from giao_reference import primitive_moment as reference_moment

ORIGIN = np.array([0.37, -0.29, 0.43])
FIELDS = [gi.MagneticField(), gi.MagneticField((0.17, -0.11, 0.23), (0.2, -0.3, 0.1))]
POWERS = [
    (x, y, z) for x in range(3) for y in range(3) for z in range(3) if x + y + z <= 2
]


def richardson(function, step=4e-4):
    coarse = (function(step) - function(-step)) / (2 * step)
    fine = (function(step / 2) - function(-step / 2)) / step
    return (4 * fine - coarse) / 3


def shells(centers):
    return [
        gi.Shell(centers[0], 1, [0.7, 1.2], [[0.6, -0.2], [0.3, 0.8]]),
        gi.Shell(centers[1], 2, [0.9], [1.0]),
    ]


@pytest.mark.parametrize("powers", POWERS + [(3, 1, 0)])
@pytest.mark.parametrize("field", FIELDS)
def test_moment_basis_and_origin_finite_differences(powers, field):
    centers = np.array([[0.1, -0.2, 0.3], [-0.4, 0.2, 0.1]])
    basis = gi.Basis(shells(centers))
    derivatives, origin = gi.moment_nuclear_derivatives(
        basis, powers, origin=ORIGIN, field=field, threads=1
    )
    value = gi.moment(basis, powers, origin=ORIGIN, field=field)
    for center in range(2):
        for axis in range(3):

            def displaced(step):
                changed = centers.copy()
                changed[center, axis] += step
                return gi.moment(
                    gi.Basis(shells(changed)), powers, origin=ORIGIN, field=field
                )

            np.testing.assert_allclose(
                derivatives[center, axis], richardson(displaced), atol=3e-10, rtol=3e-10
            )
    for axis in range(3):
        direction = np.eye(3)[axis]
        numerical = richardson(
            lambda step: gi.moment(
                basis, powers, origin=ORIGIN + step * direction, field=field
            )
        )
        np.testing.assert_allclose(origin[axis], numerical, atol=3e-10, rtol=3e-10)
    np.testing.assert_allclose(
        derivatives, derivatives.swapaxes(-1, -2).conj(), atol=2e-13
    )
    np.testing.assert_allclose(origin, origin.swapaxes(-1, -2).conj(), atol=2e-13)
    # Raw London AO matrices are translation-covariant, not invariant at B != 0.
    wave = [np.cross(field.B, center) / 2 for center in centers]
    ao_wave = np.repeat(wave, [6, 6], axis=0)
    phase = 1j * (ao_wave[:, None, :] - ao_wave[None, :, :]).transpose(2, 0, 1)
    np.testing.assert_allclose(
        derivatives.sum(axis=0) + origin, phase * value, atol=3e-13
    )
    for axis in range(3):
        gauge = np.array(field.gauge_origin)
        numerical = richardson(
            lambda step: gi.moment(
                basis,
                powers,
                origin=ORIGIN,
                field=gi.MagneticField(field.B, gauge + step * np.eye(3)[axis]),
            )
        )
        np.testing.assert_allclose(numerical, 0, atol=3e-10)


@pytest.mark.parametrize("angular_momentum", range(5))
@pytest.mark.parametrize("field", FIELDS)
def test_primitive_and_shell_derivatives_through_g(angular_momentum, field):
    a = gi.Shell((0.1, -0.2, 0.3), angular_momentum, [0.8], [1.0])
    b = gi.Shell((-0.3, 0.4, 0.2), 0, [0.6], [1.0])
    powers = (1, 1, 0)
    block = gi.moment_center_derivatives_shell(
        a, b, list(map(int, powers)), origin=ORIGIN, field=field
    )
    origins = gi.moment_origin_derivatives_shell(
        a, b, list(map(int, powers)), origin=ORIGIN, field=field
    )
    for k, angular in enumerate(gi.cartesian_components(angular_momentum)):
        pa = gi.PrimitiveGaussian(0.8, a.center, angular)
        pb = gi.PrimitiveGaussian(0.6, b.center, (0, 0, 0))
        np.testing.assert_allclose(
            block[:, :, k, 0],
            gi.primitive_moment_center_derivatives(
                pa, pb, powers, origin=ORIGIN, field=field
            ),
            atol=3e-14,
        )
        np.testing.assert_allclose(
            origins[:, k, 0],
            gi.primitive_moment_origin_derivatives(
                pa, pb, powers, origin=ORIGIN, field=field
            ),
            atol=3e-14,
        )
        for center in range(2):
            for axis in range(3):

                def displaced(step):
                    ca, cb = np.array(a.center), np.array(b.center)
                    (ca if center == 0 else cb)[axis] += step
                    return gi.primitive_moment(
                        gi.PrimitiveGaussian(0.8, ca, angular),
                        gi.PrimitiveGaussian(0.6, cb, (0, 0, 0)),
                        powers,
                        origin=ORIGIN,
                        field=field,
                    )

                np.testing.assert_allclose(
                    block[center, axis, k, 0],
                    richardson(displaced),
                    atol=2e-10,
                    rtol=2e-10,
                )


def test_moment_out_threading_same_shell_and_errors():
    basis = gi.Basis([gi.Shell((0.2, -0.1, 0.3), 1, [0.8], [1.0])])
    result = gi.moment_nuclear_derivatives(basis, (1, 0, 0), field=FIELDS[1], threads=1)
    out = tuple(np.empty_like(x) for x in result)
    filled = gi.moment_nuclear_derivatives(
        basis,
        (1, 0, 0),
        field=FIELDS[1],
        out=out,
        threads=2 if gi.openmp_enabled() else 1,
    )
    for actual, expected, buffer in zip(filled, result, out, strict=True):
        assert actual is buffer
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(
        result[0][0],
        sum(
            gi.moment_center_derivatives_shell(
                basis.shells[0], basis.shells[0], (1, 0, 0), field=FIELDS[1]
            )
        ),
        atol=1e-14,
    )
    for bad in [np.empty(1), (None,), (np.empty(result[0].shape, dtype=float), None)]:
        with pytest.raises((TypeError, ValueError)):
            gi.moment_nuclear_derivatives(basis, (1, 0, 0), out=bad)
    with pytest.raises(ValueError, match="overlap"):
        gi.moment_nuclear_derivatives(basis, (1, 0, 0), out=(out[0], out[0][0]))
    for bad in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            gi.moment_nuclear_derivatives(basis, (1, 0, 0), threads=bad)
    empty = gi.moment_nuclear_derivatives(gi.Basis([]), (0, 0, 0), threads=1)
    assert empty[0].shape == (0, 3, 0, 0)
    assert empty[1].shape == (3, 0, 0)


@pytest.mark.parametrize("field", FIELDS)
def test_against_reference_and_gaussian_phase_separation(field):
    centers = np.array([[0.1, -0.2, 0.3], [-0.3, 0.4, 0.2]])
    angular = [(2, 1, 1), (1, 0, 0)]
    exponents = [0.8, 0.6]
    primitives = [
        gi.PrimitiveGaussian(e, c, angular_momentum)
        for e, c, angular_momentum in zip(exponents, centers, angular, strict=True)
    ]
    ref_field = ReferenceField(field.B, field.gauge_origin)
    analytic = gi.primitive_moment_center_derivatives(
        *primitives, (1, 1, 0), origin=ORIGIN, field=field
    )
    for center in range(2):
        for axis in range(3):

            def reference_displaced(step):
                changed = centers.copy()
                changed[center, axis] += step
                a, b = [
                    ReferencePrimitive(e, c, angular_momentum)
                    for e, c, angular_momentum in zip(
                        exponents, changed, angular, strict=True
                    )
                ]
                return reference_moment(a, b, (1, 1, 0), origin=ORIGIN, field=ref_field)

            np.testing.assert_allclose(
                analytic[center, axis],
                richardson(reference_displaced),
                atol=2e-10,
                rtol=2e-10,
            )
    # Check ordinary Gaussian and London-phase centre terms separately.
    a, b = [
        gi.PrimitiveGaussian(e, c, (0, 0, 0))
        for e, c in zip(exponents, centers, strict=True)
    ]
    response = gi.primitive_moment_center_derivatives(
        a, b, (1, 0, 0), origin=ORIGIN, field=field
    )
    # Ordinary s-Gaussian derivative is 2*alpha times a p-Gaussian (unnormalized
    # with the original s normalization). Compare the remainder to i*d(k)/dR*r.
    for center in range(2):
        for axis in range(3):
            pair = [a, b]
            original = pair[center]
            coefficient = gi.primitive_normalization(original.exponent, (0, 0, 0))
            raised = np.eye(3, dtype=int)[axis]
            pair[center] = gi.PrimitiveGaussian(
                original.exponent,
                original.center,
                raised.tolist(),
                coefficient=coefficient,
                normalized=False,
            )
            ordinary = (
                2
                * original.exponent
                * gi.primitive_moment(*pair, (1, 0, 0), origin=ORIGIN, field=field)
            )
            phase = 0j
            dk = np.cross(field.B, np.eye(3)[axis]) / 2
            for q in range(3):
                powers = np.array([1, 0, 0]) + np.eye(3, dtype=int)[q]
                coordinate = gi.primitive_moment(
                    a, b, list(map(int, powers)), origin=ORIGIN, field=field
                ) + ORIGIN[q] * gi.primitive_moment(
                    a, b, (1, 0, 0), origin=ORIGIN, field=field
                )
                phase += (1j if center == 0 else -1j) * dk[q] * coordinate
            np.testing.assert_allclose(
                response[center, axis] - ordinary, phase, atol=3e-14
            )
