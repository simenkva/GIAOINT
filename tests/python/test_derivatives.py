# All loop-local finite-difference closures are consumed synchronously.
# ruff: noqa: B023

from collections.abc import Callable

import giao_integrals as gi
import numpy as np
import pytest

FIELD = gi.MagneticField(B=(0.17, -0.11, 0.23), gauge_origin=(0.2, -0.3, 0.1))


def primitive(
    exponent: float, center: np.ndarray, angular: tuple[int, int, int]
) -> gi.PrimitiveGaussian:
    return gi.PrimitiveGaussian(exponent, center, angular, coefficient=0.8)


def shifted_primitive(
    value: gi.PrimitiveGaussian, center_index: int, axis: int, step: float
) -> gi.PrimitiveGaussian:
    center = np.asarray(value.center, dtype=float)
    center[axis] += step
    return gi.PrimitiveGaussian(
        value.exponent,
        center,
        value.angular,
        coefficient=value.coefficient,
        normalized=value.normalized,
    )


def shifted_field(field: gi.MagneticField, axis: int, step: float) -> gi.MagneticField:
    magnetic = np.asarray(field.B, dtype=float)
    magnetic[axis] += step
    return gi.MagneticField(magnetic, field.gauge_origin)


def richardson(function: Callable[[float], np.ndarray | complex], step=2.0e-4):
    coarse = (function(step) - function(-step)) / (2.0 * step)
    half = step / 2.0
    fine = (function(half) - function(-half)) / (2.0 * half)
    return fine + (fine - coarse) / 3.0


@pytest.fixture
def primitives():
    return (
        primitive(0.73, np.array([0.2, -0.4, 0.1]), (1, 0, 1)),
        primitive(1.11, np.array([-0.3, 0.5, -0.2]), (0, 1, 0)),
    )


@pytest.mark.parametrize(
    ("value", "center_derivative", "magnetic_derivative"),
    [
        (
            lambda a, b, field: gi.primitive_overlap(a, b, field=field),
            gi.primitive_overlap_center_derivatives,
            gi.primitive_overlap_magnetic_derivatives,
        ),
        (
            lambda a, b, field: gi.primitive_kinetic(a, b, field=field),
            gi.primitive_kinetic_center_derivatives,
            gi.primitive_kinetic_magnetic_derivatives,
        ),
        (
            lambda a, b, field: gi.primitive_magnetic_kinetic(a, b, field=field),
            gi.primitive_magnetic_kinetic_center_derivatives,
            gi.primitive_magnetic_kinetic_magnetic_derivatives,
        ),
    ],
)
def test_primitive_pair_derivatives_match_multistep_finite_differences(
    primitives, value, center_derivative, magnetic_derivative
):
    a, b = primitives
    analytic_center = center_derivative(a, b, field=FIELD)
    analytic_magnetic = magnetic_derivative(a, b, field=FIELD)

    for center in range(2):
        for axis in range(3):
            numerical = richardson(
                lambda step: value(
                    shifted_primitive(a, center, axis, step) if center == 0 else a,
                    shifted_primitive(b, center, axis, step) if center == 1 else b,
                    FIELD,
                )
            )
            np.testing.assert_allclose(
                analytic_center[center, axis], numerical, atol=2.0e-9, rtol=2.0e-8
            )

    for axis in range(3):
        numerical = richardson(
            lambda step: value(a, b, shifted_field(FIELD, axis, step))
        )
        np.testing.assert_allclose(
            analytic_magnetic[axis], numerical, atol=2.0e-9, rtol=2.0e-8
        )


def test_primitive_attraction_derivatives_match_multistep_finite_differences(
    primitives,
):
    a, b = primitives
    nuclei = [
        gi.Nucleus(1.4, (0.1, -0.2, 0.3)),
        gi.Nucleus(0.7, (-0.6, 0.2, 0.5)),
    ]
    center = gi.primitive_nuclear_attraction_center_derivatives(
        a, b, nuclei, field=FIELD
    )
    magnetic = gi.primitive_nuclear_attraction_magnetic_derivatives(
        a, b, nuclei, field=FIELD
    )
    potential = gi.primitive_nuclear_attraction_nucleus_derivatives(
        a, b, nuclei, field=FIELD
    )

    for center_index in range(2):
        for axis in range(3):
            numerical = richardson(
                lambda step: gi.primitive_nuclear_attraction(
                    (
                        shifted_primitive(a, center_index, axis, step)
                        if center_index == 0
                        else a
                    ),
                    (
                        shifted_primitive(b, center_index, axis, step)
                        if center_index == 1
                        else b
                    ),
                    nuclei,
                    field=FIELD,
                )
            )
            np.testing.assert_allclose(
                center[center_index, axis], numerical, atol=3.0e-9, rtol=3.0e-8
            )

    for axis in range(3):
        numerical = richardson(
            lambda step: gi.primitive_nuclear_attraction(
                a, b, nuclei, field=shifted_field(FIELD, axis, step)
            )
        )
        np.testing.assert_allclose(magnetic[axis], numerical, atol=3.0e-9, rtol=3.0e-8)

    for nucleus_index in range(len(nuclei)):
        for axis in range(3):

            def displaced_nuclei(step):
                changed = list(nuclei)
                position = np.asarray(nuclei[nucleus_index].center, dtype=float)
                position[axis] += step
                changed[nucleus_index] = gi.Nucleus(
                    nuclei[nucleus_index].charge, position
                )
                return gi.primitive_nuclear_attraction(a, b, changed, field=FIELD)

            numerical = richardson(displaced_nuclei)
            np.testing.assert_allclose(
                potential[nucleus_index, axis],
                numerical,
                atol=3.0e-9,
                rtol=3.0e-8,
            )


def test_primitive_eri_derivatives_match_multistep_finite_differences():
    primitives = [
        primitive(0.8, np.array([0.1, -0.2, 0.0]), (1, 0, 0)),
        primitive(1.0, np.array([-0.3, 0.1, 0.2]), (0, 0, 0)),
        primitive(0.7, np.array([0.4, 0.2, -0.1]), (0, 1, 0)),
        primitive(1.2, np.array([-0.2, -0.4, 0.3]), (0, 0, 0)),
    ]
    center = gi.primitive_eri_center_derivatives(*primitives, field=FIELD)
    magnetic = gi.primitive_eri_magnetic_derivatives(*primitives, field=FIELD)

    for center_index in range(4):
        for axis in range(3):

            def displaced(step):
                changed = list(primitives)
                changed[center_index] = shifted_primitive(
                    changed[center_index], center_index, axis, step
                )
                return gi.primitive_eri(*changed, field=FIELD)

            numerical = richardson(displaced)
            np.testing.assert_allclose(
                center[center_index, axis], numerical, atol=4.0e-9, rtol=4.0e-8
            )

    for axis in range(3):
        numerical = richardson(
            lambda step: gi.primitive_eri(
                *primitives, field=shifted_field(FIELD, axis, step)
            )
        )
        np.testing.assert_allclose(magnetic[axis], numerical, atol=4.0e-9, rtol=4.0e-8)


def test_eri_derivative_complex_symmetries():
    values = [
        primitive(0.8, np.array([0.1, -0.2, 0.0]), (1, 0, 0)),
        primitive(1.0, np.array([-0.3, 0.1, 0.2]), (0, 0, 0)),
        primitive(0.7, np.array([0.4, 0.2, -0.1]), (0, 1, 0)),
        primitive(1.2, np.array([-0.2, -0.4, 0.3]), (0, 0, 0)),
    ]
    center = gi.primitive_eri_center_derivatives(*values, field=FIELD)
    magnetic = gi.primitive_eri_magnetic_derivatives(*values, field=FIELD)

    exchanged_center = gi.primitive_eri_center_derivatives(
        values[2], values[3], values[0], values[1], field=FIELD
    )
    exchanged_magnetic = gi.primitive_eri_magnetic_derivatives(
        values[2], values[3], values[0], values[1], field=FIELD
    )
    np.testing.assert_allclose(center, exchanged_center[[2, 3, 0, 1]])
    np.testing.assert_allclose(magnetic, exchanged_magnetic)

    reversed_center = gi.primitive_eri_center_derivatives(
        values[1], values[0], values[3], values[2], field=FIELD
    )
    reversed_magnetic = gi.primitive_eri_magnetic_derivatives(
        values[1], values[0], values[3], values[2], field=FIELD
    )
    np.testing.assert_allclose(center, reversed_center[[1, 0, 3, 2]].conj())
    np.testing.assert_allclose(magnetic, reversed_magnetic.conj())


@pytest.mark.parametrize("seed", range(8))
def test_randomized_overlap_derivatives_through_angular_momentum_four(seed):
    rng = np.random.default_rng(seed)
    components_a = gi.cartesian_components(seed % 5)
    components_b = gi.cartesian_components((3 * seed + 1) % 5)
    angular_a = components_a[seed % len(components_a)]
    angular_b = components_b[seed % len(components_b)]
    a = primitive(0.4 + rng.random(), rng.uniform(-0.7, 0.7, 3), angular_a)
    b = primitive(0.4 + rng.random(), rng.uniform(-0.7, 0.7, 3), angular_b)
    field = gi.MagneticField(rng.uniform(-0.25, 0.25, 3), rng.uniform(-0.5, 0.5, 3))
    center = seed % 2
    axis = (seed + 1) % 3
    analytic_center = gi.primitive_overlap_center_derivatives(a, b, field=field)
    numerical_center = richardson(
        lambda step: gi.primitive_overlap(
            shifted_primitive(a, center, axis, step) if center == 0 else a,
            shifted_primitive(b, center, axis, step) if center == 1 else b,
            field=field,
        ),
        step=4.0e-4,
    )
    np.testing.assert_allclose(
        analytic_center[center, axis],
        numerical_center,
        atol=2.0e-8,
        rtol=2.0e-7,
    )

    field_axis = (seed + 2) % 3
    analytic_field = gi.primitive_overlap_magnetic_derivatives(a, b, field=field)
    numerical_field = richardson(
        lambda step: gi.primitive_overlap(
            a, b, field=shifted_field(field, field_axis, step)
        ),
        step=4.0e-4,
    )
    np.testing.assert_allclose(
        analytic_field[field_axis],
        numerical_field,
        atol=2.0e-8,
        rtol=2.0e-7,
    )


def make_shells(centers):
    return [
        gi.Shell(centers[0], 0, [0.7, 1.2], [0.6, -0.2]),
        gi.Shell(centers[1], 1, [0.9], [1.0]),
    ]


def test_basis_overlap_derivatives_shapes_out_and_symmetry():
    centers = [np.array([0.1, -0.2, 0.3]), np.array([-0.4, 0.2, 0.1])]
    basis = gi.Basis(make_shells(centers))
    nuclear = np.empty((2, 3, 4, 4), dtype=np.complex128)
    magnetic = np.empty((3, 4, 4), dtype=np.complex128)
    assert gi.overlap_nuclear_derivatives(basis, field=FIELD, out=nuclear) is nuclear
    assert gi.overlap_magnetic_derivatives(basis, field=FIELD, out=magnetic) is magnetic
    np.testing.assert_allclose(nuclear, nuclear.swapaxes(-1, -2).conj())
    np.testing.assert_allclose(magnetic, magnetic.swapaxes(-1, -2).conj())

    for shell in range(2):
        for axis in range(3):

            def displaced(step):
                changed = [value.copy() for value in centers]
                changed[shell][axis] += step
                return gi.overlap(gi.Basis(make_shells(changed)), field=FIELD)

            np.testing.assert_allclose(
                nuclear[shell, axis],
                richardson(displaced),
                atol=3.0e-9,
                rtol=3.0e-8,
            )

    for axis in range(3):
        np.testing.assert_allclose(
            magnetic[axis],
            richardson(
                lambda step: gi.overlap(basis, field=shifted_field(FIELD, axis, step))
            ),
            atol=3.0e-9,
            rtol=3.0e-8,
        )


@pytest.mark.parametrize(
    ("shell_derivative", "shell_value"),
    [
        (gi.overlap_center_derivatives_shell, gi.overlap_shell),
        (gi.kinetic_center_derivatives_shell, gi.kinetic_shell),
        (
            gi.magnetic_kinetic_center_derivatives_shell,
            gi.magnetic_kinetic_shell,
        ),
    ],
)
def test_contracted_shell_center_derivatives(shell_derivative, shell_value):
    centers = [np.array([0.1, -0.2, 0.3]), np.array([-0.4, 0.2, 0.1])]
    shells = make_shells(centers)
    analytic = shell_derivative(*shells, field=FIELD)
    for center in range(2):
        for axis in range(3):

            def displaced(step):
                changed = [value.copy() for value in centers]
                changed[center][axis] += step
                return shell_value(*make_shells(changed), field=FIELD)

            np.testing.assert_allclose(
                analytic[center, axis],
                richardson(displaced),
                atol=5.0e-9,
                rtol=5.0e-8,
            )


def test_attraction_shell_and_basis_derivative_drivers():
    centers = [np.array([0.1, -0.2, 0.3]), np.array([-0.4, 0.2, 0.1])]
    shells = make_shells(centers)
    nuclei = [gi.Nucleus(1.3, (0.3, 0.1, -0.2))]
    shell_center = gi.nuclear_attraction_center_derivatives_shell(
        *shells, nuclei, field=FIELD
    )
    shell_nucleus = gi.nuclear_attraction_nucleus_derivatives_shell(
        *shells, nuclei, field=FIELD
    )
    for center in range(2):
        for axis in range(3):

            def displaced_shell(step):
                changed = [value.copy() for value in centers]
                changed[center][axis] += step
                return gi.nuclear_attraction_shell(
                    *make_shells(changed), nuclei, field=FIELD
                )

            np.testing.assert_allclose(
                shell_center[center, axis],
                richardson(displaced_shell),
                atol=8.0e-9,
                rtol=8.0e-8,
            )

    for axis in range(3):

        def displaced_nucleus(step):
            position = np.asarray(nuclei[0].center, dtype=float)
            position[axis] += step
            return gi.nuclear_attraction_shell(
                *shells, [gi.Nucleus(nuclei[0].charge, position)], field=FIELD
            )

        np.testing.assert_allclose(
            shell_nucleus[0, axis],
            richardson(displaced_nucleus),
            atol=8.0e-9,
            rtol=8.0e-8,
        )

    basis = gi.Basis(shells)
    basis_shell, basis_nucleus = gi.nuclear_attraction_nuclear_derivatives(
        basis, nuclei, field=FIELD
    )
    assert basis_shell.shape == (2, 3, 4, 4)
    assert basis_nucleus.shape == (1, 3, 4, 4)
    np.testing.assert_allclose(
        basis_shell, basis_shell.swapaxes(-1, -2).conj(), atol=2.0e-12
    )
    np.testing.assert_allclose(
        basis_nucleus, basis_nucleus.swapaxes(-1, -2).conj(), atol=2.0e-12
    )


def make_eri_shells(centers):
    return [
        gi.Shell(centers[0], 0, [0.8], [1.0]),
        gi.Shell(centers[1], 0, [1.0], [1.0]),
        gi.Shell(centers[2], 1, [0.7], [1.0]),
        gi.Shell(centers[3], 0, [1.2], [1.0]),
    ]


def test_eri_shell_derivative_drivers_and_zero_field_translation():
    centers = [
        np.array([0.1, -0.2, 0.0]),
        np.array([-0.3, 0.1, 0.2]),
        np.array([0.4, 0.2, -0.1]),
        np.array([-0.2, -0.4, 0.3]),
    ]
    shells = make_eri_shells(centers)
    center = gi.eri_center_derivatives_shell(*shells, field=FIELD)
    magnetic = gi.eri_magnetic_derivatives_shell(*shells, field=FIELD)
    assert center.shape == (4, 3, 1, 1, 3, 1)
    assert magnetic.shape == (3, 1, 1, 3, 1)

    for center_index in range(4):
        for axis in range(3):

            def displaced(step):
                changed = [value.copy() for value in centers]
                changed[center_index][axis] += step
                return gi.eri_shell(*make_eri_shells(changed), field=FIELD)

            np.testing.assert_allclose(
                center[center_index, axis],
                richardson(displaced),
                atol=8.0e-9,
                rtol=8.0e-8,
            )

    for axis in range(3):
        np.testing.assert_allclose(
            magnetic[axis],
            richardson(
                lambda step: gi.eri_shell(
                    *shells, field=shifted_field(FIELD, axis, step)
                )
            ),
            atol=8.0e-9,
            rtol=8.0e-8,
        )

    zero_center = gi.eri_center_derivatives_shell(*shells, field=gi.MagneticField())
    np.testing.assert_allclose(zero_center.sum(axis=0), 0.0, atol=3.0e-13)


@pytest.mark.parametrize(
    ("derivative", "value"),
    [
        (gi.kinetic_magnetic_derivatives, gi.kinetic),
        (
            gi.magnetic_kinetic_magnetic_derivatives,
            gi.magnetic_kinetic,
        ),
    ],
)
def test_basis_magnetic_derivative_drivers(derivative, value):
    basis = gi.Basis(
        make_shells([np.array([0.1, -0.2, 0.3]), np.array([-0.4, 0.2, 0.1])])
    )
    analytic = derivative(basis, field=FIELD)
    assert analytic.shape == (3, 4, 4)
    for axis in range(3):
        np.testing.assert_allclose(
            analytic[axis],
            richardson(
                lambda step: value(basis, field=shifted_field(FIELD, axis, step))
            ),
            atol=8.0e-9,
            rtol=8.0e-8,
        )


def test_local_derivatives_are_gauge_origin_independent():
    basis = gi.Basis(
        make_shells([np.array([0.1, -0.2, 0.3]), np.array([-0.4, 0.2, 0.1])])
    )
    shifted_origin = gi.MagneticField(
        FIELD.B, np.asarray(FIELD.gauge_origin) + np.array([0.4, -0.2, 0.3])
    )
    np.testing.assert_allclose(
        gi.overlap_nuclear_derivatives(basis, field=FIELD),
        gi.overlap_nuclear_derivatives(basis, field=shifted_origin),
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        gi.overlap_magnetic_derivatives(basis, field=FIELD),
        gi.overlap_magnetic_derivatives(basis, field=shifted_origin),
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        gi.magnetic_kinetic_nuclear_derivatives(basis, field=FIELD),
        gi.magnetic_kinetic_nuclear_derivatives(basis, field=shifted_origin),
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        gi.magnetic_kinetic_magnetic_derivatives(basis, field=FIELD),
        gi.magnetic_kinetic_magnetic_derivatives(basis, field=shifted_origin),
        atol=2.0e-12,
    )


def test_zero_field_translation_identities(primitives):
    a, b = primitives
    zero = gi.MagneticField()
    overlap = gi.primitive_overlap_center_derivatives(a, b, field=zero)
    np.testing.assert_allclose(overlap.sum(axis=0), 0.0, atol=3.0e-14)

    nuclei = [gi.Nucleus(1.2, (0.3, -0.1, 0.5))]
    basis_terms = gi.primitive_nuclear_attraction_center_derivatives(
        a, b, nuclei, field=zero
    )
    potential_terms = gi.primitive_nuclear_attraction_nucleus_derivatives(
        a, b, nuclei, field=zero
    )
    np.testing.assert_allclose(
        basis_terms.sum(axis=0) + potential_terms.sum(axis=0),
        0.0,
        atol=2.0e-13,
    )


def test_overlap_center_response_separates_gaussian_and_london_terms(
    primitives,
):
    a, b = primitives
    analytic = gi.primitive_overlap_center_derivatives(a, b, field=FIELD)
    originals = [a, b]
    for center in range(2):
        value = originals[center]
        normalization = gi.primitive_normalization(value.exponent, value.angular)
        raw_coefficient = value.coefficient * normalization
        for axis in range(3):
            ordinary = 0.0j
            angular = list(value.angular)
            if angular[axis]:
                lower = angular.copy()
                lower[axis] -= 1
                changed = gi.PrimitiveGaussian(
                    value.exponent,
                    value.center,
                    lower,
                    coefficient=raw_coefficient,
                    normalized=False,
                )
                pair = [a, b]
                pair[center] = changed
                ordinary -= angular[axis] * gi.primitive_overlap(*pair, field=FIELD)
            upper = angular.copy()
            upper[axis] += 1
            changed = gi.PrimitiveGaussian(
                value.exponent,
                value.center,
                upper,
                coefficient=raw_coefficient,
                normalized=False,
            )
            pair = [a, b]
            pair[center] = changed
            ordinary += 2.0 * value.exponent * gi.primitive_overlap(*pair, field=FIELD)

            phase_gradient = 0.5 * np.cross(np.asarray(FIELD.B), np.eye(3)[axis])
            phase_sign = 1.0 if center == 0 else -1.0
            phase = 0.0j
            for coordinate in range(3):
                powers = [0, 0, 0]
                powers[coordinate] = 1
                phase += (
                    1j
                    * phase_sign
                    * phase_gradient[coordinate]
                    * gi.primitive_moment(a, b, powers, field=FIELD)
                )
            assert abs(ordinary) > 1.0e-8
            assert abs(phase) > 1.0e-8
            np.testing.assert_allclose(
                analytic[center, axis],
                ordinary + phase,
                atol=2.0e-13,
                rtol=2.0e-12,
            )


def test_derivative_out_validation():
    shell = gi.Shell((0.0, 0.0, 0.0), 0, [1.0], [1.0])
    with pytest.raises(ValueError, match="incorrect"):
        gi.overlap_center_derivatives_shell(
            shell, shell, out=np.empty((3, 1, 1), dtype=np.complex128)
        )
    with pytest.raises(ValueError, match="complex128"):
        gi.overlap_magnetic_derivatives_shell(
            shell, shell, out=np.empty((3, 1, 1), dtype=float)
        )
