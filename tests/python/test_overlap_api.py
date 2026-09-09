import threading
import time

import giao_integrals as gi
import numpy as np
import pytest


def test_component_ordering_and_normalization():
    assert gi.cartesian_components(2) == (
        (2, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 2, 0),
        (0, 1, 1),
        (0, 0, 2),
    )
    exponent = 0.73
    assert gi.primitive_normalization(exponent, (0, 0, 0)) == pytest.approx(
        (2.0 * exponent / np.pi) ** 0.75
    )


def test_shell_api_shape_dtype_order_and_out():
    a = gi.Shell(
        (-0.3, 0.2, 0.5),
        2,
        [2.3, 0.7, 0.19],
        [0.11, -0.43, 0.82],
    )
    b = gi.Shell(
        (0.6, -0.4, 0.1),
        1,
        [1.8, 0.5],
        [[0.3, 0.8], [-0.2, 0.9]],
    )
    field = gi.MagneticField((0.19, -0.31, 0.27), (0.4, -0.2, 0.1))
    result = gi.overlap_shell(a, b, field=field)
    assert result.shape == (6, 6)
    assert result.dtype == np.complex128
    assert result.flags.c_contiguous
    assert a.components == gi.cartesian_components(2)
    assert b.ao_count == b.contraction_count * b.cartesian_count == 6

    output = np.empty_like(result)
    returned = gi.overlap_shell(a, b, field=field, out=output)
    assert returned is output
    np.testing.assert_array_equal(output, result)


def test_basis_driver_matches_shell_blocks():
    shells = [
        gi.Shell((0.0, 0.0, 0.0), 0, [3.0, 0.7], [0.2, 0.8]),
        gi.Shell((0.7, -0.2, 0.3), 1, [1.6, 0.4], [0.4, 0.7]),
        gi.Shell((-0.5, 0.8, -0.1), 2, [1.1], [1.0]),
    ]
    basis = gi.Basis(shells)
    field = gi.MagneticField((0.2, 0.1, -0.3), (-0.1, 0.4, 0.2))
    result = gi.overlap(basis, field=field)
    assert result.shape == (10, 10)
    assert basis.ao_offsets == (0, 1, 4)

    for shell_a, offset_a in zip(shells, basis.ao_offsets, strict=True):
        for shell_b, offset_b in zip(shells, basis.ao_offsets, strict=True):
            expected = gi.overlap_shell(shell_a, shell_b, field=field)
            actual = result[
                offset_a : offset_a + shell_a.ao_count,
                offset_b : offset_b + shell_b.ao_count,
            ]
            np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(result, result.conj().T, atol=2.0e-13, rtol=2.0e-13)

    output = np.empty_like(result)
    returned = gi.overlap(basis, field=field, out=output)
    assert returned is output
    np.testing.assert_array_equal(output, result)


@pytest.mark.parametrize(
    "output",
    [
        np.empty((1, 1), dtype=np.float64),
        np.empty((2, 2), dtype=np.complex128),
        [[0j]],
    ],
)
def test_out_validation(output):
    shell = gi.Shell((0.0, 0.0, 0.0), 0, [1.0], [1.0])
    with pytest.raises(ValueError):
        gi.overlap_shell(shell, shell, out=output)


def test_noncontiguous_out_is_rejected():
    shell = gi.Shell((0.0, 0.0, 0.0), 1, [1.0], [1.0])
    output = np.empty((3, 6), dtype=np.complex128)[:, ::2]
    assert output.shape == (3, 3) and not output.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        gi.overlap_shell(shell, shell, out=output)


def test_read_only_out_is_rejected():
    shell = gi.Shell((0.0, 0.0, 0.0), 0, [1.0], [1.0])
    output = np.empty((1, 1), dtype=np.complex128)
    output.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        gi.overlap_shell(shell, shell, out=output)


def test_expensive_shell_call_releases_the_gil():
    a = gi.Shell(
        (-0.3, 0.2, 0.5),
        12,
        [0.15 + index * 0.13 for index in range(16)],
        [(-1) ** index / (index + 1) for index in range(16)],
    )
    b = gi.Shell(
        (0.6, -0.4, 0.1),
        12,
        [0.18 + index * 0.11 for index in range(16)],
        [(-1) ** (index + 1) / (index + 2) for index in range(16)],
    )
    entered = threading.Event()
    finished = threading.Event()

    def compute():
        entered.set()
        gi.overlap_shell(a, b, field=gi.MagneticField((0.2, -0.3, 0.1)))
        finished.set()

    worker = threading.Thread(target=compute)
    worker.start()
    assert entered.wait(timeout=1.0)
    time.sleep(0.05)
    released = not finished.is_set()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert released, "the worker held the GIL for the complete C++ shell call"


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: gi.PrimitiveGaussian(0.0, (0.0, 0.0, 0.0)),
        lambda: gi.PrimitiveGaussian(1.0, (0.0, 0.0)),
        lambda: gi.PrimitiveGaussian(1.0, (0.0, 0.0), (-1, 0, 0)),
        lambda: gi.PrimitiveGaussian(1.0, (0.0, 0.0, 0.0), normalized=1),
        lambda: gi.MagneticField((0.0, np.nan, 0.0)),
        lambda: gi.Shell((0.0, 0.0, 0.0), -1, [1.0], [1.0]),
        lambda: gi.Shell((0.0, 0.0, 0.0), 0, [], []),
        lambda: gi.Shell((0.0, 0.0, 0.0), 0, [1.0], [1.0, 2.0]),
        lambda: gi.Shell((0.0, 0.0, 0.0), 0, [1.0], [0.0]),
        lambda: gi.Shell((0.0, 0.0, 0.0), 0, [1.0], [1.0], normalize=1),
    ],
)
def test_data_models_reject_invalid_inputs(constructor):
    with pytest.raises(ValueError):
        constructor()
