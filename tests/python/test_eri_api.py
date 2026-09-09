from __future__ import annotations

import giao_integrals as gi
import numpy as np
import pytest


def small_basis():
    return gi.Basis(
        [
            gi.Shell((-0.2, 0.1, 0.3), 0, (0.8, 0.3), (0.4, 0.7)),
            gi.Shell((0.5, -0.4, 0.2), 1, (1.1,), (1.0,)),
        ]
    )


def scatter_batches(basis, batches):
    count = basis.ao_count
    offsets = basis.ao_offsets
    result = np.zeros((count,) * 4, dtype=np.complex128)
    for batch in batches:
        assert batch.quartets.dtype == np.uint32
        assert batch.shapes.dtype == np.int64
        assert batch.offsets.dtype == np.int64
        assert batch.values.dtype == np.complex128
        for index, quartet_array in enumerate(batch.quartets):
            quartet = tuple(int(value) for value in quartet_array)
            slices = tuple(
                slice(offsets[shell], offsets[shell] + batch.shapes[index, axis])
                for axis, shell in enumerate(quartet)
            )
            block = batch.block(index)
            a, b, c, d = quartet
            result[slices] = block
            result[
                slice(offsets[c], offsets[c] + block.shape[2]),
                slice(offsets[d], offsets[d] + block.shape[3]),
                slice(offsets[a], offsets[a] + block.shape[0]),
                slice(offsets[b], offsets[b] + block.shape[1]),
            ] = block.transpose(2, 3, 0, 1)
            result[
                slice(offsets[b], offsets[b] + block.shape[1]),
                slice(offsets[a], offsets[a] + block.shape[0]),
                slice(offsets[d], offsets[d] + block.shape[3]),
                slice(offsets[c], offsets[c] + block.shape[2]),
            ] = block.conj().transpose(1, 0, 3, 2)
            result[
                slice(offsets[d], offsets[d] + block.shape[3]),
                slice(offsets[c], offsets[c] + block.shape[2]),
                slice(offsets[b], offsets[b] + block.shape[1]),
                slice(offsets[a], offsets[a] + block.shape[0]),
            ] = block.conj().transpose(3, 2, 1, 0)
    return result


def test_shell_quartet_shape_dtype_and_exact_out():
    basis = small_basis()
    a, b = basis.shells
    expected = gi.eri_shell(a, b, a, b)
    assert expected.shape == (1, 3, 1, 3)
    assert expected.dtype == np.complex128
    assert expected.flags.c_contiguous
    output = np.empty_like(expected)
    assert gi.eri_shell(a, b, a, b, out=output) is output
    np.testing.assert_array_equal(output, expected)
    with pytest.raises(ValueError):
        gi.eri_shell(a, b, a, b, out=np.empty(expected.shape, dtype=np.complex64))
    with pytest.raises(ValueError):
        gi.eri_shell(
            a, b, a, b, out=np.empty(expected.shape[::-1], dtype=np.complex128)
        )
    with pytest.raises(ValueError):
        gi.eri_shell(
            a, b, a, b, out=np.empty(expected.shape, dtype=np.complex128)[..., ::-1]
        )


def test_default_is_streamed_and_matches_guarded_full_tensor():
    basis = small_basis()
    field = gi.MagneticField((0.3, -0.5, 0.2), (0.1, 0.4, -0.2))
    stream = gi.eri(basis, field=field, target_bytes=1)
    assert not isinstance(stream, np.ndarray)
    batches = list(stream)
    assert len(batches) > 1
    full = gi.eri(basis, field=field, storage="full", max_bytes=4096)
    np.testing.assert_allclose(scatter_batches(basis, batches), full, atol=3.0e-13)


def test_full_tensor_guard_and_option_validation():
    basis = small_basis()
    with pytest.raises(ValueError, match="explicit max_bytes"):
        gi.eri(basis, storage="full")
    with pytest.raises(MemoryError, match="requires 4096 bytes"):
        gi.eri(basis, storage="full", max_bytes=4095)
    with pytest.raises(ValueError, match="storage"):
        gi.eri(basis, storage="dense", max_bytes=4096)
    with pytest.raises(ValueError, match="only valid"):
        gi.eri(basis, max_bytes=4096)
    with pytest.raises(ValueError, match="positive integer"):
        list(gi.eri_batches(basis, target_bytes=0))


def test_explicit_quartets_and_index_validation():
    basis = small_basis()
    batches = list(
        gi.eri_batches(basis, quartets=[(1, 0, 1, 0), (0, 0, 0, 0)], target_bytes=1)
    )
    assert [tuple(batch.quartets[0]) for batch in batches] == [
        (1, 0, 1, 0),
        (0, 0, 0, 0),
    ]
    with pytest.raises(IndexError):
        list(gi.eri_batches(basis, quartets=[(0, 0, 0, 2)]))
    with pytest.raises(ValueError):
        list(gi.eri_batches(basis, quartets=[(0, 0, 0)]))
    with pytest.raises(ValueError):
        list(gi.eri_batches(basis, quartets=[(0, 0, 0, 0.5)]))


def test_contracted_shell_is_explicit_primitive_sum():
    shell = gi.Shell((0.1, -0.2, 0.3), 0, (1.2, 0.4), (0.3, 0.8), normalize=False)
    field = gi.MagneticField((0.2, 0.1, -0.4))
    actual = gi.eri_shell(shell, shell, shell, shell, field=field)[0, 0, 0, 0]
    expected = 0.0j
    for ea, ca in zip((1.2, 0.4), (0.3, 0.8), strict=True):
        for eb, cb in zip((1.2, 0.4), (0.3, 0.8), strict=True):
            for ec, cc in zip((1.2, 0.4), (0.3, 0.8), strict=True):
                for ed, cd in zip((1.2, 0.4), (0.3, 0.8), strict=True):
                    primitives = [
                        gi.PrimitiveGaussian(
                            exponent, shell.center, coefficient=coefficient
                        )
                        for exponent, coefficient in (
                            (ea, ca),
                            (eb, cb),
                            (ec, cc),
                            (ed, cd),
                        )
                    ]
                    expected += gi.primitive_eri(*primitives, field=field)
    np.testing.assert_allclose(actual, expected, atol=3.0e-13)


def test_general_contractions_preserve_contraction_major_ao_order():
    exponents = (1.1, 0.35)
    coefficients = np.array(((0.3, 0.8), (-0.2, 0.6)))
    shell = gi.Shell((0.1, -0.2, 0.3), 0, exponents, coefficients, normalize=False)
    actual = gi.eri_shell(shell, shell, shell, shell)
    assert actual.shape == (2, 2, 2, 2)
    for contractions in np.ndindex((2, 2, 2, 2)):
        expected = 0.0j
        for primitives in np.ndindex((2, 2, 2, 2)):
            functions = [
                gi.PrimitiveGaussian(
                    exponents[primitive],
                    shell.center,
                    coefficient=coefficients[contraction, primitive],
                )
                for contraction, primitive in zip(contractions, primitives, strict=True)
            ]
            expected += gi.primitive_eri(*functions)
        np.testing.assert_allclose(
            actual[contractions], expected, atol=4.0e-13, rtol=2.0e-12
        )
