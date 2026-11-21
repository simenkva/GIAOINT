"""Selected entries must equal the unchanged full-shell kernel bit for bit."""

from itertools import combinations_with_replacement

import giao_integrals as gi
import numpy as np
import pytest
from giao_integrals._giao_integrals import _cholesky_columns, _cholesky_diagonal


@pytest.mark.parametrize(
    "field", [None, gi.MagneticField((0.05, -0.03, 0.2), (0.37, -0.29, 0.43))]
)
@pytest.mark.parametrize("contracted", [False, True])
def test_selected_diagonal_and_columns_equal_full_shells(field, contracted):
    coeff = [[0.7, -0.2], [0.1, 0.8]] if contracted else [0.7, -0.2]
    basis = gi.Basis(
        [
            gi.Shell((0.1, -0.2, 0.3), 0, [0.8, 1.2], coeff),
            gi.Shell((-0.3, 0.4, 0.2), 1, [0.8, 1.2], coeff),
            gi.Shell((0.2, 0.1, -0.3), 2, [0.6], [1.0]),
        ]
    )
    shells = basis.shells
    n = basis.ao_count
    spans = [
        slice(o, o + s.ao_count) for o, s in zip(basis.ao_offsets, shells, strict=True)
    ]
    pairs = list(combinations_with_replacement(range(len(shells)), 2))
    diagonal = np.zeros((n, n))
    for batch in gi.eri_batches(
        basis, quartets=[(a, b, b, a) for a, b in pairs], field=field, threads=1
    ):
        for k, (a, b, _, _) in enumerate(batch.quartets):
            values = np.einsum("ijji->ij", batch.block(k)).real
            diagonal[spans[a], spans[b]] = values
            diagonal[spans[b], spans[a]] = values.T
    for threads in [1, 2] if gi.openmp_enabled() else [1]:
        np.testing.assert_array_equal(
            _cholesky_diagonal(basis, pairs, field=field, threads=threads), diagonal
        )
    for sa, sb in [(0, 1), (1, 1), (2, 1)]:
        pivots = [
            (basis.ao_offsets[sa] + i, basis.ao_offsets[sb] + j)
            for i in range(shells[sa].ao_count)
            for j in range(shells[sb].ao_count)
        ][:8]
        real = field is None
        expected = np.empty((len(pivots), n, n), dtype=float if real else complex)
        for m in range(len(shells)):
            for k in range(m if real else 0, len(shells)):
                block = gi.eri_shell(
                    shells[m], shells[k], shells[sb], shells[sa], field=field
                )
                values = np.stack(
                    [
                        block[
                            :, :, nu - basis.ao_offsets[sb], mu - basis.ao_offsets[sa]
                        ]
                        for mu, nu in pivots
                    ]
                )
                if real:
                    values = values.real
                    if m == k:
                        values = (values + values.transpose(0, 2, 1)) * 0.5
                expected[:, spans[m], spans[k]] = values
                if real and m != k:
                    expected[:, spans[k], spans[m]] = values.transpose(0, 2, 1)
        for threads in [1, 2] if gi.openmp_enabled() else [1]:
            actual = _cholesky_columns(basis, pivots, field=field, threads=threads)
            np.testing.assert_array_equal(actual, expected.reshape(len(pivots), -1))


def test_selected_native_input_validation():
    basis = gi.Basis(
        [
            gi.Shell((0, 0, 0), 0, [0.8], [1.0]),
            gi.Shell((0.1, 0.2, 0.3), 1, [0.6], [1.0]),
        ]
    )
    for pairs in [[(1, 0)], [(0, 0), (0, 0)], [(0, 9)]]:
        with pytest.raises((ValueError, IndexError)):
            _cholesky_diagonal(basis, pairs, threads=1)
    for pivots in [[(0, 9)], [(0, 0), (1, 1)]]:
        with pytest.raises((ValueError, IndexError)):
            _cholesky_columns(basis, pivots, threads=1)
    assert _cholesky_columns(basis, [], threads=1).shape == (0, 16)


@pytest.mark.parametrize("field", [None, gi.MagneticField((0.03, -0.02, 0.05))])
def test_selected_long_contraction_pair_cache_fallback(field):
    # 65**2 primitive pairs exceeds the bounded 4096-pair cache.
    exponents = np.linspace(0.5, 2.0, 65).tolist()
    a = gi.Shell((0.1, -0.2, 0.3), 0, exponents, [[0.1] * 65, [0.2] * 65])
    b = gi.Shell((-0.3, 0.2, 0.5), 0, [0.7], [1.0])
    basis = gi.Basis([a, b])
    block = gi.eri_shell(a, a, b, b, field=field)[:, :, 0, 0]
    actual = _cholesky_columns(basis, [(2, 2)], field=field, threads=1)
    np.testing.assert_array_equal(actual.reshape(3, 3)[:2, :2], block)
