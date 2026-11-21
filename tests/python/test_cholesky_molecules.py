"""Molecular contraction reuse must retain pivoting, bounds and determinism."""

import importlib

import giao_integrals as gi
import numpy as np
import pytest

WATER = [
    ("O", (0.0, 0.1, 0.2)),
    ("H", (0.1, 1.4, -0.9)),
    ("H", (-0.2, -1.3, -1.0)),
]
FIELDS = [None, gi.MagneticField((0.03, -0.02, 0.05), (0.7, -0.4, 1.1))]


@pytest.mark.parametrize("field", FIELDS)
def test_molecular_cache_reconstruction_and_threads(field, monkeypatch):
    mol = gi.Molecule(WATER, "cc-pVDZ", unit="bohr", field=field)
    exact = mol.eri(max_bytes=32 * 1024**2, threads=1)
    module = importlib.import_module("giao_integrals.cholesky")
    original = module._cholesky_columns
    requests = []

    def record(basis, pivots, **kwargs):
        requests.append(len(pivots))
        return original(basis, pivots, **kwargs)

    monkeypatch.setattr(module, "_cholesky_columns", record)
    reference = mol.cholesky_eri(threshold=1e-8, threads=1)
    assert max(requests) > 1  # contraction-equivalent columns really are shared
    assert len(requests) < len(reference.vectors)
    assert reference.max_residual_diagonal <= 1e-8
    error = np.max(np.abs(reference.reconstruct() - exact))
    assert error <= reference.max_residual_diagonal + 5e-13
    reconstructed = reference.reconstruct()
    np.testing.assert_allclose(
        reconstructed, reconstructed.transpose(2, 3, 0, 1), atol=2e-8, rtol=0
    )
    np.testing.assert_allclose(
        reconstructed.conj(), reconstructed.transpose(1, 0, 3, 2), atol=2e-8, rtol=0
    )
    if gi.openmp_enabled():
        for threads in (2, 4):
            other = mol.cholesky_eri(threshold=1e-8, threads=threads)
            np.testing.assert_array_equal(other.pivots, reference.pivots)
            np.testing.assert_array_equal(other.vectors, reference.vectors)
            assert other.max_residual_diagonal == reference.max_residual_diagonal


@pytest.mark.parametrize("field,rank", zip(FIELDS, (656, 668), strict=True))
def test_cc_pvtz_reproducer_rank_and_bound(field, rank):
    mol = gi.Molecule(WATER, "cc-pVTZ", unit="bohr", field=field)
    count = 2 if gi.openmp_enabled() else 1
    result = mol.cholesky_eri(threshold=1e-8, threads=count)
    assert mol.ao_count == 65
    assert len(result.vectors) == rank  # published 0.11.0 reference ranks
    assert result.max_residual_diagonal <= 1e-8
    exact = mol.eri(max_bytes=512 * 1024**2, threads=count)
    for start in range(0, mol.ao_count, 4):
        block = (slice(start, start + 4), slice(None), slice(None), slice(None))
        error = np.max(np.abs(result.reconstruct(block) - exact[block]))
        assert error <= result.max_residual_diagonal + 5e-13
