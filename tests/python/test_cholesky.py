"""Reconstruction tests use unscreened production ERIs as the independent oracle."""

import tracemalloc

import giao_integrals as gi
import numpy as np
import pytest

WATER = [("O", (0, 0, 0)), ("H", (0.1, 0.2, 0.96)), ("H", (0.9, -0.2, -0.3))]
ORIGIN = (0.37, -0.29, 0.43)
FIELD = (0.05, -0.03, 0.2)


def system(name, field):
    if name == "df":
        # Displaced centers ensure nonzero imaginary d/f integrals in a field.
        return gi.Basis(
            [
                gi.Shell((0.2, -0.1, 0.3), 2, [0.8], [1.0]),
                gi.Shell((-0.3, 0.4, -0.2), 3, [0.6], [1.0]),
            ]
        )
    pytest.importorskip("basis_set_exchange")
    atoms = [("O", (0.2, -0.1, 0.3))] if name == "atom" else WATER
    return gi.Molecule(
        atoms, "6-31g" if name == "6-31g" else "sto-3g", field=field
    ).basis


@pytest.mark.parametrize("name", ["sto-3g", "6-31g", "atom", "df"])
@pytest.mark.parametrize("magnetic", [False, True])
def test_thresholds_reconstruction_symmetry_and_diagonal(name, magnetic):
    field = gi.MagneticField(FIELD if magnetic else (0, 0, 0), ORIGIN)
    basis = system(name, field)
    exact = gi.eri(basis, field=field, storage="full", max_bytes=basis.ao_count**4 * 16)
    np.testing.assert_allclose(exact, exact.transpose(2, 3, 0, 1), atol=1e-13)
    np.testing.assert_allclose(exact.conj(), exact.transpose(1, 0, 3, 2), atol=1e-13)
    # W is Hermitian PSD; V generally is not Hermitian at finite field.
    w = exact.transpose(1, 0, 2, 3).reshape(basis.ao_count**2, -1)
    np.testing.assert_allclose(w, w.conj().T, atol=1e-13)
    assert np.linalg.eigvalsh(w).min() > -1e-12
    for tau in (1e-4, 1e-6, 1e-8, 1e-10, 0):
        result = gi.cholesky_eri(basis, field=field, threshold=tau)
        reconstructed = result.reconstruct()
        atol = tau + 2e-12 if tau else 2e-12
        assert np.max(np.abs(exact - reconstructed)) <= atol
        assert result.max_residual_diagonal <= max(tau, 2e-12)
        residual = w.diagonal().real - np.sum(abs(result.vectors) ** 2, axis=0).ravel()
        assert abs(result.max_residual_diagonal - max(0, residual.max())) < 2e-12
        np.testing.assert_allclose(
            reconstructed, reconstructed.transpose(2, 3, 0, 1), atol=2 * atol, rtol=0
        )
        np.testing.assert_allclose(
            reconstructed.conj(),
            reconstructed.transpose(1, 0, 3, 2),
            atol=2 * atol,
            rtol=0,
        )
        assert result.vectors.dtype == np.complex128
        assert result.pivots.shape == (len(result.vectors), 2)
        assert len(np.unique(result.pivots, axis=0)) == len(result.vectors)
        block = (slice(0, 2), slice(1, 3), slice(0, 1), slice(2, 4))
        np.testing.assert_allclose(
            result.reconstruct(block), reconstructed[block], atol=1e-14
        )
        if not magnetic:
            np.testing.assert_array_equal(result.vectors.imag, 0)
            np.testing.assert_allclose(
                result.vectors, result.vectors.transpose(0, 2, 1), atol=2e-12
            )


@pytest.mark.parametrize("tau", [1e-4, 1e-6, 1e-8, 1e-10])
def test_gauge_origin_independence(tau):
    pytest.importorskip("basis_set_exchange")
    mol = gi.Molecule(WATER, "sto-3g", field=gi.MagneticField(FIELD, ORIGIN))
    shifted = gi.Molecule(
        WATER, "sto-3g", field=gi.MagneticField(FIELD, (-0.61, 0.72, 0.18))
    )
    np.testing.assert_allclose(mol.eri(), shifted.eri(), atol=2e-13)
    np.testing.assert_allclose(
        mol.cholesky_eri(threshold=tau).reconstruct(),
        shifted.cholesky_eri(threshold=tau).reconstruct(),
        atol=tau + 2e-12,
        rtol=0,
    )


def test_zero_field_matches_independent_real_pivoted_cholesky():
    pytest.importorskip("basis_set_exchange")
    mol = gi.Molecule(WATER, "sto-3g")
    w = mol.eri().real.reshape(mol.ao_count**2, -1).copy()
    rows = []
    pivots = []
    while w.diagonal().max() > 1e-8:
        p = int(w.diagonal().argmax())
        row = w[:, p] / np.sqrt(w[p, p])
        rows.append(row)
        pivots.append(divmod(p, mol.ao_count))
        w -= np.outer(row, row)
    result = mol.cholesky_eri()
    # Ordered pairs (mu,nu) and (nu,mu) tie exactly in real arithmetic.
    np.testing.assert_array_equal(
        np.sort(result.pivots, axis=1), np.sort(pivots, axis=1)
    )
    np.testing.assert_allclose(result.vectors.reshape(len(rows), -1), rows, atol=1e-12)


def test_memory_streaming_output_cap_and_no_dense_backend(monkeypatch):
    from importlib import import_module

    import giao_integrals.eri as unused  # noqa: F401

    eri_module = import_module("giao_integrals.eri")

    def forbidden(*args, **kwargs):
        raise AssertionError("dense ERIs must never be called")

    basis = gi.Basis(
        [gi.Shell((0, 0, 0), 0, [x], [1.0]) for x in np.geomspace(0.1, 10, 20)]
    )
    dense_bytes = basis.ao_count**4 * 16
    cap = dense_bytes // 4
    with pytest.raises(MemoryError):
        gi.eri(basis, storage="full", max_bytes=cap)
    monkeypatch.setattr(eri_module, "_eri_full", forbidden)
    tracemalloc.start()
    result = gi.cholesky_eri(basis, threshold=1e-8, max_bytes=cap)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert result.vectors.nbytes + result.pivots.nbytes <= cap
    assert peak < dense_bytes
    with pytest.raises(MemoryError, match="Cholesky ERI output requires"):
        gi.cholesky_eri(basis, max_bytes=1)
    empty = gi.cholesky_eri(basis, threshold=100, max_bytes=0)
    assert empty.vectors.shape == (0, 20, 20)


def test_conservative_diagonal_screening_and_threads():
    basis = gi.Basis([gi.Shell((x, 0, 0), 0, [0.8], [1.0]) for x in [0, 8]])
    exact = gi.eri(basis, storage="full", max_bytes=256)
    for screening in [0, 1e-8, 100]:
        result = gi.cholesky_eri(basis, screening_threshold=screening)
        np.testing.assert_allclose(result.reconstruct(), exact, atol=1e-8, rtol=0)
        residual = np.einsum("ijji->ij", exact).real - np.sum(
            abs(result.vectors) ** 2, axis=0
        )
        assert abs(result.max_residual_diagonal - max(0, residual.max())) < 1e-13
    if gi.openmp_enabled():
        parallel = gi.cholesky_eri(basis, threads=2)
        np.testing.assert_allclose(parallel.reconstruct(), exact, atol=1e-8)
    else:
        with pytest.raises(ValueError, match="OpenMP-enabled"):
            gi.cholesky_eri(basis, threads=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": -1},
        {"threshold": np.nan},
        {"threshold": np.inf},
        {"screening_threshold": -1},
        {"screening_threshold": np.inf},
        {"threads": 0},
        {"threads": True},
        {"threads": 1.5},
        {"max_bytes": -1},
        {"max_bytes": True},
        {"max_bytes": 1.5},
    ],
)
def test_validation(kwargs):
    basis = gi.Basis([gi.Shell((0, 0, 0), 0, [0.8], [1.0])])
    with pytest.raises(ValueError):
        gi.cholesky_eri(basis, **kwargs)


def test_empty_basis():
    result = gi.cholesky_eri(gi.Basis([]), max_bytes=0)
    assert result.vectors.shape == (0, 0, 0)
    assert result.pivots.shape == (0, 2)
    assert result.max_residual_diagonal == 0
    assert result.reconstruct().shape == (0, 0, 0, 0)


def test_single_center_finite_field_phase_cancellation():
    center = (0.3, -0.2, 0.1)
    shells = [
        gi.Shell(center, angular, [0.7 + 0.1 * angular], [1.0]) for angular in [0, 2, 3]
    ]
    basis = gi.Basis(shells)
    field = gi.MagneticField(FIELD, ORIGIN)
    result = gi.cholesky_eri(basis, field=field, threshold=1e-8)
    offsets = basis.ao_offsets
    for quartet in [(0, 1, 2, 1), (1, 2, 0, 2)]:
        exact = gi.eri_shell(*(shells[s] for s in quartet), field=field)
        slices = tuple(
            slice(offsets[s], offsets[s] + shells[s].ao_count) for s in quartet
        )
        np.testing.assert_allclose(result.reconstruct(slices), exact, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(result.vectors.imag, 0)
