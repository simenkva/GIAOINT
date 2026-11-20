"""Runtime thread policy and bitwise equality across every ERI entry point."""

from importlib import import_module

import giao_integrals as gi
import numpy as np
import pytest
from giao_integrals import _threads


@pytest.fixture(autouse=True)
def reset_threads(monkeypatch):
    monkeypatch.delenv("GIAO_NUM_THREADS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    gi.set_num_threads(None)
    yield
    gi.set_num_threads(None)


def molecule(field):
    data = {
        "elements": {
            "1": {
                "electron_shells": [
                    {
                        "angular_momentum": [0],
                        "exponents": ["0.8"],
                        "coefficients": [["1"]],
                    },
                    {
                        "angular_momentum": [1],
                        "exponents": ["0.6"],
                        "coefficients": [["1"]],
                    },
                ]
            }
        }
    }
    return gi.Molecule(
        [("H", (0, 0, 0)), ("H", (0.7, 0.1, -0.2)), ("H", (8, 0, 0))],
        data,
        field=field,
    )


def test_build_info():
    info = gi.build_info()
    assert info["version"] == gi.__version__
    assert info["compiler"]
    assert info["openmp_enabled"] == gi.openmp_enabled()
    assert info["default_threads"] >= 1
    assert info["physical_cores"] >= 1
    if gi.openmp_enabled():
        assert info["openmp_runtime_path"]
    else:
        assert info["openmp_runtime_path"] is None


def test_policy_precedence(monkeypatch):
    monkeypatch.setattr(_threads, "openmp_enabled", lambda: True)
    monkeypatch.setattr(_threads, "physical_core_count", lambda: 7)
    assert _threads.resolve_num_threads() == 7
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert _threads.resolve_num_threads() == 3
    monkeypatch.setenv("GIAO_NUM_THREADS", "4")
    assert _threads.resolve_num_threads() == 4
    gi.set_num_threads(2)
    assert _threads.resolve_num_threads() == 2
    assert _threads.resolve_num_threads(1) == 1
    gi.set_num_threads(None)
    assert _threads.resolve_num_threads() == 4


@pytest.mark.parametrize("value", [0, -1, True, np.bool_(True), 1.5, "2", 2**31])
def test_invalid_thread_counts(value):
    with pytest.raises(ValueError, match="positive integer"):
        gi.set_num_threads(value)
    with pytest.raises(ValueError, match="positive integer"):
        _threads.resolve_num_threads(value)


@pytest.mark.parametrize("value", ["0", "-1", "", "hello", "2,3", "1.5"])
def test_invalid_environment(monkeypatch, value):
    monkeypatch.setenv("GIAO_NUM_THREADS", value)
    with pytest.raises(ValueError, match="GIAO_NUM_THREADS"):
        gi.build_info()
    assert _threads.resolve_num_threads(1) == 1


def test_serial_fallback_is_visible(monkeypatch):
    monkeypatch.setattr(_threads, "openmp_enabled", lambda: False)
    monkeypatch.setattr(_threads, "_warned_serial", False)
    monkeypatch.setenv("GIAO_NUM_THREADS", "2")
    with pytest.warns(RuntimeWarning, match="using threads=1"):
        assert _threads.resolve_num_threads() == 1
    with pytest.raises(ValueError, match="OpenMP-enabled"):
        gi.set_num_threads(2)
    with pytest.raises(ValueError, match="OpenMP-enabled"):
        _threads.resolve_num_threads(2)


def assert_batches_equal(first, second):
    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        for name in ("quartets", "shapes", "offsets", "values"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        assert a.screened_count == b.screened_count
        assert a.requested_count == b.requested_count


@pytest.mark.parametrize("magnetic", [False, True])
@pytest.mark.parametrize("screening", [0.0, 0.01])
def test_bitwise_equality_every_entry_point(magnetic, screening):
    field = gi.MagneticField(
        (0.05, -0.03, 0.2) if magnetic else (0, 0, 0), (0.37, -0.29, 0.43)
    )
    mol = molecule(field)
    kw = dict(screening_threshold=screening)
    full = mol.eri(threads=1, **kw)
    batches = list(mol.eri_batches(threads=1, target_bytes=2048, **kw))
    if screening:
        assert sum(batch.screened_count for batch in batches) > 0
    cholesky = mol.cholesky_eri(threads=1, **kw)
    core = import_module("giao_integrals._giao_integrals")
    quartets = [(0, 1, 2, 3), (0, 0, 0, 0), (1, 1, 1, 1)]
    bounds = core._EriSchwarzBounds(mol.basis, field=field) if screening else None
    raw = core._eri_batch(
        mol.basis, quartets, field=field, threads=1, bounds=bounds, **kw
    )
    counts = {1}
    if gi.openmp_enabled():
        counts.update(
            (2, max(gi.openmp_max_threads(), gi.build_info()["physical_cores"]))
        )
    for threads in sorted(counts):
        np.testing.assert_array_equal(mol.eri(threads=threads, **kw), full)
        np.testing.assert_array_equal(
            mol.integrals(eri=True, threads=threads, **kw).eri, full
        )
        np.testing.assert_array_equal(
            gi.eri(
                mol.basis,
                field=field,
                storage="full",
                max_bytes=full.nbytes,
                threads=threads,
                **kw,
            ),
            full,
        )
        assert_batches_equal(
            list(
                gi.eri(mol.basis, field=field, target_bytes=2048, threads=threads, **kw)
            ),
            batches,
        )
        assert_batches_equal(
            list(mol.eri_batches(target_bytes=2048, threads=threads, **kw)), batches
        )
        np.testing.assert_array_equal(
            core._eri_full(mol.basis, field=field, threads=threads, **kw), full
        )
        packed = core._eri_batch(
            mol.basis, quartets, field=field, threads=threads, bounds=bounds, **kw
        )
        for a, b in zip(raw, packed, strict=True):
            np.testing.assert_array_equal(a, b)
        factors = mol.cholesky_eri(threads=threads, **kw)
        np.testing.assert_array_equal(factors.pivots, cholesky.pivots)
        np.testing.assert_array_equal(factors.vectors, cholesky.vectors)
        assert factors.max_residual_diagonal == cholesky.max_residual_diagonal
    gi.set_num_threads(max(counts))
    # None defaults must reach the same backend policy, including native bindings.
    np.testing.assert_array_equal(mol.eri(**kw), full)
    np.testing.assert_array_equal(core._eri_full(mol.basis, field=field, **kw), full)
