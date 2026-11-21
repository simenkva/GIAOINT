"""Streaming layout, exact unscreened equivalence and derivative-specific bounds."""

# Loop-local closures are evaluated synchronously.
# ruff: noqa: B023

from itertools import product

import giao_integrals as gi
import numpy as np
import pytest
from test_moment_derivatives import FIELDS, richardson


def basis():
    return gi.Basis(
        [
            gi.Shell((0.1, -0.2, 0.3), 0, [0.8, 1.2], [0.7, -0.2]),
            gi.Shell((-0.3, 0.4, 0.2), 1, [0.6], [1.0]),
        ]
    )


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("explicit", [False, True])
def test_streamed_derivative_blocks_and_thread_equality(field, explicit):
    b = basis()
    qs = list(product(range(2), repeat=4)) if explicit else None
    kwargs = {"quartets": qs, "target_bytes": 3000, "field": field}
    reference = list(gi.eri_derivative_batches(b, threads=1, **kwargs))
    parallel = list(
        gi.eri_derivative_batches(b, threads=2 if gi.openmp_enabled() else 1, **kwargs)
    )
    assert len(reference) == len(parallel) > 1
    seen = []
    for serial, threaded in zip(reference, parallel, strict=True):
        assert isinstance(serial, gi.EriDerivativeBatch)
        assert serial.shapes.shape[1] == 6
        assert serial.offsets[-1] == serial.values.size
        assert serial.screened_count == 0
        if serial.values.nbytes > kwargs["target_bytes"]:
            assert len(serial.quartets) == 1
        for attr in ["quartets", "shapes", "offsets", "values"]:
            np.testing.assert_array_equal(
                getattr(serial, attr), getattr(threaded, attr)
            )
        for row, q in enumerate(serial.quartets):
            seen.append(tuple(q))
            expected = gi.eri_center_derivatives_shell(
                *[b.shells[int(k)] for k in q], field=field
            )
            np.testing.assert_array_equal(serial.block(row), expected)
            assert np.shares_memory(serial.block(row), serial.values)
    if explicit:
        assert seen == qs
    else:
        canonical = [
            tuple(q)
            for batch in gi.eri_batches(b, field=field, threads=1)
            for q in batch.quartets
        ]
        assert seen == canonical


@pytest.mark.parametrize("field", FIELDS)
def test_derivative_bounds_include_phase_and_contractions(field):
    from giao_integrals._giao_integrals import _EriDerivativeSchwarzBounds

    b = gi.Basis(
        [
            gi.Shell((0.1, -0.2, 0.3), 1, [0.8, 1.2], [[0.7, -0.2], [0.3, 0.4]]),
            gi.Shell((-0.3, 0.4, 0.2), 0, [0.6], [1.0]),
        ]
    )
    bounds = _EriDerivativeSchwarzBounds(b, field=field, threads=1)
    for q in product(range(2), repeat=4):
        block = gi.eri_center_derivatives_shell(*[b.shells[k] for k in q], field=field)
        assert np.max(np.abs(block)) <= bounds.quartet_bound(q) * (1 + 1e-13)


@pytest.mark.parametrize("field", FIELDS)
def test_screening_is_safe_per_component_and_thread_identical(field):
    b = gi.Basis(
        [
            gi.Shell((0.1, -0.2, 0.3), 0, [0.8], [1.0]),
            gi.Shell((30, -20, 10), 1, [0.9], [1.0]),
        ]
    )
    all_q = list(product(range(2), repeat=4))
    threshold = 1e-8
    serial = list(
        gi.eri_derivative_batches(
            b,
            quartets=all_q,
            field=field,
            threads=1,
            target_bytes=2000,
            screening_threshold=threshold,
        )
    )
    parallel = list(
        gi.eri_derivative_batches(
            b,
            quartets=all_q,
            field=field,
            threads=2 if gi.openmp_enabled() else 1,
            target_bytes=2000,
            screening_threshold=threshold,
        )
    )
    assert sum(x.requested_count for x in serial) == len(all_q)
    assert 0 < sum(x.screened_count for x in serial) < len(all_q)
    kept = set()
    for a, c in zip(serial, parallel, strict=True):
        np.testing.assert_array_equal(a.values, c.values)
        np.testing.assert_array_equal(a.quartets, c.quartets)
        assert a.screened_count == c.screened_count
        for row, q in enumerate(a.quartets):
            q = tuple(q)
            kept.add(q)
            np.testing.assert_array_equal(
                a.block(row),
                gi.eri_center_derivatives_shell(
                    *[b.shells[int(k)] for k in q], field=field
                ),
            )
    for q in all_q:
        if q not in kept:
            block = gi.eri_center_derivatives_shell(
                *[b.shells[k] for k in q], field=field
            )
            assert np.max(np.abs(block)) < threshold


@pytest.mark.parametrize("field", FIELDS)
def test_streamed_derivative_finite_difference_and_molecule(field):
    centers = np.array([[0.1, -0.2, 0.3], [-0.3, 0.4, 0.2]])

    def shells(c):
        return [gi.Shell(c[0], 0, [0.8], [1.0]), gi.Shell(c[1], 1, [0.6], [1.0])]

    b = gi.Basis(shells(centers))
    q = (0, 1, 0, 1)
    batch = next(gi.eri_derivative_batches(b, quartets=[q], field=field, threads=1))
    block = batch.block(0)
    # Both occurrences of a shell move together in an atom/shell derivative.
    for center in range(2):
        for axis in range(3):

            def value(step):
                changed = centers.copy()
                changed[center, axis] += step
                s = shells(changed)
                return gi.eri_shell(*[s[k] for k in q], field=field)

            expected = richardson(value)
            actual = block[
                [k for k, shell in enumerate(q) if shell == center], axis
            ].sum(axis=0)
            np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=3e-9)
    molecule = gi.Molecule(
        [("H", (0.1, -0.2, 0.3)), ("H", (-0.3, 0.4, 0.2))],
        "STO-3G",
        unit="bohr",
        field=field,
    )
    result = next(molecule.eri_derivative_batches(quartets=[q], threads=1))
    expected = next(
        gi.eri_derivative_batches(molecule.basis, quartets=[q], field=field, threads=1)
    )
    np.testing.assert_array_equal(result.values, expected.values)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_target(bad):
    with pytest.raises(ValueError):
        list(gi.eri_derivative_batches(basis(), target_bytes=bad))


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_invalid_screening(bad):
    with pytest.raises(ValueError):
        list(gi.eri_derivative_batches(basis(), screening_threshold=bad))


def test_empty_invalid_quartets_and_native_worker_exception():
    assert list(gi.eri_derivative_batches(gi.Basis([]), threads=1)) == []
    assert list(gi.eri_derivative_batches(basis(), quartets=[], threads=1)) == []
    for q in [(0, 0, 0, 9), (0, 0, 0, True), (0, 0, 0, 0.5)]:
        with pytest.raises((ValueError, IndexError)):
            list(gi.eri_derivative_batches(basis(), quartets=[q]))
    b = gi.Basis(
        [gi.Shell((0, 0, 0), 0, [0.8], [1.0]), gi.Shell((0, 0, 0), 33, [0.6], [1.0])]
    )
    with pytest.raises(ValueError, match="angular order"):
        list(
            gi.eri_derivative_batches(
                b,
                quartets=[(1, 0, 0, 0), (0, 0, 0, 0)],
                threads=2 if gi.openmp_enabled() else 1,
            )
        )


def test_streaming_memory_does_not_scale_as_nao_fourth_power():
    import tracemalloc

    b = gi.Basis([gi.Shell((k * 0.1, 0, 0), 0, [0.8], [1.0]) for k in range(120)])
    dense_bytes = 120**4 * 12 * 16
    assert dense_bytes > 30 * 1024**3
    tracemalloc.start()
    iterator = gi.eri_derivative_batches(b, target_bytes=4096, threads=1)
    batch = next(iterator)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert batch.values.nbytes <= 4096
    assert peak < 8 * 1024**2
    iterator.close()


@pytest.mark.parametrize("field", FIELDS)
def test_screening_discards_nonzero_derivatives_within_cutoff(field):
    from giao_integrals._giao_integrals import _EriDerivativeSchwarzBounds

    b = gi.Basis(
        [
            gi.Shell((0.1, -0.2, 0.3), 0, [0.8], [1.0]),
            gi.Shell((4, 0, 0), 1, [0.9], [1.0]),
        ]
    )
    quartets = list(product(range(2), repeat=4))
    bounds = _EriDerivativeSchwarzBounds(b, field=field, threads=1)
    threshold = float(np.median([bounds.quartet_bound(q) for q in quartets]))
    kept = {
        tuple(q)
        for batch in gi.eri_derivative_batches(
            b, quartets=quartets, field=field, screening_threshold=threshold, threads=1
        )
        for q in batch.quartets
    }
    dropped = []
    for q in quartets:
        if q not in kept:
            maximum = np.max(
                np.abs(
                    gi.eri_center_derivatives_shell(
                        *[b.shells[k] for k in q], field=field
                    )
                )
            )
            assert maximum < threshold
            dropped.append(maximum)
    assert any(value > 1e-20 for value in dropped)


def test_native_screening_bounds_are_tied_to_basis_and_field():
    from giao_integrals._giao_integrals import (
        _eri_derivative_batch,
        _EriDerivativeSchwarzBounds,
    )

    original = basis()
    bounds = _EriDerivativeSchwarzBounds(original, field=FIELDS[1], threads=1)
    for changed_basis, changed_field in [(basis(), FIELDS[1]), (original, FIELDS[0])]:
        with pytest.raises(ValueError, match="do not match"):
            _eri_derivative_batch(
                changed_basis,
                [(0, 0, 0, 0)],
                field=changed_field,
                screening_threshold=1e-8,
                threads=1,
                bounds=bounds,
            )
    with pytest.raises(ValueError, match="requires derivative bounds"):
        _eri_derivative_batch(
            original, [(0, 0, 0, 0)], threads=1, screening_threshold=1e-8
        )
