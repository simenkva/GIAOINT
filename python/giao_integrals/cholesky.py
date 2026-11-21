"""On-demand pivoted Cholesky of the Hermitian ordered-pair Coulomb matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._giao_integrals import Basis, MagneticField, _cholesky_columns, _cholesky_diagonal
from ._threads import resolve_num_threads
from .eri import eri_schwarz_bounds


@dataclass(frozen=True)
class CholeskyERI:
    r"""Coulomb factors in the AO order and normalization of :func:`eri`.

    ``vectors`` is complex128 with shape (rank, nao, nao). The exact convention is

    .. math::

        (\mu\nu|\lambda\sigma) \simeq
        \sum_P L_{P\mu\nu} L^*_{P\sigma\lambda}.

    Equivalently, W[(mu,nu),(lambda,sigma)] = (nu mu|lambda sigma)
    is approximated by sum_P conj(L[P,mu,nu]) * L[P,lambda,sigma].
    All nao**2 ordered pairs participate. ``pivots`` contains AO index pairs.
    At zero field the vectors are real symmetric (stored as complex128).
    ``max_residual_diagonal`` is the measured largest remaining diagonal of W.
    """

    vectors: npt.NDArray[np.complex128]
    pivots: npt.NDArray[np.int64]
    threshold: float
    max_residual_diagonal: float

    def reconstruct(
        self, slices: tuple[slice, slice, slice, slice] | None = None
    ) -> npt.NDArray[np.complex128]:
        """Reconstruct four AO slices; omitted slices allocate the full tensor.

        ``slices`` is a tuple of four Python slices in (mu, nu, lambda, sigma)
        order. This helper is intended for tests and small tensor blocks.
        """
        if slices is None:
            slices = (slice(None),) * 4
        if len(slices) != 4 or not all(isinstance(s, slice) for s in slices):
            raise ValueError("slices must contain four slices")
        a, b, c, d = slices
        return np.einsum(
            "Pmn,Psl->mnls",
            self.vectors[:, a, b],
            self.vectors[:, d, c].conj(),
            optimize=True,
        )


def cholesky_eri(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    threshold: float = 1e-8,
    screening_threshold: float = 0.0,
    threads: int | None = None,
    max_bytes: int | None = None,
) -> CholeskyERI:
    """Factor ERIs without forming a full AO tensor or ordered-pair matrix.

    See :class:`CholeskyERI` for the reconstruction and conjugation convention.
    Pivoted Cholesky acts on W[(mu,nu),(lambda,sigma)] = (nu mu|lambda sigma).
    Stop when its largest residual diagonal is <= ``threshold``. For exact
    integrals this bounds every reconstructed ERI error by ``threshold``.
    Roundoff limits stopping to 64*eps*max(initial diagonal); in particular,
    threshold=0 requests numerical exhaustion rather than exact arithmetic.
    The returned residual reports the achieved tolerance, including this floor.

    Diagonals come from (MN|NM); columns use (MN|SL) for pivot (L,S).
    Up to eight candidate columns from the pivot shell pair are cached together.
    Storage is O(rank*nao**2) plus bounded selected-column/native workspace.
    Only needed diagonal elements and up to eight pivot columns are evaluated;
    complete shell quartets are not constructed. ``threads`` controls the native
    selected-integral tasks. None selects the configured default; 1 is serial.
    NumPy/BLAS has its own thread settings and runs outside OpenMP regions.

    ``screening_threshold`` enables existing Schwarz bounds only to skip initial
    diagonal blocks whose bound is already <= threshold. These use conservative
    upper diagonals until measured at the end; columns are never screened.
    ``max_bytes`` caps the returned vectors and pivot indices, with no default
    cap. It does not cap workspace or total process memory. Exceeding it raises
    MemoryError before allocating another output row.
    """
    threshold = float(threshold)
    screening_threshold = float(screening_threshold)
    for name, value in (
        ("threshold", threshold),
        ("screening_threshold", screening_threshold),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    threads = resolve_num_threads(threads)
    if max_bytes is not None and (
        isinstance(max_bytes, (bool, np.bool_))
        or not isinstance(max_bytes, (int, np.integer))
        or max_bytes < 0
    ):
        raise ValueError("max_bytes must be a non-negative integer or None")

    n = basis.ao_count
    shells = basis.shells
    # On one center every AO has the same London phase, which cancels in
    # each density. The exact zero-field ERI backend is therefore applicable.
    single_center = all(s.center == shells[0].center for s in shells)
    integral_field = None if single_center else field
    offsets = basis.ao_offsets
    spans = [slice(o, o + s.ao_count) for o, s in zip(offsets, shells, strict=False)]
    owners = np.repeat(np.arange(len(shells)), [s.ao_count for s in shells])
    diagonal = np.zeros((n, n))
    bounds = (
        eri_schwarz_bounds(basis, field=integral_field) ** 2
        if screening_threshold
        else None
    )
    skipped = []
    quartets = []
    for a in range(len(shells)):
        for b in range(a, len(shells)):
            if bounds is not None and bounds[a, b] <= min(
                threshold, screening_threshold
            ):
                diagonal[spans[a], spans[b]] = bounds[a, b]
                diagonal[spans[b], spans[a]] = bounds[a, b]
                skipped.append((a, b, b, a))
            else:
                quartets.append((a, b, b, a))

    def read_diagonal(quartets):
        if not quartets:
            return
        measured = _cholesky_diagonal(
            basis,
            [(int(a), int(b)) for a, b, _, _ in quartets],
            field=integral_field,
            threads=threads,
        )
        for a, b, _, _ in quartets:
            diagonal[spans[a], spans[b]] = measured[spans[a], spans[b]]
            diagonal[spans[b], spans[a]] = measured[spans[b], spans[a]]

    read_diagonal(quartets)
    residual = diagonal.ravel().copy()
    floor = 64 * np.finfo(float).eps * float(np.max(residual, initial=0.0))
    tolerance = max(threshold, floor)
    # A common London phase also cancels identically on a single center.
    # In these cases use real arithmetic and exact pair symmetry internally.
    real = field is None or all(x == 0 for x in field.B) or single_center
    dtype = np.float64 if real else np.complex128
    row_bytes = n * n * np.dtype(np.complex128).itemsize + 2 * 8
    limit = n * n if max_bytes is None else min(n * n, max_bytes // row_bytes)
    factors = np.empty((0, n * n), dtype=dtype)
    pivots = []
    candidates = np.empty(0, dtype=int)
    columns = np.empty((0, n * n), dtype=dtype)

    while float(np.max(residual, initial=0.0)) > tolerance:
        pivot = int(np.argmax(residual))
        rank = len(pivots)
        if rank >= limit:
            raise MemoryError(
                f"Cholesky ERI output requires at least {(rank + 1)*row_bytes} "
                f"bytes, exceeding max_bytes={max_bytes}"
            )
        if rank == len(factors):
            capacity = min(limit, max(8, 2 * len(factors)))
            grown = np.empty((capacity, n * n), dtype=dtype)
            grown[:rank] = factors[:rank]
            factors = grown
        match = np.flatnonzero(candidates == pivot)
        if not len(match):
            mu, nu = divmod(pivot, n)
            a, b = int(owners[mu]), int(owners[nu])
            indices = (np.arange(n * n).reshape(n, n)[spans[a], spans[b]]).ravel()
            indices = indices[residual[indices] > tolerance]
            candidates = indices[np.argsort(-residual[indices], kind="stable")[:8]]
            # Ensure the globally chosen pivot wins ties even within a shell pair.
            if pivot not in candidates:
                candidates[-1] = pivot
            cmu, cnu = np.divmod(candidates, n)
            columns = _cholesky_columns(
                basis,
                list(zip(cmu.tolist(), cnu.tolist(), strict=True)),
                field=integral_field,
                threads=threads,
            )
            match = np.flatnonzero(candidates == pivot)
        column = columns[int(match[0])].copy()
        if rank:
            column -= factors[:rank, pivot].conj() @ factors[:rank]
        if real:
            square = column.reshape(n, n)
            column = ((square + square.T) * 0.5).ravel()
        column /= np.sqrt(residual[pivot])
        factors[rank] = column
        pivots.append(divmod(pivot, n))
        residual -= np.abs(column) ** 2
        if float(np.min(residual, initial=0.0)) < -max(floor * 16, 1e-12):
            raise RuntimeError(
                "Coulomb Cholesky residual lost positive semidefiniteness"
            )
        np.maximum(residual, 0, out=residual)
        residual[pivot] = 0

    if skipped:
        read_diagonal(skipped)
        # Replace conservative upper diagonals by actual final residuals.
        residual = diagonal.ravel() - np.sum(
            np.abs(factors[: len(pivots)]) ** 2, axis=0
        )
    vectors = np.array(factors[: len(pivots)], dtype=np.complex128).reshape(
        len(pivots), n, n
    )
    return CholeskyERI(
        vectors,
        np.asarray(pivots, dtype=np.int64).reshape(-1, 2),
        threshold,
        float(np.max(residual, initial=0.0)),
    )
