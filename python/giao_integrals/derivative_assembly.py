"""Bounded ERI derivative contractions with explicit complex density indices."""

from __future__ import annotations

from operator import index

import numpy as np

from ._giao_integrals import _contract_eri_derivative_batch
from ._threads import resolve_num_threads
from .eri import eri_derivative_batches


def eri_derivative_contraction(
    basis,
    shell_atoms,
    natom,
    *,
    field=None,
    coulomb_pairs=(),
    exchange_pairs=(),
    gamma=None,
    target_bytes=64 * 1024**2,
    screening_threshold=0.0,
    threads=None,
    out=None,
):
    r"""Return literal per-atom ERI derivative contractions, complex128 (natom,3).

    With I[mu,nu,lam,sig]=(mu nu|lam sig) in eri()'s conjugated-bra convention,
    return, for each atom A and nuclear axis r, the sum of:

    Coulomb pairs (P,Q): sum dI/dR[A,r] * P[nu,mu] * Q[sig,lam].
    Exchange pairs (P,Q): sum dI/dR[A,r] * P[nu,lam] * Q[sig,mu].
    Dense gamma: sum dI/dR[A,r] * gamma[mu,nu,lam,sig].

    Densities are held fixed under dR; no density/orbital response is included.
    P, Q and gamma are never conjugated implicitly. No 1/2, spin, exchange
    minus, charge or force sign is applied. Non-Hermitian complex densities
    are supported; the result is not forced real. Supplied pairs and gamma
    are added. P/Q are converted to C-contiguous complex128 (O(nao**2) each).
    gamma must already be C-contiguous float64/complex128 of shape (nao,)*4;
    it is read without copying. At least one density contribution is required.

    shell_atoms maps each basis shell to [0,natom); repeated slots add. Native
    work streams canonical derivative quartets, expanding exactly these orbits:
    identity (0,1,2,3), pair exchange (2,3,0,1), conjugate reversal (1,0,3,2),
    and conjugate exchanged reversal (3,2,1,0). Equal shell-quartet images are
    counted once. AO axes AND centre slots are permuted; only the derivative
    is conjugated for the last two images. No real eight-fold symmetry is used.

    target_bytes and opt-in screening have eri_derivative_batches semantics.
    The per-component screening bound does not bound the contracted total;
    screening is zero by default. Workspace is a bounded derivative batch,
    twelve complex partial sums per retained quartet and native workspaces;
    no nao**4 derivative tensor or extra dense gamma copy is formed. Native
    contraction/evaluation release the GIL and use configured threads=None.
    Each quartet uses fixed local reduction order, followed by fixed canonical
    quartet/slot order: results are bitwise invariant to thread scheduling and
    batch boundaries. out is an optional exact writable C-contiguous complex128
    (natom,3) buffer, returned exactly; it must not overlap any input density.
    """
    try:
        if isinstance(natom, (bool, np.bool_)):
            raise TypeError
        natom = index(natom)
        atoms = [
            index(a) if not isinstance(a, (bool, np.bool_)) else -1 for a in shell_atoms
        ]
    except TypeError as error:
        raise ValueError("natom and shell_atoms must be integers") from error
    if (
        natom < 0
        or len(atoms) != len(basis.shells)
        or any(a < 0 or a >= natom for a in atoms)
    ):
        raise ValueError("shell_atoms must map every shell to an atom in [0,natom)")
    n = basis.ao_count
    original_arrays = []

    def pairs(values):
        result = []
        for pair in values:
            if len(pair) != 2:
                raise ValueError("each density pair must contain P and Q")
            original_arrays.extend(a for a in pair if isinstance(a, np.ndarray))
            p, q = (np.ascontiguousarray(a, dtype=np.complex128) for a in pair)
            if p.shape != (n, n) or q.shape != (n, n):
                raise ValueError("density matrices must have shape (nao,nao)")
            result.append((p, q))
        return result

    coulomb = pairs(coulomb_pairs)
    exchange = pairs(exchange_pairs)
    if gamma is not None and (
        not isinstance(gamma, np.ndarray)
        or gamma.shape != (n,) * 4
        or not gamma.flags.c_contiguous
        or gamma.dtype not in (np.dtype("float64"), np.dtype("complex128"))
    ):
        raise ValueError(
            "gamma must be a C-contiguous float64/complex128 (nao,)*4 ndarray"
        )
    if not coulomb and not exchange and gamma is None:
        raise ValueError("at least one density pair or gamma is required")
    count = resolve_num_threads(threads)
    if out is None:
        result = np.zeros((natom, 3), dtype=np.complex128)
    else:
        if (
            not isinstance(out, np.ndarray)
            or out.shape != (natom, 3)
            or out.dtype != np.complex128
            or not out.flags.c_contiguous
            or not out.flags.writeable
        ):
            raise ValueError("out must be writable C-contiguous complex128 (natom,3)")
        result = out
        densities = [a for pair in coulomb + exchange for a in pair]
        densities.extend(original_arrays)
        if gamma is not None:
            densities.append(gamma)
        for density in densities:
            if np.shares_memory(result, density):
                raise ValueError("out must not overlap input densities")
        result.fill(0)
    for batch in eri_derivative_batches(
        basis,
        field=field,
        target_bytes=target_bytes,
        screening_threshold=screening_threshold,
        threads=count,
    ):
        _contract_eri_derivative_batch(
            basis,
            atoms,
            natom,
            batch.quartets.tolist(),
            batch.offsets.tolist(),
            batch.values,
            coulomb,
            exchange,
            gamma,
            result,
            threads=count,
        )
        del batch
    return result
