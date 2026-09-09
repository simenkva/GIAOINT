"""Independent high-precision definitions of the complex Boys function."""

from __future__ import annotations

from numbers import Integral

import mpmath as mp


def _validate_order(order: int) -> int:
    if isinstance(order, bool) or not isinstance(order, Integral) or order < 0:
        raise ValueError("Boys order must be a non-negative integer")
    return int(order)


def boys_hypergeometric(order: int, argument: complex, *, dps: int = 80):
    """Evaluate ``F_n(z)`` from its entire confluent-hypergeometric form."""

    order = _validate_order(order)
    with mp.workdps(dps):
        z = mp.mpc(argument)
        return mp.hyp1f1(order + mp.mpf("0.5"), order + mp.mpf("1.5"), -z) / (
            2 * order + 1
        )


def boys_quadrature(order: int, argument: complex, *, dps: int = 80):
    """Evaluate ``F_n(z)`` directly from its defining finite integral."""

    order = _validate_order(order)
    with mp.workdps(dps):
        z = mp.mpc(argument)
        return mp.quad(lambda t: t ** (2 * order) * mp.exp(-z * t * t), [0, 1])


def boys_values(
    maximum_order: int,
    argument: complex,
    *,
    dps: int = 80,
    scaled: bool = False,
):
    """Return ``F_0..F_nmax`` using the hypergeometric definition."""

    maximum_order = _validate_order(maximum_order)
    with mp.workdps(dps):
        z = mp.mpc(argument)
        factor = mp.exp(z) if scaled else mp.mpf(1)
        return tuple(
            factor * mp.hyp1f1(n + mp.mpf("0.5"), n + mp.mpf("1.5"), -z) / (2 * n + 1)
            for n in range(maximum_order + 1)
        )
