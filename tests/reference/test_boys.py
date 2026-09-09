import mpmath as mp
import pytest
from giao_reference import boys_hypergeometric, boys_quadrature


@pytest.mark.parametrize(
    "argument",
    [
        0.0,
        1.0e-12 + 2.0e-12j,
        0.75,
        2.0 + 3.0j,
        2.0 - 3.0j,
        -4.0 + 0.2j,
        -25.0 - 7.0j,
        40.0 + 25.0j,
    ],
)
@pytest.mark.parametrize("order", [0, 1, 4, 12, 32])
def test_hypergeometric_and_direct_quadrature_agree(argument, order):
    hypergeometric = boys_hypergeometric(order, argument, dps=90)
    quadrature = boys_quadrature(order, argument, dps=90)
    with mp.workdps(80):
        assert abs(hypergeometric - quadrature) <= mp.mpf("1e-75") * max(
            1, abs(hypergeometric)
        )


def test_reference_rejects_invalid_order():
    with pytest.raises(ValueError):
        boys_hypergeometric(-1, 0.0)
