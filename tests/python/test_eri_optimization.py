"""Regression values captured from the M8 recursive kernel before its removal.

The fixture contains 96 fixed-seed quartets with independently varied centers,
exponents, signed coefficients, normalization, and Cartesian powers through g.
The independent 80-digit OS oracle remains in test_eri_against_reference.py.
"""

import json
from pathlib import Path

import giao_integrals as gi
import numpy as np
import pytest

RECORDS = [
    json.loads(line)
    for line in (Path(__file__).parents[1] / "data/eri_recursive_m8.jsonl")
    .read_text()
    .splitlines()
]


@pytest.mark.parametrize("record", RECORDS, ids=range(len(RECORDS)))
def test_optimized_eri_against_recursive_baseline(record):
    primitives = [gi.PrimitiveGaussian(**p) for p in record["primitives"]]
    for field, expected in zip(
        [(0, 0, 0), record["field"]], record["expected"], strict=True
    ):
        actual = gi.primitive_eri(
            *primitives, field=gi.MagneticField(field, record["gauge"])
        )
        np.testing.assert_allclose(
            actual, complex(*expected), atol=5.0e-14, rtol=5.0e-13
        )
