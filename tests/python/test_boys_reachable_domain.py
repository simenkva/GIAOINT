import math
import random

import giao_integrals as gi
from giao_reference import boys_hypergeometric


def test_reachable_argument_map_for_documented_input_envelope():
    """Map T from exponents [0.05, 20], 4-bohr boxes, and |B| <= 1."""

    generator = random.Random(410404)
    for _ in range(120):
        exponent_a = 10 ** generator.uniform(-1.3, 1.3)
        exponent_b = 10 ** generator.uniform(-1.3, 1.3)
        exponent = exponent_a + exponent_b
        center_a = tuple(generator.uniform(-2.0, 2.0) for _ in range(3))
        center_b = tuple(generator.uniform(-2.0, 2.0) for _ in range(3))
        nucleus = tuple(generator.uniform(-2.0, 2.0) for _ in range(3))

        direction = tuple(generator.gauss(0.0, 1.0) for _ in range(3))
        norm = math.sqrt(sum(value * value for value in direction))
        magnitude = generator.random()
        field = tuple(magnitude * value / norm for value in direction)
        separation = tuple(center_b[i] - center_a[i] for i in range(3))
        pair_wave_vector = (
            0.5 * (field[1] * separation[2] - field[2] * separation[1]),
            0.5 * (field[2] * separation[0] - field[0] * separation[2]),
            0.5 * (field[0] * separation[1] - field[1] * separation[0]),
        )
        product_center = tuple(
            (exponent_a * center_a[i] + exponent_b * center_b[i]) / exponent
            for i in range(3)
        )
        complex_center = tuple(
            product_center[i] - 0.5j * pair_wave_vector[i] / exponent for i in range(3)
        )
        argument = exponent * sum(
            (complex_center[i] - nucleus[i]) ** 2 for i in range(3)
        )

        values = gi.boys(argument, 32)
        for order in (0, 1, 8, 16, 32):
            expected = complex(boys_hypergeometric(order, argument, dps=80))
            assert abs(values[order] - expected) <= 3e-14 + 8e-13 * abs(expected)
