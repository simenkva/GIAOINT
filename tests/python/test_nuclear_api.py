import giao_integrals as gi
import numpy as np
import pytest


def test_nucleus_validation_and_properties():
    nucleus = gi.Nucleus(8.0, (0.1, -0.2, 0.3))
    assert nucleus.charge == 8.0
    assert nucleus.center == (0.1, -0.2, 0.3)
    with pytest.raises(ValueError):
        gi.Nucleus(0.0, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        gi.Nucleus(1.0, (0.0, np.nan, 0.0))


def test_shell_shape_dtype_out_and_empty_nuclei():
    shell = gi.Shell((0.0, 0.0, 0.0), 1, [1.2, 0.4], [0.3, 0.8])
    nuclei = [gi.Nucleus(1.0, (0.2, -0.1, 0.4))]
    result = gi.nuclear_attraction_shell(shell, shell, nuclei)
    assert result.shape == (3, 3)
    assert result.dtype == np.complex128
    output = np.empty_like(result)
    returned = gi.nuclear_attraction_shell(shell, shell, nuclei, out=output)
    assert returned is output
    np.testing.assert_array_equal(output, result)
    np.testing.assert_array_equal(
        gi.nuclear_attraction_shell(shell, shell, []), np.zeros_like(result)
    )


def test_shell_out_validation():
    shell = gi.Shell((0.0, 0.0, 0.0), 0, [1.0], [1.0])
    nucleus = gi.Nucleus(1.0, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="complex128"):
        gi.nuclear_attraction_shell(
            shell, shell, [nucleus], out=np.empty((1, 1), dtype=float)
        )
