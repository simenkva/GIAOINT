"""The installed documentation must describe the actual derivative signature."""

import giao_integrals as gi
import pytest


@pytest.mark.parametrize(
    ("name", "shape", "property_argument"),
    [
        ("moment_magnetic_derivatives", "(3, nao, nao)", "powers"),
        ("momentum_nuclear_derivatives", "(nshell, 3, nao, nao)", "component"),
        ("momentum_magnetic_derivatives", "(3, nao, nao)", "component"),
        ("gradient_nuclear_derivatives", "(nshell, 3, nao, nao)", "component"),
        ("gradient_magnetic_derivatives", "(3, nao, nao)", "component"),
    ],
)
def test_property_docstring_matches_returned_shape(name, shape, property_argument):
    function = getattr(gi, name)
    doc = function.__doc__
    assert f"shape {shape}" in doc
    assert property_argument + ":" in doc
    assert "gauge origin" in doc
    assert "London" in doc
    assert "no explicit B dependence" in doc
    if name == "moment_magnetic_derivatives":
        assert "component:" not in doc
        assert "moment origin" in doc
    else:
        assert "moment origin" not in doc
    if "magnetic" in name:
        assert "(nshell, 3, nao, nao)" not in doc
    else:
        assert "shape (3, nao, nao)" not in doc
    basis = gi.Basis([gi.Shell((0.1, -0.2, 0.3), 0, [0.8], [1.0])])
    argument = (1, 0, 0) if property_argument == "powers" else "x"
    result = function(basis, argument, threads=1)
    assert result.shape == ((3, 1, 1) if "magnetic" in name else (1, 3, 1, 1))


@pytest.mark.parametrize("name", ["overlap", "kinetic", "magnetic_kinetic"])
def test_mixed_operator_docstrings_are_specific(name):
    doc = getattr(gi, name + "_mixed_derivatives").__doc__
    assert "shape (nshell, 3, 3, nao, nao)" in doc
    if name == "magnetic_kinetic":
        assert "explicit B dependence" in doc
        assert "no explicit B dependence" not in doc
    else:
        assert "no explicit B dependence" in doc
        assert "physical operator" not in doc


def test_primitive_potential_docstring_has_no_ao_dimensions():
    doc = gi.primitive_nuclear_attraction_nucleus_mixed_derivatives.__doc__
    assert "shape (nnucleus, 3, 3)" in doc
    assert "bra AO" not in doc
