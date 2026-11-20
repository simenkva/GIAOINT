import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[2] / "examples"


@pytest.mark.parametrize(
    "name", ["water_matrices.py", "eri_batches.py", "molecular_integrals.py"]
)
def test_examples_run(name, capsys):
    if name == "molecular_integrals.py":
        pytest.importorskip("basis_set_exchange")
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    assert capsys.readouterr().out


def test_atom_cholesky_timing(monkeypatch, capsys):
    pytest.importorskip("basis_set_exchange")
    import giao_integrals as gi

    path = EXAMPLES / "atom_cholesky_timing.py"
    counts = ["1", "2"] if gi.openmp_enabled() else ["1"]
    monkeypatch.setattr(
        "sys.argv",
        [str(path), "--basis", "STO-3G", "--uncontracted", "--threads", *counts],
    )
    runpy.run_path(str(path), run_name="__main__")
    output = capsys.readouterr().out
    assert "Atom: Ne" in output
    assert "uncontracted: True" in output
    assert "max residual" in output
    if gi.openmp_enabled():
        assert output.splitlines()[-1].endswith("yes")
