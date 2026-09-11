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
