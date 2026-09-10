import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[2] / "examples"


@pytest.mark.parametrize("name", ["water_matrices.py", "eri_batches.py"])
def test_examples_run(name, capsys):
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    assert capsys.readouterr().out
