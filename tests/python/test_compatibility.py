from importlib.metadata import version

import giao_integrals as gi
import pytest
from giao_integrals._compat import deprecated


def test_runtime_and_distribution_versions_match():
    assert gi.__version__ == version("giao-integrals")


def test_deprecated_preserves_callable_and_emits_actionable_warning():
    @deprecated(replacement="new_api", removal="0.10")
    def old_api(value: int) -> int:
        """A test callable."""

        return value + 1

    with pytest.warns(FutureWarning, match=r"new_api.*0\.10"):
        assert old_api(4) == 5
    assert old_api.__name__ == "old_api"
    assert old_api.__doc__ == "A test callable."
    assert "new_api" in old_api.__deprecated__
