import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "benchmarks" / "compare_results.py"
SPEC = importlib.util.spec_from_file_location("compare_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
compare_results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_results)


def record(name: str, throughput: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "case": name,
        "operator": "overlap",
        "field": "finite",
        "shell_class": "s-p",
        "primitive_shape": "2x2",
        "threads": 1,
        "integrals_per_second": throughput,
    }


def write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in records))


def test_compare_reports_ratio_and_optional_threshold(tmp_path, capsys):
    baseline = tmp_path / "baseline.jsonl"
    current = tmp_path / "current.jsonl"
    write_records(baseline, [record("case-a", 100.0)])
    write_records(current, [record("case-a", 92.0)])

    assert compare_results.main([str(baseline), str(current)]) == 0
    assert "0.9200" in capsys.readouterr().out
    assert (
        compare_results.main([str(baseline), str(current), "--minimum-ratio", "0.95"])
        == 1
    )


def test_compare_rejects_missing_and_malformed_cases(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    current = tmp_path / "current.jsonl"
    write_records(baseline, [record("case-a", 100.0)])
    write_records(current, [record("case-b", 100.0)])
    assert compare_results.main([str(baseline), str(current)]) == 2

    current.write_text("not-json\n")
    with pytest.raises(SystemExit, match="2"):
        compare_results.main([str(baseline), str(current)])
