from pathlib import Path

from exlightnet.report import md_table, build_report


def test_nested_table_values_are_readable_not_python_reprs():
    table = md_table(["Evidence"], [[{"rows": 123117, "counts": {"normal": 12507, "attack": 110610}, "seeds": [42, 43]}]])
    assert "{'" not in table and "[42" not in table
    assert "rows: 123117" in table and "normal: 12507" in table


def test_report_candidate_note_is_consistent(tmp_path):
    build_report(tmp_path)
    text = (tmp_path / "REPORT.md").read_text()
    assert "consistent (94 - 1 constant = 93)" in text
    assert "versus the paper's description of 93" not in text
