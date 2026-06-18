"""Tests for ``devtools/failure_context.py``."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from devtools import failure_context as fc


@pytest.fixture()
def stub_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the module's REPO_ROOT/TESTMON_DB/WITNESS_DIR to a temp tree."""
    monkeypatch.setattr(fc, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(fc, "TESTMON_DB", tmp_path / ".testmondata")
    monkeypatch.setattr(fc, "WITNESS_DIR", tmp_path / "tests" / "witnesses")
    return tmp_path


def _seed_testmon(db_path: Path, mapping: dict[str, list[str]]) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE test_execution (id INTEGER PRIMARY KEY, environment_id INTEGER,
            test_name TEXT, duration FLOAT, failed BIT, forced BIT);
        CREATE TABLE file_fp (id INTEGER PRIMARY KEY, filename TEXT,
            method_checksums BLOB, mtime FLOAT, fsha TEXT);
        CREATE TABLE test_execution_file_fp (test_execution_id INTEGER, fingerprint_id INTEGER);
        """
    )
    file_id = 1
    test_id = 1
    for test_name, files in mapping.items():
        conn.execute("INSERT INTO test_execution(id, test_name) VALUES (?, ?)", (test_id, test_name))
        for filename in files:
            conn.execute("INSERT INTO file_fp(id, filename) VALUES (?, ?)", (file_id, filename))
            conn.execute(
                "INSERT INTO test_execution_file_fp(test_execution_id, fingerprint_id) VALUES (?, ?)",
                (test_id, file_id),
            )
            file_id += 1
        test_id += 1
    conn.commit()
    conn.close()


def test_parse_failure_id_splits_path_and_name() -> None:
    assert fc._parse_failure_id("tests/foo.py::test_bar") == ("tests/foo.py", "test_bar")


def test_parse_failure_id_rejects_missing_separator() -> None:
    with pytest.raises(ValueError):
        fc._parse_failure_id("tests/foo.py")


def test_testmon_dependencies_returns_sorted_unique(stub_repo: Path) -> None:
    _seed_testmon(
        stub_repo / ".testmondata",
        {"tests/unit/foo.py::test_a": ["polylogue/b.py", "polylogue/a.py"]},
    )
    deps = fc._testmon_dependencies("tests/unit/foo.py::test_a")
    assert deps == ["polylogue/a.py", "polylogue/b.py"]


def test_testmon_dependencies_handles_missing_db(stub_repo: Path) -> None:
    assert fc._testmon_dependencies("tests/foo.py::test_missing") == []


def test_related_fixtures_extracts_parametrized_test(stub_repo: Path) -> None:
    test_file = stub_repo / "tests" / "unit" / "sample.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "def test_thing(tmp_path, workspace_env, value: int = 1) -> None:\n    pass\n",
        encoding="utf-8",
    )
    fixtures = fc._related_fixtures("tests/unit/sample.py", "test_thing[case-a]")
    assert "tmp_path" in fixtures
    assert "workspace_env" in fixtures
    assert "value" in fixtures


def test_similar_witnesses_scores_source_test_match(stub_repo: Path) -> None:
    witness_dir = stub_repo / "tests" / "witnesses"
    witness_dir.mkdir(parents=True)
    (witness_dir / "blob.witness.json").write_text(
        json.dumps(
            {
                "witness_id": "storage.blob",
                "provenance": {"source_test": "tests/unit/storage/test_blob.py"},
            }
        ),
        encoding="utf-8",
    )
    (witness_dir / "unrelated.witness.json").write_text(
        json.dumps({"witness_id": "unrelated", "provenance": {"source_test": "tests/other.py"}}),
        encoding="utf-8",
    )
    results = fc._similar_witnesses("tests/unit/storage/test_blob.py::test_layout")
    assert results, "expected at least one witness match"
    assert results[0]["witness_id"] == "storage.blob"


def test_build_envelope_shape(stub_repo: Path) -> None:
    _seed_testmon(
        stub_repo / ".testmondata",
        {"tests/unit/foo.py::test_a": ["polylogue/a.py"]},
    )
    envelope = fc.build_envelope("tests/unit/foo.py::test_a", days=7)
    assert envelope["failure_id"] == "tests/unit/foo.py::test_a"
    assert envelope["test_file"] == "tests/unit/foo.py"
    assert envelope["test_name"] == "test_a"
    assert envelope["testmon_dependencies"] == ["polylogue/a.py"]
    assert "recent_changes" in envelope
    assert "related_fixtures" in envelope
    assert "similar_witnesses" in envelope
    assert envelope["metadata"]["days_window"] == 7
    assert envelope["metadata"]["testmon_db_present"] is True


def test_main_emits_json_envelope(stub_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _seed_testmon(
        stub_repo / ".testmondata",
        {"tests/unit/foo.py::test_a": ["polylogue/a.py"]},
    )
    exit_code = fc.main(["tests/unit/foo.py::test_a"])
    assert exit_code == 0
    parsed: dict[str, Any] = json.loads(capsys.readouterr().out)
    assert parsed["failure_id"] == "tests/unit/foo.py::test_a"


def test_main_rejects_bad_failure_id(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = fc.main(["not-a-failure-id"])
    assert exit_code == 2
    assert "failure id must be" in capsys.readouterr().err


def test_command_registered_in_catalog() -> None:
    from devtools.command_catalog import COMMANDS

    assert "workspace failure-context" in COMMANDS
    spec = COMMANDS["workspace failure-context"]
    assert spec.module == "devtools.failure_context"
    assert callable(spec.resolve_main())
