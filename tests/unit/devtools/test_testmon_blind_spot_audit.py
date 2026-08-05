"""Synthetic, mutation-proof tests for the testmon blind-spot audit."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from devtools.testmon_blind_spot_audit import (
    BlindSpotFinding,
    BlindSpotReport,
    audit_blind_spots,
    classify_source_ast,
    main,
)


def _write_coverage(path: Path, files: dict[str, tuple[int, int]]) -> None:
    path.write_text(
        json.dumps(
            {
                "meta": {"version": "7.15.2"},
                "files": {
                    name: {
                        "summary": {"num_statements": statements, "covered_lines": covered_lines},
                        "executed_lines": [],
                    }
                    for name, (statements, covered_lines) in files.items()
                },
            }
        ),
        encoding="utf-8",
    )


def _write_testmon(path: Path, filenames: tuple[str, ...] = ()) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE file_fp (id INTEGER PRIMARY KEY, filename TEXT, fsha TEXT)")
        connection.executemany(
            "INSERT INTO file_fp(filename, fsha) VALUES (?, ?)",
            [(filename, f"synthetic-{index}") for index, filename in enumerate(filenames)],
        )
        connection.commit()
    finally:
        connection.close()


def _finding(report: BlindSpotReport, path: str) -> BlindSpotFinding:
    return next(finding for finding in report.findings if finding.path == path)


def test_audit_separates_declaration_only_from_unfingerprinted_validator(tmp_path: Path) -> None:
    (tmp_path / "declarations.py").write_text(
        '"""A declaration-only synthetic module."""\n'
        "from typing import Final\n"
        "VALUE: Final[int] = 1\n"
        "\n"
        "class Marker:\n"
        "    name: str\n",
        encoding="utf-8",
    )
    (tmp_path / "validator.py").write_text(
        "def validate_payload(payload: dict[str, object]) -> bool:\n    return bool(payload)\n",
        encoding="utf-8",
    )
    coverage = tmp_path / "coverage.json"
    testmon = tmp_path / "testmondata"
    _write_coverage(coverage, {"declarations.py": (0, 0), "validator.py": (1, 0)})
    _write_testmon(testmon)

    report = audit_blind_spots(
        coverage_json_path=coverage,
        testmon_db_path=testmon,
        source_root=tmp_path,
    )

    declaration = _finding(report, "declarations.py")
    validator = _finding(report, "validator.py")
    assert declaration.ast_classification == "declaration-only"
    assert declaration.status == "declaration-only-unfingerprinted"
    assert declaration.safe is True
    assert validator.ast_classification == "executable"
    assert validator.status == "executable-validator-unfingerprinted"
    assert validator.safe is False
    assert len(report.risks) == 1


def test_fingerprint_presence_clears_the_executable_blind_spot(tmp_path: Path) -> None:
    source = tmp_path / "validator_fixture.py"
    source.write_text(
        "def validate(value: object) -> bool:\n    return value is not None\n",
        encoding="utf-8",
    )
    coverage = tmp_path / "coverage.json"
    testmon = tmp_path / "testmondata"
    _write_coverage(coverage, {"validator_fixture.py": (1, 1)})
    _write_testmon(testmon, ("validator_fixture.py",))

    report = audit_blind_spots(
        coverage_json_path=coverage,
        testmon_db_path=testmon,
        source_root=tmp_path,
    )

    finding = _finding(report, "validator_fixture.py")
    assert finding.testmon_fingerprinted is True
    assert finding.status == "fingerprinted"
    assert finding.safe is True


def test_evaluated_function_and_class_headers_are_executable(tmp_path: Path) -> None:
    source = tmp_path / "declaration_headers.py"
    source.write_text(
        "def configured(value=DEFAULT_VALUE, *, named=KEYWORD_DEFAULT):\n"
        "    pass\n\n"
        "class Configured(Base, Mixin, metaclass=METACLASS):\n"
        "    pass\n",
        encoding="utf-8",
    )

    assert classify_source_ast(source) == "executable"


def test_fingerprinted_source_read_error_is_unsafe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "unreadable_fixture.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    coverage = tmp_path / "coverage.json"
    testmon = tmp_path / "testmondata"
    _write_coverage(coverage, {"unreadable_fixture.py": (0, 0)})
    _write_testmon(testmon, ("unreadable_fixture.py",))

    original_read_text = Path.read_text

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path == source:
            raise OSError("synthetic source read failure")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    report = audit_blind_spots(
        coverage_json_path=coverage,
        testmon_db_path=testmon,
        source_root=tmp_path,
    )

    finding = _finding(report, "unreadable_fixture.py")
    assert finding.testmon_fingerprinted is True
    assert finding.status == "source-unreadable"
    assert finding.safe is False
    assert (
        main(
            [
                "--coverage-json",
                str(coverage),
                "--testmon-db",
                str(testmon),
                "--source-root",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert "source-unreadable" in capsys.readouterr().out


def test_mutation_proof_fixture_or_ast_change_never_makes_validator_safe(tmp_path: Path) -> None:
    """A coverage fixture cannot bless a validator after its AST becomes executable."""
    source = tmp_path / "same_fixture.py"
    source.write_text(
        '"""Only declarations before the mutation."""\nVALUE: int = 1\n',
        encoding="utf-8",
    )
    coverage = tmp_path / "coverage.json"
    testmon = tmp_path / "testmondata"
    _write_coverage(coverage, {"same_fixture.py": (0, 0)})
    _write_testmon(testmon)

    safe_declaration = audit_blind_spots(
        coverage_json_path=coverage,
        testmon_db_path=testmon,
        source_root=tmp_path,
    )
    assert _finding(safe_declaration, "same_fixture.py").safe is True

    source.write_text(
        "def validate_payload(payload: dict[str, object]) -> bool:\n"
        "    if not payload:\n"
        "        return False\n"
        "    return True\n",
        encoding="utf-8",
    )
    executable_with_stale_fixture = audit_blind_spots(
        coverage_json_path=coverage,
        testmon_db_path=testmon,
        source_root=tmp_path,
    )
    mutated = _finding(executable_with_stale_fixture, "same_fixture.py")
    assert mutated.ast_classification == "executable"
    assert mutated.status == "executable-validator-unfingerprinted"
    assert mutated.safe is False

    _write_coverage(coverage, {"same_fixture.py": (2, 2)})
    executable_with_changed_fixture = audit_blind_spots(
        coverage_json_path=coverage,
        testmon_db_path=testmon,
        source_root=tmp_path,
    )
    changed = _finding(executable_with_changed_fixture, "same_fixture.py")
    assert changed.status == "executable-validator-unfingerprinted"
    assert changed.safe is False


def test_audit_is_read_only_and_main_returns_risk_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "validator.py"
    source.write_text("def validate(value: object) -> bool:\n    return True\n", encoding="utf-8")
    coverage = tmp_path / "coverage.json"
    testmon = tmp_path / "testmondata"
    _write_coverage(coverage, {"validator.py": (1, 0)})
    _write_testmon(testmon)
    coverage_before = hashlib.sha256(coverage.read_bytes()).digest()
    testmon_before = hashlib.sha256(testmon.read_bytes()).digest()

    assert (
        main(
            [
                "--coverage-json",
                str(coverage),
                "--testmon-db",
                str(testmon),
                "--source-root",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert "executable-validator-unfingerprinted" in capsys.readouterr().out
    assert hashlib.sha256(coverage.read_bytes()).digest() == coverage_before
    assert hashlib.sha256(testmon.read_bytes()).digest() == testmon_before
