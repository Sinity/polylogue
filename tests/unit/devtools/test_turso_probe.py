from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from devtools import turso_probe


def test_probe_reports_missing_python_binding_and_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(turso_probe, "_find_python_turso", lambda: None)
    monkeypatch.setattr(turso_probe, "_find_tursodb", lambda: None)

    payload = turso_probe.run_probe(scratch_dir=tmp_path)

    assert payload["ok"] is True
    assert payload["tursodb"] is None
    assert payload["compatibility_blockers"] == ["python_binding"]
    rows = cast(list[dict[str, object]], payload["results"])
    results = {row["name"]: row for row in rows}
    assert results["python_binding"]["status"] == "skip"
    assert results["python_runtime_api"]["status"] == "skip"
    assert results["python_readonly_uri"]["status"] == "skip"
    assert results["python_multiprocess_wal"]["status"] == "skip"
    assert results["tursodb_binary"]["status"] == "skip"


def test_probe_classifies_polylogue_sql_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(turso_probe, "_find_python_turso", lambda: None)

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        sql = command[-1]
        failing = "STORED" in sql or "fts5" in sql or "journal_size_limit" in sql
        return subprocess.CompletedProcess(
            command,
            1 if failing else 0,
            stdout="" if failing else "ok",
            stderr="unsupported" if failing else "",
        )

    monkeypatch.setattr(turso_probe, "_run_command", fake_run)

    payload = turso_probe.run_probe(tursodb="/fake/tursodb", scratch_dir=tmp_path)

    assert payload["ok"] is True
    assert payload["unexpected"] == []
    assert payload["compatibility_blockers"] == [
        "python_binding",
        "stored_generated_columns",
        "fts5_virtual_table",
    ]
    rows = cast(list[dict[str, object]], payload["results"])
    results = {row["name"]: row for row in rows}
    assert results["strict_tables"]["status"] == "pass"
    assert results["stored_generated_columns"]["status"] == "fail"
    assert results["stored_generated_columns"]["expected"] is True
    assert results["fts5_virtual_table"]["status"] == "fail"
    assert results["wal_journal_size_limit"]["status"] == "fail"
    assert results["attach_experimental"]["status"] == "pass"


def test_python_readonly_uri_failure_is_a_compatibility_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        turso_probe,
        "_python_binding_probe",
        lambda: turso_probe.ProbeResult(
            name="python_binding",
            status="pass",
            expected_status="pass",
            summary="available",
            command=["python", "-c", "import turso"],
        ),
    )
    monkeypatch.setattr(
        turso_probe,
        "_python_runtime_api_probe",
        lambda: turso_probe.ProbeResult(
            name="python_runtime_api",
            status="pass",
            expected_status="pass",
            summary="runtime ok",
            command=["python", "-c", "import turso"],
        ),
    )
    monkeypatch.setattr(
        turso_probe,
        "_python_readonly_uri_probe",
        lambda *, scratch_dir: turso_probe.ProbeResult(
            name="python_readonly_uri",
            status="fail",
            expected_status="fail",
            summary=f"readonly incompatible in {scratch_dir}",
            command=["python", "-c", "import turso"],
        ),
    )
    monkeypatch.setattr(
        turso_probe,
        "_python_multiprocess_probe",
        lambda *, scratch_dir: turso_probe.ProbeResult(
            name="python_multiprocess_wal",
            status="pass",
            expected_status="pass",
            summary=f"multiprocess ok in {scratch_dir}",
            command=["python", "-c", "import turso"],
        ),
    )
    monkeypatch.setattr(turso_probe, "_find_tursodb", lambda: None)

    payload = turso_probe.run_probe(scratch_dir=tmp_path)

    assert payload["ok"] is True
    assert payload["unexpected"] == []
    assert payload["compatibility_blockers"] == ["python_readonly_uri"]


def test_main_json_emits_probe_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_probe(*, tursodb: str | None, scratch_dir: Path | None) -> dict[str, Any]:
        _ = scratch_dir
        return {
            "ok": True,
            "tursodb": tursodb,
            "results": [],
            "compatibility_blockers": [],
            "unexpected": [],
            "recommendation": "ok",
        }

    monkeypatch.setattr(
        turso_probe,
        "run_probe",
        fake_probe,
    )

    assert turso_probe.main(["--json", "--scratch-dir", str(tmp_path), "--tursodb", "/fake"]) == 0
    assert '"tursodb": "/fake"' in capsys.readouterr().out
