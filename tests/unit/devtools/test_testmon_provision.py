from __future__ import annotations

import json
from pathlib import Path

import testmon.db

from devtools.testmon_provision import main


def _seed_graph(root: Path, environment_name: str) -> None:
    data = root / ".cache" / "testmon" / "testmondata"
    data.parent.mkdir(parents=True)
    database = testmon.db.DB(str(data))
    try:
        database.con.execute(
            "INSERT INTO environment (environment_name, system_packages, python_version) VALUES (?, '', '')",
            (environment_name,),
        )
        execution = database.con.execute(
            "INSERT INTO test_execution (environment_id, test_name, duration, failed, forced) "
            "VALUES (last_insert_rowid(), 'tests/unit/test_sample.py::test_sample', 0, 0, 0)"
        )
        assert execution.rowcount == 1
        fingerprint = database.con.execute(
            "INSERT INTO file_fp (filename, method_checksums, mtime, fsha) "
            "VALUES ('tests/unit/test_sample.py', '', 0, '')"
        )
        assert fingerprint.rowcount == 1
        database.con.execute(
            "INSERT INTO test_execution_file_fp (test_execution_id, fingerprint_id) VALUES (?, ?)",
            (execution.lastrowid, fingerprint.lastrowid),
        )
        database.con.commit()
    finally:
        database.con.close()


def test_stale_provisioned_graph_is_discarded_not_inherited(
    tmp_path: Path, capsys: object, monkeypatch: object
) -> None:
    """A copied graph for another environment is dropped during provisioning.

    Anti-vacuity: keeping the file makes this red. Failing the command instead
    would fail every lane, because the seed a workspace copies is routinely
    older than the environment it is provisioned into.
    """
    _seed_graph(tmp_path, "polylogue-stale-environment")
    sidecar = tmp_path / ".cache" / "testmon" / "testmondata-wal"
    sidecar.write_bytes(b"")
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]

    assert main([]) == 0
    assert not (tmp_path / ".cache" / "testmon" / "testmondata").exists()
    # A surviving sidecar reads as damaged state and refuses the lane's
    # verification, which is worse than the stale seed it came with.
    assert not sidecar.exists()
    assert "discarded" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_current_provisioned_graph_passes_with_json_result(tmp_path: Path, capsys: object, monkeypatch: object) -> None:
    from devtools.testmon_bootstrap import testmon_environment_digest

    _seed_graph(tmp_path, testmon_environment_digest(tmp_path))
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]

    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["state"] == "valid"
    assert payload["discarded"] is False
    assert (tmp_path / ".cache" / "testmon" / "testmondata").exists()


def test_absent_graph_provisions_without_a_seed(tmp_path: Path, capsys: object, monkeypatch: object) -> None:
    """A workspace with no seeded graph provisions.

    The cache is untracked, so a fresh checkout has no graph to copy; refusing
    here fails every lane before it starts.
    """
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]

    assert main([]) == 0
    assert "absent" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_seed_removal_failure_is_a_typed_error_result(tmp_path: Path, capsys: object, monkeypatch: object) -> None:
    """A seed that cannot be removed reports state error, never a traceback.

    Anti-vacuity: letting the repair error escape leaves lane provisioning
    with no JSON result to classify.
    """
    _seed_graph(tmp_path, "polylogue-stale-environment")
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]

    def refuse(_root: Path) -> tuple[Path, ...]:
        from devtools.testmon_bootstrap import NativeTestmonRepairError

        raise NativeTestmonRepairError("refusing to remove hard-linked owned SQLite path")

    monkeypatch.setattr("devtools.testmon_provision.remove_invalid_native_testmon_state", refuse)  # type: ignore[attr-defined]

    assert main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["state"] == "error"
    assert payload["discarded"] is False
    assert "hard-linked" in payload["reason"]
