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


def test_stale_provisioned_graph_fails_before_affected_verify(
    tmp_path: Path, capsys: object, monkeypatch: object
) -> None:
    """A copied graph for another environment is rejected during provisioning.

    Anti-vacuity: changing the command to accept an absent environment would
    make this red because the stale environment remains in the seeded graph.
    """
    _seed_graph(tmp_path, "polylogue-stale-environment")
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]

    assert main([]) == 1
    assert "absent" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_current_provisioned_graph_passes_with_json_result(tmp_path: Path, capsys: object, monkeypatch: object) -> None:
    from devtools.testmon_bootstrap import testmon_environment_digest

    _seed_graph(tmp_path, testmon_environment_digest(tmp_path))
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]

    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["state"] == "valid"
