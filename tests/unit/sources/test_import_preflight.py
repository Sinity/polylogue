"""Import preflight truthfulness tests for unsupported/degraded shapes."""

from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.operations.import_operations import prepare_import_source_admission
from polylogue.sources.import_preflight import ImportPreflightStatus
from tests.infra.antigravity_parser import parse_trajectory_db


def _chatgpt_payload() -> dict[str, object]:
    return {
        "id": "chatgpt-preflight-fixture",
        "conversation_id": "chatgpt-preflight-fixture",
        "title": "Supported ChatGPT fixture",
        "create_time": 1704067200.0,
        "current_node": "root",
        "mapping": {
            "root": {
                "id": "root",
                "message": {
                    "id": "root-message",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["hello"]},
                },
                "children": [],
            }
        },
    }


def test_preflight_accepts_supported_json_file(tmp_path: Path) -> None:
    source = tmp_path / "chatgpt.json"
    source.write_text(json.dumps(_chatgpt_payload()))

    result = prepare_import_source_admission(source).preflight

    assert result.status is ImportPreflightStatus.SUPPORTED
    assert result.admissible is True
    assert result.supported_count == 1
    assert result.providers == (Provider.CHATGPT,)
    assert result.error_code == ""


def test_preflight_accepts_antigravity_trajectory_sqlite(tmp_path: Path) -> None:
    source = tmp_path / "renamed-trajectory.sqlite"
    with sqlite3.connect(source) as connection:
        connection.executescript(
            """
            CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);
            CREATE TABLE steps (idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT);
            INSERT INTO trajectory_meta VALUES ('trajectory-preflight', 'cascade-preflight');
            INSERT INTO steps VALUES (0, 'message', 'v1', '{"role":"user","text":"hello"}');
            """
        )

    result = prepare_import_source_admission(source).preflight

    assert result.status is ImportPreflightStatus.SUPPORTED
    assert result.providers == (Provider.ANTIGRAVITY,)
    assert result.supported_count == 1


@pytest.mark.parametrize(
    "steps_sql",
    [
        "",
        'INSERT INTO steps VALUES (0, \'message\', \'future-format\', \'{"role":"user","text":"hi"}\');',
    ],
    ids=["empty-trajectory", "only-unsupported-steps"],
)
def test_preflight_refuses_what_the_production_evidence_gate_refuses(tmp_path: Path, steps_sql: str) -> None:
    """A trajectory whose steps yield no message is refused as production refuses it.

    Anti-vacuity: report an event-only trajectory as supported and preflight
    promises an import that ``admit_parsed_sessions_for_publication``
    removes on every production write path.
    """
    from polylogue.sources.dispatch import admit_parsed_sessions_for_publication

    source = tmp_path / "quiet-trajectory.sqlite"
    with sqlite3.connect(source) as connection:
        connection.executescript(
            f"""
            CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);
            CREATE TABLE steps (idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT);
            INSERT INTO trajectory_meta VALUES ('trajectory-quiet', 'cascade-quiet');
            {steps_sql}
            """
        )
    sessions = list(parse_trajectory_db(source, fallback_id=source.stem))

    result = prepare_import_source_admission(source).preflight

    assert admit_parsed_sessions_for_publication(sessions, provider=Provider.ANTIGRAVITY, source_path=None) == []
    assert result.admissible is False


def test_preflight_rejects_unknown_json_shape(tmp_path: Path) -> None:
    source = tmp_path / "unknown.json"
    source.write_text(json.dumps({"not": "an export"}))

    result = prepare_import_source_admission(source).preflight

    assert result.status is ImportPreflightStatus.UNSUPPORTED
    assert result.admissible is False
    assert result.error_code == "unsupported_import_source"
    assert result.unsupported_count == 1
    assert "unsupported" in result.summary()


def test_preflight_rejects_malformed_json(tmp_path: Path) -> None:
    source = tmp_path / "broken.json"
    source.write_text('{"mapping": ')

    result = prepare_import_source_admission(source).preflight

    assert result.status is ImportPreflightStatus.MALFORMED
    assert result.admissible is False
    assert result.error_code == "malformed_import_source"
    assert result.malformed_count == 1


def test_preflight_accepts_supported_zip_member(tmp_path: Path) -> None:
    source = tmp_path / "export.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("conversations.json", json.dumps(_chatgpt_payload()))

    result = prepare_import_source_admission(source).preflight

    assert result.status is ImportPreflightStatus.SUPPORTED
    assert result.supported_count == 1
    assert result.providers == (Provider.CHATGPT,)
    assert result.samples == ("export.zip:conversations.json: chatgpt",)


def test_preflight_marks_mixed_directory_as_degraded(tmp_path: Path) -> None:
    source = tmp_path / "fixture-world"
    source.mkdir()
    (source / "chatgpt.json").write_text(json.dumps(_chatgpt_payload()))
    (source / "unknown.json").write_text(json.dumps({"not": "an export"}))

    result = prepare_import_source_admission(source).preflight

    assert result.status is ImportPreflightStatus.DEGRADED
    assert result.admissible is True
    assert result.supported_count == 1
    assert result.unsupported_count == 1
    assert "degraded" in result.summary()


def test_preflight_rejects_zip_without_parseable_members(tmp_path: Path) -> None:
    source = tmp_path / "notes.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("README.txt", "not an export")

    result = prepare_import_source_admission(source).preflight

    assert result.status is ImportPreflightStatus.UNSUPPORTED
    assert result.admissible is False
    assert result.error_code == "unsupported_import_source"


def test_preflight_accepts_complete_high_ratio_conversation(tmp_path: Path) -> None:
    source = tmp_path / "compressed-preflight.zip"
    payload = _chatgpt_payload()
    payload["padding"] = "x" * (16 * 1024 * 1024)
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("conversations.json", json.dumps(payload))
    with zipfile.ZipFile(source) as archive:
        member = archive.infolist()[0]
        assert member.file_size / member.compress_size > 1000

    result = prepare_import_source_admission(source).preflight

    assert result.status is ImportPreflightStatus.SUPPORTED
    assert result.supported_count == 1
    assert result.providers == (Provider.CHATGPT,)


def test_preflight_inspects_conversational_evidence_after_the_former_prefix(tmp_path: Path) -> None:
    """An empty first eight trajectories cannot hide a later admitted session."""
    source = tmp_path / "wide-trajectory.sqlite"
    with sqlite3.connect(source) as connection:
        connection.executescript(
            """
            CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);
            CREATE TABLE steps (trajectory_id TEXT, idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT);
            """
        )
        for index in range(40):
            connection.execute(
                "INSERT INTO trajectory_meta VALUES (?, ?)",
                (f"trajectory-{index:03d}", f"cascade-{index:03d}"),
            )
            if index == 39:
                connection.execute(
                    'INSERT INTO steps VALUES (?, 0, \'message\', \'v1\', \'{"role":"user","text":"hello"}\')',
                    (f"trajectory-{index:03d}",),
                )

    result = prepare_import_source_admission(source).preflight

    assert result.status is ImportPreflightStatus.DEGRADED
    assert result.providers == (Provider.ANTIGRAVITY,)
    assert result.supported_count == 1
    from polylogue.sources.sqlite_inspection import inspect_sqlite_source

    inspection = inspect_sqlite_source(source, preflight=True)
    assert inspection.produced["sessions"] == 40
    assert inspection.admitted == 1
    assert inspection.produced["session_refs"] == []


@pytest.mark.parametrize("retained_export", [False, True])
def test_sqlite_preflight_cancellation_reaches_the_worker(tmp_path: Path, retained_export: bool) -> None:
    """Dropping progress or classifying callback failure as malformed makes this red."""
    from polylogue.sources.import_preflight import preflight_import_bindings
    from polylogue.sources.source_staging import bind_source_input
    from polylogue.sources.sqlite_export import write_logical_export
    from polylogue.sources.sqlite_inspection import inspect_sqlite_source

    source = tmp_path / "trajectories.sqlite"
    with sqlite3.connect(source) as connection:
        connection.executescript(
            "CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);"
            "CREATE TABLE steps (trajectory_id TEXT, idx INTEGER, step_type TEXT, "
            "step_format TEXT, step_payload TEXT);"
        )
        connection.executemany(
            "INSERT INTO trajectory_meta VALUES (?, ?)",
            ((f"trajectory-{index}", f"cascade-{index}") for index in range(2048)),
        )
    if retained_export:
        export = tmp_path / "retained.sqlite"
        with export.open("wb") as handle:
            write_logical_export(source, handle)
        source = export

    calls = 0
    cancellation = ValueError("synthetic cancellation")

    def check_stop() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise cancellation

    with bind_source_input(source) as binding, pytest.raises(ValueError) as raised:
        preflight_import_bindings(
            [(binding, source.name)], source_path=str(source), single_file=True, check_stop=check_stop
        )
    assert raised.value is cancellation
    assert calls == 2
    # A subsequent real inspection can use the same source after worker settlement.
    assert inspect_sqlite_source(source, preflight=True).produced["sessions"] == 2048


def test_several_unidentified_trajectory_rows_are_refused(tmp_path: Path) -> None:
    """Several trajectory rows without ids are refused, not told apart by position.

    A row's position or rowid moves with deletions and VACUUM, so an id minted
    from it would re-point one trajectory onto another's archive identity
    (#5711). Anti-vacuity: mint ``<fallback>:trajectory-<n>`` for each
    unidentified row and the export parses into two sessions.
    """
    from polylogue.sources.sqlite_export import LogicalExportError

    source = tmp_path / "unnamed.sqlite"
    with sqlite3.connect(source) as connection:
        connection.executescript(
            """
            CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);
            CREATE TABLE steps (idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT);
            INSERT INTO trajectory_meta VALUES (NULL, NULL);
            INSERT INTO trajectory_meta VALUES ('', NULL);
            """
        )

    with pytest.raises(LogicalExportError):
        list(parse_trajectory_db(source, fallback_id="unnamed"))


def test_generated_trajectory_id_avoids_a_native_id(tmp_path: Path) -> None:
    """A generated row id never reuses a provider-native trajectory id.

    Anti-vacuity: take ``<fallback>:trajectory-0`` without checking the native
    ids and both rows share one ``provider_session_id``.
    """

    source = tmp_path / "x.sqlite"
    with sqlite3.connect(source) as connection:
        connection.executescript(
            """
            CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);
            CREATE TABLE steps (idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT);
            INSERT INTO trajectory_meta VALUES (NULL, NULL);
            INSERT INTO trajectory_meta VALUES ('x:trajectory-0', NULL);
            """
        )

    sessions = list(parse_trajectory_db(source, fallback_id="x"))

    identities = [session.provider_session_id for session in sessions]
    assert len(identities) == 2
    assert len(set(identities)) == 2
