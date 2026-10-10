"""Original source-cell intake, independently of retired job scheduling."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.browser_capture.capture_jobs import canonical_digest, canonical_json
from polylogue.browser_capture.models import BrowserCaptureEnvelope
from polylogue.browser_capture.receiver import write_capture_envelope
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, json_document, loads
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.live import WatchSource


def _envelope() -> JSONDocument:
    fixture = Path(__file__).parents[2] / "fixtures/chatgpt/native-browser-capture-v1.json"
    envelope = json_document(loads(fixture.read_bytes()))
    envelope.pop("raw_provider_payload")
    session = envelope["session"]
    metadata = envelope["provider_meta"]
    provenance = envelope["provenance"]
    assert isinstance(session, dict) and isinstance(metadata, dict) and isinstance(provenance, dict)
    session_metadata = session["provider_meta"]
    assert isinstance(session_metadata, dict)
    session["attachments"] = []
    metadata["capture_fidelity"] = "dom_fallback"
    session_metadata["capture_fidelity"] = "dom_fallback"
    provenance["provider_meta"] = {"original": {"value": "retained"}}
    return envelope


def _source_cell(root: Path, envelope: JSONDocument, *, digest: str | None = None) -> Path:
    path = root / "capture-jobs/registry.sqlite3"
    path.parent.mkdir(parents=True)
    payload = {"version": 1, "jobs": [], "queue": [{"id": "source-only", "envelope": envelope}], "revisions": []}
    checkpoint = {"sequence": 1, "payload": payload, "digest": digest or canonical_digest(payload)}
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE capture_jobs (job_id TEXT PRIMARY KEY, checkpoint_json TEXT)")
        connection.execute(
            "INSERT INTO capture_jobs VALUES (?, ?)", ("retired-bookkeeping", canonical_json(checkpoint))
        )
    return path


@pytest.mark.asyncio
async def test_owned_source_preparation_extracts_the_original_cell_without_a_receiver(tmp_path: Path) -> None:
    from polylogue.daemon.cli import _prepare_owned_source_roots

    root = tmp_path / "spool"
    original = _envelope()
    database = _source_cell(root, original)
    before = database.read_bytes()
    assert not list(root.rglob("*.json"))
    await _prepare_owned_source_roots([WatchSource("browser-capture", root)])
    artifacts = list((root / "chatgpt").glob("*.json"))
    assert len(artifacts) == 1
    acquired = json_document(loads(artifacts[0].read_bytes()))
    original_session = parse_payload(Provider.CHATGPT, original, "original")[0]
    acquired_session = parse_payload(Provider.CHATGPT, acquired, "original")[0]
    assert acquired_session.messages == original_session.messages
    assert acquired_session.attachments == original_session.attachments
    assert acquired_session.provider_session_id == original_session.provider_session_id
    provenance = acquired["provenance"]
    assert isinstance(provenance, dict)
    metadata = provenance["provider_meta"]
    assert isinstance(metadata, dict)
    binding = metadata.pop("polylogue_source_cell")
    assert isinstance(binding, dict)
    assert binding["database"] == "capture-jobs/registry.sqlite3"
    assert binding["rowid"] == 1 and binding["queue_index"] == 0
    assert (
        binding["checkpoint_sha256"]
        == hashlib.sha256(
            sqlite3.connect(database).execute("SELECT CAST(checkpoint_json AS BLOB) FROM capture_jobs").fetchone()[0]
        ).hexdigest()
    )
    assert acquired == original
    assert parse_payload(Provider.CHATGPT, acquired, "original") == parse_payload(
        Provider.CHATGPT, original, "original"
    )
    retained = artifacts[0].read_bytes()
    await _prepare_owned_source_roots([WatchSource("browser-capture", root)])
    assert artifacts[0].read_bytes() == retained
    assert database.read_bytes() == before
    assert not (root / "capture-jobs/v2").exists()
    assert not list((root / ".staging").iterdir())


def test_identical_original_already_in_spool_is_not_rewritten(tmp_path: Path) -> None:
    from polylogue.browser_capture.source_checkpoint import extract_source_checkpoints

    original = _envelope()
    _source_cell(tmp_path, original)
    resident = write_capture_envelope(BrowserCaptureEnvelope.model_validate(original), spool_path=tmp_path)
    before = resident.path.read_bytes()
    result = extract_source_checkpoints(tmp_path)
    assert result.duplicates == 1 and result.published == 0
    assert resident.path.read_bytes() == before


@pytest.mark.parametrize("failure", ["digest", "provenance"])
def test_source_cell_refusal_preserves_original_and_closes_custody(tmp_path: Path, failure: str) -> None:
    from polylogue.browser_capture.source_checkpoint import SourceCheckpointError, extract_source_checkpoints
    from polylogue.storage.sqlite.connection_profile import retained_native_sql_owners_on_current_thread

    original = _envelope()
    if failure == "provenance":
        provenance = original["provenance"]
        assert isinstance(provenance, dict)
        metadata = provenance["provider_meta"]
        assert isinstance(metadata, dict)
        metadata["polylogue_source_cell"] = {"database": "different"}
    path = _source_cell(tmp_path, original, digest="sha256:" + "0" * 64 if failure == "digest" else None)
    before = path.read_bytes()
    owners = retained_native_sql_owners_on_current_thread()
    with pytest.raises(SourceCheckpointError, match="source_checkpoint_" + failure):
        extract_source_checkpoints(tmp_path)
    assert path.read_bytes() == before
    assert not list(tmp_path.rglob("*.json"))
    assert retained_native_sql_owners_on_current_thread() == owners


@pytest.mark.asyncio
async def test_original_cell_delivers_into_an_empty_archive_and_retries_after_restart(
    workspace_env: dict[str, Path], tmp_path: Path
) -> None:
    from polylogue.api import Polylogue
    from polylogue.config import Source, get_config
    from polylogue.daemon.cli import _prepare_owned_source_roots
    from polylogue.storage.blob_store import BlobStore
    from tests.infra.archive_scenarios import open_index_db
    from tests.infra.daemon_operations import daemon_serving_archive

    del workspace_env
    root = tmp_path / "browser-capture"
    original = _envelope()
    original_session = original["session"]
    assert isinstance(original_session, dict)
    original_session["attachments"] = [
        {
            "provider_attachment_id": "source-cell-asset",
            "name": "source-cell.txt",
            "mime_type": "text/plain",
            "attachment_kind": "file_service_file",
            "provider_meta": {"inline_base64": "c291cmNlIGFzc2V0"},
        }
    ]
    turns = original_session["turns"]
    assert isinstance(turns, list) and isinstance(turns[1], dict)
    turns[1]["attachments"] = original_session["attachments"]
    _source_cell(root, original)
    configuration = get_config()
    configuration.sources = [Source(name="browser-capture", path=root)]
    for _restart in range(2):
        await _prepare_owned_source_roots([WatchSource("browser-capture", root)])
        with daemon_serving_archive(configuration.archive_root, session_derivation=True):
            async with Polylogue(archive_root=configuration.archive_root, db_path=configuration.db_path) as api:
                await api.parse_sources(configuration.sources)
    expected = parse_payload(Provider.CHATGPT, original, "original")[0]
    with open_index_db(configuration.archive_root / "index.db") as connection:
        session = connection.execute("SELECT native_id FROM sessions").fetchall()
        messages = connection.execute("SELECT native_id FROM messages ORDER BY position").fetchall()
        blocks = connection.execute(
            "SELECT m.native_id, b.block_type, b.text FROM messages AS m "
            "JOIN blocks AS b ON b.message_id=m.message_id ORDER BY m.position,b.position"
        ).fetchall()
        attachment = connection.execute(
            "SELECT blob_hash, byte_count, acquisition_status FROM attachments WHERE display_name=?",
            ("source-cell.txt",),
        ).fetchone()
    assert [row[0] for row in session] == [expected.provider_session_id]
    assert [row[0] for row in messages] == [row.provider_message_id for row in expected.messages]
    # This DOM fixture has plain message text; normal lowering supplies its
    # TEXT blocks. Compare the actual material to the original input prose.
    assert [tuple(row) for row in blocks] == [
        (message.provider_message_id, "text", message.text) for message in expected.messages
    ]
    assert attachment is not None
    digest = hashlib.sha256(b"source asset").digest()
    assert bytes(attachment[0]) == digest and attachment[1:] == (len(b"source asset"), "acquired")
    assert BlobStore(configuration.archive_root / "blob").read_all(digest.hex()) == b"source asset"
    assert len(list((root / "chatgpt").glob("*.json"))) == 1


def test_cancelled_source_extraction_discards_stages_and_closes_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections.abc import Iterator

    from polylogue.browser_capture import source_checkpoint
    from polylogue.core.compute import DaemonOperationCancelled
    from polylogue.storage.sqlite.connection_profile import retained_native_sql_owners_on_current_thread

    path = _source_cell(tmp_path, _envelope())
    before = path.read_bytes()
    owners = retained_native_sql_owners_on_current_thread()

    def cancelled_chunks(_envelope: object) -> Iterator[bytes]:
        yield b"{"
        raise DaemonOperationCancelled("neutral cancelled source extraction")

    monkeypatch.setattr(source_checkpoint, "json_chunks", cancelled_chunks)
    with pytest.raises(DaemonOperationCancelled):
        source_checkpoint.extract_source_checkpoints(tmp_path)
    assert retained_native_sql_owners_on_current_thread() == owners
    assert path.read_bytes() == before
    assert not list((tmp_path / ".staging").iterdir())
    assert not list(tmp_path.rglob("*.json"))


def test_source_snapshot_is_coherent_and_next_acquisition_observes_changed_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.browser_capture import source_checkpoint
    from polylogue.browser_capture.capture_stream import CaptureSummary
    from polylogue.browser_capture.receiver import BrowserCaptureWriteResult, admit_staged_capture
    from polylogue.core.staged_body import StagedBody

    original = _envelope()
    path = _source_cell(tmp_path, original)
    second = _envelope()
    session = second["session"]
    assert isinstance(session, dict)
    session["provider_session_id"] = "different-conversation"
    payload = {"version": 1, "queue": [{"id": "second", "envelope": second}]}
    checkpoint = {"sequence": 1, "payload": payload, "digest": canonical_digest(payload)}
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("INSERT INTO capture_jobs VALUES (?, ?)", ("second", canonical_json(checkpoint)))
    changed = _envelope()
    changed_session = changed["session"]
    assert isinstance(changed_session, dict)
    changed_session["provider_session_id"] = "third-conversation"
    changed_payload = {"version": 1, "queue": [{"id": "third", "envelope": changed}]}
    changed_checkpoint = {"sequence": 2, "payload": changed_payload, "digest": canonical_digest(changed_payload)}
    admit = admit_staged_capture
    updated = False

    def update_second(staged: StagedBody, summary: CaptureSummary, *, spool_path: Path) -> BrowserCaptureWriteResult:
        nonlocal updated
        if not updated:
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute(
                    "UPDATE capture_jobs SET checkpoint_json=? WHERE job_id='second'",
                    (canonical_json(changed_checkpoint),),
                )
            updated = True
        return admit(staged, summary, spool_path=spool_path)

    monkeypatch.setattr(source_checkpoint, "admit_staged_capture", update_second)
    assert source_checkpoint.extract_source_checkpoints(tmp_path).published == 2
    assert len(list((tmp_path / "chatgpt").glob("*.json"))) == 2
    assert source_checkpoint.extract_source_checkpoints(tmp_path).published == 1
    assert len(list((tmp_path / "chatgpt").glob("*.json"))) == 3


def test_original_cell_conflicting_bytes_obey_resident_capture_authority(tmp_path: Path) -> None:
    from polylogue.browser_capture.source_checkpoint import extract_source_checkpoints

    original = _envelope()
    _source_cell(tmp_path, original)
    resident = _envelope()
    session = resident["session"]
    provenance = resident["provenance"]
    assert isinstance(session, dict) and isinstance(provenance, dict)
    turns = session["turns"]
    assert isinstance(turns, list) and isinstance(turns[0], dict)
    turns[0]["text"] = "newer retained bytes"
    provenance["captured_at"] = "2099-01-01T00:00:00Z"
    published = write_capture_envelope(BrowserCaptureEnvelope.model_validate(resident), spool_path=tmp_path)
    before = published.path.read_bytes()
    result = extract_source_checkpoints(tmp_path)
    assert result.superseded == 1 and result.duplicates == 0 and result.published == 0
    assert published.path.read_bytes() == before
