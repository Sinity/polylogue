"""Owned inactive generations may write only their derived index tier."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from io import BytesIO
from pathlib import Path

import pytest

import polylogue.sources.revision_backfill as revision_backfill_module
from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.daemon.bulk_rebuild import resolve_or_start_daemon_bulk_rebuild_transaction
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.sources.revision_backfill import (
    backfill_historical_revision_evidence,
    census_historical_revision_evidence,
    validate_frozen_source_authority,
)
from polylogue.storage.blob_store import PreparedBlob
from polylogue.storage.fts.drift_sampling import sample_fts_drift_to_ops_sync
from polylogue.storage.fts.fts_lifecycle import rebuild_fts_index_sync
from polylogue.storage.index_generation import IndexGenerationStore, source_revision_snapshot
from polylogue.storage.sqlite.archive_tiers.archive import (
    ArchiveStore,
    InactiveCandidateDurableWriteError,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.revision_governance import FrozenSourceRemediationRequiredError
from polylogue.storage.sqlite.durable_change_train import DurableChangeTrainError
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt
from tests.infra.revision_backfill_benchmark import build_independent_raw_corpus


def _file_evidence(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, hashlib.sha256(path.read_bytes()).hexdigest()


def _blob_evidence(root: Path) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (str(path.relative_to(root)), path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def _symlink_evidence(path: Path) -> tuple[int, int, str]:
    stat = path.lstat()
    target = (
        f"symlink:{path.readlink()}" if path.is_symlink() else f"file:{hashlib.sha256(path.read_bytes()).hexdigest()}"
    )
    return stat.st_dev, stat.st_ino, target


def _optional_path_evidence(path: Path) -> tuple[int, int, str] | None:
    if not path.exists() and not path.is_symlink():
        return None
    return _symlink_evidence(path)


def _assert_no_candidate_bookkeeping(root: Path) -> None:
    assert not (root / ".index-generations").exists()
    assert not (root / ".index-rebuild-transactions").exists()


def _chatgpt_bundle(*native_ids: str) -> bytes:
    sessions = []
    for native_id in native_ids:
        node_id = f"{native_id}-node"
        sessions.append(
            {
                "id": native_id,
                "conversation_id": native_id,
                "title": native_id,
                "create_time": 1_700_000_000,
                "update_time": 1_700_000_001,
                "current_node": node_id,
                "mapping": {
                    node_id: {
                        "id": node_id,
                        "parent": None,
                        "children": [],
                        "message": {
                            "id": f"{native_id}-message",
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": [native_id]},
                            "create_time": 1_700_000_000,
                        },
                    }
                },
            }
        )
    return json.dumps(sessions, sort_keys=True).encode()


def _prepare_frozen_source(root: Path, monkeypatch: pytest.MonkeyPatch, *, raw_count: int = 1) -> Path:
    build_independent_raw_corpus(root, raw_count=raw_count, avg_payload_bytes=1_000)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    census = census_historical_revision_evidence(root)
    assert census.scanned == raw_count
    assert census.classified_full == raw_count
    with sqlite3.connect(root / "source.db") as source:
        source.execute(
            """
            UPDATE raw_sessions
            SET revision_authority = 'byte_proven', baseline_raw_id = raw_id,
                predecessor_raw_id = NULL, acquisition_generation = 0
            """
        )
        source.commit()
    return write_valid_rebuild_receipt(root, root.parent / "schema-inference-receipt.json")


def test_resumed_candidate_validates_only_selected_raw_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    receipt_path = _prepare_frozen_source(root, monkeypatch, raw_count=2)
    original_validate = revision_backfill_module.validate_frozen_source_authority
    selections: list[tuple[str, ...] | None] = []

    def record_validation(
        archive_root: Path,
        *,
        selected_raw_ids: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        selections.append(None if selected_raw_ids is None else tuple(selected_raw_ids))
        original_validate(archive_root, selected_raw_ids=selected_raw_ids, **kwargs)

    monkeypatch.setattr(revision_backfill_module, "validate_frozen_source_authority", record_validation)
    first = rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            schema_inference_receipt_path=receipt_path,
            raw_batch_size=1,
            promote=False,
        )
    )

    assert first.status == "paused"
    assert first.transaction is not None
    assert selections == [None]

    selections.clear()
    second = rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            operation_id=str(first.transaction["operation_id"]),
            schema_inference_receipt_path=receipt_path,
            raw_batch_size=1,
            promote=False,
        )
    )

    assert second.status == "replayed"
    assert len(selections) == 1
    assert selections[0] is not None
    assert len(selections[0]) == 1


def test_frozen_source_admission_treats_non_session_census_as_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"[]",
            source_path="empty-conversations.json",
            acquired_at_ms=1,
        )
    census_historical_revision_evidence(root)

    with sqlite3.connect(root / "source.db") as source:
        assert source.execute(
            "SELECT status, member_count FROM raw_membership_census WHERE raw_id = ?",
            (raw_id,),
        ).fetchone() == ("non_session", 0)
        assert source.execute(
            "SELECT revision_authority FROM raw_sessions WHERE raw_id = ?",
            (raw_id,),
        ).fetchone() == ("quarantined",)

    validate_frozen_source_authority(root)


def test_candidate_rebuild_does_not_require_the_missing_active_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    receipt_path = _prepare_frozen_source(root, monkeypatch)
    (root / "index.db").unlink()

    result = rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            schema_inference_receipt_path=receipt_path,
            promote=False,
        )
    )

    assert result.status == "replayed"
    assert result.transaction is not None


def test_real_no_promote_candidate_preserves_frozen_durable_tiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    receipt_path = _prepare_frozen_source(root, monkeypatch)
    generation_store = IndexGenerationStore.for_archive_root(root)
    anchor = root / ".index-active-pointer"
    anchor_before = _optional_path_evidence(anchor)
    active_target_before = generation_store.active_pointer.resolve(strict=True)
    active_pointer_before = _symlink_evidence(generation_store.active_pointer)
    active_index_before = _file_evidence(active_target_before)
    source_before = _file_evidence(root / "source.db")
    user_before = _file_evidence(root / "user.db")
    with sqlite3.connect(root / "ops.db") as ops:
        ops.execute(
            """
            INSERT INTO convergence_debt (
                debt_id, stage, target_type, target_id, status, priority,
                attempts, last_error, created_at_ms, updated_at_ms
            ) VALUES ('candidate-guard-debt', 'insights', 'session_id',
                      'claude-ai-export:frozen-source', 'deferred', 0, 1,
                      'candidate must not resolve live debt', 1, 1)
            """
        )
        ops.commit()
    ops_before = _file_evidence(root / "ops.db")
    blobs_before = _blob_evidence(root / "blob")

    result = rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            schema_inference_receipt_path=receipt_path,
            promote=False,
        )
    )

    assert result.status == "replayed"
    assert result.transaction is not None
    assert result.transaction["status"] == "ready"
    generation = generation_store.load(str(result.transaction["generation_id"]))
    assert generation.state == "inactive"
    assert generation_store.active_pointer.resolve(strict=True) == active_target_before
    assert _optional_path_evidence(anchor) == anchor_before
    assert _symlink_evidence(generation_store.active_pointer) == active_pointer_before
    assert _file_evidence(active_target_before) == active_index_before
    assert _file_evidence(root / "source.db") == source_before
    assert _file_evidence(root / "user.db") == user_before
    assert _file_evidence(root / "ops.db") == ops_before
    assert _blob_evidence(root / "blob") == blobs_before
    with sqlite3.connect(root / "ops.db") as ops:
        assert ops.execute(
            "SELECT stage, target_id FROM convergence_debt WHERE debt_id = 'candidate-guard-debt'"
        ).fetchone() == ("insights", "claude-ai-export:frozen-source")
    with sqlite3.connect(f"file:{generation.index_path}?mode=ro", uri=True) as candidate:
        assert candidate.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_owned_candidate_refuses_source_user_and_blob_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    _prepare_frozen_source(root, monkeypatch)
    generation_store = IndexGenerationStore.for_archive_root(root)
    generation = generation_store.create(source_snapshot=source_revision_snapshot(root))
    generation_root = Path(generation.index_path).parent
    anchor = root / ".index-active-pointer"
    anchor.write_text(str(generation.index_path), encoding="utf-8")
    poisoned_anchor_before = _optional_path_evidence(anchor)
    source_before = _file_evidence(root / "source.db")
    user_before = _file_evidence(root / "user.db")
    ops_before = _file_evidence(root / "ops.db")
    blobs_before = _blob_evidence(root / "blob")

    with ArchiveStore.open_owned_inactive_generation(
        generation_root,
        generation_id=generation.generation_id,
        owner_id=generation.owner_id,
    ) as candidate:
        candidate._conn.execute("CREATE TABLE candidate_index_probe (value INTEGER) STRICT")
        rebuild_fts_index_sync(candidate._conn)
        assert sample_fts_drift_to_ops_sync(candidate._conn, archive_root=root) == 0
        candidate.commit()
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            candidate._ensure_source_conn().execute("UPDATE raw_sessions SET parse_error = 'candidate-write'")
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            candidate._conn.execute("CREATE TABLE user_tier.candidate_user_probe (value INTEGER) STRICT")
        assert candidate._blob_publisher is not None
        blob_publisher = candidate._blob_publisher
        existing_blob_hash = next(blob_publisher.iter_all())
        staged_path = tmp_path / "prepared-candidate-blob"
        staged_path.write_bytes(b"candidate-write")
        prepared = PreparedBlob(
            hash_hex=hashlib.sha256(b"candidate-write").hexdigest(),
            size_bytes=len(b"candidate-write"),
            temporary_path=staged_path,
        )
        source_path = tmp_path / "candidate-source"
        source_path.write_bytes(b"candidate-write")
        refusing_blob_calls = (
            lambda: blob_publisher.prepare_from_path(source_path),
            lambda: blob_publisher.prepare_from_fileobj(BytesIO(b"candidate-write")),
            lambda: blob_publisher.prepare_from_bytes(b"candidate-write"),
            lambda: blob_publisher.publish_prepared(prepared),
            lambda: blob_publisher.publish_many((prepared,)),
            lambda: blob_publisher.discard_prepared(prepared),
            lambda: blob_publisher.write_from_path(source_path),
            lambda: blob_publisher.write_from_fileobj(BytesIO(b"candidate-write")),
            lambda: blob_publisher.write_from_bytes(b"candidate-write"),
            lambda: blob_publisher.remove(existing_blob_hash),
            lambda: blob_publisher.cleanup_orphans({existing_blob_hash}, dry_run=False),
        )
        for refusing_call in refusing_blob_calls:
            with pytest.raises(InactiveCandidateDurableWriteError, match="may not publish"):
                refusing_call()
        assert blob_publisher.flush() == ()
        blob_publisher.discard_pending()
        assert staged_path.read_bytes() == b"candidate-write"
        assert not tuple((root / "blob").glob(".blob.*"))
        with pytest.raises(InactiveCandidateDurableWriteError, match="may not publish"):
            candidate.write_raw_payload(
                provider=Provider.CODEX,
                payload=b"candidate-write",
                source_path="candidate-write.jsonl",
                acquired_at_ms=1,
            )
        with pytest.raises(InactiveCandidateDurableWriteError, match="may not mutate user.db"):
            candidate.add_user_tags(("candidate:session",), ("candidate",))
        with pytest.raises(InactiveCandidateDurableWriteError, match="may not mutate user.db"):
            candidate.set_user_metadata(("candidate:session",), (("candidate", True),))
        with pytest.raises(InactiveCandidateDurableWriteError, match="may not mutate user.db"):
            candidate.post_blackboard_note("candidate write")

    assert _file_evidence(root / "source.db") == source_before
    assert _file_evidence(root / "user.db") == user_before
    assert _file_evidence(root / "ops.db") == ops_before
    assert _blob_evidence(root / "blob") == blobs_before
    assert _optional_path_evidence(anchor) == poisoned_anchor_before
    with sqlite3.connect(generation.index_path) as candidate_index:
        assert candidate_index.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'candidate_index_probe'"
        ).fetchone() == (1,)


@pytest.mark.parametrize("anchor_state", ["missing", "poisoned"])
def test_candidate_requires_current_parser_census_before_generation_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    anchor_state: str,
) -> None:
    root = tmp_path / "archive"
    build_independent_raw_corpus(root, raw_count=1, avg_payload_bytes=1_000)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    receipt_path = write_valid_rebuild_receipt(root, root.parent / "schema-inference-receipt.json")
    anchor = root / ".index-active-pointer"
    if anchor_state == "poisoned":
        anchor.write_text(
            str(root / ".index-generations" / "gen-poisoned" / "index.db"),
            encoding="utf-8",
        )
    anchor_before = _optional_path_evidence(anchor)

    with pytest.raises(FrozenSourceRemediationRequiredError, match="complete current-parser source census"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=receipt_path,
                promote=False,
            )
        )

    assert _optional_path_evidence(anchor) == anchor_before
    _assert_no_candidate_bookkeeping(root)


@pytest.mark.parametrize("anchor_state", ["missing", "poisoned"])
def test_daemon_candidate_requires_source_admission_before_transaction_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    anchor_state: str,
) -> None:
    root = tmp_path / "archive"
    build_independent_raw_corpus(root, raw_count=1, avg_payload_bytes=1_000)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    receipt_path = write_valid_rebuild_receipt(root, root.parent / "daemon-schema-inference-receipt.json")
    anchor = root / ".index-active-pointer"
    if anchor_state == "poisoned":
        anchor.write_text(
            str(root / ".index-generations" / "gen-poisoned" / "index.db"),
            encoding="utf-8",
        )
    anchor_before = _optional_path_evidence(anchor)

    with pytest.raises(FrozenSourceRemediationRequiredError, match="complete current-parser source census"):
        resolve_or_start_daemon_bulk_rebuild_transaction(
            root,
            schema_inference_receipt_path=receipt_path,
        )

    assert _optional_path_evidence(anchor) == anchor_before
    _assert_no_candidate_bookkeeping(root)


@pytest.mark.parametrize("route", ["offline", "daemon"])
@pytest.mark.parametrize("anchor_state", ["missing", "poisoned"])
def test_valid_candidate_admission_does_not_repair_active_pointer_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    anchor_state: str,
) -> None:
    root = tmp_path / "archive"
    receipt_path = _prepare_frozen_source(root, monkeypatch)
    anchor = root / ".index-active-pointer"
    if anchor.exists() or anchor.is_symlink():
        anchor.unlink()
    if anchor_state == "poisoned":
        anchor.write_text(
            str(root / ".index-generations" / "gen-poisoned" / "index.db"),
            encoding="utf-8",
        )
    anchor_before = _optional_path_evidence(anchor)

    if route == "offline":
        result = rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=receipt_path,
                promote=False,
            )
        )
        assert result.transaction is not None
        assert result.transaction["status"] == "ready"
    else:
        transaction = resolve_or_start_daemon_bulk_rebuild_transaction(
            root,
            schema_inference_receipt_path=receipt_path,
        )
        assert transaction.status == "running"

    assert _optional_path_evidence(anchor) == anchor_before


def test_candidate_rejects_poisoned_current_parser_logical_keys_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    _prepare_frozen_source(root, monkeypatch)
    with sqlite3.connect(root / "source.db") as source:
        source.execute(
            "UPDATE raw_sessions SET logical_source_key = ?",
            ("codex-session:poisoned-census-key",),
        )
        source.execute(
            "UPDATE raw_authority_parser_census SET logical_keys_json = ?",
            (json.dumps(["codex-session:poisoned-census-key"]),),
        )
        source.commit()
    receipt_path = write_valid_rebuild_receipt(root, root.parent / "poisoned-census-receipt.json")
    anchor = root / ".index-active-pointer"
    anchor_before = _optional_path_evidence(anchor)

    with pytest.raises(FrozenSourceRemediationRequiredError, match="re-derived different current-parser logical keys"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=receipt_path,
                promote=False,
            )
        )

    assert _optional_path_evidence(anchor) == anchor_before
    _assert_no_candidate_bookkeeping(root)


def test_candidate_rejects_extra_membership_binding_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    _prepare_frozen_source(root, monkeypatch)
    with sqlite3.connect(root / "source.db") as source:
        raw_id = str(source.execute("SELECT raw_id FROM raw_sessions").fetchone()[0])
        source.execute(
            """
            INSERT INTO raw_session_memberships (
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count, acquisition_generation,
                revision_authority, decision, decided_at_ms
            ) VALUES (?, 'codex-session:stale-extra', 'stale-extra', ?, ?, 0, 0,
                      'byte_proven', 'applied', 0)
            """,
            (raw_id, raw_id, b"\x00" * 32),
        )
        source.commit()
    receipt_path = write_valid_rebuild_receipt(root, root.parent / "extra-membership-receipt.json")

    with pytest.raises(FrozenSourceRemediationRequiredError, match="frozen durable authority bindings"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=receipt_path,
                promote=False,
            )
        )

    _assert_no_candidate_bookkeeping(root)


@pytest.mark.parametrize("link_shape", ["linked", "self-linked"])
def test_candidate_rejects_poisoned_typed_append_census_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    link_shape: str,
) -> None:
    root = tmp_path / "archive"
    _prepare_frozen_source(root, monkeypatch)
    with sqlite3.connect(root / "source.db") as source:
        baseline_raw_id, logical_key = source.execute("SELECT raw_id, logical_source_key FROM raw_sessions").fetchone()
    append_payload = b'{"type":"response_item","payload":{"type":"message","id":"append"}}\n'
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        append_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=append_payload,
            source_path="current/append.jsonl",
            source_index=-1,
            acquired_at_ms=2,
            revision=RawRevisionEnvelope(
                logical_source_key=str(logical_key),
                kind=RawRevisionKind.APPEND,
                source_revision="append-revision",
                predecessor_source_revision=str(baseline_raw_id),
                predecessor_raw_id=str(baseline_raw_id),
                baseline_raw_id=str(baseline_raw_id),
                acquisition_generation=1,
                append_start_offset=0,
                append_end_offset=len(append_payload),
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
    with sqlite3.connect(root / "source.db") as source:
        source.execute(
            "UPDATE raw_sessions SET logical_source_key = ? WHERE raw_id = ?",
            ("codex-session:poisoned-append-key", append_raw_id),
        )
        if link_shape == "self-linked":
            source.execute(
                "UPDATE raw_sessions SET predecessor_raw_id = raw_id, baseline_raw_id = raw_id WHERE raw_id = ?",
                (append_raw_id,),
            )
        source.execute(
            """
            INSERT INTO raw_authority_parser_census (
                raw_id, parser_fingerprint, status, logical_keys_json, detail, censused_at_ms
            )
            SELECT ?, parser_fingerprint, 'complete', ?, 'poisoned typed append census', 0
            FROM raw_authority_parser_census LIMIT 1
            """,
            (append_raw_id, json.dumps(["codex-session:poisoned-append-key"])),
        )
        source.commit()
    receipt_path = write_valid_rebuild_receipt(root, root.parent / "poisoned-append-receipt.json")

    with pytest.raises(FrozenSourceRemediationRequiredError, match="typed continuation identity"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=receipt_path,
                promote=False,
            )
        )

    _assert_no_candidate_bookkeeping(root)


def test_candidate_requires_complete_source_authority_before_generation_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    build_independent_raw_corpus(root, raw_count=1, avg_payload_bytes=1_000)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    census = census_historical_revision_evidence(root)
    assert census.scanned == 1
    receipt_path = write_valid_rebuild_receipt(root, root.parent / "schema-inference-receipt.json")

    with pytest.raises(FrozenSourceRemediationRequiredError, match="complete frozen source authority"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=receipt_path,
                promote=False,
            )
        )

    _assert_no_candidate_bookkeeping(root)


@pytest.mark.parametrize("drift", ["asserted", "stale-byte-proven"])
def test_candidate_rejects_authority_drift_in_frozen_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    root = tmp_path / "archive"
    _prepare_frozen_source(root, monkeypatch)
    with sqlite3.connect(root / "source.db") as source:
        if drift == "asserted":
            source.execute("UPDATE raw_sessions SET revision_authority = 'asserted', baseline_raw_id = NULL")
        else:
            source.execute("UPDATE raw_sessions SET acquisition_generation = 7")
        source.commit()
    receipt_path = write_valid_rebuild_receipt(root, root.parent / "post-drift-receipt.json")
    anchor = root / ".index-active-pointer"
    anchor_before = _optional_path_evidence(anchor)

    with pytest.raises(FrozenSourceRemediationRequiredError, match="re-derived different byte authority"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=receipt_path,
                promote=False,
            )
        )

    assert _optional_path_evidence(anchor) == anchor_before
    _assert_no_candidate_bookkeeping(root)


def test_candidate_rejects_membership_authority_drift_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_chatgpt_bundle("membership-a", "membership-b"),
            source_path="conversations.json",
            acquired_at_ms=1,
        )
    result = backfill_historical_revision_evidence(root)
    assert result.replayed_logical_sources == 2
    with sqlite3.connect(root / "source.db") as source:
        decisions = source.execute(
            """
            SELECT logical_source_key, decision
            FROM raw_session_memberships WHERE raw_id = ?
            ORDER BY logical_source_key
            """,
            (raw_id,),
        ).fetchall()
        assert decisions == [
            ("chatgpt:membership-a", "applied"),
            ("chatgpt:membership-b", "applied"),
        ]
        source.execute(
            """
            UPDATE raw_session_memberships SET decision = 'superseded_prefix'
            WHERE raw_id = ? AND logical_source_key = ?
            """,
            (raw_id, "chatgpt:membership-a"),
        )
        source.commit()
    receipt_path = write_valid_rebuild_receipt(root, root.parent / "membership-drift-receipt.json")
    anchor = root / ".index-active-pointer"
    anchor_before = _optional_path_evidence(anchor)

    with pytest.raises(FrozenSourceRemediationRequiredError, match="different membership authority"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=receipt_path,
                promote=False,
            )
        )

    assert _optional_path_evidence(anchor) == anchor_before
    _assert_no_candidate_bookkeeping(root)


def test_active_bootstrap_still_rejects_candidate_durable_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    _prepare_frozen_source(root, monkeypatch)
    generation_store = IndexGenerationStore.for_archive_root(root)
    generation = generation_store.create(source_snapshot=source_revision_snapshot(root))

    with pytest.raises(DurableChangeTrainError, match="unsafe file"):
        initialize_active_archive_root(Path(generation.index_path).parent)
