from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, ItemsView, Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.archive.ingest_flags import (
    COMPACT_BROWSER_CAPTURE_INGEST_FLAG,
    DOM_FALLBACK_INGEST_FLAG,
    NATIVE_BROWSER_CAPTURE_INGEST_FLAG,
)
from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.core.errors import SchemaSkew
from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind
from polylogue.pipeline.parsed_tree_size import estimate_parsed_tree_bytes
from polylogue.sources import revision_backfill
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.revision_backfill import (
    LEGACY_PAGE_IMAGE_CENSUS_DETAIL,
    _browser_snapshot_fidelity,
    _lineage_aware_replay_schedule,
    backfill_historical_revision_evidence,
    census_historical_revision_evidence,
    uncensused_historical_revision_raw_ids,
)
from polylogue.storage.artifacts.inspection import inspect_raw_artifact
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.index_generation import IndexGeneration, IndexGenerationStore
from polylogue.storage.raw_authority import iter_parser_census_logical_keys, raw_authority_parser_fingerprint
from polylogue.storage.sqlite import runtime_indexes, schema_bootstrap
from polylogue.storage.sqlite.archive_tiers import revision_governance as archive_revision_governance
from polylogue.storage.sqlite.archive_tiers import write as archive_tier_write
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write_shard import ShardRefusedError
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.connection_profile import StaleContinuationError
from polylogue.storage.sqlite.runtime_indexes import DEFERRED_SECONDARY_INDEX_NAMES
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.retained_parser_payloads import (
    _chatgpt_session,
)
from tests.infra.revision_backfill_benchmark import (
    REVISION_CHAIN_SHAPE,
    WHALE_BEARING_SHAPE,
    build_independent_raw_corpus,
    build_revision_chain_corpus,
    build_whale_bearing_corpus,
)


def _seed_historical_revision(archive: ArchiveStore, raw_id: str, revision: RawRevisionEnvelope) -> None:
    """Plant a retained legacy revision shape below live raw admission."""
    with archive._ensure_source_conn():
        archive._ensure_source_conn().execute(
            """
            UPDATE raw_sessions
            SET logical_source_key = ?, revision_kind = ?, source_revision = ?,
                predecessor_source_revision = ?, predecessor_raw_id = ?, baseline_raw_id = ?,
                append_start_offset = ?, append_end_offset = ?, acquisition_generation = ?, revision_authority = ?
            WHERE raw_id = ?
            """,
            (
                revision.logical_source_key,
                revision.kind.value,
                revision.source_revision,
                revision.predecessor_source_revision,
                revision.predecessor_raw_id,
                revision.baseline_raw_id,
                revision.append_start_offset,
                revision.append_end_offset,
                revision.acquisition_generation,
                revision.authority.value,
                raw_id,
            ),
        )


def test_revision_backfill_archive_readers_use_declared_tier_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Archive read helpers acquire cancellable frames with explicit tier identity."""
    root = tmp_path / "archive"
    bootstrap_archive_root(root)
    real_read_frame = cast(Any, revision_backfill).read_frame
    opened: list[tuple[Path, ArchiveTier | None, str]] = []

    @contextmanager
    def capture_read_frame(path: str | Path, **kwargs: Any) -> Iterator[Any]:
        with real_read_frame(path, **kwargs) as frame:
            opened.append((Path(path).resolve(), kwargs.get("tier"), str(kwargs.get("timeout_class"))))
            yield frame

    monkeypatch.setattr(revision_backfill, "read_frame", capture_read_frame)

    assert revision_backfill._expand_frozen_revision_link_selection(root, []) == ()
    assert revision_backfill._replay_representative_raw_ids([], root) == {}
    assert revision_backfill.uncensused_historical_revision_raw_ids(root, ["missing-raw"]) == ()

    frames = revision_backfill._RebindingEvidenceFrames(
        source_db_path=root / "source.db",
        index_db_path=root / "index.db",
    )
    try:
        assert set(frames.current(None)) == {ArchiveTier.SOURCE, ArchiveTier.INDEX}
    finally:
        frames.close()

    assert opened
    assert all(
        (path == (root / "source.db").resolve() and tier is ArchiveTier.SOURCE)
        or (path == (root / "index.db").resolve() and tier is ArchiveTier.INDEX)
        for path, tier, _timeout in opened
    )
    assert all(timeout == "background-read" for _path, _tier, timeout in opened)

    assert (root / "source.db").resolve() in {path for path, _tier, _timeout in opened}
    assert (root / "index.db").resolve() in {path for path, _tier, _timeout in opened}


def test_revision_backfill_profile_preserves_stale_source_tier_diagnostic(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    bootstrap_archive_root(root)
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute("PRAGMA user_version = 999")

    with pytest.raises(SchemaSkew, match="source schema skew"):
        revision_backfill._expand_frozen_revision_link_selection(root, [])


def test_current_parser_source_census_keeps_progress_after_elapsed_frame_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: Any
) -> None:
    """Elapsed read-frame time cannot discard a valid caller-owned Source snapshot."""
    root = tmp_path / "archive"
    bootstrap_archive_root(root)
    raw_ids: list[str] = []

    def write_terminal_non_session(archive: ArchiveStore, index: int) -> str:
        source_path = f"synthetic/census-{index}.jsonl"
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=f"not valid codex jsonl {index}".encode(),
            source_path=source_path,
            acquired_at_ms=index + 1,
        )
        archive.record_raw_failure_evidence(
            raw_id,
            provider=Provider.CODEX,
            source_path=source_path,
            source_index=0,
            acquired_at_ms=index + 1,
            kind=RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT,
        )
        archive.mark_raw_parse_failed(
            raw_id,
            provider=Provider.CODEX,
            error=ValueError("synthetic terminal corrupt source"),
            preserve_existing_failure_evidence=True,
        )
        with archive._ensure_source_conn():
            archive_revision_governance.record_current_parser_source_census(archive._ensure_source_conn(), raw_id)
        return raw_id

    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for index in range(2):
            raw_ids.append(write_terminal_non_session(archive, index))

    real_measurement = revision_backfill.parser_census_identity_measurement
    measured_raws = 0

    @contextmanager
    def advance_after_measurement(**kwargs: Any) -> Iterator[Any]:
        nonlocal measured_raws
        with real_measurement(**kwargs) as measured:
            measured_raws += 1
            frozen_clock.advance(301)
            yield measured

    monkeypatch.setattr(revision_backfill, "parser_census_identity_measurement", advance_after_measurement)
    assert revision_backfill.uncensused_historical_revision_raw_ids(root, raw_ids) == ()
    assert measured_raws == 2


def test_current_parser_source_census_refuses_reused_rowid_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deleted maximum rowid cannot admit a concurrent replacement.

    Anti-vacuity: replace the only raw row after frontier capture, confirm
    SQLite reuses its rowid, and require the census to refuse the mixed read.
    """
    root = tmp_path / "archive"
    bootstrap_archive_root(root)

    def write_terminal_non_session(archive: ArchiveStore, index: int) -> str:
        source_path = f"synthetic/reused-rowid-{index}.jsonl"
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=f"not valid codex jsonl {index}".encode(),
            source_path=source_path,
            acquired_at_ms=index + 1,
        )
        archive.record_raw_failure_evidence(
            raw_id,
            provider=Provider.CODEX,
            source_path=source_path,
            source_index=0,
            acquired_at_ms=index + 1,
            kind=RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT,
        )
        archive.mark_raw_parse_failed(
            raw_id,
            provider=Provider.CODEX,
            error=ValueError("synthetic terminal corrupt source"),
            preserve_existing_failure_evidence=True,
        )
        with archive._ensure_source_conn():
            archive_revision_governance.record_current_parser_source_census(archive._ensure_source_conn(), raw_id)
        return raw_id

    with ArchiveStore.open_existing(root, read_only=False) as archive:
        original_raw_id = write_terminal_non_session(archive, 0)
        original_rowid = int(
            archive._ensure_source_conn()
            .execute("SELECT rowid FROM raw_sessions WHERE raw_id = ?", (original_raw_id,))
            .fetchone()[0]
        )

    real_measurement = revision_backfill.parser_census_identity_measurement
    replaced = False
    replacement_raw_id: str | None = None

    @contextmanager
    def replace_after_observation(**kwargs: Any) -> Iterator[Any]:
        nonlocal replaced, replacement_raw_id
        with real_measurement(**kwargs) as measured:
            if not replaced:
                replaced = True
                with ArchiveStore.open_existing(root, read_only=False) as archive:
                    with archive._ensure_source_conn():
                        archive._ensure_source_conn().execute(
                            "DELETE FROM raw_sessions WHERE raw_id = ?", (original_raw_id,)
                        )
                    replacement_raw_id = write_terminal_non_session(archive, 1)
                    replacement_rowid = (
                        archive._ensure_source_conn()
                        .execute("SELECT rowid FROM raw_sessions WHERE raw_id = ?", (replacement_raw_id,))
                        .fetchone()[0]
                    )
                    assert int(replacement_rowid) == original_rowid
            yield measured

    monkeypatch.setattr(revision_backfill, "parser_census_identity_measurement", replace_after_observation)
    with pytest.raises(StaleContinuationError):
        revision_backfill.uncensused_historical_revision_raw_ids(root, [original_raw_id])
    assert replaced
    assert replacement_raw_id is not None and replacement_raw_id != original_raw_id


def _bundle(*sessions: dict[str, object]) -> bytes:
    return json.dumps(list(sessions), sort_keys=True).encode()


def test_owned_empty_generation_uses_cold_build_policy_and_finishes_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real replay route gets both cold-build flags and restores its reader shape.

    Anti-vacuity: the recording wrapper proves the production caller actually
    removed the deferred indexes before replay; the final-schema assertion
    proves the same route recreated them before it returned. Repeating the
    same session through the fresh writer is separately refused by
    ``test_fresh_build_refuses_duplicate_session_instead_of_replacing``.
    """
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("cold-generation", "hello", "world")),
            source_path="export/conversations.json",
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                logical_source_key="chatgpt-export:cold-generation",
                kind=RawRevisionKind.FULL,
                source_revision="cold-generation-v1",
                acquisition_generation=0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        # The frozen candidate validates the classifier's persisted
        # BYTE_PROVEN lineage, including a singleton full revision's own
        # baseline raw id.  Seed the fixture through that classifier instead
        # of asserting an incomplete authority row directly.
        archive.classify_raw_revision_cohort_for_rebuild_repair("chatgpt-export:cold-generation")
    census_historical_revision_evidence(tmp_path)

    generation = IndexGenerationStore.for_archive_root(tmp_path).create(source_snapshot="cold-build-test")
    generation_root = Path(generation.index_path).parent
    deferred_calls: list[tuple[str, ...]] = []
    stamped_tiers: list[str] = []
    original_defer = runtime_indexes.defer_secondary_indexes_sync
    original_stamp = schema_bootstrap.stamp_derived_schema_identity

    def record_defer(conn: sqlite3.Connection) -> tuple[str, ...]:
        dropped = original_defer(conn)
        deferred_calls.append(dropped)
        return dropped

    def record_stamp(conn: sqlite3.Connection, tier: str) -> None:
        stamped_tiers.append(tier)
        original_stamp(conn, tier)

    monkeypatch.setattr(runtime_indexes, "defer_secondary_indexes_sync", record_defer)
    monkeypatch.setattr(schema_bootstrap, "stamp_derived_schema_identity", record_stamp)
    backfill_historical_revision_evidence(
        generation_root,
        owned_inactive_generation=(generation.generation_id, generation.owner_id),
    )

    with sqlite3.connect(generation.index_path) as conn:
        index_names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
        assert set(DEFERRED_SECONDARY_INDEX_NAMES) <= index_names
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM action_pairs").fetchone()[0] == 0
    assert deferred_calls == [DEFERRED_SECONDARY_INDEX_NAMES]
    assert stamped_tiers == ["index"]


def test_retained_replay_terminal_fts_verifies_nonempty_membership(tmp_path: Path) -> None:
    """A settled retained replay leaves every replayed searchable block indexed.

    Anti-vacuity: this invokes only the retained replay route. Removing its
    FTS rebuild leaves ``messages_fts`` incomplete and fails the direct
    canonical membership assertion below.
    """
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("retained-readiness", "retained source", "retained reply")),
            source_path="export/conversations.json",
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                logical_source_key="chatgpt-export:retained-readiness",
                kind=RawRevisionKind.FULL,
                source_revision="retained-readiness-v1",
                acquisition_generation=0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        archive.classify_raw_revision_cohort_for_rebuild_repair("chatgpt-export:retained-readiness")
        archive.commit()

    census_historical_revision_evidence(tmp_path)
    result = backfill_historical_revision_evidence(tmp_path)
    assert result.replayed_logical_sources == 1

    with sqlite3.connect(tmp_path / "index.db") as conn:
        from polylogue.storage.fts.fts_lifecycle import fts_invariant_snapshot_sync

        messages = fts_invariant_snapshot_sync(conn).messages

    assert messages.ready
    assert messages.source_rows == messages.indexed_rows > 0
    assert (messages.missing_rows, messages.excess_rows, messages.duplicate_rows, messages.identity_mismatch_rows) == (
        0,
        0,
        0,
        0,
    )


@pytest.mark.parametrize("deferred_indexes", [False, True])
def test_owned_nonempty_generation_refuses_cold_build_deferral(tmp_path: Path, deferred_indexes: bool) -> None:
    """A resumed candidate is never silently treated as a fresh writer target."""
    bootstrap_archive_root(tmp_path)
    generation = IndexGenerationStore.for_archive_root(tmp_path).create(source_snapshot="nonempty-cold-build-test")
    generation_root = Path(generation.index_path).parent
    with ArchiveStore.open_owned_inactive_generation(
        generation_root,
        generation_id=generation.generation_id,
        owner_id=generation.owner_id,
        defer_secondary_indexes=deferred_indexes,
    ) as archive:
        archive._conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES ('present', 'codex-session', zeroblob(32))"
        )
        archive.commit()

    with pytest.raises(ValueError, match="empty archive generation"):
        backfill_historical_revision_evidence(
            generation_root,
            owned_inactive_generation=(generation.generation_id, generation.owner_id),
        )


@pytest.mark.parametrize("padded_native_id", [False, True])
def test_frozen_inactive_generation_replays_through_sealed_session_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, padded_native_id: bool
) -> None:
    """The frozen candidate copies the live shard transport, never source rows.

    Anti-vacuity: replacing the writer's shard copy with inline row binding
    leaves the logical projection green but makes ``copies`` zero.

    The completed finished-build measurement owns the one chosen transport
    profile: four thread workers.  This law verifies that profile's real
    sealed handoff without turning transport safety into a 1/4/12 grid.
    """
    import polylogue.storage.sqlite.archive_tiers.write as archive_tier_write

    ingest_workers = 4
    root = tmp_path / "shard-thread-4"
    bootstrap_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session(f"sealed-shard-{ingest_workers}", "hello", "world")),
            source_path="export/conversations.json",
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                logical_source_key=f"chatgpt-export:sealed-shard-{ingest_workers}",
                kind=RawRevisionKind.FULL,
                source_revision=f"sealed-shard-{ingest_workers}-v1",
                acquisition_generation=0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        archive.classify_raw_revision_cohort_for_rebuild_repair(f"chatgpt-export:sealed-shard-{ingest_workers}")
    census_historical_revision_evidence(root, ingest_workers=ingest_workers)
    source_before = (root / "source.db").read_bytes()
    generation = IndexGenerationStore.for_archive_root(root).create(source_snapshot="sealed-shard-test")
    copies = 0
    original_copy = archive_tier_write.copy_shard_session_rows

    def counting_copy(*args: object, **kwargs: object) -> object:
        nonlocal copies
        copies += 1
        return original_copy(*args, **kwargs)  # type: ignore[arg-type]

    binding_calls: list[str] = []
    if padded_native_id:
        original_binding = revision_backfill._required_shard_prepared_rows

        def bind_padded_native_id(raw_id: str, session: ParsedSession, bindings: Any) -> Any:
            # The shard already holds the canonical identity. Exercise the
            # real replay writer with an equivalent provider spelling.
            binding_calls.append(raw_id)
            return original_binding(
                raw_id,
                session.model_copy(update={"provider_session_id": f" {session.provider_session_id} "}),
                bindings,
            )

        monkeypatch.setattr(revision_backfill, "_required_shard_prepared_rows", bind_padded_native_id)
    monkeypatch.setattr(archive_tier_write, "copy_shard_session_rows", counting_copy)
    result = backfill_historical_revision_evidence(
        Path(generation.index_path).parent,
        owned_inactive_generation=(generation.generation_id, generation.owner_id),
        ingest_workers=ingest_workers,
        use_session_shards=True,
    )

    assert result.replayed_logical_sources == 1
    assert copies == 1
    if padded_native_id:
        assert binding_calls
    assert (root / "source.db").read_bytes() == source_before
    with sqlite3.connect(generation.index_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] > 0
    assert not list(Path(generation.index_path).parent.glob(".frozen-replay-shards-*"))


def test_frozen_inactive_generation_refuses_corrupt_required_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt sealed-route handoff cannot fall back to an inline index write."""
    root = tmp_path / "corrupt-shard"
    bootstrap_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("corrupt-shard", "hello", "world")),
            source_path="export/conversations.json",
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                logical_source_key="chatgpt-export:corrupt-shard",
                kind=RawRevisionKind.FULL,
                source_revision="corrupt-shard-v1",
                acquisition_generation=0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        archive.classify_raw_revision_cohort_for_rebuild_repair("chatgpt-export:corrupt-shard")
    census_historical_revision_evidence(root)
    source_before = (root / "source.db").read_bytes()
    generation = IndexGenerationStore.for_archive_root(root).create(source_snapshot="corrupt-shard-test")
    original_add_raw = revision_backfill._FrozenReplayShardTransport.add_raw

    def corrupt_after_seal(
        transport: revision_backfill._FrozenReplayShardTransport,
        raw_id: str,
        sessions: object,
        *,
        prepared_artifact: object = None,
    ) -> None:
        original_add_raw(transport, raw_id, sessions, prepared_artifact=prepared_artifact)  # type: ignore[arg-type]
        transport.path_for_raw(raw_id).write_bytes(b"not a sqlite shard")

    monkeypatch.setattr(revision_backfill._FrozenReplayShardTransport, "add_raw", corrupt_after_seal)
    with pytest.raises(ShardRefusedError, match="required session shard refused"):
        backfill_historical_revision_evidence(
            Path(generation.index_path).parent,
            owned_inactive_generation=(generation.generation_id, generation.owner_id),
            use_session_shards=True,
        )

    assert (root / "source.db").read_bytes() == source_before
    with sqlite3.connect(generation.index_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_active_prepared_replay_requires_captured_authority_plan(tmp_path: Path) -> None:
    """A sealed active replay cannot classify large raw bytes under the writer."""
    with pytest.raises(ValueError, match="captured source authority plan"):
        backfill_historical_revision_evidence(
            tmp_path,
            selected_raw_ids=[],
            prepared_inputs={},
            use_session_shards=True,
        )


def test_browser_snapshot_fidelity_derives_from_parser_ingest_flags() -> None:
    """``MembershipRevision.browser_snapshot_fidelity`` must reflect the parser's
    own ingest flags -- until this was wired up, every ``MembershipRevision``
    built during census carried ``browser_snapshot_fidelity=None``
    regardless of content, making ``classify_membership_revisions``'s entire
    dom/native/direct-export precedence dead code in production
    (polylogue-z1c6)."""
    assert _browser_snapshot_fidelity([]) is None
    assert _browser_snapshot_fidelity(["capture:temporary-chat"]) is None
    assert _browser_snapshot_fidelity([DOM_FALLBACK_INGEST_FLAG]) == "dom"
    assert _browser_snapshot_fidelity([NATIVE_BROWSER_CAPTURE_INGEST_FLAG]) == "native"
    assert _browser_snapshot_fidelity([COMPACT_BROWSER_CAPTURE_INGEST_FLAG]) == "native"
    # Native takes precedence if a parser somehow reports both.
    assert _browser_snapshot_fidelity([DOM_FALLBACK_INGEST_FLAG, NATIVE_BROWSER_CAPTURE_INGEST_FLAG]) == "native"


def test_current_parser_receipt_reselection_repairs_legacy_empty_membership_keys(tmp_path: Path) -> None:
    """Current receipts with legacy empty keys are re-censused when authority has a key."""
    bootstrap_archive_root(tmp_path)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        legacy_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"legacy-empty-receipt",
            source_path="legacy-empty.jsonl",
            acquired_at_ms=1,
        )
        _seed_historical_revision(
            archive,
            legacy_raw_id,
            RawRevisionEnvelope("codex-session:legacy-membership", RawRevisionKind.FULL, "legacy-v1", 0),
        )
        canonical_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"canonical-receipt",
            source_path="canonical.jsonl",
            acquired_at_ms=2,
        )
        _seed_historical_revision(
            archive,
            canonical_raw_id,
            RawRevisionEnvelope("codex-session:canonical-membership", RawRevisionKind.FULL, "canonical-v1", 0),
        )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        for raw_id, logical_key, receipt_keys in (
            (legacy_raw_id, "codex-session:legacy-membership", "[]"),
            (
                canonical_raw_id,
                "codex-session:canonical-membership",
                json.dumps(["codex-session:canonical-membership"]),
            ),
        ):
            conn.execute(
                """
                INSERT INTO raw_session_memberships (
                    raw_id, logical_source_key, provider_session_id, source_revision,
                    normalized_content_hash, message_count
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (raw_id, logical_key, logical_key.rsplit(":", 1)[1], "revision-1", bytes(32), 1),
            )
            conn.execute(
                """
                INSERT INTO raw_authority_parser_census (
                    raw_id, parser_fingerprint, status, logical_keys_json, detail
                ) VALUES (?, ?, 'complete', ?, 'parser-observed: legacy receipt shape')
                """,
                (raw_id, raw_authority_parser_fingerprint(), receipt_keys),
            )
        conn.commit()

    assert uncensused_historical_revision_raw_ids(tmp_path, [legacy_raw_id, canonical_raw_id]) == (legacy_raw_id,)


def test_previous_dynamic_parser_fingerprint_requires_reobservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parser receipt gate compares with current executable semantics, without a revision allowlist."""
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"previous-parser-semantics",
            source_path="previous-parser-semantics.jsonl",
            acquired_at_ms=1,
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_authority_parser_census (
                raw_id, parser_fingerprint, status, logical_keys_json, detail
            ) VALUES (?, ?, 'complete', '[]', 'parser-observed: previous semantic closure')
            """,
            (raw_id, raw_authority_parser_fingerprint()),
        )
        conn.commit()

    previous_fingerprint = raw_authority_parser_fingerprint()
    changed_fingerprint = previous_fingerprint[:-1] + ("0" if previous_fingerprint[-1] != "0" else "1")
    monkeypatch.setattr(revision_backfill, "raw_authority_parser_fingerprint", lambda: changed_fingerprint)

    assert uncensused_historical_revision_raw_ids(tmp_path, [raw_id]) == (raw_id,)


def test_antigravity_trajectory_page_image_is_terminal_during_frozen_backfill(tmp_path: Path) -> None:
    """A retained Antigravity trajectory page image is not replay authority.

    The SQLite bytes deliberately carry a valid trajectory schema and message,
    so removing the Antigravity page-image refusal parses a session and fails
    the zero-membership assertion.
    """
    bootstrap_archive_root(tmp_path)
    trajectory_path = tmp_path / "antigravity" / "conversations" / "page-image.sqlite"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(trajectory_path) as conn:
        conn.executescript(
            """
            CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);
            CREATE TABLE steps (
                idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT,
                status TEXT, error_details TEXT
            );
            CREATE TABLE conversation_summaries (cascade_id TEXT, title TEXT, last_modified_time TEXT);
            CREATE TABLE parent_references (cascade_id TEXT, parent_id TEXT);
            INSERT INTO trajectory_meta VALUES ('page-image-trajectory', 'page-image-cascade');
            INSERT INTO conversation_summaries VALUES ('page-image-cascade', 'Page image', NULL);
            INSERT INTO steps VALUES (0, 'message', 'v1', '{"role":"user","text":"must not replay"}', NULL, NULL);
            """
        )
    page_image = trajectory_path.read_bytes()

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        page_image_raw_id = archive.write_raw_payload(
            provider=Provider.ANTIGRAVITY,
            payload=page_image,
            source_path=str(trajectory_path),
            acquired_at_ms=1,
        )
        valid_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=(
                b'{"type":"session_meta","payload":{"id":"after-page-image"}}\n'
                b'{"type":"response_item","payload":{"type":"message","role":"user",'
                b'"content":[{"type":"input_text","text":"still processable"}]}}\n'
            ),
            source_path="after-page-image.jsonl",
            acquired_at_ms=2,
        )

    result = backfill_historical_revision_evidence(tmp_path)

    assert result.scanned == 2
    assert result.replayed_logical_sources == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        membership = conn.execute(
            "SELECT status, member_count, detail FROM raw_membership_census WHERE raw_id = ?",
            (page_image_raw_id,),
        ).fetchone()
        parser = conn.execute(
            "SELECT status, logical_keys_json, detail FROM raw_authority_parser_census WHERE raw_id = ?",
            (page_image_raw_id,),
        ).fetchone()
        later_parser = conn.execute(
            "SELECT status, logical_keys_json FROM raw_authority_parser_census WHERE raw_id = ?",
            (valid_raw_id,),
        ).fetchone()
    assert membership is not None
    assert membership[0:2] == ("non_session", 0)
    assert LEGACY_PAGE_IMAGE_CENSUS_DETAIL in str(membership[2])
    assert parser is not None
    assert parser[0] == "complete"
    assert tuple(iter_parser_census_logical_keys(parser[1])) == ()
    assert str(parser[2]).startswith("parser-observed:")
    assert later_parser is not None
    assert later_parser[0] == "complete"
    assert tuple(iter_parser_census_logical_keys(later_parser[1])) == ("codex-session:after-page-image",)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        sessions = conn.execute("SELECT origin, native_id FROM sessions ORDER BY origin, native_id").fetchall()
    assert sessions == [("codex-session", "after-page-image")]


def test_backfill_terminalizes_source_only_declared_artifact(tmp_path: Path) -> None:
    """Replay turns a decoded fact-sidecar raw into terminal source authority.

    This exercises the same retained-raw replay path as recovery: the
    source-only raw starts pending, the parser confirms it is a workflow
    artifact, and the source tier must retain both typed artifact evidence and
    a successful parse receipt so it is not selected forever.
    """
    bootstrap_archive_root(tmp_path)
    source_path = str(tmp_path / ".claude" / "projects" / "proj" / "subagents" / "workflows" / "wf" / "journal.jsonl")
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=b'{"contentKey":"workflow-artifact","agentId":"agent"}\n',
            source_path=source_path,
            acquired_at_ms=1,
        )

    backfill_historical_revision_evidence(tmp_path)

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT parsed_at_ms IS NOT NULL FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (1,)
        assert conn.execute("SELECT parse_as_session FROM raw_artifacts WHERE raw_id = ?", (raw_id,)).fetchone() == (0,)
        assert conn.execute("SELECT status FROM raw_membership_census WHERE raw_id = ?", (raw_id,)).fetchone() == (
            "non_session",
        )
        assert conn.execute(
            "SELECT status, logical_keys_json FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
        ).fetchone() == ("complete", "[]")


@pytest.mark.asyncio
async def test_backfill_terminalizes_detected_unknown_empty_artifact(tmp_path: Path) -> None:
    """Detected provider evidence must survive an empty retained replay."""
    bootstrap_archive_root(tmp_path)
    source_path = str(tmp_path / ".claude" / "projects" / "proj" / "subagents" / "workflows" / "wf" / "journal.jsonl")
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.UNKNOWN,
            payload=(
                b'{"type":"file-history-snapshot","messageId":"history-message",'
                b'"sessionId":"history-only-session","snapshot":{"trackedFileBackups":{}}}\n'
            ),
            source_path=source_path,
            acquired_at_ms=1,
        )

    backfill_historical_revision_evidence(tmp_path)

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT origin, detected_provider, parsed_at_ms IS NOT NULL FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (
            "unknown-export",
            "claude-code",
            1,
        )
        assert conn.execute("SELECT parse_as_session FROM raw_artifacts WHERE raw_id = ?", (raw_id,)).fetchone() == (0,)
        assert conn.execute(
            "SELECT status, logical_keys_json FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
        ).fetchone() == ("complete", "[]")

        terminal_artifact_id = str(
            conn.execute("SELECT artifact_id FROM raw_artifacts WHERE raw_id = ?", (raw_id,)).fetchone()[0]
        )

    backend = SQLiteBackend(db_path=tmp_path / "index.db")
    try:
        record = await backend.get_raw_session(raw_id)
        assert record is not None
        refreshed = inspect_raw_artifact(record, blob_store=BlobStore(tmp_path / "blob"))
        assert refreshed.observation_id == terminal_artifact_id
        assert await backend.save_artifact_observation(refreshed) is False
    finally:
        await backend.close()

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            """
            SELECT COUNT(*) FROM raw_artifacts
            WHERE origin = 'claude-code-session' AND source_path = ? AND source_index = 0
            """,
            (source_path,),
        ).fetchone() == (1,)


def test_terminal_artifact_receipts_roll_back_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed terminal census cannot expose only its artifact carrier."""
    bootstrap_archive_root(tmp_path)
    source_path = str(tmp_path / ".claude" / "projects" / "proj" / "subagents" / "workflows" / "wf" / "journal.jsonl")
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=b'{"contentKey":"workflow-artifact","agentId":"agent"}\n',
            source_path=source_path,
            acquired_at_ms=1,
        )

    def fail_census(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected terminal census failure")

    monkeypatch.setattr(ArchiveStore, "replace_raw_membership_census", fail_census)
    with pytest.raises(RuntimeError, match="injected terminal census failure"):
        backfill_historical_revision_evidence(tmp_path)

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts WHERE raw_id = ?", (raw_id,)).fetchone() == (0,)
        assert conn.execute("SELECT parsed_at_ms FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (None,)
        assert conn.execute(
            "SELECT COUNT(*) FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (0,)


def test_backfill_preserves_latest_terminal_artifact_observation(tmp_path: Path) -> None:
    """A delayed older replay cannot replace a newer coordinate carrier."""
    bootstrap_archive_root(tmp_path)
    source_path = str(tmp_path / ".claude" / "projects" / "proj" / "subagents" / "workflows" / "wf" / "journal.jsonl")
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        older_raw_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=b'{"contentKey":"workflow-artifact","agentId":"old"}\n',
            source_path=source_path,
            acquired_at_ms=1,
            raw_id="z-older-artifact",
        )
        newer_raw_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=b'{"contentKey":"workflow-artifact","agentId":"new"}\n',
            source_path=source_path,
            acquired_at_ms=2,
            raw_id="a-newer-artifact",
        )

    backfill_historical_revision_evidence(tmp_path)

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (1,)
        assert conn.execute("SELECT raw_id, last_observed_at_ms FROM raw_artifacts").fetchone() == (newer_raw_id, 2)
        assert older_raw_id > newer_raw_id
        assert conn.execute(
            "SELECT COUNT(*) FROM raw_sessions WHERE raw_id IN (?, ?) AND parsed_at_ms IS NOT NULL",
            (older_raw_id, newer_raw_id),
        ).fetchone() == (2,)

    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        assert all(row[2] for row in archive.raw_membership_census_rows([older_raw_id, newer_raw_id]))


def test_backfill_uses_raw_observation_order_for_equal_time_artifacts(tmp_path: Path) -> None:
    """Legacy receipt-free observations use raw insertion order, not raw-id order."""
    bootstrap_archive_root(tmp_path)
    source_path = str(tmp_path / ".claude" / "projects" / "proj" / "subagents" / "workflows" / "wf" / "journal.jsonl")
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        older_raw_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=b'{"contentKey":"workflow-artifact","agentId":"old"}\n',
            source_path=source_path,
            acquired_at_ms=1,
            raw_id="a-older-artifact",
        )
        newer_raw_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=b'{"contentKey":"workflow-artifact","agentId":"new"}\n',
            source_path=source_path,
            acquired_at_ms=1,
            raw_id="z-newer-artifact",
        )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        # Legacy rows can lack receipts entirely. The fallback must compare
        # both observations through raw_sessions, never one rowid per table.
        conn.execute("DELETE FROM blob_refs WHERE ref_id IN (?, ?)", (older_raw_id, newer_raw_id))
        conn.commit()

    backfill_historical_revision_evidence(tmp_path)

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (1,)
        assert conn.execute("SELECT raw_id, last_observed_at_ms FROM raw_artifacts").fetchone() == (newer_raw_id, 1)
        assert older_raw_id < newer_raw_id


def test_backfill_preserves_latest_repeated_artifact_observation(tmp_path: Path) -> None:
    """A -> B -> A reacquisition restores A as the coordinate authority."""
    bootstrap_archive_root(tmp_path)
    source_path = str(tmp_path / ".claude" / "projects" / "proj" / "subagents" / "workflows" / "wf" / "journal.jsonl")
    payload_a = b'{"contentKey":"workflow-artifact","agentId":"a"}\n'
    payload_b = b'{"contentKey":"workflow-artifact","agentId":"b"}\n'
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_a = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=payload_a,
            source_path=source_path,
            acquired_at_ms=1,
        )
        raw_b = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=payload_b,
            source_path=source_path,
            acquired_at_ms=2,
        )
        assert (
            archive.write_raw_payload(
                provider=Provider.CLAUDE_CODE,
                payload=payload_a,
                source_path=source_path,
                acquired_at_ms=3,
            )
            == raw_a
        )

    backfill_historical_revision_evidence(tmp_path)

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (1,)
        assert conn.execute("SELECT raw_id, last_observed_at_ms FROM raw_artifacts").fetchone() == (raw_a, 3)
        assert raw_a != raw_b


def test_historical_backfill_selects_prefix_newest_independent_of_acquisition_order(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    baseline = (
        b'{"type":"session_meta","payload":{"id":"session-1","timestamp":"2026-06-01T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":'
        b'[{"type":"input_text","text":"old"}]}}\n'
    )
    newest = baseline + (
        b'{"type":"response_item","payload":{"type":"message","role":"assistant","content":'
        b'[{"type":"output_text","text":"new"}]}}\n'
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        newest_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=newest,
            source_path="session.jsonl",
            acquired_at_ms=1,
        )
        baseline_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline,
            source_path="session.jsonl",
            acquired_at_ms=2,
        )
        legacy_append_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"response_item","payload":{"type":"message","id":"legacy-suffix"}}\n',
            source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=3,
        )

    result = backfill_historical_revision_evidence(tmp_path)

    assert result.scanned == 3
    assert result.classified_full == 2
    assert result.replayed_logical_sources == 1
    assert result.quarantined == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        parser_census = conn.execute(
            """
            SELECT status, COUNT(*)
            FROM raw_authority_parser_census
            WHERE parser_fingerprint = ?
            GROUP BY status ORDER BY status
            """,
            (raw_authority_parser_fingerprint(),),
        ).fetchall()
    # polylogue-39kcs: all three raws are census-complete, including the
    # legacy append fragment. Its receipt records the durable identity set
    # available to byte revision governance (empty here because this fixture
    # has no membership binding) -- a ``failed`` receipt there matched neither branch of
    # ``uncensused_historical_revision_raw_ids``'s gate, so the fragment was
    # re-selected for census forever and raw-replay planning never started.
    # The fragment is still ``quarantined`` (asserted above); only the
    # census receipt changed, not its authority.
    assert parser_census == [("complete", 3)]
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT message_count, raw_id FROM sessions").fetchone() == (2, newest_raw_id)

    with sqlite3.connect(tmp_path / "index.db") as conn:
        row = conn.execute("SELECT rowid FROM blocks ORDER BY rowid LIMIT 1").fetchone()
        assert row is not None
        conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (row[0],))
        conn.execute(
            "INSERT INTO messages_fts(rowid, text) VALUES (?, 'stale-only-token')",
            (row[0],),
        )
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'stale' ").fetchone()[0] == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET parsed_at_ms = NULL WHERE logical_source_key IS NOT NULL")
        conn.commit()

    backfill_historical_revision_evidence(tmp_path)

    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'stale'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'old'").fetchone()[0] == 1
        assert set(conn.execute("SELECT raw_id, decision FROM raw_revision_applications")) == {
            (baseline_raw_id, "superseded"),
            (newest_raw_id, "selected_baseline"),
        }
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE parsed_at_ms IS NOT NULL").fetchone()[0] == 2
        assert conn.execute(
            "SELECT revision_kind, revision_authority, parsed_at_ms FROM raw_sessions WHERE raw_id = ?",
            (legacy_append_raw_id,),
        ).fetchone() == ("unknown", "quarantined", None)

    parsed_baseline = parse_payload(
        Provider.CODEX,
        list(_iter_json_stream(BytesIO(baseline), "session.jsonl")),
        "session",
        source_path="session.jsonl",
    )[0]
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_parsed_for_retained_raw(
            parsed_baseline,
            raw_id=baseline_raw_id,
            source_path="session.jsonl",
            acquired_at_ms=3,
        )
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT message_count, raw_id FROM sessions").fetchone() == (2, newest_raw_id)


def test_incremental_target_expands_new_logical_key_across_source_paths(tmp_path: Path) -> None:
    """A newly parsed path must not split an already-known byte cohort."""
    bootstrap_archive_root(tmp_path)
    baseline = (
        b'{"type":"session_meta","payload":{"id":"shared","timestamp":"2026-07-15T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":'
        b'[{"type":"input_text","text":"old"}]}}\n'
    )
    newest = baseline + (
        b'{"type":"response_item","payload":{"type":"message","role":"assistant","content":'
        b'[{"type":"output_text","text":"new"}]}}\n'
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        old_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline,
            source_path="first/shared.jsonl",
            acquired_at_ms=1,
        )
    assert backfill_historical_revision_evidence(tmp_path, selected_raw_ids=[old_raw_id]).replayed_logical_sources == 1

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        new_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=newest,
            source_path="moved/shared.jsonl",
            acquired_at_ms=2,
        )

    result = backfill_historical_revision_evidence(tmp_path, selected_raw_ids=[new_raw_id])

    assert result.scanned == 2
    assert result.replayed_logical_sources == 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id, message_count, raw_id FROM sessions").fetchall() == [
            ("shared", 2, new_raw_id)
        ]


def test_backfill_resumes_after_index_receipt_commits_before_source_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap_archive_root(tmp_path)
    payload = (
        b'{"type":"session_meta","payload":{"id":"session-1","timestamp":"2026-06-01T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"one","role":"user","content":'
        b'[{"type":"input_text","text":"one"}]}}\n'
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path="session.jsonl",
            acquired_at_ms=1,
        )

    # polylogue-1r9c: mark_raw_parse_succeeded's real implementation moved to
    # revision_governance.py, and apply_raw_revision_replay (also in that
    # module) calls it as a direct module-internal function reference, not
    # through `self.` dynamic dispatch -- so the spy must patch the
    # revision_governance module attribute, not the ArchiveStore delegator
    # method (which only intercepts *external* callers).
    original_mark = archive_revision_governance.mark_raw_parse_succeeded

    def crash_after_index_commit(
        store: archive_revision_governance.RawRevisionGovernanceHost, raw_id: str, *, provider: Provider
    ) -> None:
        raise RuntimeError("crash after index receipt")

    monkeypatch.setattr(archive_revision_governance, "mark_raw_parse_succeeded", crash_after_index_commit)
    with pytest.raises(RuntimeError, match="crash after index receipt"):
        backfill_historical_revision_evidence(tmp_path)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone()[0] == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT parsed_at_ms FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (None,)

    monkeypatch.setattr(archive_revision_governance, "mark_raw_parse_succeeded", original_mark)
    resumed = backfill_historical_revision_evidence(tmp_path)
    assert resumed.replayed_logical_sources == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT parsed_at_ms IS NOT NULL FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (1,)


def test_backfill_resumes_after_only_some_source_markers_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap_archive_root(tmp_path)
    baseline = (
        b'{"type":"session_meta","payload":{"id":"session-1","timestamp":"2026-06-01T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"one","role":"user","content":'
        b'[{"type":"input_text","text":"one"}]}}\n'
    )
    newest = baseline + (
        b'{"type":"response_item","payload":{"type":"message","id":"two","role":"assistant","content":'
        b'[{"type":"output_text","text":"two"}]}}\n'
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_ids = {
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path="session.jsonl",
                acquired_at_ms=index,
            )
            for index, payload in enumerate((baseline, newest), start=1)
        }

    # polylogue-1r9c: see the sibling test above -- patch the
    # revision_governance module attribute, the actual internal call target.
    original_mark = archive_revision_governance.mark_raw_parse_succeeded
    calls = 0

    def crash_after_one_marker(
        store: archive_revision_governance.RawRevisionGovernanceHost, raw_id: str, *, provider: Provider
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            original_mark(store, raw_id, provider=provider)
            return
        raise RuntimeError("crash between source markers")

    monkeypatch.setattr(archive_revision_governance, "mark_raw_parse_succeeded", crash_after_one_marker)
    with pytest.raises(RuntimeError, match="between source markers"):
        backfill_historical_revision_evidence(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE parsed_at_ms IS NOT NULL").fetchone()[0] == 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        accepted_before = conn.execute("SELECT raw_id, content_hash FROM sessions").fetchone()
        assert conn.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone()[0] == 2

    monkeypatch.setattr(archive_revision_governance, "mark_raw_parse_succeeded", original_mark)
    backfill_historical_revision_evidence(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE parsed_at_ms IS NOT NULL").fetchone()[0] == 2
        assert {
            str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions WHERE parsed_at_ms IS NOT NULL")
        } == raw_ids
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT raw_id, content_hash FROM sessions").fetchone() == accepted_before
        assert conn.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone()[0] == 2


def test_cold_rebuild_restores_overlapping_multi_session_bundles(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    bundle_a = _bundle(_chatgpt_session("s1", "old"), _chatgpt_session("s2", "only-two"))
    bundle_b = _bundle(
        _chatgpt_session("s1", "old", "extended"),
        _chatgpt_session("s3", "only-three"),
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_a = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=bundle_a,
            source_path="conversations.json",
            acquired_at_ms=1,
        )
        raw_b = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=bundle_b,
            source_path="conversations.json",
            acquired_at_ms=2,
        )

    result = backfill_historical_revision_evidence(tmp_path)
    assert result.replayed_logical_sources == 3
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert set(conn.execute("SELECT native_id, message_count FROM sessions")) == {
            ("s1", 2),
            ("s2", 1),
            ("s3", 1),
        }
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert set(conn.execute("SELECT raw_id FROM raw_sessions WHERE parsed_at_ms IS NOT NULL")) == {
            (raw_a,),
            (raw_b,),
        }
        assert conn.execute(
            "SELECT COUNT(*) FROM raw_session_memberships WHERE decision IN ('ambiguous', 'deferred')"
        ).fetchone() == (0,)

    (tmp_path / "index.db").unlink()
    bootstrap_archive_root(tmp_path)
    rebuilt = backfill_historical_revision_evidence(tmp_path)
    assert rebuilt.replayed_logical_sources == 3
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert set(conn.execute("SELECT native_id, message_count FROM sessions")) == {
            ("s1", 2),
            ("s2", 1),
            ("s3", 1),
        }


def test_divergent_bundle_member_does_not_block_safe_members(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_a = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("s1", "base", "left"), _chatgpt_session("s2", "safe")),
            source_path="conversations.json",
            acquired_at_ms=1,
        )
        raw_b = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("s1", "base", "right"), _chatgpt_session("s3", "safe")),
            source_path="conversations.json",
            acquired_at_ms=2,
        )

    result = backfill_historical_revision_evidence(tmp_path)
    # s1's own two revisions (base+left vs base+right) are a genuine,
    # irreducible fork with no prior head for this fresh archive -- the
    # presence-guarantee fallback (polylogue-lb39z item 5) now materializes
    # a deterministic winner instead of leaving s1 permanently headless, so
    # only the LOSING side of that fork stays quarantined (1, not 2). s2/s3
    # are each single-member "safe" cohorts and were never at risk.
    assert result.quarantined == 1
    winner_raw_id = max(raw_a, raw_b)
    loser_raw_id = raw_a if winner_raw_id == raw_b else raw_b
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert set(conn.execute("SELECT native_id FROM sessions")) == {("s1",), ("s2",), ("s3",)}
        s1_raw_id = conn.execute("SELECT raw_id FROM sessions WHERE native_id = 's1'").fetchone()[0]
        assert s1_raw_id == winner_raw_id
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert set(conn.execute("SELECT raw_id FROM raw_sessions WHERE parsed_at_ms IS NULL")) == {(loser_raw_id,)}
        assert conn.execute("SELECT COUNT(*) FROM raw_session_memberships WHERE decision = 'ambiguous'").fetchone() == (
            1,
        )


def test_stale_pre_fix_identity_split_folds_into_one_ambiguous_cohort(tmp_path: Path) -> None:
    """polylogue-eqnv: two raws of the SAME physical document, one carrying a
    ``logical_source_key`` assigned by a since-superseded parser (a stale
    ``raw_authority_parser_census`` receipt persisted before an identity-bug
    fix -- the exact shape of the pre-#3179/z1c6 dispatch bug), must not be
    replayed as two independent byte-proven singletons. The retire-to-
    membership-governance fallback must bucket both raws under the identity
    the RETIREMENT reparse actually recomputes, not the stale key either raw
    was originally censused under, so they land in ONE membership cohort and
    get jointly arbitrated (here: genuinely divergent content -> ambiguous,
    neither materializes) instead of each becoming an independent
    membership-governance "singleton winner" -- which would reproduce the
    exact fidelity-downgrade bug this retirement path exists to prevent, one
    layer down.
    """
    bootstrap_archive_root(tmp_path)
    correct_key = "chatgpt-export:s1"
    stale_key = "chatgpt-export:s1-0"

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_correct = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("s1", "base", "left")),
            source_path="conversations.json",
            acquired_at_ms=1,
        )
        raw_stale = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("s1", "base", "right")),
            source_path="conversations.json",
            acquired_at_ms=2,
        )

    # Simulate the durable pre-existing state a since-fixed parser identity
    # bug leaves behind: both raws already censused and typed 'full', one
    # under the correct key a fresh reparse yields, the other under a stale,
    # superseded key -- both stamped with the SAME parser fingerprint, so
    # the ordinary quiescence gate would never re-derive either.
    with sqlite3.connect(tmp_path / "source.db") as conn:
        for raw_id, key in ((raw_correct, correct_key), (raw_stale, stale_key)):
            conn.execute(
                """
                UPDATE raw_sessions
                SET logical_source_key = ?, revision_kind = 'full', source_revision = raw_id,
                    baseline_raw_id = raw_id, acquisition_generation = 0, revision_authority = 'quarantined'
                WHERE raw_id = ?
                """,
                (key, raw_id),
            )
            conn.execute(
                """
                INSERT INTO raw_authority_parser_census
                    (raw_id, parser_fingerprint, status, logical_keys_json, detail)
                VALUES (?, ?, 'complete', ?, 'pre-seeded for test')
                """,
                (raw_id, raw_authority_parser_fingerprint(), json.dumps([key])),
            )
        conn.commit()

    result = backfill_historical_revision_evidence(tmp_path, selected_raw_ids=[raw_correct, raw_stale])
    # Both raws now correctly fold into ONE cohort under the freshly
    # re-derived identity -- the bug this test guards against. That cohort
    # is a genuine, irreducible fork (base+left vs base+right) with no
    # prior head, so the presence-guarantee fallback (polylogue-lb39z item
    # 5) now deterministically accepts one side instead of leaving the
    # correctly-unified cohort permanently headless; this is a single
    # cohort's own arbitration outcome, not a reappearance of the
    # independent-singleton-winners bug (which would have produced TWO
    # accepted sessions under two different keys).
    assert result.replayed_logical_sources == 1

    with sqlite3.connect(tmp_path / "source.db") as conn:
        memberships = conn.execute(
            "SELECT raw_id, logical_source_key, decision FROM raw_session_memberships ORDER BY raw_id"
        ).fetchall()
    assert {row[0] for row in memberships} == {raw_correct, raw_stale}
    # Both raws must converge on the SAME (freshly re-derived) identity --
    # not the stale key either was originally censused under.
    assert {row[1] for row in memberships} == {correct_key}
    assert {row[2] for row in memberships} == {"applied", "ambiguous"}

    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)


def test_divergent_bundle_member_preserves_last_accepted_session(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("s1", "base", "accepted")),
            source_path="first.json",
            acquired_at_ms=1,
        )
    backfill_historical_revision_evidence(tmp_path)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        accepted = conn.execute("SELECT message_count, content_hash FROM sessions WHERE native_id = 's1'").fetchone()
        accepted_head = conn.execute(
            "SELECT accepted_content_hash FROM raw_revision_heads WHERE logical_source_key = 'chatgpt-export:s1'"
        ).fetchone()
    assert accepted is not None
    assert accepted_head is not None

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("s1", "base", "divergent")),
            source_path="second.json",
            acquired_at_ms=2,
        )
    result = backfill_historical_revision_evidence(tmp_path)

    assert result.quarantined == 2
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert (
            conn.execute("SELECT message_count, content_hash FROM sessions WHERE native_id = 's1'").fetchone()
            == accepted
        )
        assert (
            conn.execute(
                "SELECT accepted_content_hash FROM raw_revision_heads WHERE logical_source_key = 'chatgpt-export:s1'"
            ).fetchone()
            == accepted_head
        )


def test_targeted_rebuild_expands_same_session_across_source_paths_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        selected_raw = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("shared", "old")),
            source_path="first.json",
            acquired_at_ms=1,
        )
        archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("shared", "old", "new")),
            source_path="second.json",
            acquired_at_ms=2,
        )
        unrelated_raw = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("unrelated", "no")),
            source_path="third.json",
            acquired_at_ms=3,
        )

    # Production ordinary convergence starts from the durable membership
    # census established by ingestion/offline rebuild, not an empty source-v7
    # authority catalog.
    backfill_historical_revision_evidence(tmp_path)
    (tmp_path / "index.db").unlink()
    bootstrap_archive_root(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        unrelated_before = conn.execute(
            "SELECT parser_fingerprint, status, member_count, detail FROM raw_membership_census WHERE raw_id = ?",
            (unrelated_raw,),
        ).fetchone()

    from polylogue.sources import revision_backfill

    original_parse = revision_backfill._parse_retained_raw
    opened: list[str] = []

    def observed_parse(archive: ArchiveStore, raw_id: str) -> tuple[list[ParsedSession], int, RawRevisionKind]:
        opened.append(raw_id)
        return original_parse(archive, raw_id)

    monkeypatch.setattr(revision_backfill, "_parse_retained_raw", observed_parse)
    result = backfill_historical_revision_evidence(tmp_path, selected_raw_ids=[selected_raw])
    assert result.replayed_logical_sources == 1
    assert result.scanned == 2
    assert unrelated_raw not in opened
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id, message_count FROM sessions").fetchall() == [("shared", 2)]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert (
            conn.execute(
                "SELECT parser_fingerprint, status, member_count, detail FROM raw_membership_census WHERE raw_id = ?",
                (unrelated_raw,),
            ).fetchone()
            == unrelated_before
        )


def test_membership_census_retains_only_one_logical_cohort_at_scale(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    independent_raw_count = 64
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        for index in range(independent_raw_count):
            payload = _bundle(_chatgpt_session(f"session-{index}", f"message-{index}"))
            archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=payload,
                source_path=f"bundle-{index}.json",
                acquired_at_ms=index + 1,
            )
        shared_payloads = [
            _bundle(_chatgpt_session("shared", "base")),
            _bundle(_chatgpt_session("shared", "base", "new")),
        ]
        for index, payload in enumerate(shared_payloads, start=1):
            archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=payload,
                source_path=f"shared-{index}.json",
                acquired_at_ms=independent_raw_count + index,
            )
    raw_count = independent_raw_count + len(shared_payloads)

    retained: list[tuple[int, int]] = []
    result = backfill_historical_revision_evidence(
        tmp_path,
        retention_observer=lambda count, payload_bytes: retained.append((count, payload_bytes)),
    )

    assert result.scanned == raw_count
    assert result.replayed_logical_sources == independent_raw_count + 1
    assert len(retained) == independent_raw_count + 1
    assert max(count for count, _payload_bytes in retained) == 2
    assert max(payload_bytes for _count, payload_bytes in retained) == sum(map(len, shared_payloads))
    assert sum(count for count, _payload_bytes in retained) == raw_count


def test_historical_backfill_reparses_multi_gib_shaped_raw_instead_of_spilling_archive_wide(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cache miss reparses durable bytes rather than retaining a giant cohort tree."""
    bootstrap_archive_root(tmp_path)
    # polylogue-9ykn: a session_meta-only stream carries no positive
    # conversational evidence and is refused -- append one real message
    # record so this fixture keeps testing the cache/reparse mechanics it is
    # named for, not the now-refused empty shape.
    payload = (
        b'{"type":"session_meta","payload":{"id":"multi-gib-shaped"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"hello"}]}}\n'
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path="multi-gib-shaped.jsonl",
            acquired_at_ms=1,
        )
    declared_multi_gib = 3 * 1024 * 1024 * 1024
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?", (declared_multi_gib, raw_id))
        conn.commit()

    original = revision_backfill._parse_retained_raw
    parses = 0

    def counted(*args: object, **kwargs: object) -> tuple[list[ParsedSession], int, RawRevisionKind]:
        nonlocal parses
        parses += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(revision_backfill, "_parse_retained_raw", counted)
    retained: list[tuple[int, int]] = []
    result = backfill_historical_revision_evidence(
        tmp_path,
        retention_observer=lambda count, payload_bytes: retained.append((count, payload_bytes)),
    )

    assert result.replayed_logical_sources == 1
    assert retained == [(1, declared_multi_gib)]
    # The former archive-wide spill served the second lookup from a retained
    # pickle. A bounded cache deliberately reparses the durable source row.
    assert parses >= 2


def _append_chain_archive(root: Path) -> tuple[str, str]:
    """Two revisions of one logical session: an accepted-cohort replay fixture."""
    bootstrap_archive_root(root)
    baseline = (
        b'{"type":"session_meta","payload":{"id":"chain","timestamp":"2026-07-01T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":'
        b'[{"type":"input_text","text":"old"}]}}\n'
    )
    newest = baseline + (
        b'{"type":"response_item","payload":{"type":"message","role":"assistant","content":'
        b'[{"type":"output_text","text":"new"}]}}\n'
    )
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        newest_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=newest,
            source_path="chain.jsonl",
            acquired_at_ms=1,
        )
        baseline_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline,
            source_path="chain.jsonl",
            acquired_at_ms=2,
        )
    return baseline_raw_id, newest_raw_id


_CHAIN_META = b'{"type":"session_meta","payload":{"id":"chain","timestamp":"2026-07-01T00:00:00Z"}}\n'


def _chain_turn(index: int) -> bytes:
    return (
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":'
        b'[{"type":"input_text","text":"turn-%d"}]}}\n' % index
    )


def _growing_chain_archive(root: Path, *, turns: int) -> list[str]:
    """One rollout file re-captured while it grows, oldest capture first.

    The first capture is the file as it exists between session start and the
    first turn: a ``session_meta`` header and nothing else. That is a strict
    byte prefix of every later capture and still parses to NO session -- the
    parser refuses a session with no conversational evidence.
    """
    bootstrap_archive_root(root)
    payload = _CHAIN_META
    payloads = [payload]
    for index in range(turns):
        payload = payload + _chain_turn(index)
        payloads.append(payload)
    raw_ids: list[str] = []
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for acquired_at_ms, capture in enumerate(payloads, start=1):
            raw_ids.append(
                archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=capture,
                    source_path="chain.jsonl",
                    acquired_at_ms=acquired_at_ms,
                )
            )
    return raw_ids


def _census_facts(root: Path, raw_id: str) -> tuple[str | None, str, tuple[str, ...] | None]:
    with sqlite3.connect(root / "source.db") as conn:
        logical_key, authority = conn.execute(
            "SELECT logical_source_key, revision_authority FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone()
        receipt = conn.execute(
            "SELECT logical_keys_json FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
        ).fetchone()
    keys = tuple(iter_parser_census_logical_keys(receipt[0])) if receipt is not None else None
    return logical_key, str(authority), keys


def test_byte_proof_refuses_a_head_between_forks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """polylogue-asp4b (sibling append): two valid extensions, no chosen head.

    One rollout path holds a shared capture and two captures that each extend
    it differently -- the same file re-scanned after a fork, or two machines
    appending to one synced path. Both are valid extensions of the baseline and
    NEITHER is a byte prefix of the other, so byte comparison alone cannot say
    which is the file's current state.

    Wrong outcome prevented: the census picks the largest capture as the
    cohort's head and binds the other fork to its learned identity on
    containment with the baseline alone. Anti-vacuity: deleting the whole-
    cohort verdict guard in ``classify_untyped_full_revision_groups``
    (``any(decision.authority is not RawRevisionAuthority.BYTE_PROVEN ...)``)
    makes this red -- the group is returned, a head is chosen, and the losing
    fork is never parsed.
    """
    bootstrap_archive_root(tmp_path)
    shared = _CHAIN_META + _chain_turn(0)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_ids = {
            name: archive.write_raw_payload(
                provider=Provider.CODEX, payload=payload, source_path="chain.jsonl", acquired_at_ms=index + 1
            )
            for index, (name, payload) in enumerate(
                (
                    ("shared", shared),
                    ("fork_a", shared + _chain_turn(1)),
                    ("fork_b", shared + _chain_turn(2)),
                )
            )
        }
        assert archive.classify_untyped_full_revision_groups(sorted(raw_ids.values())) == {}

    original = revision_backfill._parse_retained_raw
    parsed: list[str] = []

    def counted(archive: ArchiveStore, raw_id: str) -> tuple[list[ParsedSession], int, RawRevisionKind]:
        parsed.append(raw_id)
        return original(archive, raw_id)

    monkeypatch.setattr(revision_backfill, "_parse_retained_raw", counted)

    backfill_historical_revision_evidence(tmp_path, max_payload_bytes=None)

    # Every member is opened: nothing inherits an identity byte proof cannot
    # establish for it.
    assert set(raw_ids.values()) <= set(parsed)
    # The shared capture is the only member byte proof can place; a fork with a
    # sibling is quarantined rather than crowned.
    assert _census_facts(tmp_path, raw_ids["shared"])[1] == "byte_proven"
    assert _census_facts(tmp_path, raw_ids["fork_a"])[1] == "quarantined"
    assert _census_facts(tmp_path, raw_ids["fork_b"])[1] == "quarantined"


def test_chain_member_identity_refuted_by_its_own_parse(tmp_path: Path) -> None:
    """polylogue-irtix (C): containment must not bind a member the parser refutes.

    Input: two captures of one Codex rollout at one ``source_path`` -- a
    header-only capture taken before the first turn, and the finished file.
    The header-only bytes are a strict byte prefix of the finished file, so the
    byte-growth census proves the chain; parsing those bytes alone yields no
    session at all.

    Wrong outcome prevented: the header-only raw is recorded as a FULL revision
    of ``codex-session:chain`` with a ``parser-observed`` census receipt naming
    a membership its own parse denies. Anti-vacuity: dropping the smallest
    member back into ``head_by_older`` (inheriting on containment alone) makes
    this red -- the key becomes ``codex-session:chain`` and the receipt
    ``("codex-session:chain",)``.
    """
    header_only, finished = _growing_chain_archive(tmp_path, turns=1)

    backfill_historical_revision_evidence(tmp_path, max_payload_bytes=None)

    assert _census_facts(tmp_path, finished) == ("codex-session:chain", "byte_proven", ("codex-session:chain",))
    # The refuted member keeps the classification its OWN bytes support.
    assert _census_facts(tmp_path, header_only) == (None, "quarantined", ())


def test_chain_inherits_only_its_interior_members(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The identity spot-check costs parses per CHAIN, never per member.

    Opposite direction of the test above: a blanket "parse every member"
    refusal would also keep a refuted member unbound, so this pins the
    optimization polylogue-nh44 bought. Five captures whose smallest member is
    header-only: the census parses the smallest, ascends until one capture's own
    parse lands on the head's key, and inherits for every member bracketed
    between that capture and the head.
    """
    raw_ids = _growing_chain_archive(tmp_path, turns=4)
    original = revision_backfill._parse_retained_raw
    parsed: list[str] = []

    def counted(archive: ArchiveStore, raw_id: str) -> tuple[list[ParsedSession], int, RawRevisionKind]:
        parsed.append(raw_id)
        return original(archive, raw_id)

    monkeypatch.setattr(revision_backfill, "_parse_retained_raw", counted)

    backfill_historical_revision_evidence(tmp_path, max_payload_bytes=None)

    # raw_ids[0] is the refuted header-only capture, raw_ids[1] the smallest
    # capture whose own parse agrees, raw_ids[-1] the head. Nothing between
    # them is opened.
    assert set(parsed) == {raw_ids[0], raw_ids[1], raw_ids[-1]}
    assert _census_facts(tmp_path, raw_ids[0])[0] is None
    for inherited in raw_ids[2:-1]:
        assert _census_facts(tmp_path, inherited) == ("codex-session:chain", "byte_proven", ("codex-session:chain",))


def _independent_growing_chains(root: Path, *, chains: int, turns: int) -> dict[str, list[str]]:
    """``chains`` rollout files, each re-captured while it grows, oldest capture first."""
    bootstrap_archive_root(root)
    raw_ids: dict[str, list[str]] = {}
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for chain in range(chains):
            session = f"chain-{chain}"
            payload = b'{"type":"session_meta","payload":{"id":"%s","timestamp":"2026-07-01T00:00:00Z"}}\n' % (
                session.encode()
            )
            captures = [payload]
            for index in range(turns):
                payload = payload + _chain_turn(index)
                captures.append(payload)
            raw_ids[session] = [
                archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=capture,
                    source_path=f"{session}.jsonl",
                    acquired_at_ms=chain * 100 + acquired_at_ms,
                )
                for acquired_at_ms, capture in enumerate(captures, start=1)
            ]
    return raw_ids


def test_chain_census_finds_learned_keys_without_scanning_every_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A deferred chain member's key lookup costs the same for 2 chains or 6.

    Input: independently growing rollout files, each with superseded
    byte-prefix captures. Every chain head and probe is looked up once per
    deferred member, so a lookup that walks every learned logical key makes
    the census quadratic in the number of files.

    Wrong outcome prevented: ``provisional_full_raw_ids.items()`` scanned per
    lookup. Anti-vacuity: restore the linear scan in the census's
    ``bound_logical_key`` and each run scans once per deferred member instead
    of at most once per phase. The interior members still inherit their own
    chain's key, which pins the reverse map's semantics.
    """
    original_state = revision_backfill._RevisionCensusState
    scans: list[int] = []

    class CountingKeys(dict[str, set[str]]):
        def items(self) -> ItemsView[str, set[str]]:  # type: ignore[override]
            scans[-1] += 1
            return super().items()

    def counting_state(*args: Any, **kwargs: Any) -> Any:
        state = original_state(*args, **kwargs)
        state.provisional_full_raw_ids = CountingKeys(state.provisional_full_raw_ids)
        return state

    monkeypatch.setattr(revision_backfill, "_RevisionCensusState", counting_state)

    for chains in (2, 6):
        root = tmp_path / f"chains-{chains}"
        raw_ids = _independent_growing_chains(root, chains=chains, turns=4)
        scans.append(0)
        backfill_historical_revision_evidence(root, max_payload_bytes=None)
        for session, chain in raw_ids.items():
            key = f"codex-session:{session}"
            assert _census_facts(root, chain[0])[0] is None
            for inherited in chain[2:-1]:
                assert _census_facts(root, inherited) == (key, "byte_proven", (key,))

    # The decode prefetcher snapshots the learned keys once per replay phase
    # when it runs; a per-member lookup scan would add one per deferred member.
    assert max(scans) <= 1, scans


def test_backfill_replay_reparses_when_spill_cache_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Baseline (pre-fix) shape: an unbounded (envelope=None) backfill with no
    explicit spill-cache bound reparses the accepted revision during replay
    even though census already parsed it once. This is exactly the CLI
    rebuild-index path's behavior before max_cached_payload_bytes decoupled
    caching from the resource envelope. Paired with the fixed-behavior test
    below to pin both sides of the regression.
    """
    _append_chain_archive(tmp_path)
    original = revision_backfill._parse_retained_raw
    parse_calls: list[str] = []

    def counted(archive: ArchiveStore, raw_id: str) -> tuple[list[ParsedSession], int, RawRevisionKind]:
        parse_calls.append(raw_id)
        return original(archive, raw_id)

    monkeypatch.setattr(revision_backfill, "_parse_retained_raw", counted)

    result = backfill_historical_revision_evidence(tmp_path, max_payload_bytes=None)

    assert result.replayed_logical_sources == 1
    # polylogue-nh44 + polylogue-irtix (C): a two-member cohort has no interior,
    # so both endpoints are parsed once during census; replay then reparses the
    # accepted revision again from blob because nothing was cached (a 3rd call,
    # duplicating the head) instead of reusing census output.
    assert len(parse_calls) == 3
    assert len(set(parse_calls)) == 2


def test_census_skips_parse_for_byte_proven_superseded_revisions_at_scale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """polylogue-nh44 regression at the bead's own recorded corpus shape: a
    growing-file cohort (one re-scanned Codex rollout, 50 superseded captures
    plus the winner) must census-parse only the winner, never the 50 byte-
    proven-superseded snapshots. Measured on this exact shape: 52->2 parse
    calls (1 unique raw parsed instead of 51), ~3.3x wall-time reduction for
    the cohort (see PR body for the before/after numbers)."""
    raw_ids = build_revision_chain_corpus(tmp_path, **REVISION_CHAIN_SHAPE)
    original = revision_backfill._parse_retained_raw
    parse_calls: list[str] = []

    def counted(archive: ArchiveStore, raw_id: str) -> tuple[list[ParsedSession], int, RawRevisionKind]:
        parse_calls.append(raw_id)
        return original(archive, raw_id)

    monkeypatch.setattr(revision_backfill, "_parse_retained_raw", counted)

    result = backfill_historical_revision_evidence(tmp_path)

    assert result.scanned == len(raw_ids)
    # polylogue-irtix (C): the cohort's smallest capture is the rollout's
    # ``session_meta`` header alone, which parses to no session, so it is no
    # longer counted as a classified full revision.
    assert result.classified_full == len(raw_ids) - 1
    assert result.replayed_logical_sources == 1
    assert result.quarantined == 0
    # Three raws are ever independently parsed: the winner, the header-only
    # capture whose own parse refutes the head's identity, and the smallest
    # capture that agrees with it. The 48 captures bracketed between that
    # capture and the winner are bound by byte-prefix proof.
    assert set(parse_calls) == {raw_ids[0], raw_ids[1], raw_ids[-1]}
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT raw_id FROM sessions").fetchone() == (raw_ids[-1],)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM raw_sessions WHERE revision_kind = 'full' AND logical_source_key IS NOT NULL"
            ).fetchone()[0]
            == len(raw_ids) - 1
        )
        assert conn.execute(
            "SELECT COUNT(*) FROM raw_authority_parser_census WHERE parser_fingerprint = ? AND status = 'complete'",
            (raw_authority_parser_fingerprint(),),
        ).fetchone()[0] == len(raw_ids)


def test_backfill_replay_reuses_spill_cache_when_bound_explicitly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """max_cached_payload_bytes caches census parse output independently of
    max_payload_bytes (the resource-envelope block), so an unbounded backfill
    still avoids reparsing accepted revisions during replay. Mutation:
    reverting to the paired baseline test's call (omitting
    max_cached_payload_bytes) reproduces the doubled parse count above --
    this is the anti-vacuity pairing for the CLI rebuild-index fix.
    """
    _append_chain_archive(tmp_path)
    original = revision_backfill._parse_retained_raw
    parse_calls: list[str] = []

    def counted(archive: ArchiveStore, raw_id: str) -> tuple[list[ParsedSession], int, RawRevisionKind]:
        parse_calls.append(raw_id)
        return original(archive, raw_id)

    monkeypatch.setattr(revision_backfill, "_parse_retained_raw", counted)

    result = backfill_historical_revision_evidence(
        tmp_path,
        max_payload_bytes=None,
    )

    assert result.replayed_logical_sources == 1
    # polylogue-nh44 + polylogue-irtix (C): census parses each of the cohort's
    # two endpoints once; replay hits the census-populated spill cache instead
    # of reparsing the head from blob a second time (contrast the 3-call
    # baseline above).
    assert len(parse_calls) == 2
    assert len(set(parse_calls)) == 2


def test_parallel_census_matches_sequential_archive_state(tmp_path: Path) -> None:
    """Parsing spread across a process pool must produce byte-identical
    archive state to the sequential path. Only read-only blob->ParsedSession
    decode is parallelized; archive writes apply in fixed pending-rows order
    regardless of worker completion order, so parallel and sequential runs
    are authority-equivalent (not merely "close enough").
    """
    sequential_root = tmp_path / "sequential"
    parallel_root = tmp_path / "parallel"
    for root in (sequential_root, parallel_root):
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            for index in range(6):
                payload = _bundle(_chatgpt_session(f"session-{index}", f"hello {index}", f"world {index}"))
                archive.write_raw_payload(
                    provider=Provider.CHATGPT,
                    payload=payload,
                    source_path=f"chat-{index}.json",
                    acquired_at_ms=index,
                )

    seq_result = backfill_historical_revision_evidence(sequential_root, ingest_workers=1)
    par_result = backfill_historical_revision_evidence(parallel_root, ingest_workers=4)

    assert seq_result == par_result

    def _sessions(root: Path) -> list[tuple[object, ...]]:
        with sqlite3.connect(root / "index.db") as conn:
            return conn.execute("SELECT native_id, message_count, raw_id FROM sessions ORDER BY native_id").fetchall()

    assert _sessions(sequential_root) == _sessions(parallel_root)


def _state_db_bytes_for_session(tmp_path: Path, *, session_id: str, message_text: str) -> bytes:
    """Variant of _single_session_state_db_bytes with a distinct session id."""
    db_path = tmp_path / f"state-source-{session_id}.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (19);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, source TEXT, model_config TEXT, parent_session_id TEXT,
                started_at REAL, ended_at REAL, end_reason TEXT, title TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT,
                timestamp REAL NOT NULL, tool_calls TEXT, observed INTEGER DEFAULT 0,
                active INTEGER DEFAULT 1, compacted INTEGER DEFAULT 0
            );
            """
        )
        conn.execute(
            "INSERT INTO sessions (id, source, model_config, started_at, ended_at, end_reason, title) "
            "VALUES (?, 'cli', '{}', 1.0, 8.0, 'completed', ?)",
            (session_id, session_id),
        )
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (1, ?, 'user', ?, 2.0)",
            (session_id, message_text),
        )
    return db_path.read_bytes()


def test_parallel_census_quarantines_legacy_hermes_sqlite_page_images(tmp_path: Path) -> None:
    """Legacy SQLite page images cannot re-enter replay through the pool.

    The historical #3113 path accepted these as Hermes state-db raws. Current
    acquisition retains declared logical exports, so admitting an old page
    image would recreate a second source authority during a future reindex.
    Two independent raws still exercise the parallel census boundary.
    """
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        for index in range(2):
            payload = _state_db_bytes_for_session(tmp_path, session_id=f"hermes-{index}", message_text=f"hi {index}")
            archive.write_raw_payload(
                provider=Provider.HERMES,
                payload=payload,
                source_path=str(tmp_path / f"hermes-home-{index}" / "state.db"),
                acquired_at_ms=index,
            )

    result = backfill_historical_revision_evidence(tmp_path, ingest_workers=4)

    assert result.scanned == 2
    assert result.replayed_logical_sources == 0
    assert result.quarantined == 2
    with sqlite3.connect(tmp_path / "index.db") as conn:
        rows = conn.execute("SELECT native_id, message_count FROM sessions ORDER BY native_id").fetchall()
    assert rows == []


def test_independent_raw_corpus_fixture_backfills_cleanly(tmp_path: Path) -> None:
    """polylogue-amg1 benchmark fixture sanity: every synthetic raw census-and-replays
    to exactly one session with no quarantine, at both recorded payload shapes' scale
    (downscaled here for test speed; devtools/scripts run the full recorded counts)."""
    raw_ids = build_independent_raw_corpus(tmp_path, raw_count=12, avg_payload_bytes=5_000)

    result = backfill_historical_revision_evidence(tmp_path)

    assert result.scanned == 12
    assert result.replayed_logical_sources == 12
    assert result.quarantined == 0
    with sqlite3.connect(tmp_path / "index.db") as conn:
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    assert session_count == 12
    assert len(set(raw_ids)) == 12


def test_census_batch_crash_loses_at_most_one_batch_and_resumes_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-amg1 crash-mid-batch proof: a fault partway through an
    uncommitted census batch must discard exactly that batch (not a partial
    raw, not prior committed batches), and a resume must converge to the
    same terminal state as an uninterrupted run with zero duplication."""
    raw_count = 10
    batch_size = 4
    root = tmp_path / "archive"
    build_independent_raw_corpus(root, raw_count=raw_count, avg_payload_bytes=1_000)

    original_bind: Callable[..., None] = ArchiveStore.bind_raw_revision
    calls = 0
    # Crash on the 7th bind call: batch 1 (calls 1-4) has already committed;
    # batch 2 (calls 5-8) is interrupted after its 3rd call (7), before it
    # reaches batch_size and self-commits.
    crash_at_call = 7

    def crash_partway(self: ArchiveStore, raw_id: str, revision: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == crash_at_call:
            raise RuntimeError("injected crash mid-batch")
        original_bind(self, raw_id, revision, **kwargs)

    monkeypatch.setattr(ArchiveStore, "bind_raw_revision", crash_partway)
    with pytest.raises(RuntimeError, match="injected crash mid-batch"):
        backfill_historical_revision_evidence(root, commit_batch_size=batch_size)

    with sqlite3.connect(root / "source.db") as conn:
        complete_after_crash = conn.execute(
            "SELECT COUNT(*) FROM raw_sessions WHERE revision_kind != 'unknown'"
        ).fetchone()[0]
    # Exactly one fully-committed batch survives the crash -- never a partial one.
    assert complete_after_crash == batch_size

    monkeypatch.setattr(ArchiveStore, "bind_raw_revision", original_bind)
    result = backfill_historical_revision_evidence(root, commit_batch_size=batch_size)

    assert result.scanned == raw_count
    assert result.replayed_logical_sources == raw_count
    assert result.quarantined == 0
    with sqlite3.connect(root / "index.db") as conn:
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        application_count = conn.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone()[0]
    assert session_count == raw_count
    # One application receipt per raw, no duplicates from the retried batch.
    assert application_count == raw_count
    with sqlite3.connect(root / "source.db") as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE revision_kind != 'unknown'").fetchone()[0]
            == raw_count
        )


def test_backfill_resumes_after_replay_batch_crash_discards_whole_batch_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-oikv: with ``commit_batch_size`` set, the REPLAY phase now
    batches index.db writes + terminal source.db markers across MULTIPLE
    independent cohorts (not just within one cohort, as the two
    unbatched-default pinned tests above still prove unmodified). A fault
    partway through an uncommitted replay batch must discard the WHOLE
    batch -- every cohort's index writes and terminal markers together,
    since neither side ever committed -- never a partial one, and a resume
    must converge to the same terminal state as an uninterrupted run with
    zero duplication (mirrors the census-phase proof above)."""
    raw_count = 10
    batch_size = 4
    root = tmp_path / "archive"
    build_independent_raw_corpus(root, raw_count=raw_count, avg_payload_bytes=1_000)

    original_apply: Callable[..., object] = ArchiveStore.apply_raw_revision_replay
    calls = 0
    # Batch 1 (cohorts 1-4) commits cleanly and resets the counter. Batch 2
    # starts (cohort 5 applies, uncommitted), then crashes on cohort 6 --
    # before batch 2 reaches batch_size and self-commits.
    crash_at_call = 6

    def crash_partway(self: ArchiveStore, plan: object, parsed_by_raw_id: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == crash_at_call:
            raise RuntimeError("injected crash mid replay-batch")
        return original_apply(self, plan, parsed_by_raw_id, **kwargs)

    monkeypatch.setattr(ArchiveStore, "apply_raw_revision_replay", crash_partway)
    with pytest.raises(RuntimeError, match="injected crash mid replay-batch"):
        backfill_historical_revision_evidence(root, commit_batch_size=batch_size)

    with sqlite3.connect(root / "index.db") as conn:
        session_count_after_crash = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    # Exactly one fully-committed batch survives the crash -- never a partial one.
    assert session_count_after_crash == batch_size
    with sqlite3.connect(root / "source.db") as conn:
        parsed_after_crash = conn.execute(
            "SELECT COUNT(*) FROM raw_sessions WHERE parsed_at_ms IS NOT NULL"
        ).fetchone()[0]
    assert parsed_after_crash == batch_size

    monkeypatch.setattr(ArchiveStore, "apply_raw_revision_replay", original_apply)
    result = backfill_historical_revision_evidence(root, commit_batch_size=batch_size)

    assert result.scanned == raw_count
    assert result.replayed_logical_sources == raw_count
    assert result.quarantined == 0
    with sqlite3.connect(root / "index.db") as conn:
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        application_count = conn.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone()[0]
    assert session_count == raw_count
    # One application receipt per raw, no duplicates from the retried batch.
    assert application_count == raw_count
    with sqlite3.connect(root / "source.db") as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE parsed_at_ms IS NOT NULL").fetchone()[0] == raw_count
        )


def test_parse_retained_raws_dedupes_identical_blob_across_paths_for_safe_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """polylogue-869u: for a path-independent provider (Codex here), rows
    sharing a ``blob_hash`` parse once and fan out -- INCLUDING across
    different ``source_path``s, since those parsers' session identity comes
    entirely from the payload bytes. Live shape: one 442MB codex rollout
    acquired 8x within 2.3s during a stampede, at up to 8 different acquired
    paths -- 8 raw rows, one blob, formerly 8 full parses."""
    descriptors = {
        "dup-1": (Provider.CODEX, "hash-A", "same.jsonl", RawRevisionKind.FULL, 10),
        "dup-2": (Provider.CODEX, "hash-A", "same.jsonl", RawRevisionKind.UNKNOWN, 10),
        "dup-3": (Provider.CODEX, "hash-A", "same.jsonl", RawRevisionKind.FULL, 10),
        "other-path": (Provider.CODEX, "hash-A", "different.jsonl", RawRevisionKind.FULL, 10),
        "other-bytes": (Provider.CODEX, "hash-B", "same.jsonl", RawRevisionKind.FULL, 20),
    }

    class FakeArchive:
        archive_root = Path("/synthetic/archive")
        source_db_path = Path("/synthetic/archive/source.db")

        def raw_revision_descriptor(self, raw_id: str) -> tuple[Provider, str, str, RawRevisionKind, int]:
            return descriptors[raw_id]

    parsed: list[str] = []

    def fake_parse(raw_id: str, *args: object) -> tuple[str, list[ParsedSession], None]:
        parsed.append(raw_id)
        return raw_id, [], None

    monkeypatch.setattr(revision_backfill, "census_parse_worker", fake_parse)

    results = revision_backfill._parse_retained_raws(
        FakeArchive(),  # type: ignore[arg-type]
        list(descriptors),
        ingest_workers=1,
    )

    # one parse per distinct blob_hash: dup-2/dup-3/other-path all reuse
    # dup-1's outcome despite other-path having a different source_path.
    assert sorted(parsed) == ["dup-1", "other-bytes"]
    assert set(results) == set(descriptors)
    sessions, size, kind = results["dup-2"]  # type: ignore[misc]
    assert (sessions, size, kind) == ([], 10, RawRevisionKind.UNKNOWN)
    _sessions, _size, dup3_kind = results["dup-3"]  # type: ignore[misc]
    assert dup3_kind == RawRevisionKind.FULL
    _sessions, other_path_size, other_path_kind = results["other-path"]  # type: ignore[misc]
    assert (other_path_size, other_path_kind) == (10, RawRevisionKind.FULL)


def test_parse_retained_raws_fans_out_exceptions_to_duplicate_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptors = {
        "dup-1": (Provider.CODEX, "hash-A", "same.jsonl", RawRevisionKind.FULL, 10),
        "dup-2": (Provider.CODEX, "hash-A", "same.jsonl", RawRevisionKind.FULL, 10),
    }

    class FakeArchive:
        archive_root = Path("/synthetic/archive")
        source_db_path = Path("/synthetic/archive/source.db")

        def raw_revision_descriptor(self, raw_id: str) -> tuple[Provider, str, str, RawRevisionKind, int]:
            return descriptors[raw_id]

    def failing_parse(raw_id: str, *args: object) -> tuple[str, list[ParsedSession], None]:
        raise ValueError(f"boom {raw_id}")

    monkeypatch.setattr(revision_backfill, "census_parse_worker", failing_parse)

    results = revision_backfill._parse_retained_raws(
        FakeArchive(),  # type: ignore[arg-type]
        list(descriptors),
        ingest_workers=1,
    )

    assert isinstance(results["dup-1"], ValueError)
    assert results["dup-2"] is results["dup-1"]


def test_thread_parse_matches_sequential_archive_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Shared compute window widths preserve canonical archive rows."""
    sequential_root = tmp_path / "sequential"
    thread_root = tmp_path / "threaded"
    for root in (sequential_root, thread_root):
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            for index in range(6):
                payload = _bundle(_chatgpt_session(f"session-{index}", f"hello {index}", f"world {index}"))
                archive.write_raw_payload(
                    provider=Provider.CHATGPT,
                    payload=payload,
                    source_path=f"chat-{index}.json",
                    acquired_at_ms=index,
                )

    seq_result = backfill_historical_revision_evidence(sequential_root, ingest_workers=1)

    thread_result = backfill_historical_revision_evidence(thread_root, ingest_workers=4)

    assert seq_result == thread_result

    def _sessions(root: Path) -> list[tuple[object, ...]]:
        with sqlite3.connect(root / "index.db") as conn:
            return conn.execute("SELECT native_id, message_count, raw_id FROM sessions ORDER BY native_id").fetchall()

    assert _sessions(sequential_root) == _sessions(thread_root)


def test_thread_parse_normalizes_derived_timestamps_matching_sequential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stateless workers still apply the retained timestamp authority contract."""
    bootstrap_archive_root(tmp_path)
    raw_ids: list[str] = []
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        for index in range(2):
            native_id = f"timestamp-session-{index}"
            payload = (
                f'{{"type":"session_meta","payload":{{"id":"{native_id}"}}}}\n'
                f'{{"timestamp":"2025-02-14T11:08:01.474463+00:00","type":"response_item",'
                f'"payload":{{"type":"message","id":"m-{index}","role":"user",'
                f'"content":[{{"type":"input_text","text":"hello {index}"}}]}}}}\n'
            ).encode()
            raw_ids.append(
                archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=payload,
                    source_path=f"timestamp-{index}.jsonl",
                    acquired_at_ms=index,
                )
            )

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        sequential = revision_backfill._parse_retained_raws(archive, raw_ids, ingest_workers=1)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        threaded = revision_backfill._parse_retained_raws(archive, raw_ids, ingest_workers=2)

    for raw_id in raw_ids:
        sequential_sessions, _size, _kind = sequential[raw_id]  # type: ignore[misc]
        threaded_sessions, _thread_size, _thread_kind = threaded[raw_id]  # type: ignore[misc]
        assert threaded_sessions == sequential_sessions


def _append_delta_without_self_describing_identity(text: str) -> bytes:
    """A Codex append-delta payload with no ``session_meta`` record of its
    own -- the polylogue-u19l shape: the parser has no self-describing
    identity to read and must fall back to whatever fallback_id it is
    given."""
    return (
        b'{"type":"response_item","payload":{"type":"message","id":"m0","role":"user",'
        b'"content":[{"type":"input_text","text":"' + text.encode() + b'"}]}}\n'
    )


def _write_append_raw_with_recovered_identity(
    archive: ArchiveStore, *, raw_id: str, native_id: str, source_path: str, payload: bytes, acquired_at_ms: int
) -> None:
    """Write an APPEND-kind raw whose own bytes carry no identity, recording
    ``native_id`` as the write-time recovery hint (``write_raw_payload``'s
    ``native_id`` -- see ``sources/live/batch.py``'s
    ``_append_payload_for_provider``) -- deliberately at a ``source_path``
    whose stem does NOT equal ``native_id``, so a dispatch path that falls
    back to ``Path(source_path).stem`` instead of the recorded native_id
    diverges observably from one that recovers it correctly."""
    assert Path(source_path).stem != native_id
    archive.write_raw_payload(
        provider=Provider.CODEX,
        payload=payload,
        source_path=source_path,
        acquired_at_ms=acquired_at_ms,
        raw_id=raw_id,
        native_id=native_id,
    )
    _seed_historical_revision(
        archive,
        raw_id,
        RawRevisionEnvelope(
            logical_source_key=f"codex-session:{native_id}",
            kind=RawRevisionKind.APPEND,
            source_revision=f"{raw_id}-revision",
            acquisition_generation=0,
            predecessor_source_revision=f"{raw_id}-predecessor",
            append_start_offset=0,
            append_end_offset=len(payload),
            authority=RawRevisionAuthority.QUARANTINED,
        ),
    )


def test_thread_parse_recovers_append_native_id_matching_sequential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every compute window preserves write-time APPEND native identity."""
    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        _write_append_raw_with_recovered_identity(
            archive,
            raw_id="raw-alpha",
            native_id="session-alpha",
            source_path="delta-file-one.jsonl",
            payload=_append_delta_without_self_describing_identity("hello alpha"),
            acquired_at_ms=1,
        )
        _write_append_raw_with_recovered_identity(
            archive,
            raw_id="raw-beta",
            native_id="session-beta",
            source_path="delta-file-two.jsonl",
            payload=_append_delta_without_self_describing_identity("hello beta"),
            acquired_at_ms=2,
        )

    raw_ids = ["raw-alpha", "raw-beta"]

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        sequential_results = revision_backfill._parse_retained_raws(archive, raw_ids, ingest_workers=1)

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        thread_results = revision_backfill._parse_retained_raws(archive, raw_ids, ingest_workers=4)

    expected_native_id = {"raw-alpha": "session-alpha", "raw-beta": "session-beta"}
    for raw_id in raw_ids:
        seq_sessions, _seq_size, _seq_kind = sequential_results[raw_id]  # type: ignore[misc]
        thread_sessions, _thread_size, _thread_kind = thread_results[raw_id]  # type: ignore[misc]
        assert len(seq_sessions) == 1
        assert len(thread_sessions) == 1
        assert seq_sessions[0].provider_session_id == expected_native_id[raw_id]
        # The actual regression proof: the thread path must match the
        # sequential path's recovered identity, not silently fall back to
        # the source_path stem instead.
        assert thread_sessions[0].provider_session_id == seq_sessions[0].provider_session_id


def test_thread_parse_never_touches_shared_archive_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Workers receive immutable descriptors and never query caller-owned SQLite."""

    class _NoMethodsArchive:
        archive_root = Path("/fake-root")
        source_db_path = Path("/fake-root/source.db")

    descriptors: dict[str, tuple[Provider, str, str, RawRevisionKind, int, str | None]] = {
        "raw-a": (Provider.CODEX, "hash-a", "a.jsonl", RawRevisionKind.FULL, 111, None),
        "raw-b": (Provider.CODEX, "hash-b", "b.jsonl", RawRevisionKind.FULL, 222, None),
    }

    def fake_worker(
        raw_id: str,
        provider_token: str,
        blob_hash: str,
        source_path: str,
        is_stream: bool,
        blob_root_str: str,
        source_db_path_str: str,
        kind_token: str,
        native_id: str | None,
    ) -> tuple[str, list[ParsedSession] | None, revision_backfill.RetainedParseFailure | None]:
        return raw_id, [], None

    monkeypatch.setattr(revision_backfill, "census_parse_worker", fake_worker)

    results = revision_backfill._parse_unique_retained_raws(
        _NoMethodsArchive(),  # type: ignore[arg-type]
        list(descriptors),
        descriptors=descriptors,
        ingest_workers=2,
    )

    assert results["raw-a"] == ([], 111, RawRevisionKind.FULL)
    assert results["raw-b"] == ([], 222, RawRevisionKind.FULL)


def test_thread_parse_propagates_per_raw_exception_without_poisoning_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker failure remains scoped to its raw while neighboring results survive."""
    descriptors: dict[str, tuple[Provider, str, str, RawRevisionKind, int, str | None]] = {
        "ok-1": (Provider.CODEX, "hash-A", "a.jsonl", RawRevisionKind.FULL, 10, None),
        "bad-1": (Provider.CODEX, "hash-B", "b.jsonl", RawRevisionKind.FULL, 20, None),
        "ok-2": (Provider.CODEX, "hash-C", "c.jsonl", RawRevisionKind.FULL, 30, None),
    }

    class _FakeArchive:
        archive_root = Path("/fake-root")
        source_db_path = Path("/fake-root/source.db")

    def fake_worker(
        raw_id: str,
        provider_token: str,
        blob_hash: str,
        source_path: str,
        is_stream: bool,
        blob_root_str: str,
        source_db_path_str: str,
        kind_token: str,
        native_id: str | None,
    ) -> tuple[str, list[ParsedSession] | None, revision_backfill.RetainedParseFailure | None]:
        if raw_id == "bad-1":
            raise RuntimeError(f"boom {raw_id}")
        return raw_id, [], None

    monkeypatch.setattr(revision_backfill, "census_parse_worker", fake_worker)

    results = revision_backfill._parse_unique_retained_raws(
        _FakeArchive(),  # type: ignore[arg-type]
        list(descriptors),
        descriptors=descriptors,
        ingest_workers=3,
    )

    assert isinstance(results["bad-1"], RuntimeError)
    assert "boom bad-1" in str(results["bad-1"])
    assert results["ok-1"] == ([], 10, RawRevisionKind.FULL)
    assert results["ok-2"] == ([], 30, RawRevisionKind.FULL)


def test_thread_parse_results_keyed_by_raw_id_not_completion_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Completion order cannot pair a raw with another descriptor."""
    raw_ids = [f"raw-{i}" for i in range(6)]
    descriptors: dict[str, tuple[Provider, str, str, RawRevisionKind, int, str | None]] = {
        raw_id: (Provider.CODEX, f"hash-{i}", f"path-{i}.jsonl", RawRevisionKind.FULL, 100 + i, None)
        for i, raw_id in enumerate(raw_ids)
    }
    # raw-0 (submitted first) sleeps longest; raw-5 (submitted last) returns
    # immediately -- completion order is the exact reverse of submission order.
    delay_by_raw_id = {raw_id: 0.02 * (len(raw_ids) - index) for index, raw_id in enumerate(raw_ids)}

    class _FakeArchive:
        archive_root = Path("/fake-root")
        source_db_path = Path("/fake-root/source.db")

    def fake_worker(
        raw_id: str,
        provider_token: str,
        blob_hash: str,
        source_path: str,
        is_stream: bool,
        blob_root_str: str,
        source_db_path_str: str,
        kind_token: str,
        native_id: str | None,
    ) -> tuple[str, list[ParsedSession] | None, revision_backfill.RetainedParseFailure | None]:
        time.sleep(delay_by_raw_id[raw_id])
        return raw_id, [], None

    monkeypatch.setattr(revision_backfill, "census_parse_worker", fake_worker)

    results = revision_backfill._parse_unique_retained_raws(
        _FakeArchive(),  # type: ignore[arg-type]
        raw_ids,
        descriptors=descriptors,
        ingest_workers=len(raw_ids),
    )

    for index, raw_id in enumerate(raw_ids):
        sessions, size, kind = results[raw_id]  # type: ignore[misc]
        assert sessions == []
        assert size == 100 + index
        assert kind == RawRevisionKind.FULL


# ---------------------------------------------------------------------------
# Whale-aware census spill (polylogue-odm1)
# ---------------------------------------------------------------------------

# Shrink the hot-cache budget so a modest (KB-scale) fixture reliably
# classifies as a "whale" without depending on the host's real RAM -- see
# tests/benchmarks/test_whale_census_spill_bench.py's module docstring for
# the full rationale (the class computes its budgets from
# effective_physical_memory_bytes(), whose production floor is 256 MiB).
_SHRUNK_TREE_BYTES = 64 * 1024


def test_whale_add_bypasses_sqlite_spill_and_holds_resident(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A parsed tree too large for the hot cache but within the whale
    ceiling must be held resident in ``_whales`` and must NOT be written to
    the sqlite spill at all (no pickle.dumps paid)."""
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MIN_TREE_BYTES", _SHRUNK_TREE_BYTES)
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MAX_TREE_BYTES", _SHRUNK_TREE_BYTES)

    archive_root = tmp_path / "archive"
    _small_ids, whale_id = build_whale_bearing_corpus(
        archive_root,
        small_raw_count=1,
        small_avg_payload_bytes=WHALE_BEARING_SHAPE["small_avg_payload_bytes"],
        whale_payload_bytes=200_000,
    )
    with (
        ArchiveStore.open_existing(archive_root, read_only=False) as archive,
        revision_backfill._ParsedSessionSpill(archive_root) as spill,
    ):
        sessions, payload_bytes, _kind = revision_backfill._parse_retained_raw(archive, whale_id)
        assert estimate_parsed_tree_bytes(sessions) > spill._decoded_budget, (
            "fixture must actually exceed the shrunk hot-cache budget to exercise the whale path"
        )

        spill.add(whale_id, sessions, payload_bytes=payload_bytes)

        assert whale_id in spill._whales
        assert whale_id not in spill._decoded
        row = spill.conn.execute("SELECT COUNT(*) FROM parsed_sessions WHERE raw_id = ?", (whale_id,)).fetchone()
        assert row is not None and row[0] == 0, "whale must bypass the sqlite spill write entirely"

        reloaded_sessions, reloaded_payload_bytes = spill.for_raw(archive, whale_id)
        assert reloaded_sessions is sessions, "whale reload must return the same resident objects, no round trip"
        assert reloaded_payload_bytes == payload_bytes


def test_whale_exceeding_whale_ceiling_falls_back_to_sqlite_spill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Correctness-over-speed fallback: a tree too large for EITHER tier
    (hot cache or whale ceiling) must still be spilled to sqlite exactly as
    the pre-lever code did."""
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MIN_TREE_BYTES", _SHRUNK_TREE_BYTES)
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MAX_TREE_BYTES", _SHRUNK_TREE_BYTES)
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_WHALE_CACHE_MAX_TREE_BYTES", _SHRUNK_TREE_BYTES)

    archive_root = tmp_path / "archive"
    _small_ids, whale_id = build_whale_bearing_corpus(
        archive_root,
        small_raw_count=1,
        small_avg_payload_bytes=WHALE_BEARING_SHAPE["small_avg_payload_bytes"],
        whale_payload_bytes=200_000,
    )
    with (
        ArchiveStore.open_existing(archive_root, read_only=False) as archive,
        revision_backfill._ParsedSessionSpill(archive_root) as spill,
    ):
        sessions, payload_bytes, _kind = revision_backfill._parse_retained_raw(archive, whale_id)

        spill.add(whale_id, sessions, payload_bytes=payload_bytes)

        assert whale_id not in spill._whales
        assert whale_id not in spill._decoded
        row = spill.conn.execute("SELECT COUNT(*) FROM parsed_sessions WHERE raw_id = ?", (whale_id,)).fetchone()
        assert row is not None and row[0] > 0, "must fall back to the sqlite spill when the whale ceiling is exceeded"

        reloaded_sessions, reloaded_payload_bytes = spill.for_raw(archive, whale_id)
        assert reloaded_payload_bytes == payload_bytes
        assert reloaded_sessions[0].provider_session_id == sessions[0].provider_session_id
        assert [m.text for m in reloaded_sessions[0].messages] == [m.text for m in sessions[0].messages]


def test_whale_eviction_degrades_to_sqlite_spill_courtesy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When multiple whales together exceed the whale budget, the entry
    evicted from the resident tier must be written to the sqlite spill
    (not silently dropped) so a later ``for_raw`` still finds it without a
    full reparse from raw bytes."""
    # Each whale's tree is ~83KB (measured for 20,000-char payload text under
    # this estimator); a 16KB decoded budget classifies either as a whale,
    # and a 140KB whale ceiling holds exactly one at a time but not two.
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MIN_TREE_BYTES", 16 * 1024)
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MAX_TREE_BYTES", 16 * 1024)
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_WHALE_CACHE_MAX_TREE_BYTES", 140_000)

    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        first_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=_codex_payload_of_size("whale-a", 20_000),
            source_path="odm1/whale-a.jsonl",
            acquired_at_ms=1,
        )
        second_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=_codex_payload_of_size("whale-b", 20_000),
            source_path="odm1/whale-b.jsonl",
            acquired_at_ms=2,
        )

    with (
        ArchiveStore.open_existing(archive_root, read_only=False) as archive,
        revision_backfill._ParsedSessionSpill(archive_root) as spill,
    ):
        first_sessions, first_payload_bytes, _kind = revision_backfill._parse_retained_raw(archive, first_id)
        spill.add(first_id, first_sessions, payload_bytes=first_payload_bytes)
        assert first_id in spill._whales

        second_sessions, second_payload_bytes, _kind = revision_backfill._parse_retained_raw(archive, second_id)
        spill.add(second_id, second_sessions, payload_bytes=second_payload_bytes)

        # The second whale evicted the first from the resident tier.
        assert second_id in spill._whales
        assert first_id not in spill._whales

        # But the first must have been degraded into the sqlite spill on
        # eviction, not dropped -- for_raw() must still resolve it without
        # touching _parse_retained_raw again (proven by content equality
        # despite the archive already being past that raw in the loop).
        row = spill.conn.execute("SELECT COUNT(*) FROM parsed_sessions WHERE raw_id = ?", (first_id,)).fetchone()
        assert row is not None and row[0] > 0, "evicted whale must be degraded into the sqlite spill, not dropped"

        reloaded_sessions, reloaded_payload_bytes = spill.for_raw(archive, first_id)
        assert reloaded_payload_bytes == first_payload_bytes
        assert [m.text for m in reloaded_sessions[0].messages] == [m.text for m in first_sessions[0].messages]


def _codex_payload_of_size(session_id: str, text_len: int) -> bytes:
    session_meta = (
        json.dumps(
            {"type": "session_meta", "payload": {"id": session_id, "timestamp": "2026-06-01T00:00:00Z"}},
            separators=(",", ":"),
        )
        + "\n"
    )
    response_item = (
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "one",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "x" * text_len}],
                },
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    return (session_meta + response_item).encode()


def test_backfill_replays_whale_bearing_page_byte_identical_to_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end proof (through ``backfill_historical_revision_evidence``,
    not the ``_ParsedSessionSpill`` internals) that a whale-bearing rebuild
    page still replays every session -- large and small -- to correct
    content when the whale-residency lever is active. No schema change, no
    output drift: counts and message text must match what an equivalent
    small-only corpus already proves the pipeline produces."""
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MIN_TREE_BYTES", _SHRUNK_TREE_BYTES)
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MAX_TREE_BYTES", _SHRUNK_TREE_BYTES)

    archive_root = tmp_path / "archive"
    small_raw_ids, whale_raw_id = build_whale_bearing_corpus(
        archive_root,
        small_raw_count=6,
        small_avg_payload_bytes=5_000,
        whale_payload_bytes=200_000,
    )

    result = backfill_historical_revision_evidence(archive_root, commit_batch_size=200, replay_commit_batch_size=1)

    assert result.scanned == len(small_raw_ids) + 1
    assert result.replayed_logical_sources == len(small_raw_ids) + 1
    assert result.quarantined == 0

    with sqlite3.connect(archive_root / "index.db") as conn:
        session_ids = {row[0] for row in conn.execute("SELECT raw_id FROM sessions")}
        assert session_ids == {*small_raw_ids, whale_raw_id}
        for raw_id in [*small_raw_ids, whale_raw_id]:
            message_count = conn.execute("SELECT message_count FROM sessions WHERE raw_id = ?", (raw_id,)).fetchone()
            assert message_count == (1,)


def _pipeline_equivalence_corpus(root: Path) -> None:
    """Mixed corpus exercising BOTH replay phases: independent single-session
    raws (byte-proven cohorts) plus a multi-session bundle (membership
    cohorts), so the pipelined decode is proven over each ``for_raw`` call
    site in ``backfill_historical_revision_evidence``."""
    bootstrap_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for index in range(10):
            payload = _bundle(_chatgpt_session(f"pipe-{index}", f"hello {index}", f"world {index}"))
            archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=payload,
                source_path=f"pipe-{index}.json",
                acquired_at_ms=index,
            )
        bundle = _bundle(
            _chatgpt_session("pipe-shared-a", "alpha", "beta"),
            _chatgpt_session("pipe-shared-b", "gamma", "delta"),
        )
        archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=bundle,
            source_path="pipe-bundle.json",
            acquired_at_ms=100,
        )


def _index_content_manifest(root: Path) -> dict[str, list[tuple[object, ...]]]:
    """Full ordered row dump of the content tables the replay writes --
    the same equivalence currency as PR #3469's MANIFESTS IDENTICAL proof."""
    order_column = {
        "sessions": "session_id",
        "messages": "message_id",
        "blocks": "block_id",
        "session_links": "src_session_id, dst_origin, dst_native_id, link_type",
    }
    with sqlite3.connect(root / "index.db") as conn:
        return {
            table: conn.execute(f"SELECT * FROM {table} ORDER BY {order}").fetchall()
            for table, order in order_column.items()
        }


@pytest.mark.parametrize(
    "spill_payload_cap",
    [1, 512 * 1024 * 1024],
    ids=["reparse-fallback-lane", "sqlite-spill-lane"],
)
def test_spill_decode_keeps_canonical_archive_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spill_payload_cap: int
) -> None:
    """Shared compute window widths preserve replay across disk-spill policies."""
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MIN_TREE_BYTES", 1)
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MAX_TREE_BYTES", 1)
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_WHALE_CACHE_MAX_TREE_BYTES", 1)

    serial_root = tmp_path / "serial"
    pipelined_root = tmp_path / "pipelined"
    for root in (serial_root, pipelined_root):
        _pipeline_equivalence_corpus(root)

    serial_result = backfill_historical_revision_evidence(serial_root, ingest_workers=1)
    pipelined_result = backfill_historical_revision_evidence(pipelined_root, ingest_workers=2)

    assert serial_result == pipelined_result
    assert _index_content_manifest(serial_root) == _index_content_manifest(pipelined_root)


def test_spill_decode_respects_batched_replay_commits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Batched replay commits preserve the canonical rows with bounded spill memory."""
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MIN_TREE_BYTES", 1)
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MAX_TREE_BYTES", 1)
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_WHALE_CACHE_MAX_TREE_BYTES", 1)

    serial_root = tmp_path / "serial"
    pipelined_root = tmp_path / "pipelined"
    for root in (serial_root, pipelined_root):
        _pipeline_equivalence_corpus(root)

    serial_result = backfill_historical_revision_evidence(serial_root)
    pipelined_result = backfill_historical_revision_evidence(
        pipelined_root,
        commit_batch_size=200,
        replay_commit_batch_size=200,
    )

    assert serial_result == pipelined_result
    assert _index_content_manifest(serial_root) == _index_content_manifest(pipelined_root)


def _codex_session_payload(
    session_id: str,
    message_texts: list[str],
    *,
    forked_from_id: str | None = None,
    timestamp_adversarial: bool = False,
) -> bytes:
    """Build a codex JSONL raw with one message per ``message_texts`` entry.

    Mirrors how a real Codex resume payload looks: a ``session_meta`` record
    carrying ``forked_from_id`` when this session is a resume/fork, followed
    by ``response_item`` message records. Passing the SAME leading
    ``message_texts`` for a parent and one of its children (plus extra tail
    entries on the child) reproduces the on-disk shape #2467's deferred-tail
    extraction exists for: the child's JSONL physically re-contains the
    parent's entire prefix. ``timestamp_adversarial`` gives each message a
    clock that runs backwards against its content position, keeping replay
    parity honest about topology rather than accidentally relying on time.
    """
    meta_payload: dict[str, object] = {"id": session_id, "timestamp": "2026-06-01T00:00:00Z"}
    if forked_from_id is not None:
        meta_payload["forked_from_id"] = forked_from_id
    lines = [json.dumps({"type": "session_meta", "payload": meta_payload}, separators=(",", ":"))]
    for position, text in enumerate(message_texts):
        message: dict[str, object] = {
            "type": "message",
            "id": f"m{position}",
            "role": "user" if position % 2 == 0 else "assistant",
            "content": [{"type": "input_text", "text": text}],
        }
        if timestamp_adversarial:
            # Content position is authoritative even when the observed clock
            # is reversed. Both replay schedules consume the same evidence.
            message["timestamp"] = f"2026-01-01T00:00:{59 - position:02d}Z"
        lines.append(
            json.dumps(
                {
                    "type": "response_item",
                    "payload": message,
                },
                separators=(",", ":"),
            )
        )
    return ("\n".join(lines) + "\n").encode()


def _seed_lineage_fixture(root: Path, *, n_children: int, timestamp_adversarial: bool = False) -> None:
    """One parent (native_id sorts LAST lexicographically) plus N children
    (native_ids sort BEFORE the parent) that each replay the parent's full
    message prefix plus one new tail message -- a real Codex resume shape.
    """
    bootstrap_archive_root(root)
    parent_native_id = "zparent"
    parent_texts = [f"parent-{i}" for i in range(4)]
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=_codex_session_payload(
                parent_native_id,
                parent_texts,
                timestamp_adversarial=timestamp_adversarial,
            ),
            source_path=f"{parent_native_id}.jsonl",
            acquired_at_ms=1,
        )
        for index in range(n_children):
            child_native_id = f"achild{index}"
            child_texts = [*parent_texts, f"child-{index}-tail"]
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_codex_session_payload(
                    child_native_id,
                    child_texts,
                    forked_from_id=parent_native_id,
                    timestamp_adversarial=timestamp_adversarial,
                ),
                source_path=f"{child_native_id}.jsonl",
                acquired_at_ms=2 + index,
            )


def _lexicographic_replay_schedule(
    logical_keys: set[str],
    archive: ArchiveStore,
    spill: Any,
    archive_root: Path,
) -> revision_backfill.ReplaySchedule:
    """The pre-polylogue-5q2u schedule: sorted keys, no lineage edges. Used
    to force the old order so a lineage-aware run can be compared against it."""
    order = tuple(sorted(logical_keys))
    return revision_backfill.ReplaySchedule(
        order=order,
        topology=dict.fromkeys(order, revision_backfill.ReplayTopologyState.ROOT),
        parent_of=dict.fromkeys(order, None),
    )


def test_lineage_aware_replay_schedule_visits_parent_before_children(tmp_path: Path) -> None:
    """polylogue-5q2u: roots first, then each child only after its parent --
    NOT the lexicographic order a plain ``sorted()`` would produce (the
    parent's native id, "zparent", sorts LAST here)."""
    root = tmp_path / "archive"
    _seed_lineage_fixture(root, n_children=5)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        revision_backfill._census_historical_revision_evidence(
            archive,
            revision_backfill._ParsedSessionSpill(root),
            selected_raw_ids=None,
            max_payload_bytes=None,
        )
        archive.commit()
        _expanded, logical_keys = archive.expand_raw_membership_selection(None)
        with revision_backfill._ParsedSessionSpill(root) as spill:
            schedule = _lineage_aware_replay_schedule(set(logical_keys), archive, spill, root)
            order = list(schedule.order)
            topology = dict(schedule.topology)

    assert order[0] == "codex-session:zparent"
    parent_position = order.index("codex-session:zparent")
    for index in range(5):
        child_key = f"codex-session:achild{index}"
        assert child_key in order
        assert order.index(child_key) > parent_position
    # Lexicographic order would have put every child before the parent.
    assert sorted(logical_keys)[0] != "codex-session:zparent"
    assert topology["codex-session:zparent"] is revision_backfill.ReplayTopologyState.ROOT
    assert {topology[f"codex-session:achild{index}"] for index in range(5)} == {
        revision_backfill.ReplayTopologyState.DESCENDANT
    }


def test_lineage_aware_replay_schedule_falls_back_for_unresolvable_parent(tmp_path: Path) -> None:
    """A parent outside this call's ``logical_keys`` set (missing/external/
    cross-batch) must not crash or drop the child -- it degrades to a
    typed ``UNRESOLVED_PARENT`` root at its lexicographic position."""
    root = tmp_path / "archive"
    bootstrap_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for native_id in ("zorphan", "aorphan"):
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_codex_session_payload(native_id, ["only-message"], forked_from_id="never-ingested-parent"),
                source_path=f"{native_id}.jsonl",
                acquired_at_ms=1,
            )
        with revision_backfill._ParsedSessionSpill(root) as spill:
            revision_backfill._census_historical_revision_evidence(
                archive, spill, selected_raw_ids=None, max_payload_bytes=None
            )
            archive.commit()
            schedule = _lineage_aware_replay_schedule(
                {"codex-session:zorphan", "codex-session:aorphan"}, archive, spill, root
            )
    order = list(schedule.order)
    assert sorted(order) == ["codex-session:aorphan", "codex-session:zorphan"]
    # Neither key's parent is in the set, so both are roots -- fallback
    # degrades to lexicographic order among them.
    assert order == ["codex-session:aorphan", "codex-session:zorphan"]
    assert set(schedule.topology.values()) == {revision_backfill.ReplayTopologyState.UNRESOLVED_PARENT}
    assert set(schedule.parent_of.values()) == {None}


def test_lineage_aware_replay_schedule_reduces_deferred_tail_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-5q2u AC1: lineage-aware replay must trigger the #2467
    deferred-tail/orphaned-child normalization path (``_reextract_prefix_tail_db``)
    strictly less often than the previous lexicographic order for a
    representative parent-with-many-children fixture, where the parent's
    native id sorts lexicographically AFTER its children's.

    Anti-vacuity: reverting the ``_lineage_aware_replay_schedule`` call at the
    ``for logical_key in ...:`` call site back to ``sorted(logical_keys)``
    makes this test fail (both counts become equal and >0, since every
    child would then replay before the parent it depends on).
    """
    lineage_root = tmp_path / "lineage"
    lexicographic_root = tmp_path / "lexicographic"
    n_children = 5
    _seed_lineage_fixture(lineage_root, n_children=n_children)
    _seed_lineage_fixture(lexicographic_root, n_children=n_children)

    def _count_deferred_tail_hits(root: Path, *, force_lexicographic: bool) -> int:
        calls = 0
        original = archive_tier_write._reextract_prefix_tail_db

        def counting_wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(archive_tier_write, "_reextract_prefix_tail_db", counting_wrapper)
        if force_lexicographic:
            monkeypatch.setattr(
                revision_backfill,
                "_lineage_aware_replay_schedule",
                _lexicographic_replay_schedule,
            )
        backfill_historical_revision_evidence(root)
        monkeypatch.undo()
        return calls

    lexicographic_hits = _count_deferred_tail_hits(lexicographic_root, force_lexicographic=True)
    lineage_hits = _count_deferred_tail_hits(lineage_root, force_lexicographic=False)

    assert lexicographic_hits == n_children, (
        f"expected every one of the {n_children} children (native ids sorting before "
        f"the parent's) to hit the deferred-tail path under lexicographic order, got {lexicographic_hits}"
    )
    assert lineage_hits == 0, (
        f"lineage-aware order should replay the parent before any child, avoiding the "
        f"deferred-tail path entirely; got {lineage_hits} hits"
    )


def test_lineage_aware_replay_order_preserves_outcome_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """polylogue-5q2u AC2: lineage-aware scheduling must not change WHAT gets
    replayed/adopted -- only the order. Two archives seeded identically,
    replayed once under lineage order and once forced to the previous
    lexicographic order, must reach byte-identical index.db content
    (sessions/messages/blocks/session_links), matching the equivalence
    currency ``_index_content_manifest`` already uses for other replay-order
    equivalence proofs in this file (e.g.
    ``test_spill_decode_respects_batched_replay_commits``).
    """
    lineage_root = tmp_path / "lineage"
    lexicographic_root = tmp_path / "lexicographic"
    _seed_lineage_fixture(lineage_root, n_children=5, timestamp_adversarial=True)
    _seed_lineage_fixture(lexicographic_root, n_children=5, timestamp_adversarial=True)

    lineage_result = backfill_historical_revision_evidence(lineage_root)

    monkeypatch.setattr(
        revision_backfill,
        "_lineage_aware_replay_schedule",
        _lexicographic_replay_schedule,
    )
    lexicographic_result = backfill_historical_revision_evidence(lexicographic_root)

    assert lineage_result.replayed_logical_sources == lexicographic_result.replayed_logical_sources
    assert lineage_result.quarantined == lexicographic_result.quarantined
    assert lineage_result.adoption_deferred == lexicographic_result.adoption_deferred
    lineage_manifest = _index_content_manifest(lineage_root)
    lexicographic_manifest = _index_content_manifest(lexicographic_root)
    # Topology is the acceptance boundary: replay order may change when
    # deferred-tail work runs, never which parent edge is persisted. The raw
    # fixture reverses timestamps against message positions so a wall-clock
    # ordering shortcut cannot make these manifests agree by luck.
    assert lineage_manifest["session_links"] == lexicographic_manifest["session_links"]
    assert lineage_manifest == lexicographic_manifest


# -----------------------------------------------------------------------------
# ANTIGRAVITY .pb REPLAY DRIFT (bd polylogue-t1vl6)
# -----------------------------------------------------------------------------


def test_antigravity_pb_replay_refuses_a_drifted_trajectory(tmp_path: Path) -> None:
    """A rewritten ``.pb`` is a typed refusal, not a silent substitution.

    Antigravity decoding needs a live language-server client, so replay
    re-derives from the file on disk. Antigravity rewrites
    ``conversations/<cascade_id>.pb`` in place, so the existence and
    session-count guards both pass for a CHANGED file and current content
    would be replayed under an older revision's ``raw_id``.

    Anti-vacuity: remove the content-hash comparison and this call no longer
    raises -- it proceeds into ``iter_language_server_exports`` against bytes
    that are not the retained blob.
    """
    from polylogue.core.enums import Provider
    from polylogue.sources.revision_backfill import AntigravityTrajectoryDriftError, _parse_one_raw

    conversations = tmp_path / "conversations"
    conversations.mkdir(parents=True)
    trajectory = conversations / "cascade-1.pb"
    trajectory.write_bytes(b"live bytes after an in-place rewrite")

    with pytest.raises(AntigravityTrajectoryDriftError):
        _parse_one_raw(Provider.ANTIGRAVITY, b"retained bytes", str(trajectory))


def test_frozen_shard_replay_degrades_named_for_prefix_sharing_child(tmp_path: Path) -> None:
    """polylogue-k00uq: a prefix-sharing child must not abort the whole replay.

    The lineage fixture is a real Codex resume shape: the child's raw
    physically re-contains the parent's entire prefix, so the writer slices
    it against the already-archived parent and refuses the shard's rows
    (sealed by a parse worker with no DB read, therefore describing the
    UNSLICED session). Shard replay must record that as a counted, named
    degradation and write the unit inline -- never skip it, never abort.

    Anti-vacuity: restoring the bare ``prepared_required_raw_ids`` marking
    without the ``PreparedSessionWriteRefusedError`` handler in
    ``backfill_historical_revision_evidence`` makes this red -- the refusal
    escapes the ``sqlite3.IntegrityError``-only guard and the call raises
    instead of returning a result at all.
    """
    root = tmp_path / "lineage-shard"
    bootstrap_archive_root(root)
    parent_texts = [f"parent-{index}" for index in range(4)]
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for native_id, texts, forked_from in (
            ("zparent", parent_texts, None),
            ("achild0", [*parent_texts, "child-0-tail"], "zparent"),
        ):
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_codex_session_payload(native_id, texts, forked_from_id=forked_from),
                source_path=f"{native_id}.jsonl",
                acquired_at_ms=1,
                revision=RawRevisionEnvelope(
                    logical_source_key=f"codex-session:{native_id}",
                    kind=RawRevisionKind.FULL,
                    source_revision=f"{native_id}-v1",
                    acquisition_generation=0,
                    authority=RawRevisionAuthority.BYTE_PROVEN,
                ),
            )
            archive.classify_raw_revision_cohort_for_rebuild_repair(f"codex-session:{native_id}")
    census_historical_revision_evidence(root)
    generation = IndexGenerationStore.for_archive_root(root).create(source_snapshot="lineage-shard-test")

    result = backfill_historical_revision_evidence(
        Path(generation.index_path).parent,
        owned_inactive_generation=(generation.generation_id, generation.owner_id),
        use_session_shards=True,
    )

    assert result.shard_lowering_degraded == 1
    with sqlite3.connect(generation.index_path) as conn:
        session_ids = {row[0] for row in conn.execute("SELECT session_id FROM sessions")}
        assert session_ids == {"codex-session:zparent", "codex-session:achild0"}
        link = conn.execute(
            """SELECT resolved_dst_session_id, branch_point_message_id, inheritance
               FROM session_links WHERE src_session_id = ?""",
            ("codex-session:achild0",),
        ).fetchone()
        assert link is not None
        assert link[0] == "codex-session:zparent"
        assert link[1] is not None
        assert link[2] == "prefix-sharing"
        # The sliced tail is what landed: only the child's own divergent
        # message, with the parent's four-message prefix left to the parent.
        assert (
            conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", ("codex-session:achild0",)).fetchone()[0]
            == 1
        )


def _owned_generation_corpus(root: Path, *, raw_count: int, snapshot: str) -> IndexGeneration:
    """Seed ``raw_count`` byte-proven Codex raws and open an owned generation.

    Codex is the provider whose replay enrichment reads the index tier
    (``_replay_enrichment_reads_index``), which is what makes this the shape
    that exposed polylogue-cz17d.
    """
    bootstrap_archive_root(root)
    build_independent_raw_corpus(root, raw_count=raw_count, avg_payload_bytes=2_000, authoritative_source=True)
    census_historical_revision_evidence(root)
    return IndexGenerationStore.for_archive_root(root).create(source_snapshot=snapshot)


def test_owned_generation_spill_decode_keeps_canonical_archive_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Owned-generation replay preserves rows across cached and reparsed spill lanes."""
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MIN_TREE_BYTES", 1)
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MAX_TREE_BYTES", 1)
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_WHALE_CACHE_MAX_TREE_BYTES", 1)

    manifests: list[dict[str, list[tuple[object, ...]]]] = []
    results = []
    for name, _payload_budget in (("cached", 512 * 1024 * 1024), ("reparsed", 1)):
        root = tmp_path / name
        generation = _owned_generation_corpus(
            root,
            raw_count=4,
            snapshot=f"owned-equivalence-{name}",
        )
        results.append(
            backfill_historical_revision_evidence(
                Path(generation.index_path).parent,
                owned_inactive_generation=(generation.generation_id, generation.owner_id),
                use_session_shards=True,
            )
        )
        manifests.append(_index_content_manifest(Path(generation.index_path).parent))

    serial_result, pipelined_result = results
    assert serial_result == pipelined_result
    assert manifests[0] == manifests[1]


def _cost_probe_descriptors(count: int, *, blob_hash: str | None = None) -> dict[str, Any]:
    """``count`` retained-raw descriptors, one dedup group each unless shared."""
    return {
        f"raw-{index}": (
            Provider.CODEX,
            blob_hash if blob_hash is not None else f"hash-{index}",
            f"path-{index}.jsonl",
            RawRevisionKind.FULL,
            10,
        )
        for index in range(count)
    }


class _CostProbeArchive:
    """Protocol-shaped stand-in; enrichment skips anything not an ArchiveStore."""

    def __init__(self, descriptors: dict[str, Any], root: Path) -> None:
        self._descriptors = descriptors
        self.archive_root = root
        self.source_db_path = root / "source.db"

    def raw_revision_descriptor(self, raw_id: str) -> Any:
        return self._descriptors[raw_id]


def test_streaming_census_parse_costs_one_group_not_one_page(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """polylogue-4j9j: a census page must not hold every parsed session at once.

    This is a cost assertion, not a behavioural one. ``_parse_retained_raws``
    returned a complete dict, so the page's whole parsed tree was resident
    while the apply loop walked it one raw at a time. The streaming resolver
    parses a raw when it is read and drops it when its dedup group is spent,
    so the retained-parse count is a constant rather than the page size.

    Anti-vacuity: resolve every outcome up front (an eager resolver) and
    ``parsed`` is already full before the first read; stop releasing a spent
    group and ``live_group_count`` climbs with every read instead of
    returning to zero.
    """
    descriptors = _cost_probe_descriptors(8)
    parsed: list[str] = []

    def fake_parse(raw_id: str, *args: object) -> tuple[str, list[ParsedSession], None]:
        parsed.append(raw_id)
        return raw_id, [], None

    monkeypatch.setattr(revision_backfill, "census_parse_worker", fake_parse)
    archive = _CostProbeArchive(descriptors, tmp_path)

    with revision_backfill.stream_retained_raws(
        archive,  # type: ignore[arg-type]
        list(descriptors),
        ingest_workers=1,
    ) as outcomes:
        assert len(parsed) <= 2, "only the bounded window may be prepared"
        parses_after_each_read: list[int] = []
        for raw_id in descriptors:
            outcomes[raw_id]
            parses_after_each_read.append(len(parsed))
            # Every group here has a single member, so reading it spends it.
            assert outcomes.live_group_count == 0
        # Exactly one parse per read: the page was never materialized.
        assert all(count <= min(8, read + 1) for read, count in enumerate(parses_after_each_read, 1))
    assert sorted(parsed) == sorted(descriptors)


def test_streaming_release_waits_for_a_dedup_group_last_member(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A shared blob is parsed once and retained exactly until its last reader.

    Releasing on first read would re-parse every duplicate, which is the cost
    polylogue-869u's dedup exists to avoid; never releasing is the leak this
    bead exists to remove.
    """
    descriptors = _cost_probe_descriptors(3, blob_hash="shared")
    # Path-independent provider: all three rows share one dedup group.
    descriptors = {
        raw_id: (provider, blob, "same.jsonl", kind, size)
        for raw_id, (provider, blob, _path, kind, size) in descriptors.items()
    }
    parsed: list[str] = []

    def fake_parse(raw_id: str, *args: object) -> tuple[str, list[ParsedSession], None]:
        parsed.append(raw_id)
        return raw_id, [], None

    monkeypatch.setattr(revision_backfill, "census_parse_worker", fake_parse)
    archive = _CostProbeArchive(descriptors, tmp_path)

    with revision_backfill.stream_retained_raws(
        archive,  # type: ignore[arg-type]
        list(descriptors),
        ingest_workers=1,
    ) as outcomes:
        outcomes["raw-0"]
        assert outcomes.live_group_count == 1
        outcomes["raw-1"]
        assert outcomes.live_group_count == 1
        outcomes["raw-2"]
        assert outcomes.live_group_count == 0
    assert parsed == ["raw-0"], "the shared blob must still be parsed exactly once"


def test_streaming_parse_dispatch_is_bounded_in_flight(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Lazy reads alone do not bound memory -- dispatch is throttled too.

    A ``ThreadPoolExecutor`` resolves a future whether or not anyone reads it,
    so without a submission window the parsed graphs simply accumulate inside
    completed futures. The window is ``ingest_workers *
    _INFLIGHT_PARSES_PER_WORKER``.

    Anti-vacuity: widen ``_max_inflight`` to the page length and
    ``peak_inflight_parses`` becomes the page size.
    """
    descriptors = _cost_probe_descriptors(16)

    def fake_worker(raw_id: str, *args: object) -> tuple[str, list[ParsedSession], None]:
        return (raw_id, [], None)

    monkeypatch.setattr(revision_backfill, "census_parse_worker", fake_worker)
    archive = _CostProbeArchive(descriptors, tmp_path)
    ingest_workers = 2

    with revision_backfill.stream_retained_raws(
        archive,  # type: ignore[arg-type]
        list(descriptors),
        ingest_workers=ingest_workers,
    ) as outcomes:
        for raw_id in descriptors:
            assert not isinstance(outcomes[raw_id], Exception)
        bound = ingest_workers * revision_backfill._INFLIGHT_PARSES_PER_WORKER
        assert outcomes.peak_inflight_parses <= bound
        assert outcomes.peak_inflight_parses < len(descriptors)
        # The window must actually be used, or the bound is met by doing
        # nothing in parallel at all.
        assert outcomes.peak_inflight_parses == bound
        assert outcomes.executor_running
    assert not outcomes.executor_running, "the resolver drains its submitted work scope on exit"


def test_streaming_resolver_drains_owned_work_when_the_body_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """AC2: the pool's lifetime is owned, never left to the garbage collector."""
    descriptors = _cost_probe_descriptors(4)

    def fake_worker(raw_id: str, *args: object) -> tuple[str, list[ParsedSession], None]:
        return (raw_id, [], None)

    monkeypatch.setattr(revision_backfill, "census_parse_worker", fake_worker)
    archive = _CostProbeArchive(descriptors, tmp_path)

    escaped: list[Any] = []
    with pytest.raises(RuntimeError, match="census apply failed"):
        with revision_backfill.stream_retained_raws(
            archive,  # type: ignore[arg-type]
            list(descriptors),
            ingest_workers=2,
        ) as outcomes:
            escaped.append(outcomes)
            outcomes["raw-0"]
            raise RuntimeError("census apply failed")
    assert not escaped[0].executor_running


def test_cold_build_rebuilds_a_session_that_retains_an_agent_work_event(tmp_path: Path) -> None:
    """A retained work event survives the production cold build with its transcript.

    Anti-vacuity: admit the event raw under the transcript's logical key and
    the frozen classification re-derives different byte authority for the
    transcript (``FrozenSourceRemediationRequiredError``); replay the event
    before its transcript and the transcript's fresh write asserts an absent
    session; let the transcript's accepted head refuse the event and the live
    append records nothing; let the event write re-point ``sessions.raw_id``
    or ``content_hash`` and the transcript's accepted head disagrees with its
    materialized session.
    """
    root = tmp_path / "archive"
    build_independent_raw_corpus(root, raw_count=1, avg_payload_bytes=1_000, authoritative_source=True)
    census_historical_revision_evidence(root)
    backfill_historical_revision_evidence(root)
    session_id = "codex-session:amg1-session-000000"
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        appended = archive.append_work_event(
            session_id=session_id,
            event_type="decision",
            payload={"decision": "continue"},
            event_id="evt-cold-build",
            summary="kept going",
        )
    assert appended["content_changed"] is True

    def session_state(
        index_path: Path,
    ) -> tuple[list[Any], list[Any], list[tuple[str, str]], list[Any]]:
        with sqlite3.connect(index_path) as conn:
            header = conn.execute(
                "SELECT session_id, title, created_at_ms, updated_at_ms FROM sessions ORDER BY session_id"
            ).fetchall()
            messages = conn.execute(
                "SELECT message_id FROM messages WHERE session_id = ? ORDER BY position", (session_id,)
            ).fetchall()
            events = conn.execute(
                "SELECT event_type, payload_json FROM session_events WHERE session_id = ? ORDER BY position",
                (session_id,),
            ).fetchall()
            head_agreement = conn.execute(
                """
                SELECT h.logical_source_key, s.raw_id = h.accepted_raw_id AND s.content_hash = h.accepted_content_hash
                FROM raw_revision_heads AS h JOIN sessions AS s ON s.session_id = h.session_id
                ORDER BY h.logical_source_key
                """
            ).fetchall()
        return (
            header,
            messages,
            [(event_type, json.loads(payload)["event_id"]) for event_type, payload in events],
            head_agreement,
        )

    active = session_state(root / "index.db")
    assert active[0][0][0] == session_id
    assert len(active[1]) == 1
    assert active[2] == [("decision", "evt-cold-build")]
    assert active[3] == [(session_id, 1)]

    generation = IndexGenerationStore.for_archive_root(root).create(source_snapshot="work-event-cold-build")
    result = backfill_historical_revision_evidence(
        Path(generation.index_path).parent,
        owned_inactive_generation=(generation.generation_id, generation.owner_id),
    )

    assert result.replayed_logical_sources == 2
    assert session_state(Path(generation.index_path)) == active
