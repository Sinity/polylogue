from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.archive.ingest_flags import (
    COMPACT_BROWSER_CAPTURE_INGEST_FLAG,
    DOM_FALLBACK_INGEST_FLAG,
    NATIVE_BROWSER_CAPTURE_INGEST_FLAG,
)
from polylogue.archive.revision_authority import (
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
    parser_census_identity_measurement,
)
from polylogue.core.enums import Provider
from polylogue.core.errors import SchemaSkew
from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind
from polylogue.sources import prepared_jsonl, revision_backfill
from polylogue.sources.live import WatchSource
from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    clear_cold_build_generation,
    register_cold_build_generation,
)
from polylogue.sources.revision_backfill import (
    LEGACY_PAGE_IMAGE_CENSUS_DETAIL,
    _browser_snapshot_fidelity,
    _is_declared_provider_session_stream,
)
from polylogue.storage.artifacts.inspection import inspect_raw_artifact
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.index_generation import IndexGeneration, IndexGenerationStore
from polylogue.storage.raw_authority import iter_parser_census_logical_keys, raw_authority_parser_fingerprint
from polylogue.storage.sqlite.archive_tiers import revision_governance as archive_revision_governance
from polylogue.storage.sqlite.archive_tiers import write as archive_tier_write
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.reference_seal import ReferenceSealStaleError
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root, run_off_event_loop
from tests.infra.prepared_replay import current_fixture_parser_receipts, publish_fixture_byte_classification
from tests.infra.raw_owner_routes import replay_retained_raws, seed_parser_census
from tests.infra.retained_parser_payloads import (
    _chatgpt_session,
)
from tests.infra.retained_replay import replay_retained_components, replay_retained_components_async
from tests.infra.revision_backfill_benchmark import (
    REVISION_CHAIN_SHAPE,
    build_independent_raw_corpus,
    build_revision_chain_corpus,
)


@pytest.mark.parametrize(
    ("provider", "source_path", "expected"),
    [
        (Provider.CODEX, "2026/10/07/rollout-a.jsonl", True),
        (Provider.CLAUDE_CODE, "-home-user-repo/session.jsonl", True),
        (Provider.CLAUDE_CODE, "history.jsonl", False),
        (Provider.CHATGPT, "sessions/abc.jsonl", False),
    ],
)
def test_declared_provider_session_stream_boundary(provider: Provider, source_path: str, expected: bool) -> None:
    assert _is_declared_provider_session_stream(provider, source_path) is expected


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
    # Representative selection reads only its supplied reader; no keys means no read.
    assert revision_backfill._replay_representative_raw_ids([], cast(Any, None)) == {}

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
            canonical_source_path=source_path,
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
        return raw_id

    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for index in range(2):
            raw_ids.append(write_terminal_non_session(archive, index))
    seed_parser_census(root, raw_ids)

    real_measurement = parser_census_identity_measurement
    measured_raws = 0

    @contextmanager
    def advance_after_measurement(**kwargs: Any) -> Iterator[Any]:
        nonlocal measured_raws
        with real_measurement(**kwargs) as measured:
            measured_raws += 1
            frozen_clock.advance(301)
            yield measured

    monkeypatch.setattr(
        "polylogue.storage.sqlite.archive_tiers.revision_governance.parser_census_identity_measurement",
        advance_after_measurement,
    )
    assert current_fixture_parser_receipts(root, raw_ids) == (True, True)
    assert measured_raws == 2


def test_current_parser_source_census_refuses_reused_rowid_frontier(tmp_path: Path) -> None:
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
            canonical_source_path=source_path,
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
        return raw_id

    with ArchiveStore.open_existing(root, read_only=False) as archive:
        original_raw_id = write_terminal_non_session(archive, 0)
    seed_parser_census(root, [original_raw_id])
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        original_rowid = int(
            archive._ensure_source_conn()
            .execute("SELECT rowid FROM raw_sessions WHERE raw_id = ?", (original_raw_id,))
            .fetchone()[0]
        )

    replacement_raw_id: str | None = None

    def replace_after_observation() -> None:
        nonlocal replacement_raw_id
        # An out-of-band delete: ordinary Source custody refuses deletion.
        with closing(sqlite3.connect(root / "source.db")) as external:
            external.execute("PRAGMA foreign_keys=ON")
            external.execute("DELETE FROM raw_sessions WHERE raw_id = ?", (original_raw_id,))
            external.commit()
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            replacement_raw_id = write_terminal_non_session(archive, 1)
            replacement_rowid = archive.source_connection.execute(
                "SELECT rowid FROM raw_sessions WHERE raw_id = ?", (replacement_raw_id,)
            ).fetchone()[0]
            assert int(replacement_rowid) == original_rowid

    with pytest.raises(ReferenceSealStaleError):
        current_fixture_parser_receipts(root, [original_raw_id], after_prepare=replace_after_observation)
    assert replacement_raw_id is not None and replacement_raw_id != original_raw_id


def _bundle(*sessions: dict[str, object]) -> bytes:
    return json.dumps(list(sessions), sort_keys=True).encode()


@pytest.mark.parametrize("native_id,session_count", [("native-singleton", 1), (None, 1), (None, 2)])
def test_source_census_preserves_acquired_grouped_identity(
    tmp_path: Path, native_id: str | None, session_count: int
) -> None:
    """Parser cardinality cannot turn grouped acquired bytes into a native revision."""
    bootstrap_archive_root(tmp_path)
    sessions = tuple(
        _chatgpt_session(native_id or f"grouped-{index}", "original retained message") for index in range(session_count)
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(*sessions),
            source_path="synthetic/conversations.json",
            canonical_source_path="synthetic/conversations.json",
            native_id=native_id,
            acquired_at_ms=1,
        )
    result = replay_retained_components(tmp_path, selected_raw_ids=[raw_id])
    assert result.scanned == 1
    with ArchiveStore.open_existing(tmp_path) as archive:
        assert archive.raw_native_id(raw_id) == native_id
        members = archive.source_connection.execute(
            "SELECT logical_source_key, provider_session_id FROM raw_session_memberships WHERE raw_id=? ORDER BY logical_source_key",
            (raw_id,),
        ).fetchall()
        if native_id is None:
            assert [tuple(row) for row in members] == [
                (f"chatgpt-export:grouped-{index}", f"grouped-{index}") for index in range(session_count)
            ]
        else:
            assert members == []
        receipt = archive.source_connection.execute(
            "SELECT status, logical_keys_json FROM raw_authority_parser_census WHERE raw_id=?", (raw_id,)
        ).fetchone()
        assert receipt is not None and receipt[0] == "complete"
        assert tuple(iter_parser_census_logical_keys(receipt[1])) == tuple(
            f"chatgpt-export:{native_id or f'grouped-{index}'}" for index in range(session_count)
        )
        # One pass settles the census and publishes every grouped member.
        assert archive.count_sessions() == session_count


def test_owned_inactive_generation_binds_the_prepared_session_rows(tmp_path: Path) -> None:
    """The owned candidate writes the rows prepared off the writer, never lowers inline.

    Canonical retained preparation seals each session's message and block
    rows (``PreparedSessionWrite.rows``) before writer admission; the writer
    validates and binds that carrier. Anti-vacuity: dropping the carrier so
    the writer lowers inline leaves the logical projection green but hands
    the full replace no ``PreparedSessionRows``.
    """
    import polylogue.storage.sqlite.archive_tiers.write as archive_tier_write

    root = tmp_path / "owned-shard"
    bootstrap_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("sealed-shard", "hello", "world")),
            source_path="export/conversations.json",
            canonical_source_path="export/conversations.json",
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                logical_source_key="chatgpt-export:sealed-shard",
                kind=RawRevisionKind.FULL,
                source_revision="sealed-shard-v1",
                acquisition_generation=0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        publish_fixture_byte_classification(archive, "chatgpt-export:sealed-shard")
    replay_retained_components(root)
    source_before = (root / "source.db").read_bytes()
    # The owned candidate is the registered cold-build destination; retained
    # replay refuses any other generation.
    with write_lease("test.owned-shard.generation", archive_root=root):
        cold_build = ColdBuildGeneration.begin(
            root,
            reason="test-owned-shard",
            observed=ColdBuildGeneration.observe_source_baseline((WatchSource("fixture", root / "absent"),)),
        )
    register_cold_build_generation(cold_build)
    carriers: list[object] = []
    original_replace = archive_tier_write._replace_full_session_messages_and_blocks

    def recording_replace(*args: Any, **kwargs: Any) -> Any:
        carriers.append(kwargs.get("prepared"))
        return original_replace(*args, **kwargs)

    try:
        generation = cold_build.generation
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(archive_tier_write, "_replace_full_session_messages_and_blocks", recording_replace)
            receipts = replay_retained_raws(root)

        assert sum(receipt.replayed_logical_sources for receipt in receipts) == 1
        assert (root / "source.db").read_bytes() == source_before
        # Inactive replay defers reader models until the production candidate
        # readiness pass; inspecting FTS before that pass is premature.
        with write_lease("test.owned-shard.readiness", archive_root=root):
            cold_build.prepare_promotion_candidate()
        assert (root / "source.db").read_bytes() == source_before
        with sqlite3.connect(generation.index_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] > 0
        assert len(carriers) == 1
        assert isinstance(carriers[0], archive_tier_write.PreparedSessionRows)
    finally:
        clear_cold_build_generation()
        with write_lease("test.owned-shard.discard", archive_root=root):
            cold_build.discard()


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
            canonical_source_path="previous-parser-semantics.jsonl",
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
    monkeypatch.setattr(archive_revision_governance, "raw_authority_parser_fingerprint", lambda: changed_fingerprint)

    assert current_fixture_parser_receipts(tmp_path, [raw_id]) == (False,)


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
            canonical_source_path=str(trajectory_path),
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
            canonical_source_path="after-page-image.jsonl",
            acquired_at_ms=2,
        )

    result = replay_retained_components(tmp_path)

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
            canonical_source_path=source_path,
            acquired_at_ms=1,
        )

    replay_retained_components(tmp_path)

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
    source_path = str(tmp_path / ".claude" / "projects" / "proj" / "subagents" / "workflows" / "wf" / "journal.jsonl")

    def acquire() -> str:
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.UNKNOWN,
                payload=(
                    b'{"type":"file-history-snapshot","messageId":"history-message",'
                    b'"sessionId":"history-only-session","snapshot":{"trackedFileBackups":{}}}\n'
                ),
                source_path=source_path,
                canonical_source_path=source_path,
                acquired_at_ms=1,
            )

    raw_id = run_off_event_loop(acquire)
    await replay_retained_components_async(tmp_path)

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
            canonical_source_path=source_path,
            acquired_at_ms=1,
        )
        raw_b = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=payload_b,
            source_path=source_path,
            canonical_source_path=source_path,
            acquired_at_ms=2,
        )
        assert (
            archive.write_raw_payload(
                provider=Provider.CLAUDE_CODE,
                payload=payload_a,
                source_path=source_path,
                canonical_source_path=source_path,
                acquired_at_ms=3,
            )
            == raw_a
        )

    replay_retained_components(tmp_path)

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
            canonical_source_path="session.jsonl",
            acquired_at_ms=1,
        )
        baseline_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline,
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            acquired_at_ms=2,
        )
        legacy_append_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"response_item","payload":{"type":"message","id":"legacy-suffix"}}\n',
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=3,
        )

    result = replay_retained_components(tmp_path)

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
    # the current prepared parser receipt gate, so the fragment was
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

    replay_retained_components(tmp_path)

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
            canonical_source_path="first/shared.jsonl",
            acquired_at_ms=1,
        )
    assert replay_retained_components(tmp_path, selected_raw_ids=[old_raw_id]).replayed_logical_sources == 1

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        new_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=newest,
            source_path="moved/shared.jsonl",
            canonical_source_path="moved/shared.jsonl",
            acquired_at_ms=2,
        )

    result = replay_retained_components(tmp_path, selected_raw_ids=[new_raw_id])

    # The selected raw's component expands across both source paths. The
    # single retained preparation carries each member's own parse, so the
    # published component, not a per-member scan receipt, shows the expansion.
    assert [set(component) for component in result.components] == [{old_raw_id, new_raw_id}]
    assert result.replayed_logical_sources == 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id, message_count, raw_id FROM sessions").fetchall() == [
            ("shared", 2, new_raw_id)
        ]


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
                canonical_source_path="session.jsonl",
                acquired_at_ms=index,
            )
            for index, payload in enumerate((baseline, newest), start=1)
        }

    # The retained replay stages every parse acknowledgement into the same
    # prepared Source mutation as its Index outcome, so a crash between two
    # staged markers commits neither marker nor outcome. Patch the staging
    # call target the replay imports from revision_governance.
    original_stage = archive_revision_governance.prepare_raw_parse_success
    calls = 0

    def crash_after_one_marker(seal: Any, raw_id: str, *, provider: Provider) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            original_stage(seal, raw_id, provider=provider)
            return
        raise RuntimeError("crash between source markers")

    monkeypatch.setattr(archive_revision_governance, "prepare_raw_parse_success", crash_after_one_marker)
    with pytest.raises(RuntimeError, match="between source markers"):
        replay_retained_components(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE parsed_at_ms IS NOT NULL").fetchone()[0] == 0
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone()[0] == 0

    monkeypatch.setattr(archive_revision_governance, "prepare_raw_parse_success", original_stage)
    replay_retained_components(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert {
            str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions WHERE parsed_at_ms IS NOT NULL")
        } == raw_ids
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id, message_count FROM sessions").fetchall() == [("session-1", 2)]
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
            canonical_source_path="conversations.json",
            acquired_at_ms=1,
        )
        raw_b = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=bundle_b,
            source_path="conversations.json",
            canonical_source_path="conversations.json",
            acquired_at_ms=2,
        )

    result = replay_retained_components(tmp_path)
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
    rebuilt = replay_retained_components(tmp_path)
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
            canonical_source_path="conversations.json",
            acquired_at_ms=1,
        )
        raw_b = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("s1", "base", "right"), _chatgpt_session("s3", "safe")),
            source_path="conversations.json",
            canonical_source_path="conversations.json",
            acquired_at_ms=2,
        )

    result = replay_retained_components(tmp_path)
    # s1's own two revisions (base+left vs base+right) are a genuine,
    # irreducible fork with no prior head for this fresh archive -- the
    # conflict between two direct captures accepts the latest declared
    # capture (raw_b, acquired second) instead of leaving s1 permanently
    # headless, so only the earlier side of that fork stays quarantined
    # (1, not 2). s2/s3 are each single-member "safe" cohorts and were never
    # at risk.
    assert result.quarantined == 1
    winner_raw_id = raw_b
    loser_raw_id = raw_a
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
            canonical_source_path="conversations.json",
            acquired_at_ms=1,
        )
        raw_stale = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("s1", "base", "right")),
            source_path="conversations.json",
            canonical_source_path="conversations.json",
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

    result = replay_retained_components(tmp_path, selected_raw_ids=[raw_correct, raw_stale])
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
            canonical_source_path="first.json",
            acquired_at_ms=1,
        )
    replay_retained_components(tmp_path)
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
            canonical_source_path="second.json",
            acquired_at_ms=2,
        )
    result = replay_retained_components(tmp_path)

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
            canonical_source_path="first.json",
            acquired_at_ms=1,
        )
        sibling_raw = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("shared", "old", "new")),
            source_path="second.json",
            canonical_source_path="second.json",
            acquired_at_ms=2,
        )
        unrelated_raw = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("unrelated", "no")),
            source_path="third.json",
            canonical_source_path="third.json",
            acquired_at_ms=3,
        )

    # Production ordinary convergence starts from the durable membership
    # census established by ingestion/offline rebuild, not an empty source-v7
    # authority catalog.
    replay_retained_components(tmp_path)
    (tmp_path / "index.db").unlink()
    bootstrap_archive_root(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        unrelated_before = conn.execute(
            "SELECT parser_fingerprint, status, member_count, detail FROM raw_membership_census WHERE raw_id = ?",
            (unrelated_raw,),
        ).fetchone()

    from polylogue.sources import revision_backfill

    original_parse = revision_backfill.prepare_retained_jsonl_artifact
    opened: list[str] = []

    def observed_parse(evidence_reader: Any, raw_id: str, *, directory: Path, **kwargs: Any) -> Any:
        opened.append(raw_id)
        return original_parse(evidence_reader, raw_id, directory=directory, **kwargs)

    monkeypatch.setattr(revision_backfill, "prepare_retained_jsonl_artifact", observed_parse)
    result = replay_retained_components(tmp_path, selected_raw_ids=[selected_raw])
    assert result.replayed_logical_sources == 1
    # The selected raw expands to its same-session sibling on another path
    # and to nothing else.
    assert [set(component) for component in result.components] == [{selected_raw, sibling_raw}]
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
                canonical_source_path=f"bundle-{index}.json",
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
                canonical_source_path=f"shared-{index}.json",
                acquired_at_ms=independent_raw_count + index,
            )
    raw_count = independent_raw_count + len(shared_payloads)

    result = replay_retained_components(tmp_path)

    assert result.replayed_logical_sources == independent_raw_count + 1
    # One logical cohort is prepared and published at a time: the shared
    # session's two captures form the only multi-raw component.
    assert len(result.components) == independent_raw_count + 1
    assert max(len(component) for component in result.components) == 2
    assert sum(len(component) for component in result.components) == raw_count


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
            canonical_source_path="chain.jsonl",
            acquired_at_ms=1,
            native_id="chain",
        )
        baseline_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline,
            source_path="chain.jsonl",
            canonical_source_path="chain.jsonl",
            acquired_at_ms=2,
            native_id="chain",
        )
    return baseline_raw_id, newest_raw_id


_CHAIN_META = b'{"type":"session_meta","payload":{"id":"chain","timestamp":"2026-07-01T00:00:00Z"}}\n'


def _chain_turn(index: int, *, include_message_id: bool = False) -> bytes:
    message_id = b',"id":"turn-%d"' % index if include_message_id else b""
    return (
        b'{"type":"response_item","payload":{"type":"message"'
        + message_id
        + b',"role":"user","content":[{"type":"input_text","text":"turn-%d"}]}}\n' % index
    )


def _growing_chain_archive(root: Path, *, turns: int, include_message_ids: bool = False) -> list[str]:
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
        payload = payload + _chain_turn(index, include_message_id=include_message_ids)
        payloads.append(payload)
    raw_ids: list[str] = []
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for acquired_at_ms, capture in enumerate(payloads, start=1):
            raw_ids.append(
                archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=capture,
                    source_path="chain.jsonl",
                    canonical_source_path="chain.jsonl",
                    acquired_at_ms=acquired_at_ms,
                    native_id="chain",
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


def _observe_retained_jsonl_parse_calls(monkeypatch: pytest.MonkeyPatch, root: Path) -> list[str]:
    """Count current parser-boundary calls and bind each to its raw hash."""
    with sqlite3.connect(root / "source.db") as conn:
        raw_by_hash = {
            bytes(blob_hash).hex(): str(raw_id)
            for raw_id, blob_hash in conn.execute("SELECT raw_id, blob_hash FROM raw_sessions")
        }
    original = prepared_jsonl.prepare_jsonl_blob
    parsed: list[str] = []

    def counted(blob_path: str, source_path: str, provider_value: str, fallback_id: str, **kwargs: Any) -> Any:
        source_hash = kwargs.get("source_sha256")
        if source_hash in raw_by_hash:
            parsed.append(raw_by_hash[source_hash])
        return original(blob_path, source_path, provider_value, fallback_id, **kwargs)

    monkeypatch.setattr(prepared_jsonl, "prepare_jsonl_blob", counted)
    return parsed


def test_byte_proof_refuses_a_head_between_forks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """polylogue-asp4b (sibling append): two valid extensions, no chosen head.

    One rollout path holds a shared capture and two captures that each extend
    it differently -- the same file re-scanned after a fork, or two machines
    appending to one synced path. Both are valid extensions of the baseline and
    NEITHER is a byte prefix of the other, so byte comparison alone cannot say
    which is the file's current state.

    Wrong outcome prevented: the census picks the largest capture as the
    cohort's head and binds the other fork to its learned identity on
    containment with the baseline alone, so the losing fork is never parsed.
    """
    bootstrap_archive_root(tmp_path)
    shared = _CHAIN_META + _chain_turn(0)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_ids = {
            name: archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path="chain.jsonl",
                canonical_source_path="chain.jsonl",
                acquired_at_ms=index + 1,
                native_id="chain",
            )
            for index, (name, payload) in enumerate(
                (
                    ("shared", shared),
                    ("fork_a", shared + _chain_turn(1)),
                    ("fork_b", shared + _chain_turn(2)),
                )
            )
        }

    parsed = _observe_retained_jsonl_parse_calls(monkeypatch, tmp_path)
    replay_retained_components(tmp_path)

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

    replay_retained_components(tmp_path)

    assert _census_facts(tmp_path, finished) == ("codex-session:chain", "byte_proven", ("codex-session:chain",))
    # The refuted member keeps the classification its OWN bytes support.
    assert _census_facts(tmp_path, header_only) == (None, "quarantined", ())
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT raw_id, message_count FROM sessions").fetchall() == [(finished, 1)]


def test_chain_inherits_only_its_interior_members(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The declared native-message-id checkpoint profile parses three raws per chain.

    This profile requires stable Codex message IDs so its exact prefix grammar
    can prove each interior. ID-less Codex input remains valid and follows the
    ordinary parser path, as the neighboring refuted-member test proves.
    Five captures whose smallest member is header-only: the census parses the
    smallest, first agreeing capture, and head, then inherits for the interior.
    """
    raw_ids = _growing_chain_archive(tmp_path, turns=4, include_message_ids=True)
    parsed = _observe_retained_jsonl_parse_calls(monkeypatch, tmp_path)

    replay_retained_components(tmp_path)

    # raw_ids[0] is the refuted header-only capture, raw_ids[1] the smallest
    # capture whose own parse agrees, raw_ids[-1] the head. Nothing between
    # them is opened.
    assert set(parsed) == {raw_ids[0], raw_ids[1], raw_ids[-1]}, parsed
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
                    canonical_source_path=f"{session}.jsonl",
                    acquired_at_ms=chain * 100 + acquired_at_ms,
                    native_id=session,
                )
                for acquired_at_ms, capture in enumerate(captures, start=1)
            ]
    return raw_ids


def test_chain_census_binds_interior_members_to_their_own_chain_key(tmp_path: Path) -> None:
    """Independently growing chains each keep their own learned key.

    Input: independently growing rollout files, each with superseded
    byte-prefix captures. The interior members inherit their own chain's key
    and the header-only first capture stays unbound, for 2 chains and for 6.
    """
    for chains in (2, 6):
        root = tmp_path / f"chains-{chains}"
        raw_ids = _independent_growing_chains(root, chains=chains, turns=4)
        replay_retained_components(root)
        for session, chain in raw_ids.items():
            key = f"codex-session:{session}"
            assert _census_facts(root, chain[0])[0] is None
            for inherited in chain[2:-1]:
                assert _census_facts(root, inherited) == (key, "byte_proven", (key,))


def test_census_skips_parse_for_byte_proven_superseded_revisions_at_scale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """polylogue-nh44 regression at the bead's own recorded corpus shape: a
    growing-file cohort (one re-scanned Codex rollout, 50 superseded captures
    plus the winner) must census-parse only the winner, never the 50 byte-
    proven-superseded snapshots. Measured on this exact shape: 52->2 parse
    calls (1 unique raw parsed instead of 51), ~3.3x wall-time reduction for
    the cohort (see PR body for the before/after numbers)."""
    raw_ids = build_revision_chain_corpus(tmp_path, native_singleton=True, **REVISION_CHAIN_SHAPE)
    parse_calls = _observe_retained_jsonl_parse_calls(monkeypatch, tmp_path)

    result = replay_retained_components(tmp_path)

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
    assert set(parse_calls) == {raw_ids[0], raw_ids[1], raw_ids[-1]}, parse_calls
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


def test_census_quarantines_legacy_hermes_sqlite_page_images(tmp_path: Path) -> None:
    """Legacy SQLite page images cannot re-enter replay through the pool.

    The historical #3113 path accepted these as Hermes state-db raws. Current
    acquisition retains declared logical exports, so admitting an old page
    image would recreate a second source authority during a future reindex.
    Two independent raws still exercise the parallel census boundary.
    """
    bootstrap_archive_root(tmp_path)
    raw_ids: list[str] = []
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        for index in range(2):
            payload = _state_db_bytes_for_session(tmp_path, session_id=f"hermes-{index}", message_text=f"hi {index}")
            raw_ids.append(
                archive.write_raw_payload(
                    provider=Provider.HERMES,
                    payload=payload,
                    source_path=str(tmp_path / f"hermes-home-{index}" / "state.db"),
                    canonical_source_path=str(tmp_path / f"hermes-home-{index}" / "state.db"),
                    acquired_at_ms=index,
                )
            )

    result = replay_retained_components(tmp_path)

    assert result.scanned == 2
    assert result.replayed_logical_sources == 0
    assert result.quarantined == 2
    with sqlite3.connect(tmp_path / "index.db") as conn:
        rows = conn.execute("SELECT native_id, message_count FROM sessions ORDER BY native_id").fetchall()
    assert rows == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        page_images = conn.execute(
            "SELECT raw_id FROM raw_artifacts WHERE raw_id IN (?, ?) AND artifact_kind='binary_database' "
            "AND support_status='recognized_unparsed' AND classification_reason='legacy SQLite page image' "
            "ORDER BY raw_id",
            raw_ids,
        ).fetchall()
        memberships = conn.execute(
            "SELECT raw_id, status, member_count FROM raw_membership_census WHERE raw_id IN (?, ?) ORDER BY raw_id",
            raw_ids,
        ).fetchall()
    assert [row[0] for row in page_images] == sorted(raw_ids)
    assert memberships == [(raw_id, "non_session", 0) for raw_id in sorted(raw_ids)]


def test_independent_raw_corpus_fixture_backfills_cleanly(tmp_path: Path) -> None:
    """polylogue-amg1 benchmark fixture sanity: every synthetic raw census-and-replays
    to exactly one session with no quarantine, at both recorded payload shapes' scale
    (downscaled here for test speed; devtools/scripts run the full recorded counts)."""
    raw_ids = build_independent_raw_corpus(tmp_path, raw_count=12, avg_payload_bytes=5_000)

    result = replay_retained_components(tmp_path)

    assert result.scanned == 12
    assert result.replayed_logical_sources == 12
    assert result.quarantined == 0
    with sqlite3.connect(tmp_path / "index.db") as conn:
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    assert session_count == 12
    assert len(set(raw_ids)) == 12


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
        canonical_source_path=source_path,
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


# ---------------------------------------------------------------------------
# Whale-aware census spill (polylogue-odm1)
# ---------------------------------------------------------------------------

# Shrink the hot-cache budget so a modest (KB-scale) fixture reliably
# classifies as a "whale" without depending on the host's real RAM -- see
# tests/benchmarks/test_whale_census_spill_bench.py's module docstring for
# the full rationale (the class computes its budgets from
# effective_physical_memory_bytes(), whose production floor is 256 MiB).
_SHRUNK_TREE_BYTES = 64 * 1024


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
                canonical_source_path=f"pipe-{index}.json",
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
            canonical_source_path="pipe-bundle.json",
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


def _captured_replay_schedule(root: Path, matches: Callable[[set[str]], bool]) -> Any:
    """Run canonical retained replay and return the schedule production used.

    Production schedules each retained component as it is replayed, so the
    effective archive order is the concatenation of those per-component
    schedules in call order. The merged schedule is what the writer actually
    used; ``matches`` must accept the union of the replayed keys.
    """
    original = revision_backfill._lineage_aware_replay_schedule
    captured: list[Any] = []

    def capture(logical_keys: set[str], *args: Any, **kwargs: Any) -> Any:
        schedule = original(logical_keys, *args, **kwargs)
        captured.append(schedule)
        return schedule

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(revision_backfill, "_lineage_aware_replay_schedule", capture)
        replay_retained_components(root)
    order = tuple(key for schedule in captured for key in schedule.order)
    assert matches(set(order)), "canonical replay computed no schedule over the expected keys"
    return revision_backfill.ReplaySchedule(
        order=order,
        topology={key: value for schedule in captured for key, value in schedule.topology.items()},
        parent_of={key: value for schedule in captured for key, value in schedule.parent_of.items()},
    )


def _seed_lineage_fixture(
    root: Path, *, n_children: int, timestamp_adversarial: bool = False, children_first: bool = True
) -> None:
    """One parent (native_id sorts LAST lexicographically) plus N children
    (native_ids sort BEFORE the parent) that each replay the parent's full
    message prefix plus one new tail message -- a real Codex resume shape.

    By default every child is acquired before its parent, so neither
    acquisition order nor lexicographic order can put the parent first by
    accident: only lineage-aware scheduling does.
    """
    bootstrap_archive_root(root)
    parent_native_id = "zparent"
    parent_texts = [f"parent-{i}" for i in range(4)]
    with ArchiveStore.open_existing(root, read_only=False) as archive:

        def write_parent(acquired_at_ms: int) -> None:
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_codex_session_payload(
                    parent_native_id,
                    parent_texts,
                    timestamp_adversarial=timestamp_adversarial,
                ),
                source_path=f"{parent_native_id}.jsonl",
                canonical_source_path=f"{parent_native_id}.jsonl",
                acquired_at_ms=acquired_at_ms,
                native_id=parent_native_id,
            )

        if not children_first:
            write_parent(1)
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
                canonical_source_path=f"{child_native_id}.jsonl",
                acquired_at_ms=2 + index,
                native_id=child_native_id,
            )
        if children_first:
            write_parent(2 + n_children)


def test_lineage_aware_replay_schedule_visits_parent_before_children(tmp_path: Path) -> None:
    """polylogue-5q2u: the parent replays before each of its children even
    though the children were acquired first and sort first lexicographically.
    Production schedules per retained component; the archive order is the
    concatenation of those schedules."""
    root = tmp_path / "archive"
    _seed_lineage_fixture(root, n_children=5)
    schedule = _captured_replay_schedule(root, lambda keys: "codex-session:zparent" in keys)
    order = list(schedule.order)

    assert order[0] == "codex-session:zparent"
    parent_position = order.index("codex-session:zparent")
    for index in range(5):
        child_key = f"codex-session:achild{index}"
        assert child_key in order
        assert order.index(child_key) > parent_position


def test_lineage_aware_replay_schedule_falls_back_for_unresolvable_parent(tmp_path: Path) -> None:
    """A parent that was never ingested must not crash replay or drop the
    children that name it: both orphans still replay exactly once."""
    root = tmp_path / "archive"
    bootstrap_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for native_id in ("zorphan", "aorphan"):
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_codex_session_payload(native_id, ["only-message"], forked_from_id="never-ingested-parent"),
                source_path=f"{native_id}.jsonl",
                canonical_source_path=f"{native_id}.jsonl",
                acquired_at_ms=1,
                native_id=native_id,
            )
    schedule = _captured_replay_schedule(root, lambda keys: keys == {"codex-session:zorphan", "codex-session:aorphan"})
    assert sorted(schedule.order) == ["codex-session:aorphan", "codex-session:zorphan"]
    assert set(schedule.parent_of.values()) == {None}


def test_lineage_aware_replay_schedule_reduces_deferred_tail_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-5q2u AC1: replay must not take the #2467 deferred-tail/
    orphaned-child normalization path (``_reextract_prefix_tail_db``) for a
    parent-with-many-children fixture whose children were acquired first.

    Anti-vacuity: replaying components in acquisition order (children before
    the parent they extend) makes every child hit the deferred-tail path.
    """
    root = tmp_path / "lineage"
    n_children = 5
    _seed_lineage_fixture(root, n_children=n_children)
    calls = 0
    original = archive_tier_write._reextract_prefix_tail_db

    def counting_wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(archive_tier_write, "_reextract_prefix_tail_db", counting_wrapper)
    replay_retained_components(root)

    assert calls == 0, (
        f"lineage-aware order should replay the parent before any child, avoiding the "
        f"deferred-tail path entirely; got {calls} hits"
    )


def test_lineage_aware_replay_order_preserves_outcome_parity(tmp_path: Path) -> None:
    """polylogue-5q2u AC2: replay order must not change WHAT gets replayed or
    adopted. The same sources acquired parent-first and children-first must
    reach byte-identical index content (sessions/messages/blocks/
    session_links), using ``_index_content_manifest`` as the currency.
    """
    parent_first_root = tmp_path / "parent-first"
    children_first_root = tmp_path / "children-first"
    _seed_lineage_fixture(parent_first_root, n_children=5, timestamp_adversarial=True, children_first=False)
    _seed_lineage_fixture(children_first_root, n_children=5, timestamp_adversarial=True, children_first=True)

    parent_first = replay_retained_components(parent_first_root)
    children_first = replay_retained_components(children_first_root)

    assert parent_first.replayed_logical_sources == children_first.replayed_logical_sources
    assert parent_first.quarantined == children_first.quarantined
    assert parent_first.adoption_deferred == children_first.adoption_deferred
    parent_first_manifest = _index_content_manifest(parent_first_root)
    children_first_manifest = _index_content_manifest(children_first_root)
    # Topology is the acceptance boundary: replay order may change when
    # deferred-tail work runs, never which parent edge is persisted. The raw
    # fixture reverses timestamps against message positions so a wall-clock
    # ordering shortcut cannot make these manifests agree by luck.
    assert parent_first_manifest["session_links"] == children_first_manifest["session_links"]
    assert parent_first_manifest == children_first_manifest


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
        _parse_one_raw(Provider.ANTIGRAVITY, b"retained bytes", str(trajectory), sidecar_resolver=None)


def _owned_generation_corpus(root: Path, *, raw_count: int, snapshot: str) -> IndexGeneration:
    """Seed ``raw_count`` byte-proven Codex raws and open an owned generation.

    Codex is the provider whose replay enrichment reads the index tier
    (``_replay_enrichment_reads_index``), which is what makes this the shape
    that exposed polylogue-cz17d.
    """
    bootstrap_archive_root(root)
    build_independent_raw_corpus(root, raw_count=raw_count, avg_payload_bytes=2_000, authoritative_source=True)
    replay_retained_components(root)
    return IndexGenerationStore.for_archive_root(root).create(source_snapshot=snapshot)


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
