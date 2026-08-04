from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from polylogue.archive.artifact_taxonomy import ArtifactClassification, ArtifactKind
from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Origin, Provider
from polylogue.storage.artifacts.inspection import artifact_observation_id
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.raw_admission import (
    PriorRawHead,
    RawAdmissionArm,
    admit_raw_observation,
)
from polylogue.storage.sqlite.archive_tiers.source_write import bind_source_raw_revision
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.SOURCE)
    return conn


def _row(conn: sqlite3.Connection, raw_id: str) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT revision_kind, revision_authority, predecessor_raw_id, baseline_raw_id,
               append_start_offset, append_end_offset, blob_size, logical_source_key
        FROM raw_sessions WHERE raw_id = ?
        """,
        (raw_id,),
    ).fetchone()
    assert row is not None
    return cast(sqlite3.Row, row)


def _envelope_row(conn: sqlite3.Connection, raw_id: str) -> tuple[object, ...]:
    row = conn.execute(
        """
        SELECT logical_source_key, revision_kind, source_revision,
               predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
               append_start_offset, append_end_offset, acquisition_generation,
               revision_authority
        FROM raw_sessions WHERE raw_id = ?
        """,
        (raw_id,),
    ).fetchone()
    assert row is not None
    return tuple(row)


def test_admit_raw_observation_baseline_when_no_prior_head(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")

    result = admit_raw_observation(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/tmp/rollout.jsonl",
        payload=b"line-1\n",
        acquired_at_ms=1_767_000_000_000,
        logical_source_key="codex:/tmp/rollout.jsonl",
        prior_head=None,
    )

    assert result.arm is RawAdmissionArm.BASELINE
    row = _row(conn, result.raw_id)
    assert row["revision_kind"] == "full"
    assert row["revision_authority"] == RawRevisionAuthority.ASSERTED.value
    assert row["logical_source_key"] == "codex:/tmp/rollout.jsonl"


def test_admit_raw_observation_post_parse_arm_binds_typed_revision(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    payload = b'{"type":"user","sessionId":"s1"}\n'

    result = admit_raw_observation(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/tmp/live.jsonl",
        payload=payload,
        acquired_at_ms=1_767_000_000_000,
        logical_source_key=None,
        post_parse=True,
    )

    assert result.arm is RawAdmissionArm.POST_PARSE_PENDING
    pending = _row(conn, result.raw_id)
    assert pending["revision_kind"] == RawRevisionKind.FULL.value
    assert pending["revision_authority"] == RawRevisionAuthority.QUARANTINED.value
    assert str(pending["logical_source_key"]).startswith("pending-raw:")

    bind_source_raw_revision(
        conn,
        result.raw_id,
        RawRevisionEnvelope(
            logical_source_key="codex-session:s1",
            kind=RawRevisionKind.APPEND,
            source_revision="append-rev",
            predecessor_source_revision="prior-rev",
            predecessor_raw_id="prior-raw",
            baseline_raw_id="prior-raw",
            append_start_offset=1,
            append_end_offset=2,
            acquisition_generation=1,
            authority=RawRevisionAuthority.BYTE_PROVEN,
        ),
    )

    bound = _row(conn, result.raw_id)
    assert bound["logical_source_key"] == "codex-session:s1"
    assert bound["revision_kind"] == RawRevisionKind.APPEND.value
    assert bound["predecessor_raw_id"] == "prior-raw"


def test_archive_store_write_raw_payload_post_parse_is_deterministic_and_restart_bindable(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    payload = b'{"type":"response_item","payload":{"type":"message"}}\n'
    source_path = str(tmp_path / "append.jsonl")
    with ArchiveStore.open_existing(tmp_path, read_only=False) as store:
        raw_id = store.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path=source_path,
            source_index=-1,
            native_id="append-session",
            acquired_at_ms=10,
            post_parse=True,
        )
        assert (
            raw_id
            == hashlib.sha256(
                b"codex-session\0"
                + source_path.encode()
                + b"\0-1\0"
                + hashlib.sha256(payload).digest()
                + b"\0append-session"
            ).hexdigest()
        )

    with ArchiveStore.open_existing(tmp_path, read_only=False) as store:
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert _envelope_row(conn, raw_id) == (
                f"pending-raw:codex-session:-1:{source_path}:{raw_id}",
                "full",
                hashlib.sha256(payload).hexdigest(),
                None,
                None,
                None,
                None,
                None,
                0,
                "quarantined",
            )
        store.bind_raw_revision(
            raw_id,
            RawRevisionEnvelope(
                logical_source_key="codex-session:append-session",
                kind=RawRevisionKind.APPEND,
                source_revision="append-revision",
                predecessor_source_revision="base-revision",
                predecessor_raw_id="base-raw",
                baseline_raw_id="base-raw",
                append_start_offset=42,
                append_end_offset=99,
                acquisition_generation=3,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert _envelope_row(conn, raw_id) == (
            "codex-session:append-session",
            "append",
            "append-revision",
            "base-revision",
            "base-raw",
            "base-raw",
            42,
            99,
            3,
            "byte_proven",
        )
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)


def test_archive_store_write_raw_blob_ref_post_parse_preserves_explicit_batch_id_and_binds(
    tmp_path: Path,
) -> None:
    initialize_active_archive_root(tmp_path)
    payload = b'{"type":"session_meta","payload":{"id":"blob-session"}}\n'
    blob_hash = hashlib.sha256(payload).digest()
    raw_id = "batch-explicit-raw-id"
    source_path = str(tmp_path / "batch.jsonl")
    with ArchiveStore.open_existing(tmp_path, read_only=False) as store:
        assert store._blob_publisher is not None
        published_hash, published_size = store._blob_publisher.write_from_bytes(payload)
        store._blob_publisher.flush()
        assert published_hash == blob_hash.hex()
        assert published_size == len(payload)
        assert (
            store.write_raw_blob_ref(
                provider=Provider.CODEX,
                blob_hash_hex=blob_hash.hex(),
                blob_size=len(payload),
                source_path=source_path,
                source_index=7,
                raw_id=raw_id,
                acquired_at_ms=20,
                post_parse=True,
            )
            == raw_id
        )

    with ArchiveStore.open_existing(tmp_path, read_only=False) as store:
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert _envelope_row(conn, raw_id) == (
                f"pending-raw:codex-session:7:{source_path}:{raw_id}",
                "full",
                blob_hash.hex(),
                None,
                None,
                None,
                None,
                None,
                0,
                "quarantined",
            )
        store.bind_raw_revision(
            raw_id,
            RawRevisionEnvelope(
                logical_source_key="codex-session:blob-session",
                kind=RawRevisionKind.FULL,
                source_revision="blob-revision",
                acquisition_generation=4,
                authority=RawRevisionAuthority.ASSERTED,
            ),
        )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert _envelope_row(conn, raw_id) == (
            "codex-session:blob-session",
            "full",
            "blob-revision",
            None,
            None,
            None,
            None,
            None,
            4,
            "asserted",
        )


def test_admit_raw_observation_arm1_skip_duplicate(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    head_payload = b"line-1\n"
    baseline = admit_raw_observation(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/tmp/rollout.jsonl",
        payload=head_payload,
        acquired_at_ms=1_767_000_000_000,
        logical_source_key="codex:/tmp/rollout.jsonl",
        prior_head=None,
    )
    prior_head = PriorRawHead(
        raw_id=baseline.raw_id,
        source_revision="rev-0",
        payload=head_payload,
    )

    result = admit_raw_observation(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/tmp/rollout.jsonl",
        payload=head_payload,
        acquired_at_ms=1_767_000_001_000,
        logical_source_key="codex:/tmp/rollout.jsonl",
        prior_head=prior_head,
    )

    assert result.arm is RawAdmissionArm.SKIP_DUPLICATE
    assert result.raw_id == baseline.raw_id
    # No second raw_sessions row was written.
    count = conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0]
    assert count == 1


def test_admit_raw_observation_arm2_append_extends_head(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    head_payload = b"line-1\n"
    baseline = admit_raw_observation(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/tmp/rollout.jsonl",
        payload=head_payload,
        acquired_at_ms=1_767_000_000_000,
        logical_source_key="codex:/tmp/rollout.jsonl",
        prior_head=None,
    )
    prior_head = PriorRawHead(
        raw_id=baseline.raw_id,
        source_revision="rev-0",
        payload=head_payload,
        acquisition_generation=0,
    )
    extended_payload = head_payload + b"line-2\n"

    result = admit_raw_observation(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/tmp/rollout.jsonl",
        payload=extended_payload,
        acquired_at_ms=1_767_000_002_000,
        logical_source_key="codex:/tmp/rollout.jsonl",
        prior_head=prior_head,
    )

    assert result.arm is RawAdmissionArm.APPEND
    assert result.raw_id != baseline.raw_id
    row = _row(conn, result.raw_id)
    assert row["revision_kind"] == "append"
    assert row["revision_authority"] == RawRevisionAuthority.ASSERTED.value
    assert row["predecessor_raw_id"] == baseline.raw_id
    assert row["baseline_raw_id"] == baseline.raw_id
    assert row["append_start_offset"] == len(head_payload)
    assert row["append_end_offset"] == len(extended_payload)
    assert row["blob_size"] == len(extended_payload)


def test_admit_raw_observation_arm3_supersede_when_new_bytes_are_prefix_of_head(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    fuller_payload = b"line-1\nline-2\n"
    baseline = admit_raw_observation(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/tmp/rollout.jsonl",
        payload=fuller_payload,
        acquired_at_ms=1_767_000_000_000,
        logical_source_key="codex:/tmp/rollout.jsonl",
        prior_head=None,
    )
    prior_head = PriorRawHead(
        raw_id=baseline.raw_id,
        source_revision="rev-0",
        payload=fuller_payload,
    )
    shorter_payload = b"line-1\n"

    result = admit_raw_observation(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/tmp/rollout.jsonl",
        payload=shorter_payload,
        acquired_at_ms=1_767_000_003_000,
        logical_source_key="codex:/tmp/rollout.jsonl",
        prior_head=prior_head,
    )

    assert result.arm is RawAdmissionArm.SUPERSEDE
    row = _row(conn, result.raw_id)
    assert row["revision_kind"] == "full"
    assert row["revision_authority"] == RawRevisionAuthority.BYTE_PROVEN.value
    assert row["predecessor_raw_id"] == baseline.raw_id
    assert row["blob_size"] == len(shorter_payload)


def test_admit_raw_observation_arm5_refuses_ambiguous_bytes_with_no_reacquire(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    head_payload = b"line-1\nline-2\n"
    baseline = admit_raw_observation(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/tmp/rollout.jsonl",
        payload=head_payload,
        acquired_at_ms=1_767_000_000_000,
        logical_source_key="codex:/tmp/rollout.jsonl",
        prior_head=None,
    )
    prior_head = PriorRawHead(
        raw_id=baseline.raw_id,
        source_revision="rev-0",
        payload=head_payload,
    )
    unrelated_payload = b"totally-different-content\n"

    result = admit_raw_observation(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/tmp/rollout.jsonl",
        payload=unrelated_payload,
        acquired_at_ms=1_767_000_004_000,
        logical_source_key="codex:/tmp/rollout.jsonl",
        prior_head=prior_head,
        reacquire=None,
    )

    assert result.arm is RawAdmissionArm.REFUSED_AMBIGUOUS
    assert result.refusal_reason == "no_byte_relation_to_prior_head"
    assert result.reacquire_attempted is False
    row = _row(conn, result.raw_id)
    assert row["revision_kind"] == "unknown"
    assert row["revision_authority"] == RawRevisionAuthority.QUARANTINED.value


def test_admit_raw_observation_arm5_opportunistic_reacquire_resolves_to_append(tmp_path: Path) -> None:
    """Ambiguous first read, but a reacquire callback returns bytes that
    DO extend the head as a prefix -- the opportunistic re-read must win
    over the stale ambiguous first read (operator correction: re-acquire
    is opportunistic, only when the source is still present/readable)."""
    conn = _connect(tmp_path / "source.db")
    head_payload = b"line-1\n"
    baseline = admit_raw_observation(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/tmp/rollout.jsonl",
        payload=head_payload,
        acquired_at_ms=1_767_000_000_000,
        logical_source_key="codex:/tmp/rollout.jsonl",
        prior_head=None,
    )
    prior_head = PriorRawHead(
        raw_id=baseline.raw_id,
        source_revision="rev-0",
        payload=head_payload,
    )
    stable_extended_payload = head_payload + b"line-2\n"

    result = admit_raw_observation(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/tmp/rollout.jsonl",
        payload=b"torn-mid-rewrite-garbage",
        acquired_at_ms=1_767_000_005_000,
        logical_source_key="codex:/tmp/rollout.jsonl",
        prior_head=prior_head,
        reacquire=lambda: stable_extended_payload,
    )

    assert result.arm is RawAdmissionArm.APPEND
    assert result.reacquire_attempted is True
    assert result.reacquire_changed_outcome is True
    row = _row(conn, result.raw_id)
    assert row["blob_size"] == len(stable_extended_payload)


def test_admit_raw_observation_arm5_reacquire_returns_none_when_source_vanished(tmp_path: Path) -> None:
    """Per the operator correction: absence of the source must fall through
    to a typed refusal, never raise or block on source persistence."""
    conn = _connect(tmp_path / "source.db")
    head_payload = b"line-1\n"
    baseline = admit_raw_observation(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/tmp/rollout.jsonl",
        payload=head_payload,
        acquired_at_ms=1_767_000_000_000,
        logical_source_key="codex:/tmp/rollout.jsonl",
        prior_head=None,
    )
    prior_head = PriorRawHead(raw_id=baseline.raw_id, source_revision="rev-0", payload=head_payload)

    result = admit_raw_observation(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/tmp/rollout.jsonl",
        payload=b"unrelated-bytes",
        acquired_at_ms=1_767_000_006_000,
        logical_source_key="codex:/tmp/rollout.jsonl",
        prior_head=prior_head,
        reacquire=lambda: None,  # source vanished / unreadable
    )

    assert result.arm is RawAdmissionArm.REFUSED_AMBIGUOUS
    assert result.reacquire_attempted is True
    assert result.reacquire_changed_outcome is False
    assert result.refusal_reason == "reacquire_unavailable_or_absent"


def test_admit_raw_observation_arm4_artifact_never_touches_index_db(tmp_path: Path) -> None:
    """Structural proof for AC(c): the artifact arm writes raw_sessions +
    raw_artifacts (both source.db tables) and this module has no import of
    any index-tier writer -- so a tool-results/*.json-classified payload
    cannot reach a code path that creates an index.db `sessions` row."""
    import polylogue.storage.sqlite.archive_tiers.raw_admission as raw_admission_module

    assert raw_admission_module.__file__ is not None
    source_lines = Path(raw_admission_module.__file__).read_text()
    assert "archive_tiers.write" not in source_lines
    assert "archive_tiers.index" not in source_lines

    conn = _connect(tmp_path / "source.db")
    classification = ArtifactClassification(
        provider=Provider.CLAUDE_CODE,
        kind=ArtifactKind.TOOL_RESULT_SIDECAR,
        parse_as_session=False,
        schema_eligible=False,
        default_priority=0,
        reason="tool-results sidecar, not a conversation",
    )

    result = admit_raw_observation(
        conn,
        origin=Origin.CLAUDE_CODE_SESSION,
        source_path="/tmp/tool-results/abc.json",
        payload=b'{"tool":"bash","output":"ok"}',
        acquired_at_ms=1_767_000_000_000,
        logical_source_key="claude-code:/tmp/tool-results/abc.json",
        prior_head=None,
        artifact=classification,
    )

    assert result.arm is RawAdmissionArm.ARTIFACT
    assert result.artifact_id is not None
    expected_artifact_id = artifact_observation_id(
        source_name=Origin.CLAUDE_CODE_SESSION.value,
        source_path="/tmp/tool-results/abc.json",
        source_index=0,
    )
    assert result.artifact_id == expected_artifact_id

    raw_row = conn.execute("SELECT raw_id FROM raw_sessions WHERE raw_id = ?", (result.raw_id,)).fetchone()
    assert raw_row is not None
    artifact_row = conn.execute(
        "SELECT parse_as_session, artifact_kind, raw_id FROM raw_artifacts WHERE artifact_id = ?",
        (result.artifact_id,),
    ).fetchone()
    assert artifact_row is not None
    assert artifact_row["parse_as_session"] == 0
    assert artifact_row["raw_id"] == result.raw_id

    # There is no `sessions` table in source.db at all -- structural proof
    # this tier cannot materialize a conversation.
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}
    assert "sessions" not in tables


def test_admit_raw_observation_requires_logical_source_key(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    with pytest.raises(ValueError):
        admit_raw_observation(
            conn,
            origin=Origin.CODEX_SESSION,
            source_path="/tmp/rollout.jsonl",
            payload=b"x",
            acquired_at_ms=1_767_000_000_000,
            logical_source_key="",
            prior_head=None,
        )
