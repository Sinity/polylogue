from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.archive.revision_authority import BYTE_AUTHORITY_CENSUS_DETAIL, RawRevisionAuthority
from polylogue.core.errors import SchemaSkew
from polylogue.storage.archive_readiness import (
    CLAUDE_WORKFLOW_STAGE_NAME,
    RAW_ALIAS_BLOB_MISSING_CATEGORY,
    RawMaterializationAssessmentState,
    archive_readiness_status,
    assess_raw_materialization,
    claude_workflow_materialization_status,
    probe_archive_tier,
    raw_materialization_readiness_snapshot,
    raw_materialization_ready,
)
from polylogue.storage.raw_authority import (
    raw_authority_parser_fingerprint,
)
from polylogue.storage.sqlite import connection_profile
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.durable_tier_fixtures import initialize_runtime_source_fixture


def _category_counts(snapshot: Mapping[str, object]) -> Mapping[str, object]:
    return cast(Mapping[str, object], snapshot["category_counts"])


def _write_blob(root: Path, hex_hash: str, payload: bytes = b"{}") -> None:
    """Materialize the content-addressed blob a raw row points at."""
    blob = root / "blob" / hex_hash[:2] / hex_hash[2:]
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(payload)


def _seed_rows(conn: sqlite3.Connection, table: str, columns: tuple[str, ...], rows: Any) -> None:
    """Insert law rows into the real tier DDL, completing required columns.

    Each law names only the columns it classifies on. Required columns it does
    not name receive neutral values: a deterministic 32-byte blob hash, a
    zero size, the first acquisition instant, and a derived session identity
    (``session_id`` is generated from origin and native ID in the real DDL).
    """
    for ordinal, row in enumerate(rows):
        values: dict[str, object] = dict(zip(columns, row, strict=True))
        if table == "raw_sessions":
            raw_id = str(values["raw_id"])
            if values.get("origin") is None:
                values["origin"] = "codex-session"
            if values.get("source_path") is None:
                values["source_path"] = f"{raw_id}.json"
            if values.get("blob_hash") is None:
                values["blob_hash"] = hashlib.sha256(raw_id.encode()).digest()
            values.setdefault("blob_size", 0)
            values.setdefault("acquired_at_ms", 1)
            if values.get("source_index") is None:
                values.pop("source_index", None)
        elif table == "sessions":
            session_id = values.pop("session_id", None)
            if values.get("origin") is None:
                values["origin"] = "codex-session"
            if values.get("native_id") is None:
                prefix = f"{values['origin']}:"
                text = str(session_id) if session_id is not None else f"session-{ordinal}"
                values["native_id"] = text.removeprefix(prefix)
            values.setdefault("content_hash", bytes(32))
        elif table == "raw_membership_census":
            values.setdefault("parser_fingerprint", raw_authority_parser_fingerprint())
            values.setdefault("censused_at_ms", 1)
            if values.get("detail") is None:
                values["detail"] = ""
        elif table == "raw_session_memberships":
            raw_id = str(values["raw_id"])
            values.setdefault("logical_source_key", f"codex-session:{raw_id}-{ordinal}")
            values.setdefault("provider_session_id", f"{raw_id}-{ordinal}")
            values.setdefault("source_revision", "revision")
            values.setdefault("normalized_content_hash", bytes(32))
            values.setdefault("message_count", 1)
            if values.get("decision") is not None:
                values.setdefault("decided_at_ms", 1)
        elif table == "raw_revision_applications":
            raw_id = str(values["raw_id"])
            values.setdefault("decision_id", f"decision-{raw_id}-{ordinal}")
            values.setdefault("session_id", f"codex-session:{raw_id}")
            values.setdefault("logical_source_key", f"codex-session:{raw_id}")
            values.setdefault("source_revision", "revision")
            values.setdefault("acquisition_generation", 0)
            values.setdefault("decided_at_ms", 1)
        names = list(values)
        conn.execute(
            f"INSERT INTO {table}({', '.join(names)}) VALUES ({', '.join('?' for _ in names)})",
            [values[name] for name in names],
        )


def _stamp_index_as_current_schema(index_db: Path) -> None:
    """Declare a hand-built stub index to be at this runtime's schema.

    These fixtures assert raw-gap classification, not schema compatibility,
    and the readiness probe reads through the schema-enforcing open. Stamping
    states the precondition explicitly instead of leaving the probe to accept
    an index whose shape it never checked.
    """
    from polylogue.storage.sqlite.schema_bootstrap import stamp_derived_schema_identity

    with sqlite3.connect(index_db) as conn:
        conn.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]}")
        stamp_derived_schema_identity(conn, "index")


def test_probe_archive_tier_reports_schema_skew_without_opening_a_usable_reader(tmp_path: Path) -> None:
    """The version probe remains readable when the normal reader rejects stale schemas."""
    # Above the expected version, not below: the source tier sits at the
    # archive format floor, so one below it is 0 -- the never-provisioned
    # sentinel the reader deliberately admits rather than a stale schema.
    skewed_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE] + 1
    db_path = tmp_path / "source.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(f"PRAGMA user_version = {skewed_version}")

    with pytest.raises(SchemaSkew):
        connection_profile.open_readonly_connection(db_path)

    probe = probe_archive_tier(ArchiveTier.SOURCE, db_path)

    assert probe.user_version == skewed_version
    assert probe.version_status == "mismatch"


def test_raw_materialization_readiness_requires_populated_parser_census() -> None:
    """The durable parser census and a populated denominator gate the claim.

    Anti-vacuity: the last assertion is the only True, and it differs from the
    one above it solely by ``raw_authority_parser_census.available`` being the
    boolean the projection writes rather than a truthy string. Re-adding a
    retired census precondition would make it False.
    """
    counters_green: dict[str, object] = {"available": True, "raw_artifact_count": 1}

    assert raw_materialization_ready(counters_green) is False
    assert raw_materialization_ready({**counters_green, "raw_authority_parser_census": {"available": "yes"}}) is False
    populated = {**counters_green, "raw_authority_parser_census": {"available": True}}
    # Materialized count is informational on this compatibility snapshot; the
    # populated denominator, blocker counters, and parser census carry verdict.
    assert assess_raw_materialization(populated).state is RawMaterializationAssessmentState.POPULATED_CONVERGED
    assert raw_materialization_ready(populated) is True


def test_raw_materialization_assessment_distinguishes_unmeasured_and_convergence() -> None:
    """A pristine archive is unknown, while populated evidence gets a verdict."""
    census = {"available": True}
    pristine = {
        "available": True,
        "raw_artifact_count": 0,
        "materialized_raw_artifact_count": 0,
        "raw_authority_parser_census": census,
    }
    converged = {
        **pristine,
        "raw_artifact_count": 1,
        "materialized_raw_artifact_count": 1,
    }
    unconverged = {**converged, "unchecked": 1}

    assert assess_raw_materialization(pristine).state is RawMaterializationAssessmentState.UNMEASURED
    assert assess_raw_materialization(pristine).reason == "zero_denominator"
    assert assess_raw_materialization(converged).state is RawMaterializationAssessmentState.POPULATED_CONVERGED
    assert assess_raw_materialization(unconverged).state is RawMaterializationAssessmentState.POPULATED_UNCONVERGED
    assert raw_materialization_ready(pristine) is False
    assert raw_materialization_ready(converged) is True
    assert raw_materialization_ready(unconverged) is False

    unavailable = assess_raw_materialization({"available": False, "error": "index unavailable"})
    assert unavailable.state is RawMaterializationAssessmentState.UNMEASURED
    assert unavailable.reason == "readiness_unavailable"
    assert unavailable.detail == "index unavailable"

    known_debt_without_census = assess_raw_materialization(
        {**converged, "raw_authority_parser_census": {"available": False}, "unchecked": 1}
    )
    known_debt_without_classifier = assess_raw_materialization(
        {**converged, "debt_classifier_error": "ops.db locked", "actionable": 1}
    )
    assert known_debt_without_census.state is RawMaterializationAssessmentState.POPULATED_UNCONVERGED
    assert known_debt_without_classifier.state is RawMaterializationAssessmentState.POPULATED_UNCONVERGED

    missing_denominator_with_debt = assess_raw_materialization(
        {
            "available": True,
            "raw_authority_parser_census": {"available": True},
            "critical": 1,
        }
    )
    # An observed blocker refutes convergence even without a denominator
    # (c42f5ea3a1); only a debt-free snapshot without one stays unmeasured.
    assert missing_denominator_with_debt.state is RawMaterializationAssessmentState.POPULATED_UNCONVERGED
    assert missing_denominator_with_debt.reason == "blocking_materialization_debt"
    missing_denominator = assess_raw_materialization(
        {"available": True, "raw_authority_parser_census": {"available": True}}
    )
    assert missing_denominator.state is RawMaterializationAssessmentState.UNMEASURED
    assert missing_denominator.reason == "raw_artifact_count_unavailable"


def test_raw_materialization_snapshot_rejects_malformed_parser_receipt(tmp_path: Path) -> None:
    """Readiness shares the promotion gate's parser-receipt validation."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"malformed-census"}}\n',
            source_path="codex/malformed-census.jsonl",
            canonical_source_path="codex/malformed-census.jsonl",
            acquired_at_ms=1,
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_authority_parser_census (
                raw_id, parser_fingerprint, status, logical_keys_json, detail
            ) VALUES (?, ?, 'complete', '["codex-session:duplicate", "codex-session:duplicate"]', '')
            """,
            (raw_id, raw_authority_parser_fingerprint()),
        )
        conn.commit()

    snapshot = raw_materialization_readiness_snapshot(tmp_path)
    parser_census = cast(Mapping[str, object], snapshot["raw_authority_parser_census"])

    assert parser_census["complete_count"] == 0
    assert parser_census["incomplete_count"] == 1
    assert parser_census["non_complete_receipt_count"] == 1


def test_raw_materialization_snapshot_rejects_receipt_key_drift_from_durable_binding(tmp_path: Path) -> None:
    """Readiness and frozen promotion reject the same mismatched receipt evidence."""
    from polylogue.archive.revision_authority import RawRevisionEnvelope, RawRevisionKind
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"durable-binding"}}\n',
            source_path="codex/durable-binding.jsonl",
            canonical_source_path="codex/durable-binding.jsonl",
            acquired_at_ms=1,
            revision=RawRevisionEnvelope("codex:durable-binding", RawRevisionKind.FULL, "v1", 0),
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_authority_parser_census (
                raw_id, parser_fingerprint, status, logical_keys_json, detail
            ) VALUES (?, ?, 'complete', '["codex-session:wrong-binding"]', '')
            """,
            (raw_id, raw_authority_parser_fingerprint()),
        )
        conn.commit()

    parser_census = cast(
        Mapping[str, object], raw_materialization_readiness_snapshot(tmp_path)["raw_authority_parser_census"]
    )

    assert parser_census["complete_count"] == 0
    assert parser_census["incomplete_count"] == 1


def test_raw_materialization_snapshot_audits_validation_skipped_raws(tmp_path: Path) -> None:
    """Skipped schema validation does not exempt a raw from parser authority."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"skipped-census"}}\n',
            source_path="codex/skipped-census.jsonl",
            canonical_source_path="codex/skipped-census.jsonl",
            acquired_at_ms=1,
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET validation_status = 'skipped' WHERE raw_id = ?", (raw_id,))
        conn.execute("DELETE FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,))
        conn.commit()

    parser_census = cast(
        Mapping[str, object], raw_materialization_readiness_snapshot(tmp_path)["raw_authority_parser_census"]
    )

    assert parser_census["complete_count"] == 0
    assert parser_census["incomplete_count"] == 1
    assert parser_census["missing_receipt_count"] == 1


def test_raw_materialization_snapshot_accepts_parser_confirmed_empty_non_session(tmp_path: Path) -> None:
    """A real parser census may authoritatively establish no sessions."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"empty-census"}}\n',
            source_path="codex/empty-census.jsonl",
            canonical_source_path="codex/empty-census.jsonl",
            acquired_at_ms=1,
            post_parse=True,
        )

    from polylogue.storage.sqlite.archive_tiers.revision_governance import (
        publish_prepared_revision_source,
        replace_raw_membership_census,
    )
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
    from polylogue.storage.sqlite.write_lease import write_lease

    with (
        write_lease("test.readiness-census", archive_root=tmp_path),
        PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal,
    ):
        with seal.original_read_snapshot(), seal.source_producer():
            replace_raw_membership_census(
                seal,
                raw_id,
                [],
                parser_fingerprint=raw_authority_parser_fingerprint(),
                censused_at_ms=1,
                revision_authority=None,
            )
        permit = seal.prepare_source_mutation()
        publish_prepared_revision_source(seal, permit)

    parser_census = cast(
        Mapping[str, object], raw_materialization_readiness_snapshot(tmp_path)["raw_authority_parser_census"]
    )

    assert parser_census["complete_count"] == 1
    assert parser_census["incomplete_count"] == 0


def test_raw_materialization_snapshot_streams_parser_census_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bounded status projection iterates the real SQLite census cursor."""
    from polylogue.archive.revision_authority import RawRevisionEnvelope, RawRevisionKind
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"stream-census"}}\n',
            source_path="codex/stream-census.jsonl",
            canonical_source_path="codex/stream-census.jsonl",
            acquired_at_ms=1,
            revision=RawRevisionEnvelope("codex:stream-census", RawRevisionKind.FULL, "v1", 0),
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_authority_parser_census (
                raw_id, parser_fingerprint, status, logical_keys_json, detail
            ) VALUES (?, ?, 'complete', '["codex:stream-census"]', '')
            """,
            (raw_id, raw_authority_parser_fingerprint()),
        )
        conn.commit()

    import polylogue.storage.archive_readiness as readiness_mod

    readiness_sqlite = cast(Any, readiness_mod).sqlite3
    original_connect = readiness_sqlite.connect

    class GuardedCursor:
        def __init__(self, cursor: sqlite3.Cursor, *, census_query: bool) -> None:
            self._cursor = cursor
            self._census_query = census_query

        def fetchall(self) -> list[tuple[object, ...]]:
            if self._census_query:
                raise AssertionError("parser census readiness must not materialize its cursor")
            return self._cursor.fetchall()

        def __iter__(self) -> GuardedCursor:
            return self

        def __next__(self) -> tuple[object, ...]:
            return next(self._cursor)

        def __getattr__(self, name: str) -> object:
            return getattr(self._cursor, name)

    class GuardedConnection:
        _connection: sqlite3.Connection

        def __init__(self, connection: sqlite3.Connection) -> None:
            object.__setattr__(self, "_connection", connection)

        def execute(self, sql: str, parameters: object = ()) -> sqlite3.Cursor | GuardedCursor:
            cursor = self._connection.execute(sql, cast(Any, parameters))
            if "raw_authority_parser_census AS c" in sql:
                return GuardedCursor(cursor, census_query=True)
            return cursor

        def __getattr__(self, name: str) -> object:
            return getattr(self._connection, name)

        def __setattr__(self, name: str, value: object) -> None:
            setattr(self._connection, name, value)

    def guarded_connect(*args: object, **kwargs: object) -> GuardedConnection:
        return GuardedConnection(original_connect(*args, **kwargs))

    monkeypatch.setattr(readiness_sqlite, "connect", guarded_connect)

    snapshot = raw_materialization_readiness_snapshot(tmp_path, classify_gaps=False)

    assert cast(Mapping[str, object], snapshot["raw_authority_parser_census"])["complete_count"] == 1


def test_exact_archive_readiness_blocks_parser_census_debt(tmp_path: Path) -> None:
    """Exact readiness consumes source parser debt from its real SQLite projection."""
    from functools import partial

    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.core.enums import Provider
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.operations.raw_observation_derivation import raw_observation_frame
    from polylogue.sources.revision_backfill import PreparedRevisionReplayResult, RevisionCensusResult
    from polylogue.storage.derived.raw import RawObservationDerivation
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.prepared_replay import run_on_convergence_owner

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=(
                b'{"type":"session_meta","payload":{"id":"exact-readiness"}}\n'
                b'{"type":"response_item","payload":{"type":"message","id":"m1","role":"user",'
                b'"content":[{"type":"input_text","text":"exact readiness"}]}}\n'
            ),
            source_path="codex/exact-readiness.jsonl",
            canonical_source_path="codex/exact-readiness.jsonl",
            acquired_at_ms=1,
        )

    blocked = archive_readiness_status(tmp_path)
    assert blocked["surfaces"]["raw_artifacts"]["ready"] is False
    assert "parser_census_incomplete" in blocked["surfaces"]["raw_artifacts"]["blockers"]

    def census_phase(
        compute: BoundedComputeAdapter,
    ) -> list[tuple[str, RevisionCensusResult | PreparedRevisionReplayResult]]:
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute)
        frame = raw_observation_frame(tmp_path)
        replacement = adapter.compute(frame, raw_id)
        receipts: list[tuple[str, RevisionCensusResult | PreparedRevisionReplayResult]] = []
        try:
            # Single-pass convergence: compute commits the census and the
            # classification in place; publication reports them with its replay.
            assert not replacement.needs_source_census
            admit_stage_write(
                "test.readiness.census",
                partial(
                    adapter.publish,
                    frame,
                    replacement,
                    phase_receipt=lambda kind, receipt: receipts.append((kind, receipt)),
                ),
            )
        finally:
            replacement.close()
        return receipts

    receipts = run_on_convergence_owner(tmp_path, "test.readiness.census", census_phase)
    assert [kind for kind, _receipt in receipts] == ["census", "classification", "replay"]
    assert receipts[0][1].scanned == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        receipt = conn.execute(
            "SELECT status, logical_keys_json, detail FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
        ).fetchone()
    assert receipt is not None
    assert receipt[0] == "complete"
    assert receipt[1] == '["codex-session:exact-readiness"]'
    assert str(receipt[2]).startswith("parser-observed:")

    ready = archive_readiness_status(tmp_path)
    assert ready["surfaces"]["raw_artifacts"]["ready"] is True


def test_exact_readiness_keeps_complete_census_with_pending_replay_blocked(tmp_path: Path) -> None:
    """A committed Source phase cannot certify its unwritten Index replay."""
    from functools import partial

    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.core.enums import Provider
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.operations.raw_observation_derivation import raw_observation_frame
    from polylogue.storage.derived.raw import RawObservationDerivation
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.prepared_replay import run_on_convergence_owner

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"pending-readiness"}}\n',
            source_path="codex/pending-readiness.jsonl",
            canonical_source_path="codex/pending-readiness.jsonl",
            acquired_at_ms=1,
        )

    class StopAfterCensusError(Exception):
        pass

    def census_only(compute: BoundedComputeAdapter) -> None:
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute)
        frame = raw_observation_frame(tmp_path)
        replacement = adapter.compute(frame, raw_id)

        def stop(phase: str, _receipt: object) -> None:
            if phase == "census":
                raise StopAfterCensusError

        try:
            admit_stage_write(
                "test.readiness.partial", partial(adapter.publish, frame, replacement, phase_receipt=stop)
            )
        finally:
            replacement.close()

    with pytest.raises(StopAfterCensusError):
        run_on_convergence_owner(tmp_path, "test.readiness.partial", census_only)
    projection = raw_materialization_readiness_snapshot(tmp_path, classify_gaps=False)
    assert cast(Mapping[str, object], projection["raw_authority_parser_census"])["incomplete_count"] == 0
    assert projection["unchecked"] == 1
    surface = archive_readiness_status(tmp_path)["surfaces"]["raw_artifacts"]
    assert surface["ready"] is False
    assert "unchecked" in surface["blockers"]
    assert surface["evidence"]["materialization"]["state"] == assess_raw_materialization(projection).state.value


def test_exact_readiness_keeps_source_authority_blocker_visible(tmp_path: Path) -> None:
    """Current parser and Index rows do not erase an unresolved frontier blocker."""
    import asyncio

    from polylogue.core.enums import Provider
    from tests.infra.empty_managed_index import mutate_fixture_database
    from tests.infra.retained_replay import publish_retained_payload

    asyncio.run(
        publish_retained_payload(
            tmp_path,
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"blocked-readiness"}}\n',
            source_path="codex/blocked-readiness.jsonl",
            acquired_at_ms=1,
        )
    )
    mutate_fixture_database(
        tmp_path / "source.db",
        "INSERT INTO raw_authority_blockers(blocker_id,plan_input_digest,reason,expected_json,observed_json,created_at_ms) "
        "VALUES ('readiness-blocker',?,'unresolved_authority','{}','{}',1)",
        ("0" * 64,),
    )
    projection = raw_materialization_readiness_snapshot(tmp_path, classify_gaps=False)
    assert projection["unchecked"] == 0
    assert projection["raw_authority_blocker_count"] == 1
    surface = archive_readiness_status(tmp_path)["surfaces"]["raw_artifacts"]
    assert surface["ready"] is False
    assert "raw_authority_blocker_count" in surface["blockers"]
    assert surface["evidence"]["materialization"]["state"] == assess_raw_materialization(projection).state.value


def test_archive_readiness_status_uses_active_index_when_conventional_path_is_shadowed(tmp_path: Path) -> None:
    """Surface counts and parser census must come from one pointer-selected generation."""
    initialize_active_archive_root(tmp_path)
    active_index = tmp_path / "generations" / "candidate" / "index.db"
    active_index.parent.mkdir(parents=True)
    shadow_index = tmp_path / "index.db"
    with sqlite3.connect(shadow_index) as source, sqlite3.connect(active_index) as destination:
        source.backup(destination)

    shadow_index.unlink()
    for suffix in ("-wal", "-shm"):
        (tmp_path / f"index.db{suffix}").unlink(missing_ok=True)
    with sqlite3.connect(shadow_index):
        pass
    (tmp_path / ".index-active-pointer").write_text(str(active_index), encoding="utf-8")

    status = archive_readiness_status(tmp_path)

    assert status["checked"] is True
    assert status["reason"] is None
    assert status["surfaces"]["archive_sessions"]["ready"] is True


def test_raw_materialization_readiness_rejects_unresolved_authority_blockers() -> None:
    readiness = {
        "available": True,
        "raw_artifact_count": 1,
        "raw_authority_parser_census": {"available": True},
        "raw_authority_blocker_count": 1,
    }

    assert raw_materialization_ready(readiness) is False
    assert raw_materialization_ready({**readiness, "raw_authority_blocker_count": 0}) is True


def test_raw_materialization_snapshot_classifies_durable_authority_gaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    with sqlite3.connect(source_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
        _seed_rows(
            conn,
            "raw_sessions",
            ("raw_id", "origin", "source_path", "source_index", "revision_authority", "validation_status"),
            [
                (raw_id, "codex-session", "", source_index, authority, "valid")
                for raw_id, source_index, authority in (
                    ("append-quarantine", -1, "quarantined"),
                    ("membership-quarantine", 0, "quarantined"),
                    ("terminal-application", 0, "quarantined"),
                    ("terminal-application-error", 0, "quarantined"),
                    ("authority-pending", -1, "quarantined"),
                    ("append-proven", -1, "byte_proven"),
                    ("membership-settled", 0, "quarantined"),
                    ("membership-incomplete", 0, "quarantined"),
                    ("membership-null", 0, "quarantined"),
                    ("application-deferred", 0, "quarantined"),
                )
            ],
        )
        conn.execute(
            "UPDATE raw_sessions SET parse_error = 'database locked' WHERE raw_id = 'terminal-application-error'"
        )
        _seed_rows(
            conn,
            "raw_membership_census",
            ("raw_id", "status", "member_count", "detail", "revision_authority"),
            [
                (
                    "append-quarantine",
                    "failed",
                    0,
                    BYTE_AUTHORITY_CENSUS_DETAIL,
                    "byte_proven",
                ),
                ("membership-quarantine", "complete", 1, None, None),
                ("membership-settled", "complete", 2, None, None),
                ("membership-incomplete", "complete", 2, None, None),
                ("membership-null", "complete", 1, None, None),
            ],
        )
        _seed_rows(
            conn,
            "raw_session_memberships",
            ("raw_id", "decision"),
            [
                ("membership-quarantine", "ambiguous"),
                ("membership-settled", "applied"),
                ("membership-settled", "superseded_equivalent"),
                ("membership-incomplete", "applied"),
                ("membership-null", None),
            ],
        )
    with sqlite3.connect(index_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        _seed_rows(
            conn,
            "raw_revision_applications",
            ("raw_id", "decision", "detail"),
            [
                ("terminal-application", "superseded", "test"),
                ("terminal-application-error", "superseded", "test"),
                ("application-deferred", "deferred", "ordinary_replay:incomparable_existing_index_state"),
            ],
        )

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot.get("available") is True, snapshot
    assert snapshot["classified"] == 5
    assert snapshot["critical"] == 1
    assert snapshot["actionable"] == 1
    assert snapshot["affected_actionable"] == 1
    assert snapshot["blocked"] == 1
    assert snapshot["unchecked"] == 3
    assert snapshot["affected_unchecked"] == 3
    assert _category_counts(snapshot) == {
        "raw_id_join_gap": 3,
        "skipped": 0,
        "parse_failed": 1,
        "raw_parse_failed": 1,
        "parsed_without_index_session": 0,
        "append-authority-quarantined": 1,
        "append-authority-proven": 1,
        "membership-authority-classified": 2,
        "revision-application-terminal": 1,
        "adoption_deferred": 1,
    }

    def fail_if_classified(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("bounded readiness must not classify individual raw gaps")

    monkeypatch.setattr("polylogue.storage.archive_readiness._classify_raw_gap_rows", fail_if_classified)
    aggregate = raw_materialization_readiness_snapshot(tmp_path, classify_gaps=False)

    assert aggregate["classification"] == "not_run"
    assert aggregate["classified"] == 0
    assert aggregate["unchecked"] == 9
    assert aggregate["actionable"] == 0


def test_raw_materialization_snapshot_reads_append_census_writer_contract(tmp_path: Path) -> None:
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"append":true}\n',
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=1,
        )

    from polylogue.storage.sqlite.archive_tiers.revision_governance import (
        publish_prepared_revision_source,
        replace_raw_membership_census,
    )
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
    from polylogue.storage.sqlite.write_lease import write_lease

    with (
        write_lease("test.readiness-census", archive_root=tmp_path),
        PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal,
    ):
        with seal.original_read_snapshot(), seal.source_producer():
            replace_raw_membership_census(
                seal,
                raw_id,
                None,
                parser_fingerprint=raw_authority_parser_fingerprint(),
                censused_at_ms=0,
                detail=BYTE_AUTHORITY_CENSUS_DETAIL,
                revision_authority=RawRevisionAuthority.BYTE_PROVEN,
            )
        permit = seal.prepare_source_mutation()
        publish_prepared_revision_source(seal, permit)

    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["classified"] == 1
    assert snapshot["unchecked"] == 0
    assert _category_counts(snapshot)["append-authority-quarantined"] == 1
    parser_census = cast(Mapping[str, object], snapshot["raw_authority_parser_census"])
    assert parser_census["complete_count"] == 1
    assert parser_census["incomplete_count"] == 0


def test_raw_materialization_snapshot_ignores_skipped_raw_rows(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    with sqlite3.connect(source_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
        _seed_rows(
            conn,
            "raw_sessions",
            ("raw_id", "origin", "validation_status", "parse_error", "parsed_at_ms"),
            [
                ("raw-materializable", "chatgpt-export", "valid", None, 123),
                ("raw-skipped", "aistudio-drive", "skipped", None, None),
            ],
        )
    with sqlite3.connect(index_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["available"] is True
    assert snapshot["raw_artifact_count"] == 1
    assert snapshot["materialized_raw_artifact_count"] == 0
    assert snapshot["archive_session_count"] == 0
    assert snapshot["join_gap_count"] == 1
    assert snapshot["total"] == 1
    assert snapshot["unchecked"] == 1
    assert snapshot["affected_unchecked"] == 1
    assert snapshot["category_counts"] == {
        "raw_id_join_gap": 1,
        "skipped": 0,
        "parse_failed": 0,
        "raw_parse_failed": 0,
        "parsed_without_index_session": 1,
    }
    assert snapshot["source_family_counts"] == {"chatgpt-export": 1}


def test_raw_materialization_snapshot_counts_raw_artifacts_once(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    with sqlite3.connect(source_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
        _seed_rows(
            conn,
            "raw_sessions",
            ("raw_id", "origin", "validation_status", "parse_error", "parsed_at_ms"),
            [
                ("raw-shared", "claude-code-session", "valid", None, 123),
                ("raw-gap", "codex-session", "valid", None, 124),
            ],
        )
    with sqlite3.connect(index_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        _seed_rows(
            conn,
            "sessions",
            ("session_id", "raw_id"),
            [
                ("session-one", "raw-shared"),
                ("session-two", "raw-shared"),
            ],
        )

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["raw_artifact_count"] == 2
    assert snapshot["materialized_raw_artifact_count"] == 1
    assert snapshot["archive_session_count"] == 2
    assert snapshot["join_gap_count"] == 1
    assert snapshot["total"] == 1
    assert snapshot["source_family_counts"] == {"codex-session": 1}


def test_raw_materialization_snapshot_marks_parse_failures_actionable(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    with sqlite3.connect(source_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
        _seed_rows(
            conn,
            "raw_sessions",
            ("raw_id", "origin", "validation_status", "parse_error", "parsed_at_ms"),
            [
                ("raw-failed-one", "codex-session", "failed", "bad json", None),
                ("raw-failed-two", "aistudio-drive", "failed", "bad json", None),
            ],
        )
    with sqlite3.connect(index_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["total"] == 2
    assert snapshot["critical"] == 2
    assert snapshot["actionable"] == 2
    assert snapshot["affected_actionable"] == 2
    assert snapshot["unchecked"] == 0
    assert snapshot["affected_unchecked"] == 0
    assert snapshot["category_counts"] == {
        "raw_id_join_gap": 0,
        "skipped": 0,
        "parse_failed": 2,
        "raw_parse_failed": 2,
        "parsed_without_index_session": 0,
    }


def test_raw_materialization_snapshot_classifies_native_aliases(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    with sqlite3.connect(source_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
        _seed_rows(
            conn,
            "raw_sessions",
            (
                "raw_id",
                "origin",
                "native_id",
                "source_path",
                "blob_hash",
                "validation_status",
                "parse_error",
                "parsed_at_ms",
            ),
            [("raw-alias", "chatgpt-export", "conv-1", "capture.json", bytes.fromhex("11" * 32), "passed", None, 123)],
        )
        _seed_rows(
            conn,
            "raw_sessions",
            (
                "raw_id",
                "origin",
                "native_id",
                "source_path",
                "blob_hash",
                "validation_status",
                "parse_error",
                "parsed_at_ms",
            ),
            [("older-raw", "chatgpt-export", "conv-1", "older.json", bytes.fromhex("12" * 32), "passed", None, 122)],
        )
    _write_blob(tmp_path, "11" * 32)
    with sqlite3.connect(index_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        _seed_rows(
            conn,
            "sessions",
            ("session_id", "origin", "native_id", "raw_id"),
            [("chatgpt-export:conv-1", "chatgpt-export", "conv-1", "older-raw")],
        )

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["classification"] == "cheap_projection"
    assert snapshot["total"] == 1
    assert snapshot["classified"] == 1
    assert snapshot["affected_classified"] == 1
    assert snapshot["unchecked"] == 0
    assert snapshot["affected_unchecked"] == 0
    counts = _category_counts(snapshot)
    assert counts["materialized-alias"] == 1
    assert counts["raw_id_join_gap"] == 0


def test_raw_materialization_snapshot_reports_an_alias_whose_own_blob_is_gone(tmp_path: Path) -> None:
    """An alias reconciles identity, not bytes: a missing blob stays owed work.

    Anti-vacuity: drop the blob-existence check ahead of the alias test and the
    row is classified ``materialized-alias``, so ``affected_actionable`` falls
    back to 0 and a genuinely lost newer snapshot reads as a clean archive.
    """
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    with sqlite3.connect(source_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
        _seed_rows(
            conn,
            "raw_sessions",
            (
                "raw_id",
                "origin",
                "native_id",
                "source_path",
                "blob_hash",
                "validation_status",
                "parse_error",
                "parsed_at_ms",
            ),
            [
                # The newer snapshot: same provider session, its own blob gone.
                ("raw-newer", "chatgpt-export", "conv-1", "newer.json", bytes.fromhex("11" * 32), "passed", None, 123),
                ("older-raw", "chatgpt-export", "conv-1", "older.json", bytes.fromhex("12" * 32), "passed", None, 122),
            ],
        )
    _write_blob(tmp_path, "12" * 32)
    with sqlite3.connect(index_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        _seed_rows(
            conn,
            "sessions",
            ("session_id", "origin", "native_id", "raw_id"),
            [("chatgpt-export:conv-1", "chatgpt-export", "conv-1", "older-raw")],
        )

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    counts = _category_counts(snapshot)
    assert counts[RAW_ALIAS_BLOB_MISSING_CATEGORY] == 1
    assert counts.get("materialized-alias", 0) == 0
    assert snapshot["classified"] == 0
    assert snapshot["affected_actionable"] == 1
    assert snapshot["unchecked"] == 0


def test_raw_materialization_snapshot_classifies_stale_decode_aliases(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    with sqlite3.connect(source_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
        _seed_rows(
            conn,
            "raw_sessions",
            (
                "raw_id",
                "origin",
                "native_id",
                "source_path",
                "blob_hash",
                "validation_status",
                "parse_error",
                "parsed_at_ms",
            ),
            [
                (
                    "raw-stale-error",
                    "codex-session",
                    "session-1",
                    "session-1.jsonl",
                    bytes.fromhex("11" * 32),
                    "failed",
                    "decode: [Errno 2] No such file or directory: '/tmp/archive/blob/11/1111'",
                    None,
                ),
                (
                    "raw-current",
                    "codex-session",
                    "session-1",
                    "current.jsonl",
                    bytes.fromhex("12" * 32),
                    "passed",
                    None,
                    123,
                ),
            ],
        )
    # The decode error is stale precisely because the blob is present again.
    _write_blob(tmp_path, "11" * 32)
    with sqlite3.connect(index_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        _seed_rows(
            conn,
            "sessions",
            ("session_id", "origin", "native_id", "raw_id"),
            [("codex-session:session-1", "codex-session", "session-1", "raw-current")],
        )

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["actionable"] == 0
    assert snapshot["affected_actionable"] == 0
    assert snapshot["classified"] == 1
    counts = _category_counts(snapshot)
    assert counts["materialized-alias"] == 1
    assert counts["parse_failed"] == 0
    assert counts["raw_parse_failed"] == 1


def test_raw_materialization_snapshot_classifies_dangling_index_raw_link_as_lost_source_evidence(
    tmp_path: Path,
) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    with sqlite3.connect(source_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
        _seed_rows(
            conn,
            "raw_sessions",
            (
                "raw_id",
                "origin",
                "native_id",
                "source_path",
                "blob_hash",
                "validation_status",
                "parse_error",
                "parsed_at_ms",
            ),
            [("raw-new", "chatgpt-export", "conv-1", "capture.json", bytes.fromhex("11" * 32), "passed", None, 123)],
        )
    with sqlite3.connect(index_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        _seed_rows(
            conn,
            "sessions",
            ("session_id", "origin", "native_id", "raw_id"),
            [("chatgpt-export:conv-1", "chatgpt-export", "conv-1", "older-missing-raw")],
        )

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["lost_source_evidence_count"] == 1
    assert snapshot["classified"] == 1
    assert snapshot["unchecked"] == 0
    assert raw_materialization_ready(snapshot) is False
    counts = _category_counts(snapshot)
    assert counts.get("materialized-alias", 0) == 0
    assert counts["lost-source-evidence-alias"] == 1
    assert counts["raw_id_join_gap"] == 0


def test_lost_source_evidence_samples_include_generated_session_identity(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_runtime_source_fixture(tmp_path / "source.db")
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute(
            """
            INSERT INTO sessions (native_id, origin, raw_id, title, content_hash)
            VALUES ('missing', 'codex-session', 'raw-missing', 'missing raw', ?)
            """,
            (bytes(32),),
        )
        conn.commit()

    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["lost_source_evidence_count"] == 1
    samples = cast(list[dict[str, object]], snapshot["lost_source_evidence_samples"])
    assert samples[0]["session_id"] == "codex-session:missing"
    assert samples[0]["missing_raw_id"] == "raw-missing"


def test_raw_materialization_snapshot_marks_reverse_authority_query_failure_unavailable(
    tmp_path: Path,
) -> None:
    """A failed lost-source count cannot become a healthy zero."""

    with sqlite3.connect(tmp_path / "source.db") as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.executescript(
            """
            -- A real table (the reverse query skips anything else) whose
            -- raw_id fails only when the lost-source count reads it; the
            -- generated column is added after the row so insertion succeeds.
            CREATE TABLE sessions (raw_value INTEGER NOT NULL);
            INSERT INTO sessions (raw_value) VALUES (-9223372036854775808);
            ALTER TABLE sessions ADD COLUMN raw_id INTEGER GENERATED ALWAYS AS (abs(raw_value)) VIRTUAL;
            """
        )

    _stamp_index_as_current_schema(tmp_path / "index.db")
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["available"] is False
    assert "integer overflow" in str(snapshot["error"])
    assert raw_materialization_ready(snapshot) is False


def test_raw_materialization_snapshot_returns_unavailable_for_invalid_active_pointer(tmp_path: Path) -> None:
    (tmp_path / ".index-active-pointer").write_text("relative/index.db", encoding="utf-8")

    snapshot = raw_materialization_readiness_snapshot(tmp_path)
    status = archive_readiness_status(tmp_path)

    assert snapshot["available"] is False
    assert "invalid active index pointer" in str(snapshot["error"])
    assert status["checked"] is False
    assert "invalid active index pointer" in str(status["reason"])


def test_raw_materialization_snapshot_classifies_source_path_aliases(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    source_path = tmp_path / "cache" / "native-alias_1.jsonl.txt.json"
    source_path.parent.mkdir()
    source_path.write_text("{}", encoding="utf-8")
    with sqlite3.connect(source_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
        _seed_rows(
            conn,
            "raw_sessions",
            (
                "raw_id",
                "origin",
                "native_id",
                "source_path",
                "blob_hash",
                "validation_status",
                "parse_error",
                "parsed_at_ms",
            ),
            [
                (
                    "raw-source-alias",
                    "claude-code-session",
                    None,
                    str(source_path),
                    bytes.fromhex("12" * 32),
                    "passed",
                    None,
                    123,
                )
            ],
        )
        _seed_rows(
            conn,
            "raw_sessions",
            (
                "raw_id",
                "origin",
                "native_id",
                "source_path",
                "blob_hash",
                "validation_status",
                "parse_error",
                "parsed_at_ms",
            ),
            [
                (
                    "older-raw",
                    "claude-code-session",
                    "native-alias",
                    "older.json",
                    bytes.fromhex("13" * 32),
                    "passed",
                    None,
                    122,
                )
            ],
        )
    _write_blob(tmp_path, "12" * 32)
    with sqlite3.connect(index_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        _seed_rows(
            conn,
            "sessions",
            ("session_id", "origin", "native_id", "raw_id"),
            [("claude-code-session:native-alias", "claude-code-session", "native-alias", "older-raw")],
        )

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["classified"] == 1
    assert snapshot["unchecked"] == 0
    assert _category_counts(snapshot)["materialized-alias"] == 1


def test_raw_materialization_snapshot_keeps_unreceipted_non_session_shaped_raw_unchecked(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    blob = tmp_path / "blob" / "dd" / ("dd" * 31)
    blob.parent.mkdir(parents=True)
    blob.write_text('{"type":"file-history-snapshot","messageId":"m1"}\n', encoding="utf-8")
    with sqlite3.connect(source_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
        _seed_rows(
            conn,
            "raw_sessions",
            (
                "raw_id",
                "origin",
                "native_id",
                "source_path",
                "blob_hash",
                "validation_status",
                "parse_error",
                "parsed_at_ms",
            ),
            [
                (
                    "raw-sidecar",
                    "claude-code-session",
                    "sidecar-native",
                    str(tmp_path / "sidecar.jsonl"),
                    bytes.fromhex("dd" * 32),
                    "passed",
                    None,
                    123,
                )
            ],
        )
    with sqlite3.connect(index_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["classified"] == 0
    assert snapshot["unchecked"] == 1
    counts = _category_counts(snapshot)
    assert counts["raw_id_join_gap"] == 1


def test_raw_materialization_snapshot_accepts_currently_censused_typed_empty_claude_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real raw-only intake is complete by typed artifact and current parser receipts."""
    from types import SimpleNamespace

    import polylogue.paths as polylogue_paths
    import polylogue.sources.live.watcher as live_watcher
    from polylogue.sources.live.batch import LiveBatchProcessor
    from polylogue.sources.live.cursor import CursorStore
    from polylogue.sources.live.watcher import default_sources
    from tests.infra.archive_templates import bootstrap_archive_root
    from tests.infra.raw_owner_routes import run_ingest_files

    bootstrap_archive_root(tmp_path)
    claude_root = tmp_path / "neutral-home" / ".claude"
    claude_root.mkdir(parents=True)
    source_path = claude_root / "history.jsonl"
    source_path.write_bytes(b"")
    monkeypatch.setattr(polylogue_paths, "claude_code_path", lambda: claude_root / "projects")

    source = next(item for item in default_sources() if item.name == "claude-code-history")
    processor = LiveBatchProcessor(
        SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db")),
        (source,),
        cursor=CursorStore(tmp_path / "ops.db"),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    metrics = run_ingest_files(processor, [source_path], emit_event=False)
    assert metrics.excluded_file_count == 1 and metrics.failed_file_count == 0, metrics

    with sqlite3.connect(tmp_path / "source.db") as conn:
        raw_id = str(conn.execute("SELECT raw_id FROM raw_sessions").fetchone()[0])
        assert conn.execute(
            "SELECT artifact_kind, parse_as_session, schema_eligible, support_status, decode_error, "
            "malformed_jsonl_lines FROM raw_artifacts WHERE raw_id=?",
            (raw_id,),
        ).fetchall() == [("prompt_history_log", 0, 0, "unknown", None, 0)]
        assert conn.execute(
            "SELECT status, member_count, parser_fingerprint FROM raw_membership_census WHERE raw_id=?", (raw_id,)
        ).fetchone() == ("non_session", 0, raw_authority_parser_fingerprint())
        assert conn.execute(
            "SELECT parser_fingerprint, status, logical_keys_json FROM raw_authority_parser_census WHERE raw_id=?",
            (raw_id,),
        ).fetchone() == (raw_authority_parser_fingerprint(), "complete", "[]")

    snapshot = raw_materialization_readiness_snapshot(tmp_path, classify_gaps=True)
    assert snapshot["raw_artifact_count"] == 1
    assert snapshot["materialized_raw_artifact_count"] == 0
    assert snapshot["join_gap_count"] == 1
    assert snapshot["classified"] == 1
    assert snapshot["affected_unchecked"] == 0
    assert _category_counts(snapshot)["parsed-non-session-artifact"] == 1
    assert raw_materialization_ready(snapshot) is True

    # Default daemon status uses the bounded projection. It must consume the
    # same durable classification without turning the raw join into a session.
    import polylogue.daemon.status as daemon_status

    monkeypatch.setattr(daemon_status, "archive_root", lambda: tmp_path)
    fast_status = daemon_status._raw_materialization_readiness_info(classify_gaps=False)
    assert fast_status.raw_artifact_count == 1
    assert fast_status.materialized_raw_artifact_count == 0
    assert fast_status.join_gap_count == 1
    assert fast_status.unchecked == 0
    assert fast_status.category_counts["parsed-non-session-artifact"] == 1
    assert daemon_status._component_from_raw_materialization_readiness(fast_status).state == "ready"

    # The zero-member census proves the parser's session set, but every raw
    # artifact in the cohort must agree with that non-session reading.
    siblings = (
        ("session-sibling", 1, 1, None, 0),
        ("schema-eligible-sibling", 0, 1, None, 0),
        ("decode-error-sibling", 0, 0, "synthetic decode refusal", 0),
        ("malformed-sibling", 0, 0, None, 1),
    )
    for artifact_id, parse_as_session, schema_eligible, decode_error, malformed_lines in siblings:
        with sqlite3.connect(tmp_path / "source.db") as conn:
            conn.execute(
                """INSERT INTO raw_artifacts (
                    artifact_id, raw_id, origin, source_path, source_index,
                    artifact_kind, support_status, classification_reason,
                    parse_as_session, schema_eligible, malformed_jsonl_lines,
                    decode_error, first_observed_at_ms, last_observed_at_ms
                ) VALUES (?, ?, 'claude-code-session', ?, 1, 'session_record_stream',
                          'supported_parseable', 'synthetic mixed cohort', ?, ?, ?, ?, 1, 1)""",
                (
                    f"mixed-{artifact_id}",
                    raw_id,
                    f"{artifact_id}.jsonl",
                    parse_as_session,
                    schema_eligible,
                    malformed_lines,
                    decode_error,
                ),
            )
            conn.commit()
        mixed = raw_materialization_readiness_snapshot(tmp_path, classify_gaps=True)
        assert mixed["classified"] == 0
        assert mixed["affected_unchecked"] == 1
        assert raw_materialization_ready(mixed) is False
        fast_mixed = daemon_status._raw_materialization_readiness_info(classify_gaps=False)
        assert fast_mixed.unchecked == 1
        assert daemon_status._component_from_raw_materialization_readiness(fast_mixed).state == "degraded"
        with sqlite3.connect(tmp_path / "source.db") as conn:
            conn.execute("DELETE FROM raw_artifacts WHERE artifact_id=?", (f"mixed-{artifact_id}",))
            conn.commit()

    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_artifacts SET artifact_kind='terminal_unsupported_shape', "
            "support_status='unsupported_parseable' WHERE raw_id=?",
            (raw_id,),
        )
        conn.commit()
    unsupported = raw_materialization_readiness_snapshot(tmp_path, classify_gaps=True)
    assert unsupported["classified"] == 0
    assert unsupported["affected_unchecked"] == 1
    assert raw_materialization_ready(unsupported) is False
    assert daemon_status._raw_materialization_readiness_info(classify_gaps=False).unchecked == 1

    # Durable parser receipts cannot erase a later validation refusal or
    # retained decode failure for the same typed artifact.
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_artifacts SET artifact_kind='prompt_history_log', support_status='unknown' WHERE raw_id=?",
            (raw_id,),
        )
        conn.execute("UPDATE raw_sessions SET validation_status='failed' WHERE raw_id=?", (raw_id,))
        conn.commit()
    refused = raw_materialization_readiness_snapshot(tmp_path, classify_gaps=True)
    assert refused["classified"] == 0
    assert refused["affected_actionable"] == 1
    assert _category_counts(refused)["parse_failed"] == 1
    assert daemon_status._raw_materialization_readiness_info(classify_gaps=False).unchecked == 1
    assert raw_materialization_ready(refused) is False

    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET validation_status=NULL, parse_error='decode: malformed retained bytes' "
            "WHERE raw_id=?",
            (raw_id,),
        )
        conn.commit()
    decode_failed = raw_materialization_readiness_snapshot(tmp_path, classify_gaps=True)
    assert decode_failed["classified"] == 0
    assert decode_failed["affected_actionable"] == 1
    assert _category_counts(decode_failed)["parse_failed"] == 1
    assert raw_materialization_ready(decode_failed) is False


def test_raw_materialization_snapshot_keeps_unexplained_gaps_unchecked(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    blob = tmp_path / "blob" / "dd" / ("dd" * 31)
    blob.parent.mkdir(parents=True)
    blob.write_text('{"type":"file-history-snapshot","messageId":"m1"}\n', encoding="utf-8")
    with sqlite3.connect(source_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
        _seed_rows(
            conn,
            "raw_sessions",
            (
                "raw_id",
                "origin",
                "native_id",
                "source_path",
                "blob_hash",
                "validation_status",
                "parse_error",
                "parsed_at_ms",
            ),
            [
                (
                    "raw-sidecar",
                    "claude-code-session",
                    "sidecar-native",
                    str(tmp_path / "sidecar.jsonl"),
                    bytes.fromhex("dd" * 32),
                    "passed",
                    None,
                    123,
                ),
                (
                    "raw-session-shaped",
                    "codex-session",
                    "codex-native",
                    str(tmp_path / "session.jsonl"),
                    bytes.fromhex("ee" * 32),
                    "passed",
                    None,
                    123,
                ),
                (
                    "raw-skipped",
                    "chatgpt-export",
                    "skipped-native",
                    str(tmp_path / "skipped.json"),
                    bytes.fromhex("ff" * 32),
                    "skipped",
                    None,
                    123,
                ),
            ],
        )
    with sqlite3.connect(index_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["raw_artifact_count"] == 2
    assert snapshot["total"] == 2
    assert snapshot["classified"] == 0
    assert snapshot["unchecked"] == 2
    assert snapshot["affected_unchecked"] == 2
    counts = _category_counts(snapshot)
    assert "parsed-non-session-artifact" not in counts
    assert counts["raw_id_join_gap"] == 2


def test_raw_materialization_snapshot_classifies_same_native_lost_source_evidence(
    tmp_path: Path,
) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    with sqlite3.connect(source_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
        _seed_rows(
            conn,
            "raw_sessions",
            (
                "raw_id",
                "origin",
                "native_id",
                "source_path",
                "blob_hash",
                "validation_status",
                "parse_error",
                "parsed_at_ms",
            ),
            [
                (
                    "newer-raw",
                    "claude-code-session",
                    "session-native",
                    str(tmp_path / "session-native.jsonl"),
                    bytes.fromhex("aa" * 32),
                    "passed",
                    None,
                    123,
                )
            ],
        )
    with sqlite3.connect(index_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        _seed_rows(
            conn,
            "sessions",
            ("session_id", "origin", "native_id", "raw_id"),
            [
                (
                    "claude-code-session:session-native",
                    "claude-code-session",
                    "session-native",
                    "missing-older-raw",
                )
            ],
        )

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["lost_source_evidence_count"] == 1
    assert snapshot["classified"] == 1
    assert snapshot["unchecked"] == 0
    assert snapshot["affected_unchecked"] == 0
    assert raw_materialization_ready(snapshot) is False
    counts = _category_counts(snapshot)
    assert counts["lost-source-evidence-alias"] == 1
    assert counts["raw_id_join_gap"] == 0


def test_raw_materialization_ready_rejects_failed_debt_classifier() -> None:
    """A readiness dict carrying debt_classifier_error must not read as ready.

    paths._merge_raw_materialization_debt records classifier failures under
    this key; the composed readiness contract requires the classifier, so a
    recorded failure blocks the ready claim even when every structural count
    is clean (removing the predicate's debt_classifier_error check fails this).
    """
    clean = {
        "available": True,
        "raw_artifact_count": 1,
        "raw_authority_parser_census": {"available": True},
        "critical": 0,
        "warning": 0,
        "actionable": 0,
        "blocked": 0,
        "affected_actionable": 0,
        "affected_blocked": 0,
        "affected_open": 0,
        "lost_source_evidence_count": 0,
        "unchecked": 0,
        "affected_unchecked": 0,
    }
    assert raw_materialization_ready(clean) is True
    assert raw_materialization_ready({**clean, "debt_classifier_error": "RuntimeError: ops.db locked"}) is False


def test_claude_workflow_materialization_status_missing_ops_db_returns_none(tmp_path: Path) -> None:
    assert claude_workflow_materialization_status(tmp_path / "ops.db") is None


def test_claude_workflow_materialization_status_reads_latest_stage_event(tmp_path: Path) -> None:
    """Reads back exactly what operations/claude_workflow_convergence.py's claude_workflow
    stage persists via record_daemon_stage_event -- the wiring this bead adds
    so a materialization gap count survives past one log line (bd polylogue-uh9l).
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
    from polylogue.storage.sqlite.archive_tiers.ops_write import record_daemon_stage_event
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    ops_db = tmp_path / "ops.db"
    conn = sqlite3.connect(ops_db)
    try:
        initialize_archive_tier(conn, ArchiveTier.OPS)
        record_daemon_stage_event(
            conn,
            stage=CLAUDE_WORKFLOW_STAGE_NAME,
            status="gaps",
            observed_at_ms=1_700_000_000_000,
            payload={"gap_count": 2, "gaps": ["missing agent metadata for transcript agent-a", "unresolved call x"]},
        )
        conn.commit()
    finally:
        conn.close()

    status = claude_workflow_materialization_status(ops_db)
    assert status is not None
    assert status["status"] == "gaps"
    assert status["gap_count"] == 2
    assert status["gaps"] == ["missing agent metadata for transcript agent-a", "unresolved call x"]
    assert status["observed_at_ms"] == 1_700_000_000_000


def _pinned_index_over(tmp_path: Path) -> sqlite3.Connection:
    """An index reader with the source tier attached, as an operation pins it."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database

    initialize_runtime_source_fixture(tmp_path / "source.db")
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    conn = sqlite3.connect(tmp_path / "index.db")
    conn.execute("ATTACH DATABASE ? AS source_tier", (str(tmp_path / "source.db"),))
    return conn


def test_pinned_materialization_readiness_degrades_like_its_path_twin(tmp_path: Path) -> None:
    """A degraded pinned tier must report unavailable, not raise.

    ``raw_materialization_readiness_snapshot`` wraps its reads and returns
    ``{"available": False, "error": ...}``. Its pinned twin, added alongside it,
    had no handler, so a busy reader or a column the snapshot predates raised
    out of ``_raw_materialization_status`` -- which runs unconditionally on
    every status poll.

    Anti-vacuity: remove the ``except (OSError, sqlite3.Error)`` clause from
    ``raw_materialization_readiness_from_pinned_index`` and this goes red with
    ``sqlite3.OperationalError: no such column: s.raw_id``.
    """
    from polylogue.storage.archive_readiness import raw_materialization_readiness_from_pinned_index

    conn = _pinned_index_over(tmp_path)
    try:
        # Drop a column the projection selects: readable tier, unreadable query.
        conn.execute("DROP INDEX IF EXISTS idx_sessions_raw_id")
        # Remove the fixture's dependent triggers so SQLite can create the
        # deliberately unreadable projection, rather than refusing its setup.
        dependent = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='sessions' AND sql LIKE '%raw_id%'"
        ).fetchall()
        assert dependent
        for (trigger,) in dependent:
            conn.execute(f'DROP TRIGGER "{trigger}"')
        conn.execute("ALTER TABLE sessions DROP COLUMN raw_id")
        conn.commit()
        result = raw_materialization_readiness_from_pinned_index(conn, archive_root=tmp_path)
    finally:
        conn.close()

    assert result["available"] is False
    assert "raw_id" in str(result["error"])


def test_pinned_readiness_status_degrades_like_its_path_twin(tmp_path: Path) -> None:
    """The same missing failure contract on the readiness-status twin.

    ``archive_readiness_status`` returns ``{"checked": False, "reason": ...}``
    on ``sqlite3.Error``; the pinned variant raised instead.

    Anti-vacuity: remove the ``except (OSError, sqlite3.Error)`` clause from
    ``archive_readiness_status_from_connections`` and this goes red.
    """
    from polylogue.storage.archive_readiness import archive_readiness_status_from_connections

    conn = _pinned_index_over(tmp_path)
    try:
        conn.execute("DROP VIEW IF EXISTS threads")
        conn.execute("ALTER TABLE sessions DROP COLUMN message_count")
        conn.commit()
        result = archive_readiness_status_from_connections(conn, None, raw_materialization_readiness=None)
    finally:
        conn.close()

    assert result["checked"] is False
    assert "message_count" in str(result["reason"])


def test_pinned_materialization_readiness_defaults_to_the_bounded_contract(tmp_path: Path) -> None:
    """Exact gap classification is a diagnostic read, not a status poll.

    The path twin's caller passes ``classify_gaps=False`` because exact
    classification opens a blob and reads JSONL per unmaterialized raw. The
    pinned twin defaulted it to True and its only caller omits the kwarg, so
    every status poll walked the whole unmaterialized corpus -- which, during a
    from-scratch rebuild, is nearly every raw.

    Anti-vacuity: flip the default back to True and this goes red.
    """
    import inspect

    from polylogue.storage.archive_readiness import raw_materialization_readiness_from_pinned_index

    signature = inspect.signature(raw_materialization_readiness_from_pinned_index)
    assert signature.parameters["classify_gaps"].default is False


def test_converged_archive_reports_its_real_materialization_counts(tmp_path: Path) -> None:
    """A fully materialized archive is not an empty one.

    The counters query drove its join from the *gap* rows. A converged archive
    has none, and a join with an empty side leaves every non-aggregated total
    NULL, which the ``int(row[...] or 0)`` coercions turned into zero -- so an
    archive holding one raw artifact and one session reported
    ``raw_artifact_count == 0`` and ``archive_session_count == 0``, the
    unmeasured-state-as-a-positive-result shape.

    Anti-vacuity: restore ``FROM gaps CROSS JOIN materialization CROSS JOIN
    session_count`` and every assertion below goes red with ``0``.
    """
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.live_ingest import write_index_session

    initialize_active_archive_root(tmp_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="converged-counts",
        updated_at="2026-01-02T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="converged prose")],
            )
        ],
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"converged-counts"}}\n',
            source_path="codex/converged-counts.jsonl",
            canonical_source_path="codex/converged-counts.jsonl",
            acquired_at_ms=1,
            post_parse=True,
        )
        session_id = write_index_session(archive, session)

    # Bind the index session to its raw artifact: the index tier is
    # rebuildable, and this is the join the counters read.
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("UPDATE sessions SET raw_id = ? WHERE session_id = ?", (raw_id, session_id))
        conn.commit()

    snapshot = raw_materialization_readiness_snapshot(tmp_path, classify_gaps=False)

    assert snapshot["available"] is True
    assert snapshot["raw_artifact_count"] == 1
    assert snapshot["materialized_raw_artifact_count"] == 1
    assert snapshot["archive_session_count"] == 1
    assert snapshot["join_gap_count"] == 0
    assert snapshot["total"] == 0


def test_archive_sessions_surface_is_computed_not_asserted(tmp_path: Path) -> None:
    """polylogue-bu47u: the sessions surface published ``ready=True`` unconditionally.

    Every sibling surface derives ``ready`` from its blockers; this one was a
    literal, so an index tier whose ``messages`` relation is gone still
    certified the surface while reporting a fabricated ``message_count`` of 0.

    Anti-vacuity: restore ``surface(ready=True, blockers=[])`` for
    ``archive_sessions`` (or drop the relation-presence counts that feed it)
    and this fails; the healthy assertion below fails if the blockers are
    raised unconditionally instead.
    """

    initialize_active_archive_root(tmp_path)

    healthy = archive_readiness_status(tmp_path)
    assert healthy["surfaces"]["archive_sessions"]["ready"] is True
    assert healthy["surfaces"]["archive_sessions"]["blockers"] == []

    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("DROP TABLE IF EXISTS messages_fts")
        conn.execute("DROP TABLE messages")

    blocked = archive_readiness_status(tmp_path)
    sessions = blocked["surfaces"]["archive_sessions"]
    assert sessions["ready"] is False
    assert sessions["blockers"] == ["messages_relation_missing"]
    assert sessions["evidence"]["messages_table_present"] is False
