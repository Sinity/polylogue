"""Raw-observation laws not owned by the retired all-source scanner.

The canonical derivation laws live in ``test_raw_observation_derivation``.
This module keeps the durable raw-failure lifecycle laws that are independent
of a materialization selector, plus canonical postconditions whose fixtures
exercise retained source evidence and component isolation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterable
from contextlib import closing
from functools import partial
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.archive.revision_authority import append_source_revision
from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.enums import ArtifactSupportStatus, Origin, Provider, ValidationMode
from polylogue.core.errors import RawCASFrontierError
from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind
from polylogue.core.stage_admission import admit_stage_write
from polylogue.daemon.derivation import (
    Budget,
    DerivationRegistry,
    DerivationReport,
    converge,
)
from polylogue.daemon.status import raw_failure_info_for_root
from polylogue.logging import capture
from polylogue.operations.intake_adapters import RawMaterializationDiscovery
from polylogue.operations.raw_observation_derivation import raw_observation_frame
from polylogue.storage.derived.raw import RawObservationDerivation, RawObservationReplacement
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.raw_failure_lifecycle import read_raw_failure_lifecycle
from polylogue.storage.raw_retention import RawFrontierBlockedPaths
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceArtifact, upsert_raw_artifact
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.reference_seal import ReferenceSealError, ReferenceSealStaleError
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.index_writer import write_fixture_index_session
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.prepared_replay import run_on_convergence_owner
from tests.infra.raw_owner_routes import converge_pending_raws_async


def _codex_conversation_bytes(session_id: str = "session", text: str = "hi") -> bytes:
    return (
        b'{"type":"session_meta","payload":{"id":"' + session_id.encode() + b'"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"m-'
        + session_id.encode()
        + b'","role":"user","content":[{"type":"input_text","text":"'
        + text.encode()
        + b'"}]}}\n'
    )


def _codex_fork_bytes(session_id: str, parent_id: str, shared_record: bytes) -> bytes:
    return (
        json.dumps(
            {"type": "session_meta", "payload": {"id": session_id, "forked_from_id": parent_id}},
            separators=(",", ":"),
        ).encode()
        + b"\n"
        + shared_record
        + b'{"type":"response_item","payload":{"type":"message","id":"child-tail",'
        b'"role":"assistant","content":[{"type":"output_text","text":"child only"}]}}\n'
    )


def _chatgpt_payload(names: tuple[str, ...]) -> bytes:
    return json.dumps(
        [
            {
                "id": name,
                "title": name,
                "create_time": 1,
                "current_node": "m",
                "mapping": {
                    "m": {
                        "id": "m",
                        "parent": None,
                        "children": [],
                        "message": {
                            "id": "m",
                            "author": {"role": "user"},
                            "create_time": 1,
                            "content": {"content_type": "text", "parts": [name]},
                        },
                    }
                },
            }
            for name in names
        ]
    ).encode()


def _admit(
    root: Path,
    names: tuple[str, ...],
    *,
    path: str = "bundle.json",
    provider: Provider = Provider.CHATGPT,
    payload: bytes | None = None,
    acquired_at_ms: int = 1,
) -> str:
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        return archive.write_raw_payload(
            provider=provider,
            payload=_chatgpt_payload(names) if payload is None else payload,
            source_path=path,
            canonical_source_path=path,
            acquired_at_ms=acquired_at_ms,
        )


def _derive(
    root: Path,
    *,
    limit: int = 128,
    discovery: RawMaterializationDiscovery | None = None,
    validation_mode: ValidationMode = ValidationMode.ADVISORY,
) -> DerivationReport:
    """One fair-intake pass: the daemon's discovery page, each raw through ``converge_raw_id``."""

    async def run() -> DerivationReport:
        async with prepared_live_convergence_owner(root, validation_mode=validation_mode) as owner:
            return await converge_pending_raws_async(owner, root, limit=limit, discovery=discovery)

    return asyncio.run(run())


def _converge_raw(root: Path, compute: BoundedComputeAdapter, raw_id: str) -> DerivationReport:
    """Converge one raw on the admitted creator exactly as ``converge_raw_id`` does."""
    return converge(
        DerivationRegistry((RawObservationDerivation(root, compute_adapter=compute),)),
        raw_observation_frame(root, raw_ids=(raw_id,)),
        budget=Budget(page=1, discovery=1, inspection=2, compute=1, publication=1),
        publisher=admit_stage_write,
    )


def _inspect(
    root: Path,
    raw_id: str,
    *,
    validation_mode: ValidationMode = ValidationMode.ADVISORY,
) -> str:
    return run_on_convergence_owner(
        root,
        "test.raw.inspect",
        lambda compute: RawObservationDerivation(
            root,
            compute_adapter=compute,
            validation_mode=validation_mode,
        ).inspect(raw_observation_frame(root, validation_mode=validation_mode), (raw_id,))[raw_id],
    )


def test_canonical_replay_replaces_lost_output_without_touching_foreign_output(tmp_path: Path) -> None:
    """Output loss replays the owning component and preserves unrelated output."""
    bootstrap_archive_root(tmp_path)
    target = _admit(tmp_path, ("target-a", "target-b"), path="target.json")
    foreign = _admit(tmp_path, ("foreign",), path="foreign.json")
    first = _derive(tmp_path)
    assert first.failed == 0

    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("DELETE FROM sessions WHERE native_id = 'target-b'")
        conn.commit()
    assert _inspect(tmp_path, target) == "missing"

    replay = _derive(tmp_path)
    assert replay.failed == 0
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions ORDER BY native_id").fetchall() == [
            ("foreign",),
            ("target-a",),
            ("target-b",),
        ]
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE raw_id = ?", (foreign,)).fetchone() == (1,)


def test_prepared_retained_replay_slices_fresh_and_same_raw_fork_prefix(tmp_path: Path) -> None:
    """A retained fork reaches the writer with a precomputed tail on both replay paths."""
    bootstrap_archive_root(tmp_path)
    shared = (
        b'{"type":"response_item","payload":{"type":"message","id":"shared-user",'
        b'"role":"user","content":[{"type":"input_text","text":"shared parent text"}]}}\n'
    )
    parent_id = _admit(
        tmp_path,
        (),
        path="parent.jsonl",
        provider=Provider.CODEX,
        payload=b'{"type":"session_meta","payload":{"id":"retained-parent"}}\n' + shared,
    )
    parent_report = _derive(tmp_path, limit=1)
    assert parent_report.failed == parent_report.pending == 0, parent_report.outcomes
    assert _inspect(tmp_path, parent_id) == "valid"

    child_id = _admit(
        tmp_path,
        (),
        path="child.jsonl",
        provider=Provider.CODEX,
        payload=_codex_fork_bytes("retained-child", "retained-parent", shared),
    )
    # The committed source census re-prepares the fork within the same pass.
    first = _derive(tmp_path)
    assert first.failed == first.pending == 0, first.outcomes
    assert _inspect(tmp_path, child_id) == "valid"
    with sqlite3.connect(tmp_path / "index.db") as conn:
        child_row = conn.execute(
            "SELECT session_id, raw_id FROM sessions WHERE native_id = ?", ("retained-child",)
        ).fetchone()
        assert child_row is not None and child_row[1] == child_id
        lineage = conn.execute(
            "SELECT resolved_dst_session_id, branch_point_message_id FROM session_links WHERE src_session_id = ?",
            (child_row[0],),
        ).fetchone()
        assert lineage is not None and lineage[0] == "codex-session:retained-parent"
        assert lineage[1] is not None
        assert conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", ("codex-session:retained-child",)
        ).fetchone() == (1,)
        conn.execute("DELETE FROM raw_revision_applications WHERE raw_id = ?", (child_id,))
        conn.commit()

    assert _inspect(tmp_path, child_id) != "valid"
    for _ in range(4):
        second = _derive(tmp_path)
        assert second.failed == 0, second.outcomes
        if second.pending == 0 and _inspect(tmp_path, child_id) == "valid":
            break
    else:
        pytest.fail("same-raw fork replay did not converge")
    assert _inspect(tmp_path, child_id) == "valid"
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", ("codex-session:retained-child",)
        ).fetchone() == (1,)


def test_canonical_replay_cleans_orphaned_messages_before_replacement(tmp_path: Path) -> None:
    """Replacing a lost session cannot violate message uniqueness or touch a foreign session."""
    from polylogue.archive.message.roles import Role
    from polylogue.sources.parsers.base import ParsedMessage, ParsedSession

    bootstrap_archive_root(tmp_path)
    raw_id = _admit(
        tmp_path,
        (),
        path="orphaned-current-cohort.jsonl",
        provider=Provider.CODEX,
        payload=_codex_conversation_bytes("orphaned-current-cohort"),
    )
    assert _derive(tmp_path).failed == 0
    # The fixture Index writer requires the production measured creator.
    with closing(connect_measured(tmp_path / "index.db")) as conn:
        target_id = str(conn.execute("SELECT session_id FROM sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0])
        conn.execute("PRAGMA foreign_keys = OFF")
        foreign_id = write_fixture_index_session(
            conn,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="foreign-current-cohort",
                messages=[ParsedMessage(provider_message_id="foreign-0", role=Role.USER, text="foreign retained")],
            ),
            raw_id="foreign-current-raw",
        )
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (target_id,))
        conn.commit()

    assert _inspect(tmp_path, raw_id) == "missing"
    assert _derive(tmp_path).failed == 0
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute(
            "SELECT native_id, position FROM messages WHERE session_id = ? ORDER BY position", (target_id,)
        ).fetchall() == [("m-orphaned-current-cohort", 0)]
        assert conn.execute(
            "SELECT native_id, position FROM messages WHERE session_id = ? ORDER BY position", (foreign_id,)
        ).fetchall() == [("foreign-0", 0)]


def test_canonical_authority_refusal_blocks_only_its_raw_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused source path cannot publish while an unrelated sibling converges.

    Anti-vacuity: treating every attributed refusal as global leaves the
    healthy raw pending; omitting publication authority checks materializes
    the refused raw.
    """
    bootstrap_archive_root(tmp_path)
    healthy_path = "healthy.json"
    refused_path = "refused.json"
    healthy = _admit(tmp_path, ("healthy",), path=healthy_path)
    refused = _admit(tmp_path, ("refused",), path=refused_path)

    def blocked_raws(_archive_root: Path, raw_ids: tuple[str, ...]) -> RawFrontierBlockedPaths:
        if refused in raw_ids:
            return RawFrontierBlockedPaths(frozenset({refused_path}), None)
        return RawFrontierBlockedPaths(frozenset(), None)

    monkeypatch.setattr("polylogue.storage.raw_retention.raw_frontier_blocked_raw_ids", blocked_raws)
    report = _derive(tmp_path, limit=2)

    assert report.done == 1 and report.pending == 1 and report.failed == 0
    assert _inspect(tmp_path, healthy) == "valid"
    assert _inspect(tmp_path, refused) == "missing"
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchall() == [("healthy",)]


def test_reported_blob_size_does_not_refuse_a_valid_component(tmp_path: Path) -> None:
    """Large source metadata cannot revive the former component-size refusal."""
    bootstrap_archive_root(tmp_path)
    healthy = _admit(tmp_path, ("healthy",), path="healthy.json")
    oversized = _admit(tmp_path, ("oversized",), path="oversized.json")
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?", (64 * 1024 * 1024 + 1, oversized))
        conn.commit()

    report = _derive(tmp_path, limit=2)
    assert report.done == 2 and report.failed == 0
    assert _inspect(tmp_path, healthy) == "valid"
    assert _inspect(tmp_path, oversized) == "valid"
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert {row[0] for row in conn.execute("SELECT native_id FROM sessions")} == {"healthy", "oversized"}


def test_canonical_component_replay_is_idempotent_across_repeated_passes(tmp_path: Path) -> None:
    """A successful retained replay does not reparse on the next pass."""
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(
        tmp_path,
        ("repeated-envelope",),
        path="repeated-envelope.json",
        payload=_chatgpt_payload(("repeated-envelope",)),
    )
    first = _derive(tmp_path, limit=1)
    second = _derive(tmp_path, limit=1)

    assert first.done == 1
    assert second.done == 0
    assert first.failed == second.failed == 0
    assert _inspect(tmp_path, raw_id) == "valid"
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (1,)


def test_canonical_retryable_frontier_error_replays_from_retained_bytes(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(
        tmp_path,
        (),
        path="retry.jsonl",
        provider=Provider.CODEX,
        payload=_codex_conversation_bytes("retry"),
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET parsed_at_ms = 2, parse_error = ? WHERE raw_id = ?",
            ("OperationalError: database is locked", raw_id),
        )
        conn.commit()

    report = _derive(tmp_path)
    assert report.failed == 0
    assert _inspect(tmp_path, raw_id) == "valid"
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT parsed_at_ms IS NOT NULL, parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (1, None)


@pytest.mark.parametrize(
    ("origin", "source_path", "source_index"),
    [
        ("claude-code-session", "target.jsonl", 0),
        ("codex-session", "neighbor.jsonl", 0),
        ("codex-session", "target.jsonl", 1),
    ],
)
def test_canonical_inspection_requires_exact_failed_artifact_coordinate(
    tmp_path: Path,
    origin: str,
    source_path: str,
    source_index: int,
) -> None:
    """A deferred neighbor cannot authorize replay for another raw coordinate."""
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(
        tmp_path,
        (),
        path="target.jsonl",
        provider=Provider.CODEX,
        payload=_codex_conversation_bytes("target"),
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET parsed_at_ms = 2, parse_error = ? WHERE raw_id = ?",
            ("deferred failure", raw_id),
        )
        upsert_raw_artifact(
            conn,
            raw_id,
            ArchiveSourceArtifact(
                artifact_id="neighbor-evidence",
                origin=origin,
                source_path=source_path,
                source_index=source_index,
                artifact_kind="deferred_cas_frontier",
                classification_reason="deferred_cas_frontier",
                support_status=ArtifactSupportStatus.PARTIAL_DECODE,
                parse_as_session=True,
                schema_eligible=True,
            ),
        )
        conn.commit()
    assert _inspect(tmp_path, raw_id) == "valid"


def test_canonical_inspection_accepts_only_exact_failed_artifact_coordinate(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(
        tmp_path,
        (),
        path="target.jsonl",
        provider=Provider.CODEX,
        payload=_codex_conversation_bytes("target"),
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET parsed_at_ms = 2, parse_error = ? WHERE raw_id = ?",
            ("deferred failure", raw_id),
        )
        upsert_raw_artifact(
            conn,
            raw_id,
            ArchiveSourceArtifact(
                artifact_id="exact-evidence",
                origin="codex-session",
                source_path="target.jsonl",
                source_index=0,
                artifact_kind="deferred_cas_frontier",
                classification_reason="deferred_cas_frontier",
                support_status=ArtifactSupportStatus.PARTIAL_DECODE,
                parse_as_session=True,
                schema_eligible=True,
            ),
        )
        conn.commit()
    assert _inspect(tmp_path, raw_id) == "missing"


def test_canonical_terminal_support_cannot_authorize_deferred_replay(tmp_path: Path) -> None:
    """Contradictory terminal support wins over a deferred artifact kind."""
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(
        tmp_path,
        (),
        path="contradictory.jsonl",
        provider=Provider.CODEX,
        payload=_codex_conversation_bytes("contradictory"),
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET parsed_at_ms = 2, parse_error = ?, validation_mode='advisory' WHERE raw_id = ?",
            ("changed wording", raw_id),
        )
        upsert_raw_artifact(
            conn,
            raw_id,
            ArchiveSourceArtifact(
                artifact_id="contradictory-deferred-evidence",
                origin="codex-session",
                source_path="contradictory.jsonl",
                source_index=0,
                artifact_kind="deferred_cas_frontier",
                classification_reason="deferred_cas_frontier",
                support_status=ArtifactSupportStatus.DECODE_FAILED,
                parse_as_session=True,
                schema_eligible=True,
            ),
        )
        conn.commit()
    assert _inspect(tmp_path, raw_id) == "valid"


def test_canonical_terminal_carrier_overrides_legacy_cas_marker(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(
        tmp_path,
        (),
        path="reviewed-terminal.jsonl",
        provider=Provider.CODEX,
        payload=_codex_conversation_bytes("reviewed-terminal"),
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET parsed_at_ms = 2, parse_error = ? WHERE raw_id = ?",
            ("MembershipReplayConflictError: historical marker", raw_id),
        )
        upsert_raw_artifact(
            conn,
            raw_id,
            ArchiveSourceArtifact(
                artifact_id="reviewed-terminal-carrier",
                origin=Origin.CODEX_SESSION,
                source_path="reviewed-terminal.jsonl",
                source_index=0,
                artifact_kind=RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE.value,
                classification_reason="reviewed terminal disposition",
                support_status=ArtifactSupportStatus.UNSUPPORTED_PARSEABLE,
                parse_as_session=False,
                schema_eligible=False,
                first_observed_at_ms=2,
                last_observed_at_ms=2,
            ),
        )
        conn.commit()
    assert _inspect(tmp_path, raw_id) == "valid"


@pytest.mark.parametrize(
    ("validation_offset", "expected_materialized"),
    [(-1, 1), (0, 0), (1, 0)],
)
def test_canonical_reset_index_replays_only_when_parse_is_newer_than_validation_failure(
    tmp_path: Path,
    validation_offset: int,
    expected_materialized: int,
) -> None:
    """A newer/equal strict validation failure cannot authorize raw replay on reset."""
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(
        tmp_path,
        (),
        path=".claude/projects/-synthetic-reset/session.jsonl",
        provider=Provider.CLAUDE_CODE,
        payload=(Path(__file__).parents[2] / "fixtures" / "claude-code" / "strict-reset-validation.jsonl").read_bytes(),
    )
    assert _derive(tmp_path, validation_mode=ValidationMode.STRICT).failed == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        parsed_at_ms, validation_mode = conn.execute(
            "SELECT parsed_at_ms, validation_mode FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone()
        assert validation_mode == ValidationMode.STRICT.value
        assert (
            conn.execute(
                "SELECT 1 FROM raw_artifacts WHERE raw_id = ? AND parse_as_session = 1 AND schema_eligible = 1",
                (raw_id,),
            ).fetchone()
            is not None
        )
        conn.execute(
            "UPDATE raw_sessions SET validation_status = 'failed', validation_error = ?, validated_at_ms = ? "
            "WHERE raw_id = ?",
            ("validator rejected this observation", parsed_at_ms + validation_offset, raw_id),
        )
        conn.commit()

    active_index = tmp_path / "generations" / "active" / "index.db"
    initialize_archive_database(active_index, ArchiveTier.INDEX)
    (tmp_path / ".index-active-pointer").write_text(f"{active_index}\n", encoding="utf-8")

    expected_state = "missing" if expected_materialized else "valid"
    actual_state = _inspect(tmp_path, raw_id, validation_mode=ValidationMode.STRICT)
    assert actual_state == expected_state, (validation_offset, actual_state, expected_state)
    report = _derive(tmp_path, validation_mode=ValidationMode.STRICT)
    assert report.failed == 0, [(o.outcome.value, o.error) for o in report.outcomes]
    with sqlite3.connect(active_index) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (
            expected_materialized,
        )


def test_canonical_tied_validation_refusal_is_terminal_without_republishing_output(tmp_path: Path) -> None:
    """A tied STRICT failure is terminal refusal, not retryable output debt."""
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(
        tmp_path,
        ("tied-validation",),
        path="tied-validation.jsonl",
        provider=Provider.CODEX,
        payload=_codex_conversation_bytes("tied-validation"),
    )
    assert _derive(tmp_path, validation_mode=ValidationMode.STRICT).failed == 0

    with sqlite3.connect(tmp_path / "source.db") as conn:
        parsed_at_ms = int(
            conn.execute("SELECT parsed_at_ms FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0]
        )
        conn.execute(
            "UPDATE raw_sessions SET validation_status = 'failed', validation_error = ?, validated_at_ms = ? "
            "WHERE raw_id = ?",
            ("rejected at the same parser timestamp", parsed_at_ms, raw_id),
        )
        conn.commit()
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("DELETE FROM sessions WHERE raw_id = ?", (raw_id,))
        conn.execute("DELETE FROM raw_revision_applications WHERE raw_id = ?", (raw_id,))
        conn.commit()

    report = _derive(tmp_path, validation_mode=ValidationMode.STRICT)
    assert report.done == 0
    assert report.pending == report.failed == 0
    assert _inspect(tmp_path, raw_id, validation_mode=ValidationMode.STRICT) == "valid"
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT validation_status, validated_at_ms FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone() == ("failed", parsed_at_ms)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (0,)


def test_canonical_publish_rejects_rebuild_lease_conflict(tmp_path: Path) -> None:
    from polylogue.storage.index_generation import RebuildLease, RebuildLeaseUnavailableError

    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, ("lease-conflict",))

    def exercise(compute: BoundedComputeAdapter) -> None:
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute)
        frame = raw_observation_frame(tmp_path)
        replacement = adapter.compute(frame, raw_id)
        try:
            with RebuildLease(tmp_path):
                with pytest.raises(RebuildLeaseUnavailableError):
                    adapter.publish(frame, replacement)
        finally:
            replacement.close()

    run_on_convergence_owner(tmp_path, "test.raw.lease-conflict", exercise)


def test_canonical_publish_revalidates_the_promoted_active_generation(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, ("active-generation",))
    first_index = tmp_path / "generations" / "first" / "index.db"
    initialize_archive_database(first_index, ArchiveTier.INDEX)
    (tmp_path / ".index-active-pointer").write_text(f"{first_index}\n", encoding="utf-8")
    second_index = tmp_path / "generations" / "second" / "index.db"

    def exercise(compute: BoundedComputeAdapter) -> None:
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute)
        frame = raw_observation_frame(tmp_path)
        replacement = adapter.compute(frame, raw_id)
        try:
            initialize_archive_database(second_index, ArchiveTier.INDEX)
            (tmp_path / ".index-active-pointer").write_text(f"{second_index}\n", encoding="utf-8")
            # Publication revalidates the configured active Index and refuses
            # the moved destination with a typed stale-seal error.
            with pytest.raises(ReferenceSealStaleError):
                admit_stage_write("test.raw.promoted-generation", partial(adapter.publish, frame, replacement))
        finally:
            replacement.close()

    run_on_convergence_owner(tmp_path, "test.raw.promoted-generation", exercise)
    with sqlite3.connect(second_index) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


def test_codex_neutral_parse_survives_unrelated_source_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex parse and schema validation finish before a fresh Source binding."""
    from polylogue.schemas import validate_retained_document as validate_original
    from polylogue.sources import prepared_jsonl as prepared_jsonl_module
    from polylogue.sources.prepared_jsonl import PreparedJsonl

    bootstrap_archive_root(tmp_path)
    target = _admit(
        tmp_path,
        (),
        provider=Provider.CODEX,
        path="codex/session.jsonl",
        payload=_codex_conversation_bytes("neutral-target"),
    )
    parse_calls = 0
    validation_raw_ids: list[str] = []
    inserted: list[str] = []
    prepare_original = cast(Callable[..., PreparedJsonl], prepared_jsonl_module.prepare_jsonl_blob)
    validate_call = cast(Callable[..., object], validate_original)

    def counted_prepare(*args: object, **kwargs: object) -> PreparedJsonl:
        nonlocal parse_calls
        parse_calls += 1
        return prepare_original(*args, **kwargs)

    def commit_during_validation(*args: object, **kwargs: object) -> object:
        validation_raw_ids.append(str(kwargs["raw_id"]))
        verdict = validate_call(*args, **kwargs)
        if not inserted:
            inserted.append(
                _admit(
                    tmp_path,
                    (),
                    provider=Provider.CODEX,
                    path="codex/unrelated.jsonl",
                    payload=_codex_conversation_bytes("neutral-unrelated"),
                    acquired_at_ms=2,
                )
            )
        return verdict

    monkeypatch.setattr(prepared_jsonl_module, "prepare_jsonl_blob", counted_prepare)
    monkeypatch.setattr("polylogue.schemas.validate_retained_document", commit_during_validation)

    report = run_on_convergence_owner(
        tmp_path,
        "test.raw.codex-neutral-rebind",
        lambda compute: converge(
            DerivationRegistry((RawObservationDerivation(tmp_path, compute_adapter=compute),)),
            raw_observation_frame(tmp_path, raw_ids=(target,)),
            budget=Budget(page=1, discovery=1, inspection=2, compute=1, publication=1),
            publisher=admit_stage_write,
        ),
    )

    assert report.failed == 0, report.outcomes
    assert report.done == 1
    assert parse_calls == 1
    assert validation_raw_ids == [target], validation_raw_ids
    assert len(inserted) == 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions ORDER BY native_id").fetchall() == [("neutral-target",)]
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE raw_id = ?", (target,)).fetchone() == (1,)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE raw_id = ?", (inserted[0],)).fetchone() == (1,)


def test_changed_retained_selection_reuses_99_unchanged_parser_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed publication cohort retains only dependency-identical parses."""
    from polylogue.sources import prepared_jsonl as prepared_jsonl_module
    from polylogue.sources.prepared_jsonl import PreparedJsonl
    from tests.unit.storage.test_raw_observation_derivation import _publish_to_valid

    bootstrap_archive_root(tmp_path)
    raw_ids = [
        _admit(
            tmp_path,
            (),
            provider=Provider.CODEX,
            path=f"codex/cohort-{index:03d}.jsonl",
            payload=_codex_conversation_bytes(f"cohort-{index:03d}"),
        )
        for index in range(100)
    ]
    selected = list(raw_ids)
    parsed_paths: list[str] = []
    prepare_original = cast(Callable[..., PreparedJsonl], prepared_jsonl_module.prepare_jsonl_blob)

    def counted_prepare(*args: object, **kwargs: object) -> PreparedJsonl:
        parsed_paths.append(str(args[1]))
        artifact = prepare_original(*args, **kwargs)
        if len(parsed_paths) == 100:
            selected[50] = _admit(
                tmp_path,
                (),
                provider=Provider.CODEX,
                path="codex/cohort-replacement.jsonl",
                payload=_codex_conversation_bytes("cohort-replacement", "changed input"),
                acquired_at_ms=2,
            )
        return artifact

    monkeypatch.setattr(prepared_jsonl_module, "prepare_jsonl_blob", counted_prepare)

    def exercise(compute: BoundedComputeAdapter) -> None:
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute)
        frame = raw_observation_frame(tmp_path, raw_ids=(raw_ids[0],))
        with capture() as events:
            replacement = adapter.compute(
                frame,
                raw_ids[0],
                replay_current=True,
                select_retained_raw_ids=lambda _read: tuple(selected),
            )
            assert set(replacement.raw_ids) == set(selected)
            assert raw_ids[50] not in replacement.raw_ids
            assert _publish_to_valid(adapter, frame, replacement)
        retries = [event for event in events if event.get("event") == "storage.raw_observation.preparation_retry"]
        assert any(event["reason"] == "selection_changed" for event in retries)

    run_on_convergence_owner(tmp_path, "test.raw.dependency-local-cache", exercise)

    assert len(parsed_paths) == 101, parsed_paths
    assert parsed_paths.count("codex/cohort-replacement.jsonl") == 1
    assert all(parsed_paths.count(f"codex/cohort-{index:03d}.jsonl") == 1 for index in range(100))
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (100,)
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE raw_id=?", (raw_ids[50],)).fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE raw_id=?", (selected[50],)).fetchone() == (1,)


def test_mixed_default_retained_selection_neutralizes_only_eligible_codex_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mixed acquired selection detaches eligible JSONL while fresh-binding opaque rows."""
    from polylogue.operations.raw_observation_owner import RetainedMaterializationResult
    from polylogue.schemas import validate_retained_document as validate_original
    from polylogue.sources import prepared_jsonl as prepared_jsonl_module
    from polylogue.sources.prepared_jsonl import PreparedJsonl

    bootstrap_archive_root(tmp_path)
    codex_raw = _admit(
        tmp_path,
        (),
        provider=Provider.CODEX,
        path="codex/mixed.jsonl",
        payload=_codex_conversation_bytes("mixed-codex"),
    )
    opaque_raw = _admit(tmp_path, ("mixed-chatgpt",), path="chatgpt/mixed.json")
    inserted: list[str] = []
    parsed_codex_ids: list[str] = []
    validation_raw_ids: list[str] = []
    prepare_original = cast(Callable[..., PreparedJsonl], prepared_jsonl_module.prepare_jsonl_blob)
    validate_call = cast(Callable[..., object], validate_original)

    def counted_prepare(*args: object, **kwargs: object) -> PreparedJsonl:
        if len(args) > 2 and args[2] == Provider.CODEX.value:
            parsed_codex_ids.append(str(args[1]))
        return prepare_original(*args, **kwargs)

    def commit_during_validation(*args: object, **kwargs: object) -> object:
        raw_id = str(kwargs["raw_id"])
        validation_raw_ids.append(raw_id)
        verdict = validate_call(*args, **kwargs)
        if not inserted:
            inserted.append(
                _admit(
                    tmp_path,
                    ("unrelated-chatgpt",),
                    path="chatgpt/unrelated.json",
                    acquired_at_ms=2,
                )
            )
        return verdict

    monkeypatch.setattr(prepared_jsonl_module, "prepare_jsonl_blob", counted_prepare)
    monkeypatch.setattr("polylogue.schemas.validate_retained_document", commit_during_validation)

    async def materialize() -> RetainedMaterializationResult:
        async with prepared_live_convergence_owner(tmp_path) as owner:
            return await owner.materialize_retained_raw_ids((codex_raw, opaque_raw))

    result = asyncio.run(materialize())
    receipts = result.outcome.require_complete()

    assert receipts
    assert parsed_codex_ids == ["codex/mixed.jsonl"]
    assert validation_raw_ids.count(codex_raw) == 1
    assert len(inserted) == 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute(
            "SELECT native_id FROM sessions WHERE raw_id IN (?, ?) ORDER BY native_id",
            (codex_raw, opaque_raw),
        ).fetchall() == [("mixed-chatgpt",), ("mixed-codex",)]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        census = conn.execute(
            "SELECT raw_id, parsed_at_ms FROM raw_sessions WHERE raw_id IN (?, ?)",
            (codex_raw, opaque_raw),
        ).fetchall()
        assert len(census) == 2
        assert {str(raw_id) for raw_id, parsed_at_ms in census if parsed_at_ms is not None} == {codex_raw, opaque_raw}


@pytest.mark.parametrize("replace_sidecar", [False, True])
def test_claude_neutral_parse_uses_retained_sidecars_and_survives_source_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replace_sidecar: bool,
) -> None:
    """Claude's detached parser consumes captured CAS sidecars, then binds current Source."""
    from polylogue.schemas import validate_retained_document as validate_original
    from polylogue.sources import prepared_jsonl as prepared_jsonl_module
    from polylogue.sources import revision_backfill as revision_backfill_module
    from polylogue.sources.prepared_jsonl import PreparedJsonl

    bootstrap_archive_root(tmp_path)
    session_id = "2c9fbada-0d07-4429-8728-63f70e3c672f"
    project = tmp_path / "projects" / "-realm-project-polylogue"
    owner_path = project / f"{session_id}.jsonl"
    sidecar_path = project / session_id / "tool-results" / "toolu_capture.txt"
    sidecar_text = "retained-output-only: " + ("synthetic output " * 40)
    sidecar = _admit(
        tmp_path,
        (),
        provider=Provider.UNKNOWN,
        path=sidecar_path.as_posix(),
        payload=sidecar_text.encode(),
    )
    sibling_sidecar_path = project / session_id / "tool-results" / "toolu_sibling.txt"
    sibling_sidecar = _admit(
        tmp_path,
        (),
        provider=Provider.UNKNOWN,
        path=sibling_sidecar_path.as_posix(),
        payload=b"sibling-owned output",
    )
    sibling_path = project / session_id / "subagents" / "agent-capture.jsonl"
    sibling_payload = (
        json.dumps(
            {
                "type": "user",
                "uuid": "a-sibling",
                "sessionId": session_id,
                "timestamp": "2026-07-20T10:00:03Z",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_sibling", "content": "sibling output"}],
                },
            }
        )
        + "\n"
    ).encode()
    sibling_raw = _admit(
        tmp_path,
        (),
        provider=Provider.CLAUDE_CODE,
        path=sibling_path.as_posix(),
        payload=sibling_payload,
        acquired_at_ms=2,
    )
    pointer = f"<persisted-output>Output too large. Full output saved to: {sidecar_path}</persisted-output>"
    owner_payload = (
        json.dumps(
            {
                "type": "user",
                "uuid": "u-capture",
                "sessionId": session_id,
                "timestamp": "2026-07-20T10:00:00Z",
                "message": {"role": "user", "content": "run it"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "uuid": "a-capture",
                "parentUuid": "u-capture",
                "sessionId": session_id,
                "timestamp": "2026-07-20T10:00:01Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_capture", "name": "Bash", "input": {}}],
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "user",
                "uuid": "u-result",
                "parentUuid": "a-capture",
                "sessionId": session_id,
                "timestamp": "2026-07-20T10:00:02Z",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_capture", "content": pointer}],
                },
            }
        )
        + "\n"
    ).encode()
    target = _admit(
        tmp_path,
        (),
        provider=Provider.CLAUDE_CODE,
        path=owner_path.as_posix(),
        payload=owner_payload,
        acquired_at_ms=3,
    )
    parse_calls = 0
    validation_raw_ids: list[str] = []
    neutral_sidecar_events: list[tuple[str, dict[str, object]]] = []
    inserted: list[str] = []
    replacement_sidecars: list[str] = []
    enrichment_calls = 0
    prepare_original = cast(Callable[..., PreparedJsonl], prepared_jsonl_module.prepare_jsonl_blob)
    validate_call = cast(Callable[..., object], validate_original)
    enrich_original = cast(
        Callable[..., Iterable[object]], revision_backfill_module.iter_enriched_sessions_from_retained_read
    )

    def counted_prepare(*args: object, **kwargs: object) -> PreparedJsonl:
        nonlocal parse_calls
        parse_calls += 1
        artifact = prepare_original(*args, **kwargs)
        for session in artifact.iter_sessions():
            neutral_sidecar_events.extend((event.event_type, event.payload) for event in session.session_events)
        return artifact

    def commit_during_validation(*args: object, **kwargs: object) -> object:
        validation_raw_ids.append(str(kwargs["raw_id"]))
        return validate_call(*args, **kwargs)

    def commit_during_enrichment(*args: object, **kwargs: object) -> object:
        nonlocal enrichment_calls
        enrichment_calls += 1
        yield from enrich_original(*args, **kwargs)
        if not inserted:
            if replace_sidecar:
                replacement_sidecars.append(
                    _admit(
                        tmp_path,
                        (),
                        provider=Provider.UNKNOWN,
                        path=sidecar_path.as_posix(),
                        payload=b"new retained output after rebind",
                        acquired_at_ms=4,
                    )
                )
            inserted.append(
                _admit(
                    tmp_path,
                    (),
                    provider=Provider.CLAUDE_CODE,
                    path="projects/unrelated.jsonl",
                    payload=b'{"type":"queue-operation","operation":"compact"}\n',
                    acquired_at_ms=3,
                )
            )

    monkeypatch.setattr(prepared_jsonl_module, "prepare_jsonl_blob", counted_prepare)
    monkeypatch.setattr("polylogue.schemas.validate_retained_document", commit_during_validation)
    monkeypatch.setattr(revision_backfill_module, "iter_enriched_sessions_from_retained_read", commit_during_enrichment)

    report = run_on_convergence_owner(
        tmp_path,
        "test.raw.claude-neutral-rebind",
        lambda compute: converge(
            DerivationRegistry((RawObservationDerivation(tmp_path, compute_adapter=compute),)),
            raw_observation_frame(tmp_path, raw_ids=(target,)),
            budget=Budget(page=1, discovery=1, inspection=2, compute=1, publication=1),
            publisher=admit_stage_write,
        ),
    )

    assert report.failed == 0, report.outcomes
    assert report.done == 1
    expected_parser_calls = 2 if replace_sidecar else 1
    assert parse_calls == expected_parser_calls
    assert len(validation_raw_ids) >= 2
    assert set(validation_raw_ids) == {target}, validation_raw_ids
    assert enrichment_calls == 2
    assert sum(event_type == "claude_tool_result_sidecar" for event_type, _ in neutral_sidecar_events) == (
        expected_parser_calls
    ), neutral_sidecar_events
    assert len(inserted) == 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        rows = conn.execute(
            "SELECT b.text FROM blocks b JOIN sessions s ON s.session_id = b.session_id "
            "WHERE s.raw_id = ? AND b.block_type = 'tool_result'",
            (target,),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == ("new retained output after rebind" if replace_sidecar else sidecar_text)
        assert conn.execute(
            "SELECT COUNT(*) FROM session_events e JOIN sessions s ON s.session_id = e.session_id "
            "WHERE s.raw_id = ? AND e.event_type = 'claude_tool_result_sidecar'",
            (target,),
        ).fetchone() == (1,), "the sibling-owned file is resolved from its retained tool_result record"
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE raw_id = ?", (sidecar,)).fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE raw_id = ?", (sibling_sidecar,)).fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE raw_id = ?", (sibling_raw,)).fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE raw_id = ?", (inserted[0],)).fetchone() == (1,)
        assert all(
            conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (1,)
            for raw_id in replacement_sidecars
        )


def test_canonical_split_root_route_uses_the_explicit_archive_root(tmp_path: Path) -> None:
    """The routed archive root owns both source bytes and its active index."""
    configured_root = tmp_path / "configured"
    routed_root = tmp_path / "routed"
    configured_root.mkdir()
    bootstrap_archive_root(routed_root)
    raw_id = _admit(routed_root, ("routed",), path="routed.json")

    report = _derive(routed_root)
    assert report.failed == 0
    assert _inspect(routed_root, raw_id) == "valid"
    with sqlite3.connect(routed_root / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchall() == [("routed",)]
    assert not (configured_root / "source.db").exists()


@pytest.mark.parametrize(
    ("provider", "payload"),
    (
        (Provider.CODEX, b'{"type":"session_meta"}\n'),
        (Provider.CLAUDE_CODE, b'{"type":"queue-operation","operation":"compact"}\n'),
        (
            Provider.CLAUDE_CODE,
            b'{"sessionId":"sidecar","projectHash":"abc","startTime":"now","lastUpdated":"now","kind":"metadata"}\n',
        ),
    ),
)
def test_canonical_routed_root_classifies_parsed_sidecar_without_session_output(
    tmp_path: Path, provider: Provider, payload: bytes
) -> None:
    configured_root = tmp_path / "configured"
    routed_root = tmp_path / "routed"
    configured_root.mkdir()
    bootstrap_archive_root(routed_root)
    raw_id = _admit(routed_root, (), path="sidecar.jsonl", provider=provider, payload=payload)
    with sqlite3.connect(routed_root / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET parsed_at_ms = 2 WHERE raw_id = ?", (raw_id,))
    assert _inspect(routed_root, raw_id) != "valid"

    report = _derive(routed_root)
    assert report.failed == report.pending == 0
    assert _inspect(routed_root, raw_id) == "valid"
    assert _derive(routed_root).done == 0
    with sqlite3.connect(routed_root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
    assert not (configured_root / "source.db").exists()
    # Red twin: a historical parsed marker alone cannot replace the current
    # zero-output parser evidence, even when there is no session to compare.
    with sqlite3.connect(routed_root / "source.db") as conn:
        conn.execute("DELETE FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,))
    assert _inspect(routed_root, raw_id) != "valid"


def test_canonical_parse_failure_does_not_suppress_healthy_sibling(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    healthy = _admit(tmp_path, ("healthy",), path="healthy.json")
    poison = _admit(tmp_path, (), path="poison.json", payload=b"not json")

    report = _derive(tmp_path, limit=2)
    assert report.failed == 1
    assert _inspect(tmp_path, healthy) == "valid"
    assert _inspect(tmp_path, poison) != "valid"
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchall() == [("healthy",)]


def test_canonical_replay_refreshes_only_the_touched_derived_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One raw replay must not rebuild derived surfaces for unrelated sessions."""
    from polylogue.storage.fts import fts_lifecycle as fts_lifecycle_mod
    from polylogue.storage.sqlite import action_pairs as action_pairs_mod
    from polylogue.storage.sqlite import delegation_facts as delegation_facts_mod

    def tool_call_payload(native_id: str) -> bytes:
        return (
            f'{{"type":"session_meta","payload":{{"id":"{native_id}"}}}}\n'.encode()
            + b'{"type":"response_item","payload":{"type":"message","role":"user",'
            b'"content":[{"type":"input_text","text":"run a command"}]}}\n'
            b'{"type":"response_item","payload":{"type":"function_call","id":"fc_1",'
            b'"call_id":"call_abc","name":"exec_command","arguments":"{\\"cmd\\": \\"ls\\"}"}}\n'
            b'{"type":"response_item","payload":{"type":"function_call_output",'
            b'"call_id":"call_abc","output":"file1.txt"}}\n'
        )

    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        watched_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=tool_call_payload("unrelated-existing"),
            source_path="unrelated-existing.jsonl",
            canonical_source_path="unrelated-existing.jsonl",
            acquired_at_ms=1,
        )
        touched_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=tool_call_payload("touched-new"),
            source_path="touched-new.jsonl",
            canonical_source_path="touched-new.jsonl",
            acquired_at_ms=2,
        )

    assert _derive(tmp_path).failed == 0
    with sqlite3.connect(tmp_path / "index.db") as conn:
        watched_session_id = str(
            conn.execute("SELECT session_id FROM sessions WHERE raw_id = ?", (watched_raw_id,)).fetchone()[0]
        )
        before_action_pairs = conn.execute(
            "SELECT rowid FROM action_pairs WHERE session_id = ?", (watched_session_id,)
        ).fetchall()
        assert before_action_pairs
        conn.execute("DELETE FROM sessions WHERE raw_id = ?", (touched_raw_id,))
        conn.execute("DELETE FROM raw_revision_applications WHERE raw_id = ?", (touched_raw_id,))
        conn.commit()

    def fail_archive_wide_rebuild(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("archive-wide derived rebuild must not run for one raw component")

    monkeypatch.setattr(fts_lifecycle_mod, "rebuild_fts_index_sync", fail_archive_wide_rebuild)
    monkeypatch.setattr(action_pairs_mod, "rebuild_all_action_pairs_sync", fail_archive_wide_rebuild)
    monkeypatch.setattr(delegation_facts_mod, "rebuild_all_delegation_facts_sync", fail_archive_wide_rebuild)

    targeted = run_on_convergence_owner(
        tmp_path,
        "test.raw.touched-component",
        lambda compute: converge(
            DerivationRegistry((RawObservationDerivation(tmp_path, compute_adapter=compute),)),
            raw_observation_frame(tmp_path, raw_ids=(touched_raw_id,)),
            budget=Budget(page=1, discovery=1, inspection=2, compute=1, publication=1),
            publisher=admit_stage_write,
        ),
    )
    assert targeted.failed == 0
    assert targeted.done == 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        after_action_pairs = conn.execute(
            "SELECT rowid FROM action_pairs WHERE session_id = ?", (watched_session_id,)
        ).fetchall()
        assert after_action_pairs == before_action_pairs
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE native_id = ?", ("touched-new",)).fetchone() == (1,)


def test_canonical_bounded_passes_reach_each_independent_component(tmp_path: Path) -> None:
    """The intake's bounded discovery advances across independent components without starvation.

    A page whose raw just published is re-inspected once (valid) before the
    traversal moves on, so ``2 * len(names)`` one-raw passes reach every
    component.
    """
    bootstrap_archive_root(tmp_path)
    names = tuple(f"bounded-{index}" for index in range(4))
    for name in names:
        _admit(tmp_path, (name,), path=f"{name}.json")

    discovery = RawMaterializationDiscovery(tmp_path)
    reports = [_derive(tmp_path, limit=1, discovery=discovery) for _ in range(2 * len(names))]
    assert all(report.failed == 0 and report.done <= 1 for report in reports)
    assert sum(report.done for report in reports) == len(names)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions ORDER BY native_id").fetchall() == [
            (name,) for name in names
        ]


def test_canonical_fairness_survives_ops_reset_with_a_process_cursor(tmp_path: Path) -> None:
    """Deleting disposable ops state cannot reset the intake's process-local discovery."""
    bootstrap_archive_root(tmp_path)
    names = tuple(f"ops-reset-{index}" for index in range(4))
    for name in names:
        _admit(tmp_path, (name,), path=f"{name}.json")

    discovery = RawMaterializationDiscovery(tmp_path)
    reports = []
    for _ in range(2 * len(names)):
        reports.append(_derive(tmp_path, limit=1, discovery=discovery))
        (tmp_path / "ops.db").unlink(missing_ok=True)

    assert all(report.failed == 0 and report.done <= 1 for report in reports)
    assert sum(report.done for report in reports) == len(names)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions ORDER BY native_id").fetchall() == [
            (name,) for name in names
        ]


def test_canonical_deadline_bounds_a_pass_without_substituting_a_count_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wall-clock deadline stops between canonical components and preserves progress."""
    bootstrap_archive_root(tmp_path)
    names = tuple(f"deadline-{index}" for index in range(3))
    for name in names:
        _admit(tmp_path, (name,), path=f"{name}.json")

    clock = [0.0]
    # Patch only the pass deadline clock: freezing ``time.monotonic`` itself
    # also froze the retained-preparation worker pool's waits, so the pass
    # hung instead of expiring.
    monkeypatch.setattr("polylogue.daemon.derivation._pass_clock", lambda: clock[0])

    def exercise(compute: BoundedComputeAdapter) -> DerivationReport:
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute)
        original_compute = adapter.compute

        def compute_then_expire(frame: object, key: str) -> object:
            replacement = original_compute(frame, key)  # type: ignore[arg-type]
            # Preparatory Source phases re-prepare within the same component;
            # expire only once the component reaches its destination write.
            if replacement.prepared_writes:
                clock[0] = 2.0
            return replacement

        monkeypatch.setattr(adapter, "compute", compute_then_expire)
        return converge(
            DerivationRegistry((adapter,)),
            raw_observation_frame(tmp_path),
            budget=Budget(page=3, discovery=3, inspection=6, compute=3, publication=3, deadline_s=1.0),
            publisher=admit_stage_write,
        )

    bounded = run_on_convergence_owner(tmp_path, "test.raw.deadline", exercise)

    assert bounded.done == 1
    assert bounded.pending >= 2
    assert bounded.failed == 0
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)


def test_canonical_failed_publication_cannot_report_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A publication that makes no progress remains pending, never successful."""
    bootstrap_archive_root(tmp_path)
    _admit(tmp_path, ("publication-blocked",))

    def refuse(
        _self: RawObservationDerivation, _frame: object, replacement: RawObservationReplacement, **_kwargs: object
    ) -> bool:
        # Once publication starts the adapter owns its carrier and settles it
        # on every outcome, exactly as the real publish does in its finally.
        replacement.close()
        return False

    monkeypatch.setattr(RawObservationDerivation, "publish", refuse)
    report = _derive(tmp_path)
    assert report.done == 0
    assert report.pending + report.failed >= 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


def test_canonical_replay_does_not_replace_newer_index_authority(tmp_path: Path) -> None:
    """A stale prepared replacement cannot displace a newer accepted head."""
    bootstrap_archive_root(tmp_path)
    old_payload = _codex_conversation_bytes("same-head", "old")
    new_payload = old_payload + (
        b'{"type":"response_item","payload":{"type":"message","id":"m-new",'
        b'"role":"assistant","content":[{"type":"output_text","text":"new"}]}}\n'
    )
    old_raw_id = _admit(
        tmp_path,
        (),
        path="same-head.jsonl",
        provider=Provider.CODEX,
        payload=old_payload,
        acquired_at_ms=1,
    )

    def exercise(compute: BoundedComputeAdapter) -> str:
        # The stale replacement keeps its original creator; every later pass
        # runs on that same admitted owner.
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute)
        frame = raw_observation_frame(tmp_path)
        # Each committed preparatory Source phase re-prepares, as the kernel
        # does within its pass; the replacement that publishes is kept.
        for _ in range(3):
            old_replacement = adapter.compute(frame, old_raw_id)
            if admit_stage_write("test.raw.stale-replacement.first", partial(adapter.publish, frame, old_replacement)):
                break
        else:
            pytest.fail("the original raw did not publish after its preparatory phases")
        new_raw_id = _admit(
            tmp_path,
            (),
            path="same-head.jsonl",
            provider=Provider.CODEX,
            payload=new_payload,
            acquired_at_ms=2,
        )
        # Source classification re-prepares the new raw within one pass.
        report = _converge_raw(tmp_path, compute, new_raw_id)
        assert report.failed == report.pending == 0, report.outcomes
        assert adapter.inspect(raw_observation_frame(tmp_path), (new_raw_id,))[new_raw_id] == "valid"

        with sqlite3.connect(tmp_path / "index.db") as conn:
            head = conn.execute(
                "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = ?",
                ("codex-session:same-head",),
            ).fetchone()
            assert head == (new_raw_id,)
        try:
            admit_stage_write("test.raw.stale-replacement.late", partial(adapter.publish, frame, old_replacement))
        except (RawCASFrontierError, ReferenceSealError):
            # A consumed replacement's seal is closed; either typed refusal
            # leaves the newer head in place, which is asserted below.
            pass
        return new_raw_id

    new_raw_id = run_on_convergence_owner(tmp_path, "test.raw.stale-replacement", exercise)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = ?",
            ("codex-session:same-head",),
        ).fetchone() == (new_raw_id,)


def test_canonical_append_fragment_does_not_livelock_component_discovery(tmp_path: Path) -> None:
    """A byte-governed append member cannot keep its full component pending."""
    from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind

    key = "codex-session:growing-rollout"
    baseline = _codex_conversation_bytes("growing-rollout")
    grown = baseline + (
        b'{"type":"response_item","payload":{"type":"message","id":"m-second",'
        b'"role":"assistant","content":[{"type":"output_text","text":"tail"}]}}\n'
    )
    tail = grown[len(baseline) :]

    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as store:
        baseline_id = store.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline,
            source_path="rollout.jsonl",
            canonical_source_path="rollout.jsonl",
            acquired_at_ms=1,
        )
        store.bind_raw_revision(
            baseline_id,
            RawRevisionEnvelope(
                key,
                RawRevisionKind.FULL,
                "base",
                0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        append_id = store.write_raw_payload(
            provider=Provider.CODEX,
            payload=tail,
            source_path="rollout.jsonl",
            canonical_source_path="rollout.jsonl",
            source_index=-1,
            native_id="growing-rollout",
            acquired_at_ms=3,
        )
        store.bind_raw_revision(
            append_id,
            RawRevisionEnvelope(
                key,
                RawRevisionKind.APPEND,
                append_source_revision("base", hashlib.sha256(tail).hexdigest()),
                1,
                authority=RawRevisionAuthority.BYTE_PROVEN,
                predecessor_source_revision="base",
                predecessor_raw_id=baseline_id,
                baseline_raw_id=baseline_id,
                append_start_offset=len(baseline),
                append_end_offset=len(grown),
            ),
        )
        store.commit()

    report = _derive(tmp_path)
    assert report.failed == 0, [(str(outcome.key), outcome.outcome.value, outcome.error) for outcome in report.outcomes]
    second = _derive(tmp_path)
    assert second.failed == 0, [(str(outcome.key), outcome.outcome.value, outcome.error) for outcome in second.outcomes]
    assert second.pending == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        append_receipt = conn.execute(
            "SELECT status, detail FROM raw_authority_parser_census WHERE raw_id = ?", (append_id,)
        ).fetchone()
        assert append_receipt is not None
        assert append_receipt[1]
    assert _inspect(tmp_path, append_id) == "valid"
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT session_id FROM sessions").fetchall() == [(key,)]


def test_canonical_quarantined_append_reconciles_from_retained_full_revisions(tmp_path: Path) -> None:
    """A quarantined append with a stale predecessor is repaired from retained full bytes."""
    from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind

    key = "codex-session:quarantined-growing-rollout"
    baseline = _codex_conversation_bytes("quarantined-growing-rollout")
    grown = baseline + (
        b'{"type":"response_item","payload":{"type":"message","id":"m-second",'
        b'"role":"assistant","content":[{"type":"output_text","text":"tail"}]}}\n'
    )
    tail = grown[len(baseline) :]

    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as store:
        full_raw_ids = []
        for index, payload in enumerate((baseline, grown), start=1):
            raw_id = store.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path="quarantined-rollout.jsonl",
                canonical_source_path="quarantined-rollout.jsonl",
                acquired_at_ms=index,
            )
            store.bind_raw_revision(
                raw_id,
                RawRevisionEnvelope(
                    key,
                    RawRevisionKind.FULL,
                    raw_id,
                    index - 1,
                    authority=RawRevisionAuthority.QUARANTINED,
                ),
            )
            full_raw_ids.append(raw_id)
        append_id = store.write_raw_payload(
            provider=Provider.CODEX,
            payload=tail,
            source_path="quarantined-rollout.jsonl",
            canonical_source_path="quarantined-rollout.jsonl",
            source_index=-1,
            acquired_at_ms=3,
        )
        store.bind_raw_revision(
            append_id,
            RawRevisionEnvelope(
                key,
                RawRevisionKind.APPEND,
                append_id,
                0,
                authority=RawRevisionAuthority.QUARANTINED,
                predecessor_source_revision="0" * 64,
                append_start_offset=len(baseline),
                append_end_offset=len(grown),
            ),
        )
        store.commit()

    first = _derive(tmp_path)
    second = _derive(tmp_path)
    for report in (first, second):
        assert report.failed == 0, [
            (str(outcome.key), outcome.outcome.value, outcome.error) for outcome in report.outcomes
        ]
    assert second.pending == 0
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT session_id FROM sessions").fetchall() == [(key,)]
    assert full_raw_ids
    assert all(_inspect(tmp_path, raw_id) == "valid" for raw_id in (*full_raw_ids, append_id))
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_authority_parser_census SET parser_fingerprint = 'revision-membership-v3' WHERE raw_id = ?",
            (append_id,),
        )
    # Red twin: ignoring an obsolete refusal would wrongly certify its healthy
    # siblings without reconsidering the connected revision obligation.
    assert _inspect(tmp_path, append_id) != "valid"
    assert all(_inspect(tmp_path, raw_id) != "valid" for raw_id in full_raw_ids)


def test_raw_cas_frontier_error_is_typed_transient() -> None:
    assert RawCASFrontierError("frontier changed").is_transient is True


def test_non_codex_cas_frontier_failure_persists_provider_neutral_evidence(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=_codex_conversation_bytes("cas-frontier"),
            source_path="rollout.jsonl",
            canonical_source_path="rollout.jsonl",
            acquired_at_ms=1,
        )
        archive.mark_raw_parse_failed(
            raw_id,
            provider=Provider.CLAUDE_CODE,
            error=RawCASFrontierError("frontier changed"),
        )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT artifact_kind, support_status, parse_as_session FROM raw_artifacts WHERE raw_id = ?",
            (raw_id,),
        ).fetchone() == ("deferred_cas_frontier", "partial_decode", 1)


def test_generic_parse_state_failure_retires_prior_failure_authority(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=_codex_conversation_bytes("stale-authority"),
            source_path="stale-authority.jsonl",
            canonical_source_path="stale-authority.jsonl",
            acquired_at_ms=1,
        )
        archive.mark_raw_parse_failed(raw_id, provider=Provider.CODEX, error=RawCASFrontierError("first frontier"))
        archive.finalize_raw_parse_state(
            raw_id,
            state=RawSessionStateUpdate(
                parse_error="ValueError: later parser failure",
                payload_provider=Provider.CODEX,
            ),
        )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT artifact_kind, support_status, parse_as_session FROM raw_artifacts WHERE raw_id = ?",
            (raw_id,),
        ).fetchone() == (
            RawFailureEvidenceKind.TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER.value,
            "unknown",
            0,
        )
    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.terminal == 0
    assert lifecycle.deferred == 0
    assert lifecycle.unexplained == 1
    assert _inspect(tmp_path, raw_id) == "valid"


def test_failed_raw_lifecycle_preserves_exact_evidence_for_same_coordinate(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        old_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"revision":"old"}',
            source_path="same-coordinate.jsonl",
            canonical_source_path="same-coordinate.jsonl",
            source_index=0,
            acquired_at_ms=1,
        )
        new_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"revision":"new"}',
            source_path="same-coordinate.jsonl",
            canonical_source_path="same-coordinate.jsonl",
            source_index=0,
            acquired_at_ms=2,
        )
        archive.mark_raw_parse_failed(old_raw_id, provider=Provider.CODEX, error=RawCASFrontierError("old frontier"))
        archive.mark_raw_parse_failed(new_raw_id, provider=Provider.CODEX, error=RawCASFrontierError("new frontier"))

    with sqlite3.connect(tmp_path / "source.db") as conn:
        rows = conn.execute(
            "SELECT raw_id, origin, source_path, source_index, artifact_kind, support_status "
            "FROM raw_artifacts WHERE source_path = 'same-coordinate.jsonl' ORDER BY raw_id"
        ).fetchall()
    assert {tuple(row) for row in rows} == {
        (old_raw_id, "codex-session", "same-coordinate.jsonl", 0, "deferred_cas_frontier", "partial_decode"),
        (new_raw_id, "codex-session", "same-coordinate.jsonl", 0, "deferred_cas_frontier", "partial_decode"),
    }
    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.deferred == 2
    assert lifecycle.unexplained == 0
    assert {sample["raw_id"] for sample in lifecycle.samples} == {old_raw_id, new_raw_id}


def test_failed_raw_lifecycle_ignores_newer_ordinary_artifact_at_same_coordinate(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session

    bootstrap_archive_root(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        raw_id = write_source_raw_session(
            conn,
            origin=Origin.CODEX_SESSION,
            source_path="coexisting.jsonl",
            canonical_source_path="coexisting.jsonl",
            source_index=4,
            payload=b"unsupported",
            acquired_at_ms=1,
            parse_error="worker rejected shape",
        )
        upsert_raw_artifact(
            conn,
            raw_id,
            ArchiveSourceArtifact(
                artifact_id="failure-carrier",
                origin=Origin.CODEX_SESSION,
                source_path="coexisting.jsonl",
                source_index=4,
                artifact_kind=RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE.value,
                classification_reason=RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE.value,
                support_status=ArtifactSupportStatus.UNSUPPORTED_PARSEABLE,
                first_observed_at_ms=10,
                last_observed_at_ms=10,
            ),
        )
        upsert_raw_artifact(
            conn,
            raw_id,
            ArchiveSourceArtifact(
                artifact_id="ordinary-carrier",
                origin=Origin.CODEX_SESSION,
                source_path="coexisting.jsonl",
                source_index=4,
                artifact_kind="session_export",
                classification_reason="ordinary re-observation",
                support_status=ArtifactSupportStatus.SUPPORTED_PARSEABLE,
                first_observed_at_ms=20,
                last_observed_at_ms=20,
            ),
        )

    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.terminal == 1
    assert lifecycle.unexplained == 0
    assert lifecycle.blocking is False
    assert lifecycle.state == "degraded"
    assert lifecycle.samples[0]["artifact_kind"] == RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE.value


def test_cas_failure_evidence_rolls_back_with_parse_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.storage.sqlite.archive_tiers import revision_governance

    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"revision":"atomic"}',
            source_path="atomic.jsonl",
            canonical_source_path="atomic.jsonl",
            acquired_at_ms=1,
        )

        def fail_state_update(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("state update failed")

        monkeypatch.setattr(revision_governance, "apply_source_raw_state_update", fail_state_update)
        with pytest.raises(RuntimeError, match="state update failed"):
            archive.mark_raw_parse_failed(
                raw_id,
                provider=Provider.CODEX,
                error=RawCASFrontierError("frontier"),
            )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (None,)
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts WHERE raw_id = ?", (raw_id,)).fetchone() == (0,)


def test_deferred_cas_evidence_is_superseded_after_resolution_and_non_cas_failure(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_success = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"name":"success"}',
            source_path="success.jsonl",
            canonical_source_path="success.jsonl",
            acquired_at_ms=1,
        )
        raw_failure = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"name":"failure"}',
            source_path="failure.jsonl",
            canonical_source_path="failure.jsonl",
            acquired_at_ms=2,
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        for raw_id, source_path, neighbor_path in (
            (raw_success, "success.jsonl", "success-neighbor.jsonl"),
            (raw_failure, "failure.jsonl", "failure-neighbor.jsonl"),
        ):
            for artifact_id, path in ((f"deferred-{raw_id}", source_path), (f"neighbor-{raw_id}", neighbor_path)):
                upsert_raw_artifact(
                    conn,
                    raw_id,
                    ArchiveSourceArtifact(
                        artifact_id=artifact_id,
                        origin="codex-session",
                        source_path=path,
                        source_index=0,
                        artifact_kind="deferred_cas_frontier",
                        classification_reason="deferred_cas_frontier",
                        support_status=ArtifactSupportStatus.PARTIAL_DECODE,
                        parse_as_session=True,
                        schema_eligible=True,
                    ),
                )
        conn.commit()

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.mark_raw_parse_succeeded(raw_success, provider=Provider.CODEX)
        archive.mark_raw_parse_failed(
            raw_failure,
            provider=Provider.CODEX,
            error=ValueError("unrelated parser failure"),
        )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET parsed_at_ms = 3, parse_error = ? WHERE raw_id = ?",
            ("later unrelated parser failure", raw_success),
        )
        conn.commit()
        observations = conn.execute(
            "SELECT raw_id, source_path, artifact_kind, support_status FROM raw_artifacts "
            "WHERE raw_id IN (?, ?) ORDER BY raw_id, source_path",
            (raw_success, raw_failure),
        ).fetchall()
    assert {tuple(row) for row in observations if str(row[1]).endswith("neighbor.jsonl")} == {
        (raw_failure, "failure-neighbor.jsonl", "deferred_cas_frontier", "partial_decode"),
        (raw_success, "success-neighbor.jsonl", "deferred_cas_frontier", "partial_decode"),
    }
    assert {(row[0], row[1], row[2], row[3]) for row in observations if not str(row[1]).endswith("neighbor.jsonl")} == {
        (
            raw_failure,
            "failure.jsonl",
            RawFailureEvidenceKind.TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER.value,
            "unknown",
        ),
        (
            raw_success,
            "success.jsonl",
            RawFailureEvidenceKind.TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER.value,
            "unknown",
        ),
    }
    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.terminal == 0
    assert lifecycle.deferred == 0
    assert lifecycle.unexplained == 2
    status = raw_failure_info_for_root(tmp_path)
    assert status["terminal_rejections"] == 0
    assert status["unexplained_failures"] == 2


@pytest.mark.parametrize("provider", [Provider.CODEX, Provider.CLAUDE_CODE])
def test_neutral_jsonl_restores_exact_source_before_capture(tmp_path: Path, provider: Provider) -> None:
    from polylogue.storage.blob_store import BlobStore

    bootstrap_archive_root(tmp_path)
    payload = (
        _codex_conversation_bytes("restored-neutral")
        if provider is Provider.CODEX
        else b'{"type":"user","sessionId":"restored-neutral","uuid":"m","message":{"role":"user","content":"hi"}}\n'
    )
    source = tmp_path / "session.jsonl"
    source.write_bytes(payload)
    raw_id = _admit(tmp_path, (), provider=provider, path=str(source), payload=payload)
    blob_path = BlobStore(tmp_path / "blob").blob_path(hashlib.sha256(payload).hexdigest())
    blob_path.unlink()
    report = _derive(tmp_path, validation_mode=ValidationMode.OFF)
    assert report.failed == 0, report.outcomes
    assert blob_path.read_bytes() == payload
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (1,)


@pytest.mark.parametrize("provider", [Provider.CODEX, Provider.CLAUDE_CODE])
def test_neutral_empty_session_stream_has_clean_non_session_census(tmp_path: Path, provider: Provider) -> None:
    bootstrap_archive_root(tmp_path)
    _admit(tmp_path, (), provider=provider, path="session.jsonl", payload=b"")
    _derive(tmp_path, validation_mode=ValidationMode.OFF)
    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.terminal == 0
    assert lifecycle.unexplained == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT parsed_at_ms IS NOT NULL, parse_error FROM raw_sessions").fetchone() == (1, None)
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)
        assert conn.execute("SELECT status, member_count FROM raw_membership_census").fetchall() == [("non_session", 0)]
        assert conn.execute("SELECT status, logical_keys_json FROM raw_authority_parser_census").fetchall() == [
            ("complete", "[]")
        ]
        assert conn.execute("SELECT COUNT(*) FROM raw_session_memberships").fetchone() == (0,)


def test_empty_census_schema_exemption_requires_retained_jsonl_frontier(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, (), provider=Provider.CODEX, path="session.jsonl", payload=b"")
    assert _derive(tmp_path, validation_mode=ValidationMode.OFF).failed == 0
    assert _inspect(tmp_path, raw_id, validation_mode=ValidationMode.ADVISORY) == "valid"
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET source_path='session.json' WHERE raw_id=?", (raw_id,))
    # A document must prove its own schema policy; an empty stream's current
    # zero-member receipt cannot exempt an invalid empty JSON document.
    assert _inspect(tmp_path, raw_id, validation_mode=ValidationMode.ADVISORY) == "stale"


def test_sessionless_json_document_still_requires_schema_policy(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, (), provider=Provider.CODEX, path="session.json", payload=b"[]")
    _derive(tmp_path, validation_mode=ValidationMode.OFF)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT status, member_count FROM raw_membership_census WHERE raw_id=?", (raw_id,)
        ).fetchone() == ("non_session", 0)
        # A current census cannot replace this document's missing policy.
        conn.execute("DELETE FROM raw_artifacts WHERE raw_id=?", (raw_id,))
        conn.execute("UPDATE raw_sessions SET validation_mode=NULL WHERE raw_id=?", (raw_id,))
    assert _inspect(tmp_path, raw_id, validation_mode=ValidationMode.ADVISORY) == "stale"


@pytest.mark.parametrize("payload", [b"", b'{"display":"neutral history","timestamp":1}\n'])
def test_declared_raw_only_history_keeps_its_schema_exemption(tmp_path: Path, payload: bytes) -> None:
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, (), provider=Provider.CLAUDE_CODE, path="history.jsonl", payload=payload)
    assert _derive(tmp_path, validation_mode=ValidationMode.OFF).failed == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT parse_as_session, schema_eligible FROM raw_artifacts WHERE raw_id=?", (raw_id,)
        ).fetchall() == ([(0, 0)] if payload else [])
        assert conn.execute("SELECT parse_error FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone() == (None,)
    assert _inspect(tmp_path, raw_id, validation_mode=ValidationMode.OFF) == "valid"
    with sqlite3.connect(tmp_path / "source.db") as conn:
        # Exercise the existing typed raw-only exemption when no policy
        # receipt was captured, rather than overriding a stored OFF verdict.
        conn.execute("UPDATE raw_sessions SET validation_mode=NULL WHERE raw_id=?", (raw_id,))
    assert _inspect(tmp_path, raw_id, validation_mode=ValidationMode.ADVISORY) == "valid"


def test_neutral_retry_refreshes_validation_without_reparsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.schemas import validate_retained_document as original_validate
    from polylogue.schemas.runtime_registry import SchemaRegistry
    from polylogue.sources import prepared_jsonl
    from polylogue.storage.sqlite.reference_seal import ReferenceSealStaleError

    bootstrap_archive_root(tmp_path)
    target = _admit(tmp_path, (), provider=Provider.CODEX, path="session.jsonl", payload=_codex_conversation_bytes())
    writer = SchemaRegistry(storage_root=tmp_path / "schemas")
    reader = SchemaRegistry(storage_root=tmp_path / "schemas")
    original_init = RawObservationDerivation.__init__
    original_prepare = prepared_jsonl.prepare_jsonl_blob
    original_compute = RawObservationDerivation._compute_prepared
    parses = 0
    validations = 0
    retried = False
    versions: list[str] = []

    def initialize(self: RawObservationDerivation, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._schema_registry = reader

    def prepare(*args: Any, **kwargs: Any) -> Any:
        nonlocal parses
        parses += 1
        return original_prepare(*args, **kwargs)

    def validate(*args: Any, **kwargs: Any) -> Any:
        nonlocal validations
        validations += 1
        verdict = original_validate(*args, **kwargs)
        assert verdict.schema_resolution is not None
        versions.append(verdict.schema_resolution.package_version)
        return verdict

    def compute(self: RawObservationDerivation, *args: Any, **kwargs: Any) -> Any:
        nonlocal retried
        if not retried:
            retried = True
            writer.write_schema_version("codex", "v999", {"type": "object"}, element_kind="event_record")
            raise ReferenceSealStaleError("synthetic Source change after neutral validation")
        return original_compute(self, *args, **kwargs)

    monkeypatch.setattr(RawObservationDerivation, "__init__", initialize)
    monkeypatch.setattr(prepared_jsonl, "prepare_jsonl_blob", prepare)
    monkeypatch.setattr("polylogue.schemas.validate_retained_document", validate)
    monkeypatch.setattr(RawObservationDerivation, "_compute_prepared", compute)
    report = run_on_convergence_owner(
        tmp_path, "test.raw.neutral-retry-schema", lambda owner: _converge_raw(tmp_path, owner, target)
    )
    assert report.failed == 0, report.outcomes
    assert report.done == 1
    assert parses == 1
    assert validations == 2
    assert versions[0] != "v999"
    assert versions[1] == "v999"


def test_empty_claude_history_keeps_raw_only_admission(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    _admit(tmp_path, (), provider=Provider.CLAUDE_CODE, path="history.jsonl", payload=b"")
    report = _derive(tmp_path, validation_mode=ValidationMode.OFF)
    assert report.failed == 0, report.outcomes
    assert read_raw_failure_lifecycle(tmp_path / "source.db").terminal == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT parse_error FROM raw_sessions").fetchone() == (None,)
