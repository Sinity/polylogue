"""Raw-observation laws not owned by the retired all-source scanner.

The canonical derivation laws live in ``test_raw_observation_derivation``.
This module keeps the durable raw-failure lifecycle laws that are independent
of a materialization selector, plus canonical postconditions whose fixtures
exercise retained source evidence and component isolation.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.revision_authority import append_source_revision
from polylogue.core.enums import ArtifactSupportStatus, Origin, Provider
from polylogue.core.errors import RawCASFrontierError
from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind
from polylogue.daemon.derivation import Budget, DerivationRegistry, DerivationReport, PassCursor, converge
from polylogue.daemon.status import raw_failure_info_for_root
from polylogue.operations.raw_observation_derivation import (
    converge_raw_observations,
    raw_observation_frame,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.derived.raw import RawObservationDerivation
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.raw_failure_lifecycle import read_raw_failure_lifecycle
from polylogue.storage.raw_retention import RawFrontierBlockedPaths
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceArtifact, upsert_raw_artifact
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.archive_templates import bootstrap_archive_root


def _codex_conversation_bytes(session_id: str = "session", text: str = "hi") -> bytes:
    return (
        b'{"type":"session_meta","payload":{"id":"' + session_id.encode() + b'"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"m-'
        + session_id.encode()
        + b'","role":"user","content":[{"type":"input_text","text":"'
        + text.encode()
        + b'"}]}}\n'
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
            acquired_at_ms=acquired_at_ms,
        )


def _derive(
    root: Path,
    *,
    source_roots: tuple[Path, ...] = (),
    limit: int = 128,
    max_payload_bytes: int = 64 * 1024 * 1024,
    cursor: PassCursor | None = None,
) -> DerivationReport:
    return converge_raw_observations(
        root,
        source_roots=source_roots,
        limit=limit,
        max_payload_bytes=max_payload_bytes,
        cursor=cursor,
    )


def _inspect(root: Path, raw_id: str) -> str:
    adapter = RawObservationDerivation(root)
    return adapter.inspect(raw_observation_frame(root), (raw_id,))[raw_id]


def _seed_expanded_component(root: Path, source_paths: tuple[str, ...]) -> tuple[str, ...]:
    from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind

    key = "codex-session:expanded-stream-safety"
    with ArchiveStore.open_existing(root, read_only=False) as store:
        raw_ids = []
        for index, source_path in enumerate(source_paths):
            raw_id = store.write_raw_payload(
                provider=Provider.CODEX,
                payload=_codex_conversation_bytes("expanded-stream-safety", f"member-{index}"),
                source_path=source_path,
                acquired_at_ms=index + 1,
            )
            store.bind_raw_revision(
                raw_id,
                RawRevisionEnvelope(
                    key,
                    RawRevisionKind.FULL,
                    f"revision-{index}",
                    index,
                    authority=RawRevisionAuthority.QUARANTINED,
                ),
            )
            raw_ids.append(raw_id)
        store.commit()
    with sqlite3.connect(root / "source.db") as conn:
        conn.executemany(
            "UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?",
            ((40 * 1024 * 1024, raw_id) for raw_id in raw_ids),
        )
        conn.commit()
    return tuple(raw_ids)


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


def test_canonical_replay_cleans_orphaned_messages_before_replacement(tmp_path: Path) -> None:
    """Replacing a lost session cannot violate message uniqueness or touch a foreign session."""
    from polylogue.archive.message.roles import Role
    from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive

    bootstrap_archive_root(tmp_path)
    raw_id = _admit(
        tmp_path,
        (),
        path="orphaned-current-cohort.jsonl",
        provider=Provider.CODEX,
        payload=_codex_conversation_bytes("orphaned-current-cohort"),
    )
    assert _derive(tmp_path).failed == 0
    with sqlite3.connect(tmp_path / "index.db") as conn:
        target_id = str(conn.execute("SELECT session_id FROM sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0])
        conn.execute("PRAGMA foreign_keys = OFF")
        foreign_id = write_parsed_session_to_archive(
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


def test_canonical_scope_does_not_certify_or_rewrite_outside_observations(tmp_path: Path) -> None:
    """A bounded source pass owns only its declared source-root scope."""
    bootstrap_archive_root(tmp_path)
    source = tmp_path / "selected"
    selected = _admit(tmp_path, ("selected",), path=str(source / "one.json"))
    outside = _admit(tmp_path, ("outside",), path=str(tmp_path / "outside.json"))

    report = _derive(tmp_path, source_roots=(source,), limit=1)
    assert report.failed == 0
    assert _inspect(tmp_path, selected) == "valid"
    assert _inspect(tmp_path, outside) == "missing"
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchall() == [("selected",)]


@pytest.mark.parametrize(
    ("selected_directory", "outside_directory"),
    (("100%", "1000"), ("Case", "case")),
)
def test_canonical_source_root_scope_is_literal_and_case_sensitive(
    tmp_path: Path,
    selected_directory: str,
    outside_directory: str,
) -> None:
    """A source root selects its actual descendants, never a LIKE-expanded sibling.

    Anti-vacuity: replacing the canonical source-path interval with a LIKE
    prefix admits the ``1000`` or case-folded sibling into this scoped pass.
    """
    bootstrap_archive_root(tmp_path)
    selected_root = tmp_path / selected_directory
    selected = _admit(tmp_path, ("selected",), path=str(selected_root / "member.json"))
    outside = _admit(tmp_path, ("outside",), path=str(tmp_path / outside_directory / "member.json"))

    report = _derive(tmp_path, source_roots=(selected_root,), limit=2)

    assert report.done == 1 and report.failed == report.pending == 0
    assert _inspect(tmp_path, selected) == "valid"
    assert _inspect(tmp_path, outside) == "missing"
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions ORDER BY native_id").fetchall() == [("selected",)]


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


def test_canonical_component_budget_fails_only_the_oversized_component(tmp_path: Path) -> None:
    """A resource refusal is durable failure, not archive-wide suppression."""
    bootstrap_archive_root(tmp_path)
    healthy = _admit(tmp_path, ("healthy",), path="healthy.json")
    oversized = _admit(tmp_path, ("oversized",), path="oversized.json")
    with sqlite3.connect(tmp_path / "source.db") as conn:
        healthy_size = int(
            conn.execute("SELECT blob_size FROM raw_sessions WHERE raw_id = ?", (healthy,)).fetchone()[0]
        )
        conn.execute("UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?", (10_000, oversized))
        conn.commit()

    report = _derive(tmp_path, limit=2, max_payload_bytes=healthy_size + 1)
    assert report.failed == 1
    assert _inspect(tmp_path, healthy) == "valid"
    assert _inspect(tmp_path, oversized) != "valid"
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchall() == [("healthy",)]


def test_canonical_compute_expands_every_stream_safe_member_descriptor(tmp_path: Path) -> None:
    """Whale preparation must inspect the complete expanded stream-safe component."""
    bootstrap_archive_root(tmp_path)
    raw_ids = _seed_expanded_component(tmp_path, ("rollout-a.jsonl", "rollout-b.jsonl"))
    adapter = RawObservationDerivation(tmp_path, max_payload_bytes=8 * 1024 * 1024 * 1024, stream_safe_only=True)

    replacement = adapter.compute(raw_observation_frame(tmp_path), raw_ids[0])

    assert set(replacement.raw_ids) == set(raw_ids)
    assert replacement.raw_ids


def test_canonical_compute_excludes_a_non_stream_safe_expanded_member_before_blob_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-stream-safe expanded sibling cannot enter the widened envelope."""
    bootstrap_archive_root(tmp_path)
    raw_ids = _seed_expanded_component(tmp_path, ("rollout-a.jsonl", "export-b.json"))
    adapter = RawObservationDerivation(tmp_path, max_payload_bytes=8 * 1024 * 1024 * 1024, stream_safe_only=True)

    def forbidden_verify(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("stream-safety refusal must precede blob verification")

    monkeypatch.setattr(BlobStore, "verify", forbidden_verify)
    for _ in range(2):
        with pytest.raises(ValueError, match="stream-safe"):
            adapter.compute(raw_observation_frame(tmp_path), raw_ids[0])
    assert all(_inspect(tmp_path, raw_id) != "valid" for raw_id in raw_ids)


def test_canonical_oversized_component_remains_failed_across_repeated_envelopes(tmp_path: Path) -> None:
    """A terminal resource envelope cannot turn an unmaterialized raw into done."""
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(
        tmp_path,
        ("repeated-envelope",),
        path="repeated-envelope.json",
        payload=_chatgpt_payload(("repeated-envelope",)),
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        blob_size = int(conn.execute("SELECT blob_size FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0])

    first = _derive(tmp_path, limit=1, max_payload_bytes=max(1, blob_size - 1))
    second = _derive(tmp_path, limit=1, max_payload_bytes=max(1, blob_size - 1))

    assert first.done == 0
    assert second.done == 0
    assert first.failed >= 1
    assert second.failed >= 1
    assert _inspect(tmp_path, raw_id) != "valid"
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (0,)


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
            "UPDATE raw_sessions SET parsed_at_ms = 2, parse_error = ? WHERE raw_id = ?",
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
    """A newer/equal validation failure cannot authorize raw replay on reset."""
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(
        tmp_path,
        (),
        path="validation-history.jsonl",
        provider=Provider.CODEX,
        payload=_codex_conversation_bytes("validation-history"),
    )
    assert _derive(tmp_path).failed == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        parsed_at_ms = int(
            conn.execute("SELECT parsed_at_ms FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0]
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
    assert _inspect(tmp_path, raw_id) == expected_state
    report = _derive(tmp_path)
    assert report.failed == 0
    with sqlite3.connect(active_index) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (
            expected_materialized,
        )


def test_canonical_tied_validation_refusal_is_terminal_without_republishing_output(tmp_path: Path) -> None:
    """A tied validation failure is terminal refusal, not retryable output debt."""
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(
        tmp_path,
        ("tied-validation",),
        path="tied-validation.jsonl",
        provider=Provider.CODEX,
        payload=_codex_conversation_bytes("tied-validation"),
    )
    assert _derive(tmp_path).failed == 0

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

    report = _derive(tmp_path)
    assert report.done == 0
    assert report.pending == report.failed == 0
    assert _inspect(tmp_path, raw_id) == "valid"
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
    adapter = RawObservationDerivation(tmp_path)
    frame = raw_observation_frame(tmp_path)
    replacement = adapter.compute(frame, raw_id)
    with RebuildLease(tmp_path):
        with pytest.raises(RebuildLeaseUnavailableError):
            adapter.publish(frame, replacement)


def test_canonical_publish_revalidates_the_promoted_active_generation(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, ("active-generation",))
    first_index = tmp_path / "generations" / "first" / "index.db"
    initialize_archive_database(first_index, ArchiveTier.INDEX)
    (tmp_path / ".index-active-pointer").write_text(f"{first_index}\n", encoding="utf-8")
    adapter = RawObservationDerivation(tmp_path)
    frame = raw_observation_frame(tmp_path)
    replacement = adapter.compute(frame, raw_id)

    second_index = tmp_path / "generations" / "second" / "index.db"
    initialize_archive_database(second_index, ArchiveTier.INDEX)
    (tmp_path / ".index-active-pointer").write_text(f"{second_index}\n", encoding="utf-8")

    assert adapter.publish(frame, replacement) is False
    with sqlite3.connect(second_index) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
            acquired_at_ms=1,
        )
        touched_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=tool_call_payload("touched-new"),
            source_path="touched-new.jsonl",
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

    targeted = converge(
        DerivationRegistry((RawObservationDerivation(tmp_path),)),
        raw_observation_frame(tmp_path, raw_ids=(touched_raw_id,)),
        budget=Budget(page=1, discovery=1, inspection=2, compute=1, publication=1),
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
    """A bounded cursor advances across independent components without starvation."""
    bootstrap_archive_root(tmp_path)
    names = tuple(f"bounded-{index}" for index in range(4))
    for name in names:
        _admit(tmp_path, (name,), path=f"{name}.json")

    cursor: PassCursor | None = None
    reports = []
    for _ in names:
        report = _derive(tmp_path, limit=1, cursor=cursor)
        reports.append(report)
        cursor = report.cursor
    assert all(report.failed == 0 for report in reports)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions ORDER BY native_id").fetchall() == [
            (name,) for name in names
        ]


def test_canonical_fairness_survives_ops_reset_with_a_process_cursor(tmp_path: Path) -> None:
    """Deleting disposable ops state cannot reset the canonical bounded cursor."""
    bootstrap_archive_root(tmp_path)
    names = tuple(f"ops-reset-{index}" for index in range(4))
    for name in names:
        _admit(tmp_path, (name,), path=f"{name}.json")

    cursor: PassCursor | None = None
    reports = []
    for _ in names:
        report = _derive(tmp_path, limit=1, cursor=cursor)
        reports.append(report)
        cursor = report.cursor
        (tmp_path / "ops.db").unlink(missing_ok=True)

    assert [report.done for report in reports] == [1, 1, 1, 1]
    assert all(report.failed == 0 for report in reports)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions ORDER BY native_id").fetchall() == [
            (name,) for name in names
        ]


def test_canonical_deadline_bounds_a_pass_without_substituting_a_count_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wall-clock deadline stops between canonical components and preserves progress."""
    bootstrap_archive_root(tmp_path)
    names = tuple(f"deadline-{index}" for index in range(3))
    for name in names:
        _admit(tmp_path, (name,), path=f"{name}.json")

    adapter = RawObservationDerivation(tmp_path)
    clock = [0.0]
    monkeypatch.setattr("polylogue.daemon.derivation.time.monotonic", lambda: clock[0])
    original_compute = adapter.compute

    def compute_then_expire(frame: object, key: str) -> object:
        replacement = original_compute(frame, key)  # type: ignore[arg-type]
        clock[0] = 2.0
        return replacement

    monkeypatch.setattr(adapter, "compute", compute_then_expire)
    bounded = converge(
        DerivationRegistry((adapter,)),
        raw_observation_frame(tmp_path),
        budget=Budget(page=3, discovery=3, inspection=6, compute=3, publication=3, deadline_s=1.0),
    )

    assert bounded.done == 1
    assert bounded.pending >= 2
    assert bounded.failed == 0
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)


def test_canonical_failed_publication_cannot_report_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A publication that makes no progress remains pending, never successful."""
    bootstrap_archive_root(tmp_path)
    _admit(tmp_path, ("publication-blocked",))

    monkeypatch.setattr(RawObservationDerivation, "publish", lambda *_args, **_kwargs: False)
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
    adapter = RawObservationDerivation(tmp_path)
    frame = raw_observation_frame(tmp_path)
    old_replacement = adapter.compute(frame, old_raw_id)
    assert adapter.publish(frame, old_replacement) is True
    new_raw_id = _admit(
        tmp_path,
        (),
        path="same-head.jsonl",
        provider=Provider.CODEX,
        payload=new_payload,
        acquired_at_ms=2,
    )
    assert _derive(tmp_path).failed == 0

    with sqlite3.connect(tmp_path / "index.db") as conn:
        head = conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = ?",
            ("codex-session:same-head",),
        ).fetchone()
        assert head == (new_raw_id,)
    try:
        adapter.publish(frame, old_replacement)
    except RawCASFrontierError:
        pass
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = ?",
            ("codex-session:same-head",),
        ).fetchone() == (new_raw_id,)


def test_canonical_expanded_component_budget_blocks_before_blob_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aggregate payload limits inspect every component member before parsing any blob."""
    from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind

    bootstrap_archive_root(tmp_path)
    key = "codex-session:aggregate-budget"
    payload = _codex_conversation_bytes("aggregate-budget")
    with ArchiveStore.open_existing(tmp_path, read_only=False) as store:
        raw_ids = []
        for index in range(2):
            raw_id = store.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path="aggregate.jsonl",
                acquired_at_ms=index + 1,
            )
            store.bind_raw_revision(
                raw_id,
                RawRevisionEnvelope(
                    key,
                    RawRevisionKind.FULL,
                    raw_id,
                    0,
                    authority=RawRevisionAuthority.QUARANTINED,
                ),
            )
            raw_ids.append(raw_id)
        store.commit()
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.executemany(
            "UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?",
            ((600, raw_id) for raw_id in raw_ids),
        )
        conn.commit()

    def forbidden_verify(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("aggregate resource refusal must precede blob verification")

    monkeypatch.setattr(BlobStore, "verify", forbidden_verify)
    report = _derive(tmp_path, max_payload_bytes=1_000)
    assert report.failed >= 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


def test_canonical_already_valid_oversized_sibling_blocks_component_replay_before_blob_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid sibling is still part of the replay component's resource proof."""
    from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind

    bootstrap_archive_root(tmp_path)
    key = "codex-session:oversized-sibling"
    payloads = (_codex_conversation_bytes("small-gap"), _codex_conversation_bytes("large-done"))
    with ArchiveStore.open_existing(tmp_path, read_only=False) as store:
        raw_ids = []
        for index, payload in enumerate(payloads):
            raw_id = store.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path="shared.jsonl",
                acquired_at_ms=index + 1,
            )
            store.bind_raw_revision(
                raw_id,
                RawRevisionEnvelope(
                    key,
                    RawRevisionKind.FULL,
                    raw_id,
                    0,
                    authority=RawRevisionAuthority.QUARANTINED,
                ),
            )
            raw_ids.append(raw_id)
        store.commit()
    assert _derive(tmp_path, max_payload_bytes=10_000).failed == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET blob_size = 2_000 WHERE raw_id = ?", (raw_ids[1],))
        conn.commit()
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("DELETE FROM sessions WHERE raw_id = ?", (raw_ids[0],))
        conn.execute("DELETE FROM raw_revision_applications WHERE raw_id = ?", (raw_ids[0],))
        conn.commit()

    def forbidden_verify(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("expanded oversized sibling must block before blob verification")

    monkeypatch.setattr(BlobStore, "verify", forbidden_verify)
    report = _derive(tmp_path, max_payload_bytes=1_000)
    assert report.failed >= 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE raw_id = ?", (raw_ids[0],)).fetchone() == (0,)


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
            source_index=0,
            acquired_at_ms=1,
        )
        new_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"revision":"new"}',
            source_path="same-coordinate.jsonl",
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
            acquired_at_ms=1,
        )
        raw_failure = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"name":"failure"}',
            source_path="failure.jsonl",
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
