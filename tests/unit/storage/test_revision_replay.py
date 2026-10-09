from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from io import BytesIO
from itertools import permutations
from pathlib import Path
from typing import Any

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.revision_authority import (
    BYTE_AUTHORITY_CENSUS_DETAIL,
    HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL,
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
    append_source_revision,
)
from polylogue.archive.revision_replay import (
    ApplicationDecision,
    RevisionCandidate,
    RevisionReplayPlan,
    plan_revision_replay,
)
from polylogue.archive.session_revision_membership import (
    MembershipClassification,
    MembershipRevision,
    classify_membership_revisions,
)
from polylogue.core.enums import Provider
from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind
from polylogue.pipeline.ids import session_content_hash, session_revision_projection
from polylogue.sources.dispatch import merge_parsed_session_chunks, parse_stream_payload
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.raw_authority import iter_parser_census_logical_keys, raw_authority_parser_fingerprint
from polylogue.storage.sqlite.archive_tiers import revision_governance as archive_revision_governance
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.prepared_membership import (
    apply_prepared_aggregate_replay,
    publish_prepared_membership_classification,
    write_prepared_retained_session,
)
from tests.infra.prepared_replay import (
    apply_prepared_revision_replay,
    independent_source_connection,
    open_independent_source,
    publish_fixture_byte_classification,
    publish_membership_census,
    publish_prepared_source,
    run_on_convergence_owner,
    write_fixture_raw_session,
)


def _candidate(
    raw_id: str,
    kind: RawRevisionKind,
    generation: int,
    *,
    authority: RawRevisionAuthority = RawRevisionAuthority.BYTE_PROVEN,
    size: int = 100,
    predecessor: str | None = None,
    baseline: str | None = None,
    start: int | None = None,
    end: int | None = None,
) -> RevisionCandidate:
    return RevisionCandidate(
        raw_id=raw_id,
        logical_source_key="codex-session:session",
        kind=kind,
        source_revision=f"revision-{raw_id}",
        acquisition_generation=generation,
        authority=authority,
        blob_size=size,
        predecessor_raw_id=predecessor,
        baseline_raw_id=baseline,
        append_start_offset=start,
        append_end_offset=end,
    )


def _decisions(candidates: list[RevisionCandidate]) -> dict[str, ApplicationDecision]:
    return {item.raw_id: item.decision for item in plan_revision_replay(candidates).applications}


def _publish_membership(
    archive: ArchiveStore,
    logical_source_key: str,
    classification: MembershipClassification,
    parsed_by_raw_id: dict[str, ParsedSession],
    projections_by_raw_id: dict[str, Any],
    *,
    acquired_at_ms: int,
) -> str | None:
    """Publish one supplied classification on the canonical prepared route."""
    session_id, _decisions = publish_prepared_membership_classification(
        archive,
        logical_source_key,
        classification,
        parsed_by_raw_id,
        projections_by_raw_id,
        decided_at_ms=acquired_at_ms,
    )
    return session_id


def test_partial_append_overlap_is_ambiguous() -> None:
    baseline = _candidate("base", RawRevisionKind.FULL, 0, size=100)
    first = _candidate(
        "first", RawRevisionKind.APPEND, 1, size=100, predecessor="base", baseline="base", start=100, end=200
    )
    overlapping = _candidate(
        "overlap", RawRevisionKind.APPEND, 2, size=100, predecessor="first", baseline="base", start=150, end=250
    )

    plan = plan_revision_replay([baseline, first, overlapping])

    decisions = {item.raw_id: item for item in plan.applications}
    assert plan.accepted_raw_ids == ("base", "first")
    assert decisions["overlap"].decision is ApplicationDecision.AMBIGUOUS
    assert "inside an accepted append window" in decisions["overlap"].detail


def test_same_start_longer_append_supersedes_shorter_observation() -> None:
    baseline = _candidate("base", RawRevisionKind.FULL, 0, size=100)
    short = _candidate(
        "short", RawRevisionKind.APPEND, 1, size=100, predecessor="base", baseline="base", start=100, end=200
    )
    longer = _candidate(
        "longer", RawRevisionKind.APPEND, 2, size=100, predecessor="base", baseline="base", start=100, end=250
    )

    plan = plan_revision_replay([baseline, short, longer])

    decisions = {item.raw_id: item for item in plan.applications}
    assert plan.accepted_raw_ids == ("base", "longer")
    assert decisions["short"].decision is ApplicationDecision.DEFERRED


def test_same_start_equal_length_successors_remain_ambiguous() -> None:
    baseline = _candidate("base", RawRevisionKind.FULL, 0, size=100)
    older = _candidate("older", RawRevisionKind.APPEND, 1, predecessor="base", baseline="base", start=100, end=200)
    newer = _candidate("newer", RawRevisionKind.APPEND, 2, predecessor="base", baseline="base", start=100, end=200)

    assert _decisions([baseline, older, newer]) == {
        "base": ApplicationDecision.SELECTED_BASELINE,
        "older": ApplicationDecision.AMBIGUOUS,
        "newer": ApplicationDecision.AMBIGUOUS,
    }


def test_multiple_newest_successors_remain_ambiguous() -> None:
    baseline = _candidate("base", RawRevisionKind.FULL, 0, size=100)
    left = _candidate("left", RawRevisionKind.APPEND, 2, predecessor="base", baseline="base", start=100, end=200)
    right = _candidate("right", RawRevisionKind.APPEND, 2, predecessor="base", baseline="base", start=100, end=250)
    older = _candidate("older", RawRevisionKind.APPEND, 1, predecessor="base", baseline="base", start=100, end=150)

    assert _decisions([baseline, left, right, older]) == {
        "base": ApplicationDecision.SELECTED_BASELINE,
        "left": ApplicationDecision.AMBIGUOUS,
        "right": ApplicationDecision.AMBIGUOUS,
        "older": ApplicationDecision.AMBIGUOUS,
    }


def test_incomplete_successor_bounds_are_deferred() -> None:
    baseline = _candidate("base", RawRevisionKind.FULL, 0, size=100)
    incomplete = _candidate("incomplete", RawRevisionKind.APPEND, 1, predecessor="base", baseline="base", start=100)

    assert _decisions([baseline, incomplete]) == {
        "base": ApplicationDecision.SELECTED_BASELINE,
        "incomplete": ApplicationDecision.DEFERRED,
    }


def test_overlap_from_superseded_baseline_is_deferred() -> None:
    old = _candidate("old", RawRevisionKind.FULL, 0, size=100)
    baseline = _candidate("base", RawRevisionKind.FULL, 1, size=100)
    accepted = _candidate(
        "accepted", RawRevisionKind.APPEND, 2, predecessor="base", baseline="base", start=100, end=250
    )
    old_append = _candidate(
        "old-append", RawRevisionKind.APPEND, 3, predecessor="old", baseline="old", start=180, end=220
    )

    decisions = _decisions([old, baseline, accepted, old_append])
    assert decisions["old-append"] is ApplicationDecision.DEFERRED


def _codex_jsonl(records: list[dict[str, object]]) -> bytes:
    return b"".join(json.dumps(record, separators=(",", ":")).encode() + b"\n" for record in records)


def _parse_codex_jsonl(payload: bytes) -> ParsedSession:
    sessions = parse_stream_payload(
        Provider.CODEX,
        (json.loads(line) for line in payload.splitlines() if line),
        "fold-codex",
    )
    assert len(sessions) == 1
    return sessions[0]


def _codex_fold_payloads() -> tuple[bytes, bytes]:
    baseline = _codex_jsonl(
        [
            {"type": "session_meta", "payload": {"id": "fold-codex", "timestamp": "2026-07-12T00:00:00Z"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "m1",
                    "role": "user",
                    "timestamp": "2026-07-12T00:00:01Z",
                    "content": [{"type": "input_text", "text": "needle alpha"}],
                },
            },
        ]
    )
    append = _codex_jsonl(
        [
            {"type": "turn_context", "payload": {"cwd": "/repo", "model": "gpt-5"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "m2",
                    "role": "assistant",
                    "timestamp": "2026-07-12T00:00:02Z",
                    "content": [{"type": "output_text", "text": "needle beta"}],
                },
            },
        ]
    )
    return baseline, append


def _with_fold_attachment(session: ParsedSession) -> ParsedSession:
    """Exercise attachment persistence without inventing Codex parser behavior."""
    return session.model_copy(
        update={
            "attachments": [
                ParsedAttachment(
                    provider_attachment_id="fold-image-1",
                    message_provider_id="m1",
                    name="fold-proof.png",
                    mime_type="image/png",
                    size_bytes=4,
                    upload_origin="url",
                    source_url="https://example.invalid/fold-proof.png",
                )
            ]
        }
    )


def test_live_revision_binding_without_parser_evidence_does_not_issue_receipt(tmp_path: Path) -> None:
    """Binding acquisition metadata cannot self-certify parser authority."""
    bootstrap_archive_root(tmp_path)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"live-receipt"}}\n',
            source_path="live/codex.jsonl",
            canonical_source_path="live/codex.jsonl",
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            raw_id,
            RawRevisionEnvelope(
                "codex-session:live-receipt",
                RawRevisionKind.FULL,
                "live-receipt-v1",
                0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        receipt = conn.execute(
            "SELECT parser_fingerprint, status, logical_keys_json FROM raw_authority_parser_census WHERE raw_id = ?",
            (raw_id,),
        ).fetchone()

    assert receipt is None


def test_parser_receipt_fails_when_observed_identity_differs_from_binding(tmp_path: Path) -> None:
    """The production receipt writer cannot certify an unobserved durable key.

    Its parsed session carries a canonical identity deliberately different
    from the raw's pre-existing durable binding. Mutating receipt issuance
    back to reconstruct from ``raw_sessions`` makes this receipt complete
    with the bound key instead, so this exercises the writer shared by
    ordinary imports and retained-raw census rather than a test-local check.
    """
    bootstrap_archive_root(tmp_path)
    payload = _codex_jsonl(
        [
            {"type": "session_meta", "payload": {"id": "parser-observed-id"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "m1",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "parser proof"}],
                },
            },
        ]
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path="historical/mismatch.jsonl",
            canonical_source_path="historical/mismatch.jsonl",
            acquired_at_ms=1,
            source_index=0,
        )
        archive.bind_raw_revision(
            raw_id,
            RawRevisionEnvelope(
                "codex-session:durable-but-not-parsed",
                RawRevisionKind.FULL,
                "mismatch-v1",
                0,
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )

    publish_prepared_source(
        tmp_path,
        "test.revision.parser-census",
        lambda seal: archive_revision_governance.record_current_parser_source_census(
            seal,
            raw_id,
            parser_sessions=[
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="parser-observed-id",
                    messages=[],
                )
            ],
        ),
    )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        receipt = conn.execute(
            "SELECT status, logical_keys_json FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
        ).fetchone()

    assert receipt is not None
    assert receipt[0] == "failed"
    assert tuple(iter_parser_census_logical_keys(receipt[1])) == ("codex-session:parser-observed-id",)


def test_terminal_non_session_failure_has_complete_empty_parser_census(tmp_path: Path) -> None:
    """A typed terminal failure is a settled non-session source disposition."""

    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"not valid codex jsonl",
            source_path="terminal-corrupt.jsonl",
            canonical_source_path="terminal-corrupt.jsonl",
            acquired_at_ms=1,
        )
        archive.record_raw_failure_evidence(
            raw_id,
            provider=Provider.CODEX,
            source_path="terminal-corrupt.jsonl",
            source_index=0,
            acquired_at_ms=1,
            kind=RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT,
        )
        archive.mark_raw_parse_failed(
            raw_id,
            provider=Provider.CODEX,
            error=ValueError("terminal corrupt input"),
            preserve_existing_failure_evidence=True,
        )

    publish_prepared_source(
        tmp_path,
        "test.revision.terminal-census",
        lambda seal: archive_revision_governance.record_current_parser_source_census(seal, raw_id),
    )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        status, keys = conn.execute(
            "SELECT status, logical_keys_json FROM raw_authority_parser_census WHERE raw_id = ?",
            (raw_id,),
        ).fetchone()
    assert status == "complete"
    assert tuple(iter_parser_census_logical_keys(keys)) == ()

    from polylogue.storage.source_generation_receipts import _raw_receipt

    with sqlite3.connect(tmp_path / "source.db") as source, sqlite3.connect(tmp_path / "index.db") as index:
        with _raw_receipt(source, index, raw_id, check_stop=None) as receipt:
            assert receipt.parser_complete is True
            assert tuple(receipt.logicals) == ()


def test_byte_governed_fragment_parser_receipt_preserves_durable_membership_keys(tmp_path: Path) -> None:
    """A byte-governed receipt cannot overclaim an empty durable identity set."""
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"append":true}\n',
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=1,
        )
        with independent_source_connection(archive) as conn:
            conn.execute(
                """
                INSERT INTO raw_session_memberships (
                    raw_id, logical_source_key, provider_session_id, source_revision,
                    normalized_content_hash, message_count
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (raw_id, "codex-session:durable-append", "durable-append", "revision-1", bytes(32), 1),
            )
            conn.execute(
                """
                INSERT INTO raw_membership_census (
                    raw_id, parser_fingerprint, status, member_count, censused_at_ms, detail, revision_authority
                ) VALUES (?, ?, 'failed', 1, 1, ?, 'byte_proven')
                """,
                (raw_id, raw_authority_parser_fingerprint(), BYTE_AUTHORITY_CENSUS_DETAIL),
            )

    publish_prepared_source(
        tmp_path,
        "test.revision.byte-governed-census",
        lambda seal: archive_revision_governance.record_current_parser_source_census(seal, raw_id),
    )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        receipt = conn.execute(
            "SELECT status, logical_keys_json FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
        ).fetchone()

    assert receipt is not None
    assert receipt[0] == "complete"
    assert tuple(iter_parser_census_logical_keys(receipt[1])) == ("codex-session:durable-append",)


def test_typed_non_session_receipt_preserves_durable_membership_on_restart(tmp_path: Path) -> None:
    """Restart validation must use the same durable shape as receipt creation."""
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"not valid codex jsonl",
            source_path="terminal-corrupt.jsonl",
            canonical_source_path="terminal-corrupt.jsonl",
            acquired_at_ms=1,
        )
        archive.record_raw_failure_evidence(
            raw_id,
            provider=Provider.CODEX,
            source_path="terminal-corrupt.jsonl",
            source_index=0,
            acquired_at_ms=1,
            kind=RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT,
        )
        with independent_source_connection(archive) as conn:
            conn.execute(
                """
                INSERT INTO raw_session_memberships (
                    raw_id, logical_source_key, provider_session_id, source_revision,
                    normalized_content_hash, message_count
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (raw_id, "codex-session:typed-membership", "typed-membership", "revision-1", bytes(32), 1),
            )

    publish_prepared_source(
        tmp_path,
        "test.revision.typed-membership-census",
        lambda seal: archive_revision_governance.record_current_parser_source_census(seal, raw_id),
    )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        receipt = conn.execute(
            "SELECT status, logical_keys_json FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
        ).fetchone()

    assert receipt is not None
    assert receipt[0] == "complete"
    expected_keys = ("codex-session:typed-membership",)
    assert tuple(iter_parser_census_logical_keys(receipt[1])) == expected_keys

    from polylogue.storage.source_generation_receipts import _raw_receipt

    with sqlite3.connect(tmp_path / "source.db") as source, sqlite3.connect(tmp_path / "index.db") as index:
        with _raw_receipt(source, index, raw_id, check_stop=None) as raw_receipt:
            assert raw_receipt.parser_complete is True
            assert tuple(logical.logical_source_key for logical in raw_receipt.logicals) == expected_keys


def test_frozen_replay_skips_typed_terminal_non_session_raw(tmp_path: Path) -> None:
    """Terminal non-session evidence settles replay without dispatching its malformed bytes."""

    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"not valid codex jsonl",
            source_path="terminal-replay-corrupt.jsonl",
            canonical_source_path="terminal-replay-corrupt.jsonl",
            acquired_at_ms=1,
        )
        archive.record_raw_failure_evidence(
            raw_id,
            provider=Provider.CODEX,
            source_path="terminal-replay-corrupt.jsonl",
            source_index=0,
            acquired_at_ms=1,
            kind=RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT,
        )
        archive.mark_raw_parse_failed(
            raw_id,
            provider=Provider.CODEX,
            error=ValueError("terminal corrupt input"),
            preserve_existing_failure_evidence=True,
        )

    # The retired frozen-evidence loader is replaced by the canonical Raw
    # derivation: its census phase must settle the typed terminal raw itself.
    # Preparation commits that census in place on the writer and then reports
    # the settled typed refusal instead of dispatching the malformed bytes.
    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.core.raw_failure_evidence import RetainedRawDecodeRefusalError
    from polylogue.operations.raw_observation_derivation import raw_observation_frame
    from polylogue.storage.derived.raw import RawObservationDerivation
    from tests.infra.prepared_replay import run_on_convergence_owner

    def census_rows() -> list[tuple[object, ...]]:
        with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
            return conn.execute("SELECT * FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)).fetchall()

    assert census_rows() == []

    def census_phase(compute: BoundedComputeAdapter) -> RetainedRawDecodeRefusalError:
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute)
        frame = raw_observation_frame(tmp_path)
        with pytest.raises(RetainedRawDecodeRefusalError) as refused:
            adapter.compute(frame, raw_id)
        return refused.value

    refusal = run_on_convergence_owner(tmp_path, "test.revision.terminal-replay", census_phase)
    assert refusal.kind is RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT
    assert len(census_rows()) == 1


def test_membership_receipt_excludes_post_parse_pending_identity(tmp_path: Path) -> None:
    """A parser-derived membership receipt cannot retain its provisional raw key."""
    bootstrap_archive_root(tmp_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="post-parse-receipt",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="receipt proof")],
    )

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"post-parse-receipt"}}\n',
            source_path="live/pending.jsonl",
            canonical_source_path="live/pending.jsonl",
            acquired_at_ms=1,
            post_parse=True,
        )
        publish_membership_census(
            archive,
            raw_id,
            [session],
            parser_fingerprint=raw_authority_parser_fingerprint(),
            censused_at_ms=1,
            revision_authority=None,
        )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        receipt = conn.execute(
            "SELECT logical_keys_json FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
        ).fetchone()

    assert receipt is not None
    assert tuple(iter_parser_census_logical_keys(receipt[0])) == ("codex-session:post-parse-receipt",)


def test_replay_selects_newest_full_and_exact_contiguous_suffix_independent_of_order() -> None:
    candidates = [
        _candidate("old", RawRevisionKind.FULL, 0, size=50),
        _candidate("base", RawRevisionKind.FULL, 1),
        _candidate("append-1", RawRevisionKind.APPEND, 2, predecessor="base", baseline="base", start=100, end=140),
        _candidate(
            "append-2",
            RawRevisionKind.APPEND,
            3,
            predecessor="append-1",
            baseline="base",
            start=140,
            end=180,
        ),
    ]
    expected = {
        "old": ApplicationDecision.SUPERSEDED,
        "base": ApplicationDecision.SELECTED_BASELINE,
        "append-1": ApplicationDecision.APPLIED_APPEND,
        "append-2": ApplicationDecision.APPLIED_APPEND,
    }
    for ordering in permutations(candidates):
        assert _decisions(list(ordering)) == expected


def test_membership_reselection_reuses_equivalent_superseded_receipt(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="session",
        messages=[ParsedMessage(provider_message_id="m0", role=Role.USER, text="same")],
    )
    projection = session_revision_projection(session)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:

        def add_member(raw_id: str) -> MembershipRevision:
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=raw_id.encode(),
                source_path=f"{raw_id}.jsonl",
                canonical_source_path=f"{raw_id}.jsonl",
                acquired_at_ms=1,
                raw_id=raw_id,
            )
            publish_membership_census(
                archive, raw_id, [session], parser_fingerprint="test-parser", censused_at_ms=1, revision_authority=None
            )
            return MembershipRevision(raw_id, projection)

        members = [add_member("representative-b"), add_member("equivalent-z")]
        first = classify_membership_revisions(members)
        assert first.accepted_raw_ids == ("equivalent-z",)
        _publish_membership(
            archive,
            "codex-session:session",
            first,
            {member.raw_id: session for member in members},
            {member.raw_id: projection for member in members},
            acquired_at_ms=1,
        )

        members.append(add_member("accepted-a"))
        second = classify_membership_revisions(members)
        assert second.accepted_raw_ids == ("accepted-a",)
        _publish_membership(
            archive,
            "codex-session:session",
            second,
            {member.raw_id: session for member in members},
            {member.raw_id: projection for member in members},
            acquired_at_ms=2,
        )

        head = archive._conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'codex-session:session'"
        ).fetchone()
        assert head is not None and tuple(head) == ("accepted-a",)
        application_rows = archive._conn.execute(
            """
            SELECT raw_id, decision, accepted_raw_id
            FROM raw_revision_applications
            WHERE logical_source_key = 'codex-session:session'
            ORDER BY raw_id, decision, accepted_raw_id
            """
        ).fetchall()
        # Revision receipts are append-only evidence. When ``accepted-a``
        # becomes the new head, each equivalent member receives a current
        # supersession receipt while its historical supersession remains.
        assert [tuple(row) for row in application_rows] == [
            ("accepted-a", "selected_baseline", "accepted-a"),
            ("equivalent-z", "selected_baseline", "equivalent-z"),
            ("equivalent-z", "superseded", "accepted-a"),
            ("representative-b", "superseded", "accepted-a"),
            ("representative-b", "superseded", "equivalent-z"),
        ]
        matching_receipts = archive._conn.execute(
            """
            SELECT COUNT(*) FROM raw_revision_heads AS h
            JOIN raw_revision_applications AS a
              ON a.logical_source_key = h.logical_source_key
             AND a.accepted_raw_id = h.accepted_raw_id
             AND a.accepted_content_hash = h.accepted_content_hash
            WHERE h.logical_source_key = 'codex-session:session'
              AND a.decision IN ('selected_baseline', 'applied_append')
            """
        ).fetchone()
        assert matching_receipts is not None and tuple(matching_receipts) == (1,)


def test_headless_cohort_keeps_equivalents_quarantined_ambiguous(tmp_path: Path) -> None:
    """No accepted head means no fabricated supersession authority.

    Production dependency: ``apply_raw_membership_classification``'s
    membership write-back. Mutation that must fail this test: labeling
    ``equivalent_raw_ids`` as ``superseded_equivalent``/``byte_proven`` when
    ``accepted_raw_ids`` is empty (the pre-fix behavior that produced 914
    headless-but-byte_proven logical sources on the 2026-07-20 rebuild).
    """
    bootstrap_archive_root(tmp_path)

    def session_with(text: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="session",
            messages=[ParsedMessage(provider_message_id="m0", role=Role.USER, text=text)],
        )

    branch_a = session_with("alpha")
    branch_b = session_with("beta")

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:

        def add_member(raw_id: str, session: ParsedSession) -> MembershipRevision:
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=raw_id.encode(),
                source_path=f"{raw_id}.jsonl",
                canonical_source_path=f"{raw_id}.jsonl",
                acquired_at_ms=1,
                raw_id=raw_id,
            )
            publish_membership_census(
                archive, raw_id, [session], parser_fingerprint="test-parser", censused_at_ms=1, revision_authority=None
            )
            return MembershipRevision(raw_id, session_revision_projection(session))

        members = [
            add_member("branch-a", branch_a),
            add_member("branch-a-dup", branch_a),
            add_member("branch-b", branch_b),
        ]
        # existing_accepted_raw_id="branch-a" forces the presence-guarantee
        # fallback (which would otherwise deterministically pick "branch-b"
        # here) to be REFUSED -- exercising this test's own invariant (no
        # fabricated supersession authority when nothing is accepted)
        # requires the guarded-refusal path, not the now-default
        # fallback-applies-when-headless path (covered separately in
        # tests/unit/archive/test_session_revision_membership.py).
        classification = classify_membership_revisions(members, existing_accepted_raw_id="branch-a")
        assert classification.accepted_raw_ids == ()
        assert classification.equivalent_raw_ids

        session_by_raw = {"branch-a": branch_a, "branch-a-dup": branch_a, "branch-b": branch_b}
        _publish_membership(
            archive,
            "codex-session:session",
            classification,
            session_by_raw,
            {raw_id: session_revision_projection(session) for raw_id, session in session_by_raw.items()},
            acquired_at_ms=1,
        )

        head = archive._conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'codex-session:session'"
        ).fetchone()
        assert head is None

        membership_rows = (
            archive._ensure_source_conn()
            .execute(
                """
            SELECT raw_id, decision, revision_authority
            FROM raw_session_memberships
            WHERE logical_source_key = 'codex-session:session'
            ORDER BY raw_id
            """
            )
            .fetchall()
        )
        assert [tuple(row) for row in membership_rows] == [
            ("branch-a", "ambiguous", "quarantined"),
            ("branch-a-dup", "ambiguous", "quarantined"),
            ("branch-b", "ambiguous", "quarantined"),
        ]


def test_replay_defers_gap_and_quarantines_unproven_evidence() -> None:
    candidates = [
        _candidate("base", RawRevisionKind.FULL, 1),
        _candidate("gap", RawRevisionKind.APPEND, 2, predecessor="base", baseline="base", start=101, end=140),
        _candidate(
            "observed",
            RawRevisionKind.APPEND,
            3,
            authority=RawRevisionAuthority.QUARANTINED,
            start=100,
            end=140,
        ),
    ]
    assert _decisions(candidates) == {
        "base": ApplicationDecision.SELECTED_BASELINE,
        "gap": ApplicationDecision.DEFERRED,
        "observed": ApplicationDecision.AMBIGUOUS,
    }


def test_replay_stops_at_append_branch_without_choosing_by_raw_id() -> None:
    candidates = [
        _candidate("base", RawRevisionKind.FULL, 0),
        _candidate("left", RawRevisionKind.APPEND, 1, predecessor="base", baseline="base", start=100, end=130),
        _candidate("right", RawRevisionKind.APPEND, 1, predecessor="base", baseline="base", start=100, end=140),
    ]
    assert _decisions(candidates) == {
        "base": ApplicationDecision.SELECTED_BASELINE,
        "left": ApplicationDecision.AMBIGUOUS,
        "right": ApplicationDecision.AMBIGUOUS,
    }


def test_replay_requires_byte_proven_full_baseline() -> None:
    candidates = [
        _candidate("asserted", RawRevisionKind.FULL, 0, authority=RawRevisionAuthority.ASSERTED),
    ]
    assert _decisions(candidates) == {"asserted": ApplicationDecision.DEFERRED}


def test_replay_does_not_treat_a_duplicate_of_the_accepted_baseline_as_a_competing_head() -> None:
    """polylogue-qhk8z: a byte-identical duplicate of the accepted baseline must
    not create a false "multiple byte-proven full baselines" tie.

    ``revision_governance.prepare_raw_revision_byte_classification`` writes a duplicate
    decision's ``baseline_raw_id`` to the SAME chain root as the real
    baseline row (``predecessor_raw_id=None`` on both, per
    ``HistoricalRevisionDecision.duplicate_of_raw_id``'s contract), and
    mirrors the baseline's own ``acquisition_generation`` onto it
    (polylogue-5unky's fix). Before this fix, ``plan_revision_replay``'s
    "unique newest generation" tie-break saw two FULL BYTE_PROVEN candidates
    sharing generation 0 and misclassified this as an ambiguous multi-
    baseline fork -- even though one of the two candidates literally IS the
    baseline (``baseline_raw_id == raw_id``) and the other is only its own
    duplicate. That false ambiguity emptied ``accepted_raw_ids``, which
    routed backfill/rebuild callers into the "no accepted chain" membership-
    census fallback for a cohort that in fact has one unambiguous baseline,
    which then tripped ``ActiveByteRevisionChainError`` on retirement
    (reproduced end-to-end in
    ``test_duplicate_of_accepted_baseline_does_not_trip_membership_census_guard``).
    """
    candidates = [
        _candidate("raw-a-baseline", RawRevisionKind.FULL, 0, baseline="raw-a-baseline"),
        _candidate("raw-b-duplicate", RawRevisionKind.FULL, 0, baseline="raw-a-baseline"),
    ]
    plan = plan_revision_replay(candidates)
    assert plan.accepted_raw_ids == ("raw-a-baseline",)
    decisions = {item.raw_id: item.decision for item in plan.applications}
    assert decisions == {
        "raw-a-baseline": ApplicationDecision.SELECTED_BASELINE,
        "raw-b-duplicate": ApplicationDecision.DEFERRED,
    }


def test_cohort_classification_promotes_late_baseline_and_deferred_append(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        append_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"suffix",
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            append_raw_id,
            RawRevisionEnvelope(
                "codex-session:session",
                RawRevisionKind.APPEND,
                "revision-append",
                0,
                predecessor_source_revision="revision-base",
                append_start_offset=8,
                append_end_offset=14,
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )
        baseline_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"baseline",
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            acquired_at_ms=2,
        )
        archive.bind_raw_revision(
            baseline_raw_id,
            RawRevisionEnvelope(
                "codex-session:session",
                RawRevisionKind.FULL,
                "revision-base",
                0,
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )

        plan = publish_fixture_byte_classification(archive, "codex-session:session")

    assert {item.raw_id: item.decision for item in plan.applications} == {
        baseline_raw_id: ApplicationDecision.SELECTED_BASELINE,
        append_raw_id: ApplicationDecision.APPLIED_APPEND,
    }


def test_public_cohort_classification_commits_source_authority_before_return(tmp_path: Path) -> None:
    """The public classification route makes its source transaction visible."""
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        baseline = _write_chain_full(archive, "transaction-owner", 0)
        plan = publish_fixture_byte_classification(archive, "codex-session:session")
        assert plan.accepted_raw_ids == (baseline,)

        with sqlite3.connect(tmp_path / "source.db") as source:
            authority = source.execute(
                "SELECT revision_authority FROM raw_sessions WHERE raw_id = ?",
                (baseline,),
            ).fetchone()

    assert authority == (RawRevisionAuthority.BYTE_PROVEN.value,)


def _write_full_raw(archive: ArchiveStore, *, raw_id: str, payload: bytes, acquired_at_ms: int) -> str:
    """Acquire an undecided fixture revision for the byte-prefix proof laws."""
    written_id = archive.write_raw_payload(
        provider=Provider.CODEX,
        payload=payload,
        source_path="session.jsonl",
        canonical_source_path="session.jsonl",
        acquired_at_ms=acquired_at_ms,
        raw_id=raw_id,
    )
    archive.bind_raw_revision(
        written_id,
        RawRevisionEnvelope(
            "codex-session:session",
            RawRevisionKind.FULL,
            f"revision-{raw_id}",
            0,
            authority=RawRevisionAuthority.QUARANTINED,
        ),
    )
    return written_id


def _acquisition_generation(archive: ArchiveStore, raw_id: str) -> int:
    row = (
        archive._ensure_source_conn()
        .execute("SELECT acquisition_generation FROM raw_sessions WHERE raw_id = ?", (raw_id,))
        .fetchone()
    )
    assert row is not None
    return int(row[0])


def test_duplicate_decision_mid_chain_gets_representative_generation_not_zero(tmp_path: Path) -> None:
    """polylogue-5unky: a duplicate's ``acquisition_generation`` must mirror its
    representative's real chain position, not silently fall back to 0.

    ``_expand_duplicate_decisions`` gives every duplicate member
    ``predecessor_raw_id=None``, so a predecessor-keyed dict walk that only
    ever looks at ``predecessor_raw_id`` never reaches it. Build a 3-link
    byte chain (base -> mid -> head) plus a byte-identical duplicate of the
    *middle* link and prove the duplicate lands on generation 1 (mid's real
    chain position), not the 0 fallback the bug produced.
    """
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        base = _write_full_raw(archive, raw_id="raw-000-base", payload=b"a" * 10, acquired_at_ms=1)
        mid = _write_full_raw(archive, raw_id="raw-010-mid", payload=b"a" * 10 + b"b" * 10, acquired_at_ms=2)
        head = _write_full_raw(
            archive, raw_id="raw-020-head", payload=b"a" * 10 + b"b" * 10 + b"c" * 10, acquired_at_ms=3
        )
        mid_duplicate = _write_full_raw(
            archive, raw_id="raw-011-mid-dup", payload=b"a" * 10 + b"b" * 10, acquired_at_ms=4
        )

        publish_fixture_byte_classification(archive, "codex-session:session")

        assert _acquisition_generation(archive, base) == 0
        assert _acquisition_generation(archive, mid) == 1
        assert _acquisition_generation(archive, head) == 2
        assert _acquisition_generation(archive, mid_duplicate) == 1


def test_duplicate_generation_copy_does_not_drop_the_chain_continuing_representative(tmp_path: Path) -> None:
    """polylogue-5unky: prove the *rejected* fix's failure mode does not recur.

    The naive fix CodeRabbit flagged (mirror the representative's
    ``predecessor_raw_id`` onto its duplicate) makes the duplicate and its
    representative compete for the same key in a plain-dict
    ``{predecessor_raw_id: raw_id}`` generation walk -- whichever entry the
    dict comprehension writes last wins that slot, so the duplicate can
    silently overwrite the real chain-continuing representative and strand
    every generation downstream of it at the fallback of 0.

    This builds exactly that collision shape: a duplicate of ``mid`` that
    sorts *after* ``mid`` in ``_expand_duplicate_decisions``'s
    ``(size, raw_id)`` order -- the ordering the rejected fix's dict
    comprehension would need to let this duplicate clobber ``mid``'s real
    predecessor-keyed slot -- and proves ``head``, two links downstream of
    ``mid``, still gets its correct real generation (2), not the fallback 0
    a reintroduced collision would cause.
    """
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        base = _write_full_raw(archive, raw_id="raw-000-base", payload=b"a" * 10, acquired_at_ms=1)
        mid = _write_full_raw(archive, raw_id="raw-010-mid", payload=b"a" * 10 + b"b" * 10, acquired_at_ms=2)
        head = _write_full_raw(
            archive, raw_id="raw-020-head", payload=b"a" * 10 + b"b" * 10 + b"c" * 10, acquired_at_ms=3
        )
        # Same size and content as `mid`, and lexicographically the LATER of
        # the two raw_ids -- the exact ordering the rejected fix needed for
        # its dict comprehension to let this duplicate win mid's slot.
        mid_duplicate = _write_full_raw(
            archive, raw_id="raw-011-mid-dup", payload=b"a" * 10 + b"b" * 10, acquired_at_ms=4
        )
        assert mid < mid_duplicate  # guards the ordering assumption the collision case depends on

        publish_fixture_byte_classification(archive, "codex-session:session")

        assert _acquisition_generation(archive, base) == 0
        assert _acquisition_generation(archive, mid) == 1
        assert _acquisition_generation(archive, head) == 2
        assert _acquisition_generation(archive, mid_duplicate) == 1


def test_duplicate_of_accepted_baseline_does_not_trip_membership_census_guard(tmp_path: Path) -> None:
    """polylogue-qhk8z end-to-end reproduction: PR #3574's byte-identical-
    duplicate collapse (I4) plus polylogue-5unky's generation-mirroring fix
    together made a duplicate of the accepted baseline share that baseline's
    ``acquisition_generation``. Before the ``plan_revision_replay`` fix
    (``test_replay_does_not_treat_a_duplicate_of_the_accepted_baseline_as_a_
    competing_head``), that shared generation made ``plan_revision_replay``
    see two competing "newest" full baselines and reject the cohort as
    ambiguous, emptying ``accepted_raw_ids`` even though the cohort has one
    genuine, unambiguous baseline. Callers (``sources/revision_backfill.py``,
    ``sources/live/batch.py``) treat an empty ``accepted_raw_ids`` as "no
    accepted chain" and fall back to folding every full-only raw for this
    identity into membership governance via
    ``replace_raw_membership_census(..., retire_full_revision_governance=
    True)`` -- which raised ``ActiveByteRevisionChainError`` the moment it
    tried to retire the baseline raw_id, because the duplicate's own
    ``baseline_raw_id`` column still durably points at it. This reproduces
    that exact interaction against a real archive (mirroring the two-page
    re-export shape ``test_revision_backfill.py`` and
    ``test_rebuild_paging_content_order.py`` construct) and proves the
    cohort is now accepted outright, so the membership-census fallback path
    is never even reached -- while confirming the guard itself still fails
    closed for a raw a duplicate genuinely still depends on.
    """
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        baseline = _write_full_raw(archive, raw_id="raw-a-baseline", payload=b"hello world", acquired_at_ms=1)
        duplicate = _write_full_raw(archive, raw_id="raw-b-duplicate", payload=b"hello world", acquired_at_ms=2)

        plan = publish_fixture_byte_classification(archive, "codex-session:session")

        # The cohort has a unique byte-proven baseline -- the duplicate no
        # longer manufactures a false "multiple newest baselines" ambiguity.
        # Backfill/rebuild callers (sources/revision_backfill.py) gate the
        # membership-census fallback on exactly this ``not
        # plan.accepted_raw_ids`` check, so a non-empty result here means
        # that fallback -- and the guard inside it -- is never invoked for
        # this cohort in production.
        assert plan.accepted_raw_ids == (baseline,)
        assert _acquisition_generation(archive, duplicate) == 0

        # The membership-census guard itself must remain intact: retiring
        # the baseline directly still fails closed, because the duplicate's
        # baseline_raw_id column durably points at it -- a real dependent,
        # not a false one.
        with pytest.raises(archive_revision_governance.ActiveByteRevisionChainError):
            publish_membership_census(
                archive,
                baseline,
                [],
                parser_fingerprint=raw_authority_parser_fingerprint(),
                censused_at_ms=0,
                detail="test-duplicate-guard",
                retire_full_revision_governance=True,
                revision_authority=None,
            )
        archive.rollback()


def test_real_append_chain_folds_segmentation_distinct_full_snapshot(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)

    def parsed(*messages: tuple[str, str]) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="session",
            messages=[
                ParsedMessage(provider_message_id=message_id, role=Role.USER, text=text)
                for message_id, text in messages
            ],
        )

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        baseline = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"a" * 10,
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            baseline,
            RawRevisionEnvelope(
                "codex-session:session", RawRevisionKind.FULL, "full-0", 0, authority=RawRevisionAuthority.BYTE_PROVEN
            ),
        )
        append_one = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"b" * 5,
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=2,
        )
        archive.bind_raw_revision(
            append_one,
            RawRevisionEnvelope(
                "codex-session:session",
                RawRevisionKind.APPEND,
                append_source_revision("full-0", hashlib.sha256(b"b" * 5).hexdigest()),
                1,
                predecessor_source_revision="full-0",
                predecessor_raw_id=baseline,
                baseline_raw_id=baseline,
                append_start_offset=10,
                append_end_offset=15,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        append_two = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"c" * 5,
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=3,
        )
        archive.bind_raw_revision(
            append_two,
            RawRevisionEnvelope(
                "codex-session:session",
                RawRevisionKind.APPEND,
                append_source_revision(
                    append_source_revision("full-0", hashlib.sha256(b"b" * 5).hexdigest()),
                    hashlib.sha256(b"c" * 5).hexdigest(),
                ),
                2,
                predecessor_source_revision=append_source_revision("full-0", hashlib.sha256(b"b" * 5).hexdigest()),
                predecessor_raw_id=append_one,
                baseline_raw_id=baseline,
                append_start_offset=15,
                append_end_offset=20,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        append_plan = publish_fixture_byte_classification(archive, "codex-session:session")
        apply_prepared_revision_replay(
            archive,
            append_plan,
            {
                baseline: parsed(("m0", "zero")),
                append_one: parsed(("m1", "one")),
                append_two: parsed(("m2", "two")),
            },
            acquired_at_ms=0,
        )

        folded = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"a" * 10 + b"b" * 5 + b"c" * 5,
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            acquired_at_ms=4,
        )
        archive.bind_raw_revision(
            folded,
            RawRevisionEnvelope(
                "codex-session:session",
                RawRevisionKind.FULL,
                "full-folded",
                3,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        folded_plan = publish_fixture_byte_classification(archive, "codex-session:session")
        folded_session = parsed(("full-0", "zero"), ("full-1", "one"), ("full-2", "two"))
        before_hash = archive._conn.execute(
            "SELECT accepted_content_hash FROM raw_revision_heads WHERE logical_source_key = ?",
            ("codex-session:session",),
        ).fetchone()
        assert before_hash is not None
        assert bytes(before_hash[0]) != bytes.fromhex(session_content_hash(folded_session))
        apply_prepared_revision_replay(
            archive,
            folded_plan,
            {folded: folded_session},
            acquired_at_ms=0,
        )

        head = archive._conn.execute(
            "SELECT accepted_raw_id, accepted_frontier FROM raw_revision_heads WHERE logical_source_key = ?",
            ("codex-session:session",),
        ).fetchone()
        assert head is not None
        assert tuple(head) == (folded, 20)


def test_native_winner_persists_independent_supersession_without_prefix_claim(tmp_path: Path) -> None:
    """A unique native winner gives every dominated raw a durable terminal receipt."""
    bootstrap_archive_root(tmp_path)
    sessions = {
        "old-a": _parsed_session(("m0", "opening"), ("m1", "older answer A")),
        "old-b": _parsed_session(("m0", "opening"), ("m1", "older answer B")),
        "winner": _parsed_session(("m0", "opening"), ("m1", "current answer"), ("m2", "later reply")),
    }
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_ids = {label: _write_quarantined_member(archive, label, session) for label, session in sessions.items()}
        revisions = [
            MembershipRevision(
                raw_ids[label],
                session_revision_projection(sessions[label]),
                provider_updated_at=timestamp,
                browser_snapshot_fidelity="native",
                provider_message_ids=provider_ids,
            )
            for label, timestamp, provider_ids in (
                ("old-a", "2026-10-01T00:00:00Z", frozenset({"m0", "m1"})),
                ("old-b", "2026-10-01T00:00:00Z", frozenset({"m0", "m1"})),
                ("winner", "2026-10-02T00:00:00Z", frozenset({"m0", "m1", "m2"})),
            )
        ]
        classification = classify_membership_revisions(revisions, existing_accepted_raw_id=raw_ids["old-a"])
        assert classification.accepted_raw_ids == (raw_ids["winner"],)
        assert set(classification.superseded_raw_ids) == {raw_ids["old-a"], raw_ids["old-b"]}
        sessions_by_raw_id = {raw_ids[label]: session for label, session in sessions.items()}
        projections = {raw_id: session_revision_projection(session) for raw_id, session in sessions_by_raw_id.items()}
        _publish_membership(
            archive,
            "codex-session:session",
            classification,
            sessions_by_raw_id,
            projections,
            acquired_at_ms=10,
        )

        source = archive._ensure_source_conn()
        membership = source.execute(
            "SELECT raw_id, decision, revision_authority FROM raw_session_memberships ORDER BY raw_id"
        ).fetchall()
        assert {str(row[0]): (str(row[1]), str(row[2])) for row in membership} == {
            raw_ids["old-a"]: ("superseded_by_winner", "byte_proven"),
            raw_ids["old-b"]: ("superseded_by_winner", "byte_proven"),
            raw_ids["winner"]: ("applied", "byte_proven"),
        }
        applications = archive._conn.execute(
            "SELECT raw_id, decision, accepted_raw_id FROM raw_revision_applications ORDER BY raw_id"
        ).fetchall()
        assert {str(row[0]): (str(row[1]), str(row[2])) for row in applications} == {
            raw_ids["old-a"]: ("superseded", raw_ids["winner"]),
            raw_ids["old-b"]: ("superseded", raw_ids["winner"]),
            raw_ids["winner"]: ("selected_baseline", raw_ids["winner"]),
        }
        head = archive._conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key='codex-session:session'"
        ).fetchone()
        assert head is not None and tuple(head) == (raw_ids["winner"],)
        message_count = archive._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id='codex-session:session'"
        ).fetchone()
        assert message_count is not None and int(message_count[0]) == 3

    # New read handles prove that publication survives restart and that the
    # terminal memberships certify actual applications pointing at the winner.
    from polylogue.storage.raw_authority import (
        build_raw_replay_plans,
        raw_replay_application_receipt,
        validate_raw_replay_application_receipt,
    )

    with ArchiveStore.open_existing(tmp_path, read_only=True) as restarted:
        assert (
            restarted._conn.execute(
                "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key='codex-session:session'"
            ).fetchone()[0]
            == raw_ids["winner"]
        )
    (plan,) = build_raw_replay_plans(tmp_path, (tuple(raw_ids.values()),))
    receipt = raw_replay_application_receipt(tmp_path, plan)
    valid, problems = validate_raw_replay_application_receipt(plan, receipt)
    assert valid, problems


def test_isolated_later_raw_does_not_override_known_ambiguous_cohort(tmp_path: Path) -> None:
    """polylogue-52l2: a raw discovered for a logical identity that already
    has quarantined/ambiguous siblings must not be accepted as an
    unambiguous singleton byte-proven baseline.

    This mirrors the LIVE incremental watcher's own call sequence
    (``sources/live/batch.py``): ``bind_raw_revision`` then
    ``prepare_raw_revision_byte_classification`` directly, with no census-phase
    re-derivation or connected-component re-expansion in between (those only
    happen in the offline ``backfill_historical_revision_evidence`` path,
    which is why this bug does not reproduce through that entry point).

    ``prepare_raw_revision_byte_classification`` only ever queries
    ``raw_sessions WHERE logical_source_key = ? AND revision_kind = 'full'``.
    Retiring an ambiguous sibling to membership governance
    (``replace_raw_membership_census(..., retire_full_revision_governance=True)``,
    exactly what the backfill caller does with
    ``convertible_full_revision_raw_ids`` once a cohort is decided
    ambiguous) nulls its ``raw_sessions.logical_source_key`` -- it becomes
    invisible to that query. A THIRD raw for the same identity, discovered
    afterward, is then evaluated completely alone:
    ``classify_historical_full_revision_streams`` unconditionally accepts a
    singleton stream as a "byte-proven baseline" (there is no sibling to
    compare a byte-prefix against), so the isolated raw would permanently
    become the accepted session content -- an outcome that depends on
    incremental discovery order, not on which content is actually correct.
    """
    bootstrap_archive_root(tmp_path)

    def parsed_solo(native_id: str, *texts: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CHATGPT,
            provider_session_id=native_id,
            messages=[
                ParsedMessage(provider_message_id=f"{native_id}-{index}", role=Role.USER, text=text)
                for index, text in enumerate(texts)
            ],
        )

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_a = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"aaa-left",
            source_path="a.json",
            canonical_source_path="a.json",
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            raw_a,
            RawRevisionEnvelope(
                "chatgpt-export:s1", RawRevisionKind.FULL, raw_a, 0, authority=RawRevisionAuthority.QUARANTINED
            ),
        )
        raw_b = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"bbb-right",
            source_path="b.json",
            canonical_source_path="b.json",
            acquired_at_ms=2,
        )
        archive.bind_raw_revision(
            raw_b,
            RawRevisionEnvelope(
                "chatgpt-export:s1", RawRevisionKind.FULL, raw_b, 0, authority=RawRevisionAuthority.QUARANTINED
            ),
        )

        first_plan = publish_fixture_byte_classification(archive, "chatgpt-export:s1")
        assert first_plan.accepted_raw_ids == ()

        # Both siblings genuinely disagree (no byte-prefix relation) --
        # exactly what the backfill caller does when a cohort is decided
        # ambiguous: move it to membership governance so parsed-content
        # prefix rules can still arbitrate it later.
        for raw_id, session in (
            (raw_a, parsed_solo("s1", "base", "left")),
            (raw_b, parsed_solo("s1", "base", "right")),
        ):
            publish_membership_census(
                archive,
                raw_id,
                [session],
                parser_fingerprint=raw_authority_parser_fingerprint(),
                censused_at_ms=0,
                detail="historical non-prefix full revision governance",
                retire_full_revision_governance=True,
                revision_authority=RawRevisionAuthority.QUARANTINED,
            )

        # A THIRD raw for the same logical identity, discovered afterward.
        raw_c = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"ccc-solo",
            source_path="c.json",
            canonical_source_path="c.json",
            acquired_at_ms=3,
        )
        archive.bind_raw_revision(
            raw_c,
            RawRevisionEnvelope(
                "chatgpt-export:s1", RawRevisionKind.FULL, raw_c, 0, authority=RawRevisionAuthority.QUARANTINED
            ),
        )
        second_plan = publish_fixture_byte_classification(archive, "chatgpt-export:s1")

    # The isolated raw must not be promoted alone: this identity has known,
    # unresolved ambiguous siblings that a real classifier must weigh it
    # against, not silently outrank by discovery order.
    assert second_plan.accepted_raw_ids == ()


def test_precedence_write_refuses_a_raw_recorded_ambiguous(tmp_path: Path) -> None:
    """A raw whose OWN logical identity is durably recorded
    ``raw_session_memberships.decision = 'ambiguous'`` must never reach
    ``sessions`` through the ordinary (non-revision-authoritative) parsed-
    write path.

    ``ArchiveStore._write_parsed_precedence_result``'s only revision-
    authority awareness before this fix was a check against
    ``raw_revision_heads`` -- populated ONLY when a cohort has an ACCEPTED
    winner (``apply_raw_membership_classification``/
    ``apply_raw_revision_replay``). A cohort ``classify_membership_
    revisions`` genuinely refused to arbitrate never gets an accepted head,
    so that check stays silent and the ordinary browser-capture-precedence/
    freshness fallback below it writes the session unconditionally on the
    next reparse -- arbitrary last-writer-wins over the exact invariant this
    subsystem exists to enforce. Live evidence: 28 aistudio-drive cohorts
    recorded ambiguous nonetheless materialized a session with 641
    attachments reported unfetched despite the bytes existing in the blob
    store, because ``write_parsed_for_retained_raw`` (called from the
    one-shot importer, ``revision_authoritative=False`` by default) never
    consulted ``raw_session_memberships`` at all.
    """
    bootstrap_archive_root(tmp_path)

    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="s1",
        messages=[ParsedMessage(provider_message_id="s1-0", role=Role.USER, text="left")],
    )

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"aaa-left",
            source_path="a.json",
            canonical_source_path="a.json",
            acquired_at_ms=1,
        )
        # Durable evidence that this raw's identity was already judged
        # ambiguous -- the shape ``replace_raw_membership_census`` /
        # ``apply_raw_membership_classification`` leave behind for a
        # genuinely divergent cohort (reproduced directly here so the test
        # isolates the WRITE-PATH guard from the classifier that produces
        # this state).
        with independent_source_connection(archive) as source_conn:
            source_conn.execute(
                """
                INSERT INTO raw_session_memberships (
                    raw_id, logical_source_key, provider_session_id,
                    source_revision, normalized_content_hash, message_count,
                    decision, decided_at_ms
                ) VALUES (?, 'chatgpt-export:s1', 's1', ?, ?, 1, 'ambiguous', 1)
                """,
                (raw_id, raw_id, bytes.fromhex(raw_id)),
            )

        result = write_prepared_retained_session(archive, session, raw_id=raw_id)
        returned_raw_id, session_id = result.raw_id, result.session_id

    assert returned_raw_id == raw_id
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone() == (0,)


def test_precedence_write_allows_a_non_ambiguous_sibling_membership_on_the_same_raw(tmp_path: Path) -> None:
    """The ambiguity refusal is per-membership, not per-raw.

    One retained raw routinely lowers to many independently-arbitrated sessions
    -- a Claude Code transcript plus its subagent sidechains, a bundle member
    set. Scoping the refusal to ``raw_id`` alone suppresses every session that
    raw carries the moment a single sibling membership is ambiguous, turning a
    fidelity downgrade into outright absence.

    Measured on the live archive when this was caught: 295 raws carry a mix of
    decisions, together holding 489 sessions whose own membership is not
    ambiguous, and one raw carries 106 memberships. Their content would have
    silently vanished at the next full rebuild.
    """
    bootstrap_archive_root(tmp_path)

    ambiguous_session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="s-ambiguous",
        messages=[ParsedMessage(provider_message_id="a-0", role=Role.USER, text="left")],
    )
    settled_session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="s-settled",
        messages=[ParsedMessage(provider_message_id="b-0", role=Role.USER, text="right")],
    )

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"two-sessions",
            source_path="bundle.json",
            canonical_source_path="bundle.json",
            acquired_at_ms=1,
        )
        with independent_source_connection(archive) as source_conn:
            # One raw, two memberships, arbitrated differently -- the live shape.
            source_conn.execute(
                """
                INSERT INTO raw_session_memberships (
                    raw_id, logical_source_key, provider_session_id,
                    source_revision, normalized_content_hash, message_count,
                    decision, decided_at_ms
                ) VALUES (?, 'chatgpt-export:s-ambiguous', 's-ambiguous', ?, ?, 1, 'ambiguous', 1)
                """,
                (raw_id, raw_id, bytes.fromhex(raw_id)),
            )
            source_conn.execute(
                """
                INSERT INTO raw_session_memberships (
                    raw_id, logical_source_key, provider_session_id,
                    source_revision, normalized_content_hash, message_count,
                    decision, decided_at_ms
                ) VALUES (?, 'chatgpt-export:s-settled', 's-settled', ?, ?, 1, 'applied', 1)
                """,
                (raw_id, raw_id + "-b", bytes.fromhex(raw_id)),
            )

        ambiguous_session_id = write_prepared_retained_session(archive, ambiguous_session, raw_id=raw_id).session_id
        settled_session_id = write_prepared_retained_session(archive, settled_session, raw_id=raw_id).session_id

    with sqlite3.connect(tmp_path / "index.db") as conn:
        # The ambiguous membership is still refused ...
        assert conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id = ?", (ambiguous_session_id,)
        ).fetchone() == (0,)
        # ... and its settled sibling on the same raw is not collateral damage.
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (settled_session_id,)).fetchone() == (
            1,
        )


def test_retirement_under_an_unrecognized_marker_is_refused_at_the_write_boundary(
    tmp_path: Path,
) -> None:
    """An unknown governance marker must refuse, not read back as success.

    ``replace_raw_membership_census(..., retire_full_revision_governance=
    True)`` nulls the raw's ``logical_source_key`` and quarantines it, so the
    surviving census authority is the ONLY thing that still tells the 52l2
    guard this identity has known ambiguous evidence. A marker that does not
    translate to the typed quarantined authority is not a harmless label: the
    guard simply misses it and a later-arriving sibling is accepted as an unconditional singleton
    byte-proven baseline (polylogue-sze30 AC2).

    The second half rewrites an accepted retirement's display detail to that
    same unrecognized marker and shows the typed authority still refuses the
    isolated third raw. Delete the write-boundary check and the first half
    stops raising.
    """
    bootstrap_archive_root(tmp_path)

    def parsed_solo(native_id: str, *texts: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CHATGPT,
            provider_session_id=native_id,
            messages=[
                ParsedMessage(provider_message_id=f"{native_id}-{index}", role=Role.USER, text=text)
                for index, text in enumerate(texts)
            ],
        )

    unrecognized = "cohort resolved by upkeep pass"

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raws = []
        for label, payload in (("a", b"aaa-left"), ("b", b"bbb-right")):
            raw_id = archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=payload,
                source_path=f"{label}.json",
                canonical_source_path=f"{label}.json",
                acquired_at_ms=1,
            )
            archive.bind_raw_revision(
                raw_id,
                RawRevisionEnvelope(
                    "chatgpt-export:s1", RawRevisionKind.FULL, raw_id, 0, authority=RawRevisionAuthority.QUARANTINED
                ),
            )
            raws.append(raw_id)
        raw_a, raw_b = raws

        with pytest.raises(ValueError, match="recognized governance marker"):
            publish_membership_census(
                archive,
                raw_a,
                [parsed_solo("s1", "base", "left")],
                parser_fingerprint=raw_authority_parser_fingerprint(),
                censused_at_ms=0,
                detail=unrecognized,
                retire_full_revision_governance=True,
                revision_authority=None,
            )

        # The refusal happens before any mutation: the raw keeps its identity
        # and is not silently quarantined by a half-applied retirement.
        row = (
            archive._ensure_source_conn()
            .execute(
                "SELECT logical_source_key, revision_authority FROM raw_sessions WHERE raw_id = ?",
                (raw_a,),
            )
            .fetchone()
        )
        assert str(row[0]) == "chatgpt-export:s1"

        # A census that leaves no membership row has no logical identity to be
        # ambiguous about, so its detail stays free explanatory prose.
        publish_membership_census(
            archive,
            raw_a,
            [],
            parser_fingerprint=raw_authority_parser_fingerprint(),
            censused_at_ms=0,
            detail=unrecognized,
            retire_full_revision_governance=True,
            revision_authority=None,
        )

    # Now the hazard the refusal prevents, reached by rewriting an accepted
    # retirement's marker to the unrecognized one.
    bootstrap_archive_root(tmp_path / "hazard")
    with ArchiveStore.open_existing(tmp_path / "hazard", read_only=False) as archive:
        retired = []
        for label, payload, tail in (("a", b"aaa-left", "left"), ("b", b"bbb-right", "right")):
            raw_id = archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=payload,
                source_path=f"{label}.json",
                canonical_source_path=f"{label}.json",
                acquired_at_ms=1,
            )
            archive.bind_raw_revision(
                raw_id,
                RawRevisionEnvelope(
                    "chatgpt-export:s1", RawRevisionKind.FULL, raw_id, 0, authority=RawRevisionAuthority.QUARANTINED
                ),
            )
            publish_membership_census(
                archive,
                raw_id,
                [parsed_solo("s1", "base", tail)],
                parser_fingerprint=raw_authority_parser_fingerprint(),
                censused_at_ms=0,
                detail=HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL,
                retire_full_revision_governance=True,
                revision_authority=RawRevisionAuthority.QUARANTINED,
            )
            retired.append(raw_id)

        raw_c = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"ccc-solo",
            source_path="c.json",
            canonical_source_path="c.json",
            acquired_at_ms=3,
        )
        archive.bind_raw_revision(
            raw_c,
            RawRevisionEnvelope(
                "chatgpt-export:s1", RawRevisionKind.FULL, raw_c, 0, authority=RawRevisionAuthority.QUARANTINED
            ),
        )
        # With the recognized marker the isolated third raw stays refused.
        assert publish_fixture_byte_classification(archive, "chatgpt-export:s1").accepted_raw_ids == ()

        with independent_source_connection(archive) as conn:
            conn.executemany(
                "UPDATE raw_membership_census SET detail = ? WHERE raw_id = ?",
                [(unrecognized, raw_id) for raw_id in retired],
            )
        # Changing display wording cannot change the typed governance result.
        promoted = publish_fixture_byte_classification(archive, "chatgpt-export:s1")

    assert promoted.accepted_raw_ids == ()


def test_retired_raw_stays_fail_closed_when_census_authority_is_unknown(tmp_path: Path) -> None:
    """A missing census code cannot turn a quarantined retirement into success.

    A census whose detail has no typed translation stores a NULL
    ``revision_authority``.  The raw row itself is already durably quarantined,
    so the retirement reader must use that typed source authority as a second
    proof rather than promoting a later singleton when the census code is
    unknown.
    """
    bootstrap_archive_root(tmp_path)

    def parsed_solo(tail: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CHATGPT,
            provider_session_id="s1",
            messages=[
                ParsedMessage(provider_message_id=f"s1-{index}", role=Role.USER, text=text)
                for index, text in enumerate(("base", tail))
            ],
        )

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        retired = []
        for label, payload, tail in (("a", b"aaa-left", "left"), ("b", b"bbb-right", "right")):
            raw_id = archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=payload,
                source_path=f"{label}.json",
                canonical_source_path=f"{label}.json",
                acquired_at_ms=1,
            )
            archive.bind_raw_revision(
                raw_id,
                RawRevisionEnvelope(
                    "chatgpt-export:s1", RawRevisionKind.FULL, raw_id, 0, authority=RawRevisionAuthority.QUARANTINED
                ),
            )
            publish_membership_census(
                archive,
                raw_id,
                [parsed_solo(tail)],
                parser_fingerprint=raw_authority_parser_fingerprint(),
                censused_at_ms=0,
                detail=HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL,
                retire_full_revision_governance=True,
                revision_authority=RawRevisionAuthority.QUARANTINED,
            )
            retired.append(raw_id)

        with independent_source_connection(archive) as conn:
            conn.executemany(
                "UPDATE raw_membership_census SET detail = ?, revision_authority = NULL WHERE raw_id = ?",
                [("historical wording no longer classifies this row", raw_id) for raw_id in retired],
            )

        raw_c = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"ccc-solo",
            source_path="c.json",
            canonical_source_path="c.json",
            acquired_at_ms=3,
        )
        archive.bind_raw_revision(
            raw_c,
            RawRevisionEnvelope(
                "chatgpt-export:s1", RawRevisionKind.FULL, raw_c, 0, authority=RawRevisionAuthority.QUARANTINED
            ),
        )
        plan = publish_fixture_byte_classification(archive, "chatgpt-export:s1")

    assert plan.accepted_raw_ids == ()


def test_typed_retirement_authority_allows_detail_wording_to_change(
    tmp_path: Path,
) -> None:
    """Machine-readable authority keeps interpretation independent of prose."""
    bootstrap_archive_root(tmp_path)

    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="s1",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="hello")],
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"payload",
            source_path="capture.json",
            canonical_source_path="capture.json",
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            raw_id,
            RawRevisionEnvelope(
                "chatgpt-export:s1", RawRevisionKind.FULL, raw_id, 0, authority=RawRevisionAuthority.QUARANTINED
            ),
        )

    publish_prepared_source(
        tmp_path,
        "test.revision.membership-census-detail",
        lambda seal: archive_revision_governance.replace_raw_membership_census(
            seal,
            raw_id,
            [session],
            parser_fingerprint=raw_authority_parser_fingerprint(),
            censused_at_ms=0,
            detail="operator-facing wording may change",
            revision_authority=RawRevisionAuthority.QUARANTINED,
            retire_full_revision_governance=True,
        ),
    )
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        row = source.execute(
            "SELECT detail, revision_authority FROM raw_membership_census WHERE raw_id = ?", (raw_id,)
        ).fetchone()

    assert tuple(row) == ("operator-facing wording may change", RawRevisionAuthority.QUARANTINED.value)


def test_same_source_path_full_siblings_under_different_keys_are_not_independently_accepted(
    tmp_path: Path,
) -> None:
    """polylogue-eqnv: a raw whose byte-revision identity was assigned by a
    now-superseded parser (e.g. the pre-#3179/z1c6 dispatch bug that
    appended a spurious ``-0`` to one of two otherwise-identical Drive
    re-acquisitions of the same document) can carry a
    ``logical_source_key`` that DIFFERS from a same-``source_path``
    sibling's. Neither raw's own key ever surfaces the other in
    ``raw_membership_retired_full_revision_siblings`` (an exact-key-match
    query), so ``prepare_raw_revision_byte_classification`` evaluates each key as a
    trivial one-member chain and unconditionally accepts BOTH as
    independent byte-proven singleton baselines -- silently splitting one
    physical document into two sessions that then race on the shared
    ``(origin, native_id)`` upsert (arbitrary last-writer-wins), instead of
    ever being compared against each other.
    """
    bootstrap_archive_root(tmp_path)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_enriched = archive.write_raw_payload(
            provider=Provider.GEMINI,
            payload=b"enriched-bytes",
            source_path="doc.json",
            canonical_source_path="doc.json",
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            raw_enriched,
            RawRevisionEnvelope(
                "gemini:doc",
                RawRevisionKind.FULL,
                raw_enriched,
                0,
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )
        raw_bare = archive.write_raw_payload(
            provider=Provider.GEMINI,
            payload=b"bare-bytes",
            source_path="doc.json",
            canonical_source_path="doc.json",
            acquired_at_ms=2,
        )
        archive.bind_raw_revision(
            raw_bare,
            RawRevisionEnvelope(
                # The stale-parser identity split: same source_path, a
                # DIFFERENT logical_source_key.
                "gemini:doc-0",
                RawRevisionKind.FULL,
                raw_bare,
                0,
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )

        enriched_plan = publish_fixture_byte_classification(archive, "gemini:doc")
        bare_plan = publish_fixture_byte_classification(archive, "gemini:doc-0")

    # Neither key's lone member may be promoted alone: a same-source_path
    # sibling under a different key means this identity is genuinely
    # contested, not a clean singleton chain.
    assert enriched_plan.accepted_raw_ids == ()
    assert bare_plan.accepted_raw_ids == ()


def test_real_single_append_chain_folds_segmentation_distinct_full_snapshot(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        baseline_payload, tail = _codex_fold_payloads()
        baseline_session = _with_fold_attachment(_parse_codex_jsonl(baseline_payload))
        append_session = _parse_codex_jsonl(tail)
        folded_payload = baseline_payload + tail
        folded_session = _parse_codex_jsonl(folded_payload)
        assert session_content_hash(
            merge_parsed_session_chunks([baseline_session, append_session])[0]
        ) != session_content_hash(folded_session)
        baseline = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline_payload,
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            baseline,
            RawRevisionEnvelope(
                "codex-session:session", RawRevisionKind.FULL, "base", 0, authority=RawRevisionAuthority.BYTE_PROVEN
            ),
        )
        append = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=tail,
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=2,
        )
        append_revision = append_source_revision("base", hashlib.sha256(tail).hexdigest())
        archive.bind_raw_revision(
            append,
            RawRevisionEnvelope(
                "codex-session:session",
                RawRevisionKind.APPEND,
                append_revision,
                1,
                predecessor_source_revision="base",
                predecessor_raw_id=baseline,
                baseline_raw_id=baseline,
                append_start_offset=len(baseline_payload),
                append_end_offset=len(folded_payload),
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        apply_prepared_revision_replay(
            archive,
            archive.raw_revision_replay_plan("codex-session:session"),
            {baseline: baseline_session, append: append_session},
            acquired_at_ms=0,
        )
        folded = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=folded_payload,
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            acquired_at_ms=3,
        )
        archive.bind_raw_revision(
            folded,
            RawRevisionEnvelope(
                "codex-session:session", RawRevisionKind.FULL, "folded", 2, authority=RawRevisionAuthority.BYTE_PROVEN
            ),
        )
        before_hash = archive._conn.execute(
            "SELECT accepted_content_hash FROM raw_revision_heads WHERE logical_source_key = ?",
            ("codex-session:session",),
        ).fetchone()
        assert before_hash is not None
        assert bytes(before_hash[0]) != bytes.fromhex(session_content_hash(folded_session))
        apply_prepared_revision_replay(
            archive,
            archive.raw_revision_replay_plan("codex-session:session"),
            {folded: folded_session},
            acquired_at_ms=0,
        )
        assert archive.raw_revision_head_raw_id("codex-session:session") == folded


def test_claude_full_append_replay_persists_reduced_coverage_and_receipts(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)

    def parsed(message_id: str, text: str, *, seen: int, persisted: int) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CLAUDE_CODE,
            provider_session_id="session",
            messages=[ParsedMessage(provider_message_id=message_id, role=Role.USER, text=text)],
            session_events=[
                ParsedSessionEvent(
                    event_type="claude_parse_coverage",
                    payload={
                        "sidecar_seen": {"summary": seen},
                        "sidecar_persisted": {"summary": persisted},
                        "empty_dropped_by_record_type": {},
                    },
                )
            ],
        )

    baseline_session = parsed("m1", "baseline", seen=3, persisted=2)
    append_session = parsed("m2", "append", seen=5, persisted=4)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        baseline = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=b"base",
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            baseline,
            RawRevisionEnvelope(
                "claude-code:session", RawRevisionKind.FULL, "base", 0, authority=RawRevisionAuthority.BYTE_PROVEN
            ),
        )
        append = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=b"tail!",
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=2,
        )
        archive.bind_raw_revision(
            append,
            RawRevisionEnvelope(
                "claude-code:session",
                RawRevisionKind.APPEND,
                append_source_revision("base", hashlib.sha256(b"tail!").hexdigest()),
                1,
                predecessor_source_revision="base",
                predecessor_raw_id=baseline,
                baseline_raw_id=baseline,
                append_start_offset=4,
                append_end_offset=9,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )

        plan = archive.raw_revision_replay_plan("claude-code:session")
        session_id, applied_raw_ids = apply_prepared_revision_replay(
            archive,
            plan,
            {baseline: baseline_session, append: append_session},
            acquired_at_ms=0,
        )

        coverage = archive._conn.execute(
            "SELECT payload_json FROM session_events WHERE event_type = 'claude_parse_coverage'"
        ).fetchall()
        assert len(coverage) == 1
        assert json.loads(coverage[0][0]) == {
            "sidecar_seen": {"summary": 8},
            "sidecar_persisted": {"summary": 6},
            "empty_dropped_by_record_type": {},
        }
        stored_hash = archive._conn.execute(
            "SELECT content_hash FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        assert stored_hash is not None
        assert bytes(stored_hash[0]) == bytes.fromhex(
            session_content_hash(merge_parsed_session_chunks([baseline_session, append_session])[0])
        )
        assert applied_raw_ids == (baseline, append)
        # By identity, not by cardinality. A count of two is also what a replay
        # that wrote two rows for the append raw under different decisions
        # while dropping the baseline receipt would report, which is precisely
        # the missing-receipt mutation this assertion exists to catch.
        # Anti-vacuity: record either raw's receipt against the other raw_id,
        # or omit the baseline receipt, and this goes red while the count did
        # not move.
        assert {
            (str(row[0]), str(row[1]))
            for row in archive._conn.execute(
                "SELECT raw_id, decision FROM raw_revision_applications WHERE logical_source_key = ?",
                ("claude-code:session",),
            ).fetchall()
        } == {
            (baseline, ApplicationDecision.SELECTED_BASELINE.value),
            (append, ApplicationDecision.APPLIED_APPEND.value),
        }


def test_fold_accepts_a_legacy_codex_append_payload_after_header_normalization(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        baseline_payload, tail = _codex_fold_payloads()
        baseline_session = _with_fold_attachment(_parse_codex_jsonl(baseline_payload))
        append_session = _parse_codex_jsonl(tail)
        folded_payload = baseline_payload + tail
        folded_session = _parse_codex_jsonl(folded_payload)
        baseline = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline_payload,
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            baseline,
            RawRevisionEnvelope(
                "codex-session:session", RawRevisionKind.FULL, "base", 0, authority=RawRevisionAuthority.BYTE_PROVEN
            ),
        )
        legacy_append_payload = b'{"type":"session_meta","payload":{"id":"fold-codex"}}\n' + tail
        append = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=legacy_append_payload,
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=2,
        )
        archive.bind_raw_revision(
            append,
            RawRevisionEnvelope(
                "codex-session:session",
                RawRevisionKind.APPEND,
                append_source_revision("base", hashlib.sha256(tail).hexdigest()),
                1,
                predecessor_source_revision="base",
                predecessor_raw_id=baseline,
                baseline_raw_id=baseline,
                append_start_offset=len(baseline_payload),
                append_end_offset=len(folded_payload),
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        apply_prepared_revision_replay(
            archive,
            archive.raw_revision_replay_plan("codex-session:session"),
            {baseline: baseline_session, append: append_session},
            acquired_at_ms=0,
        )
        folded = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=folded_payload,
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            acquired_at_ms=3,
        )
        archive.bind_raw_revision(
            folded,
            RawRevisionEnvelope(
                "codex-session:session", RawRevisionKind.FULL, "folded", 2, authority=RawRevisionAuthority.BYTE_PROVEN
            ),
        )
        apply_prepared_revision_replay(
            archive,
            archive.raw_revision_replay_plan("codex-session:session"),
            {folded: folded_session},
            acquired_at_ms=0,
        )

        assert archive.raw_revision_head_raw_id("codex-session:session") == folded


@pytest.mark.parametrize("mutation", ["tail", "gap", "overlap", "predecessor", "baseline", "missing", "divergent"])
def test_real_append_fold_proof_mutations_roll_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    bootstrap_archive_root(tmp_path)

    def parsed(*messages: tuple[str, str]) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="session",
            messages=[
                ParsedMessage(provider_message_id=message_id, role=Role.USER, text=text)
                for message_id, text in messages
            ],
        )

    def state(archive: ArchiveStore) -> dict[str, object]:
        fts_matches = archive._conn.execute(
            """
            SELECT b.block_id, b.message_id, b.text
            FROM messages_fts
            JOIN blocks AS b ON b.rowid = messages_fts.rowid
            WHERE messages_fts MATCH 'needle'
            ORDER BY b.block_id
            """
        ).fetchall()
        candidate_matches = archive._conn.execute(
            """
            SELECT b.block_id, b.message_id, b.text
            FROM messages_fts
            JOIN blocks AS b ON b.rowid = messages_fts.rowid
            WHERE messages_fts MATCH 'candidate'
            ORDER BY b.block_id
            """
        ).fetchall()
        return {
            "sessions": archive._conn.execute("SELECT content_hash, message_count FROM sessions").fetchall(),
            "messages": archive._conn.execute(
                "SELECT message_id, content_hash FROM messages ORDER BY message_id"
            ).fetchall(),
            "blocks": archive._conn.execute(
                "SELECT block_id, message_id, block_type, text, search_text, content_hash FROM blocks ORDER BY block_id"
            ).fetchall(),
            "session_events": archive._conn.execute(
                "SELECT event_id, source_message_id, event_type, json_extract(payload_json, '$.summary'), "
                "payload_json FROM session_events ORDER BY event_id"
            ).fetchall(),
            "attachments": archive._conn.execute(
                "SELECT attachment_id, display_name, media_type, byte_count, blob_hash, acquisition_status FROM attachments ORDER BY attachment_id"
            ).fetchall(),
            "fts_docsize": archive._conn.execute("SELECT id, sz FROM messages_fts_docsize ORDER BY id").fetchall(),
            "fts_needle": fts_matches,
            "fts_candidate": candidate_matches,
            "head": archive._conn.execute(
                "SELECT accepted_raw_id, accepted_content_hash, accepted_frontier FROM raw_revision_heads"
            ).fetchall(),
            "receipts": archive._conn.execute(
                "SELECT decision_id FROM raw_revision_applications ORDER BY decision_id"
            ).fetchall(),
        }

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        baseline_payload, tail = _codex_fold_payloads()
        baseline_session = _with_fold_attachment(_parse_codex_jsonl(baseline_payload))
        append_session = _parse_codex_jsonl(tail)
        candidate_payload = (baseline_payload + tail).replace(b"needle beta", b"candidate X")
        assert len(candidate_payload) == len(baseline_payload + tail)
        folded_session = _parse_codex_jsonl(candidate_payload)
        baseline = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline_payload,
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            baseline,
            RawRevisionEnvelope(
                "codex-session:session", RawRevisionKind.FULL, "base", 0, authority=RawRevisionAuthority.BYTE_PROVEN
            ),
        )
        folded_payload = candidate_payload
        if mutation in {"baseline", "divergent"}:
            folded_payload = (b"X" if mutation == "baseline" else baseline_payload[:5] + b"X") + folded_payload[
                1 if mutation == "baseline" else 6 :
            ]
        # The folded snapshot is acquired before the append it claims to fold, so
        # it is not a fresher selected FULL (which would replace the head on its
        # own acquisition-order authority); only a byte-chain fold proof can
        # accept it, and every mutation below must break that proof.
        folded = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=folded_payload,
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            acquired_at_ms=3,
        )
        append = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=tail,
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=2,
        )
        append_revision = append_source_revision("base", hashlib.sha256(tail).hexdigest())
        archive.bind_raw_revision(
            append,
            RawRevisionEnvelope(
                "codex-session:session",
                RawRevisionKind.APPEND,
                append_revision,
                1,
                predecessor_source_revision="base",
                predecessor_raw_id=baseline,
                baseline_raw_id=baseline,
                append_start_offset=len(baseline_payload),
                append_end_offset=len(baseline_payload + tail),
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        chain = archive.raw_revision_replay_plan("codex-session:session")
        apply_prepared_revision_replay(
            archive, chain, {baseline: baseline_session, append: append_session}, acquired_at_ms=0
        )
        archive.bind_raw_revision(
            folded,
            RawRevisionEnvelope(
                "codex-session:session", RawRevisionKind.FULL, "folded", 2, authority=RawRevisionAuthority.BYTE_PROVEN
            ),
        )
        source = open_independent_source(archive)
        if mutation == "gap":
            source.execute(
                "UPDATE raw_sessions SET append_start_offset = ? WHERE raw_id = ?", (len(baseline_payload) + 1, append)
            )
        elif mutation == "overlap":
            source.execute(
                "UPDATE raw_sessions SET append_start_offset = ? WHERE raw_id = ?", (len(baseline_payload) - 1, append)
            )
        elif mutation == "predecessor":
            source.execute("UPDATE raw_sessions SET predecessor_source_revision = 'wrong' WHERE raw_id = ?", (append,))
        elif mutation == "missing":
            source.execute("UPDATE raw_sessions SET predecessor_raw_id = 'missing' WHERE raw_id = ?", (append,))
        elif mutation == "tail":
            # The retained append bytes no longer extend the baseline.
            tampered = b"Z" * len(tail)
            BlobStore(tmp_path / "blob").write_from_bytes(tampered)
            source.execute(
                "UPDATE raw_sessions SET blob_hash = ? WHERE raw_id = ?", (hashlib.sha256(tampered).digest(), append)
            )
        source.commit()
        source.close()
        before = state(archive)
        assert before["blocks"]
        assert before["session_events"]
        assert before["attachments"]
        assert before["fts_needle"]
        assert not before["fts_candidate"]
        plan = archive.raw_revision_replay_plan("codex-session:session")
        with pytest.raises(RuntimeError, match="conflicting accepted head"):
            apply_prepared_revision_replay(archive, plan, {folded: folded_session}, acquired_at_ms=0)
        assert state(archive) == before


def test_write_raw_and_parsed_persists_file_mtime_across_reopen(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    file_mtime_ms = 1_767_225_600_000
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="wrapper-mtime",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="no timeline")],
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        written = write_fixture_raw_session(
            archive,
            session,
            payload=b"wrapper raw",
            source_path="wrapper-mtime.jsonl",
            acquired_at_ms=1,
            file_mtime_ms=file_mtime_ms,
        )
        raw_id, _session_id = written.raw_id, written.session_id
        row = (
            archive._ensure_source_conn()
            .execute("SELECT file_mtime_ms FROM raw_sessions WHERE raw_id = ?", (raw_id,))
            .fetchone()
        )
        assert row[0] == file_mtime_ms
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        row = (
            archive._ensure_source_conn()
            .execute("SELECT file_mtime_ms FROM raw_sessions WHERE raw_id = ?", (raw_id,))
            .fetchone()
        )
        assert row[0] == file_mtime_ms


def test_retained_replay_uses_persisted_file_mtime_for_timestamp_fallback(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    file_mtime_ms = 1_767_225_600_000
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="mtime-replay",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="no timeline")],
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"retained raw",
            source_path="mtime-replay.jsonl",
            canonical_source_path="mtime-replay.jsonl",
            acquired_at_ms=1,
            file_mtime_ms=file_mtime_ms,
        )
        archive.bind_raw_revision(
            raw_id,
            RawRevisionEnvelope(
                "codex-session:session",
                RawRevisionKind.FULL,
                "mtime-revision",
                1,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        apply_prepared_aggregate_replay(
            archive,
            plan_revision_replay([_candidate(raw_id, RawRevisionKind.FULL, 1, size=len(b"retained raw"))]),
            {raw_id: session},
            acquired_at_ms=0,
        )
        timestamp_row = archive._conn.execute(
            "SELECT created_at_ms, updated_at_ms FROM sessions WHERE session_id = ?",
            ("codex-session:mtime-replay",),
        ).fetchone()
        assert tuple(timestamp_row) == (file_mtime_ms, file_mtime_ms)
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        timestamp_row = archive._conn.execute(
            "SELECT created_at_ms, updated_at_ms FROM sessions WHERE session_id = ?",
            ("codex-session:mtime-replay",),
        ).fetchone()
        assert tuple(timestamp_row) == (file_mtime_ms, file_mtime_ms)


def test_full_replay_preserves_semantic_head_and_rolls_back_regressions(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)

    def parsed(*messages: tuple[str, str], event_timestamp: str | None = None) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="session",
            messages=[
                ParsedMessage(provider_message_id=message_id, role=Role.USER, text=text)
                for message_id, text in messages
            ],
            session_events=(
                [ParsedSessionEvent(event_type="replay-evidence", timestamp=event_timestamp)]
                if event_timestamp is not None
                else []
            ),
        )

    def write_full(archive: ArchiveStore, label: str, generation: int) -> str:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=label.encode(),
            source_path="session.json",
            canonical_source_path="session.json",
            acquired_at_ms=generation,
        )
        archive.bind_raw_revision(
            raw_id,
            RawRevisionEnvelope(
                "codex-session:session",
                RawRevisionKind.FULL,
                f"revision-{label}",
                generation,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        return raw_id

    def selected_full_plan(raw_id: str, generation: int, size: int) -> RevisionReplayPlan:
        return plan_revision_replay([_candidate(raw_id, RawRevisionKind.FULL, generation, size=size)])

    def durable_index_state(archive: ArchiveStore) -> tuple[object, ...]:
        return (
            archive._conn.execute(
                "SELECT message_count, content_hash FROM sessions WHERE session_id = 'codex-session:session'"
            ).fetchone(),
            archive._conn.execute("SELECT message_id, content_hash FROM messages ORDER BY position").fetchall(),
            archive._conn.execute("SELECT block_id, search_text FROM blocks ORDER BY message_id, position").fetchall(),
            archive._conn.execute("SELECT id, sz FROM messages_fts_docsize ORDER BY id").fetchall(),
            archive._conn.execute(
                """SELECT accepted_raw_id, accepted_source_revision, accepted_content_hash,
                          accepted_frontier_kind, accepted_frontier
                   FROM raw_revision_heads WHERE logical_source_key = 'codex-session:session'"""
            ).fetchone(),
            archive._conn.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone(),
        )

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        base_session = parsed(("m0", "zero"), event_timestamp="2026-07-01T00:00:00Z")
        base = write_full(archive, "base", 1)
        _publish_membership(
            archive,
            "codex-session:session",
            MembershipClassification((base,), (), ()),
            {base: base_session},
            {base: session_revision_projection(base_session)},
            acquired_at_ms=0,
        )
        timestamp_row = archive._conn.execute(
            "SELECT created_at_ms, updated_at_ms FROM sessions WHERE session_id = ?",
            ("codex-session:session",),
        ).fetchone()
        assert tuple(timestamp_row) == (1_782_864_000_000, 1_782_864_000_000)

        later_session = parsed(("m0", "zero"), ("m1", "one"), ("m2", "two"))
        later = write_full(archive, "later", 2)
        later_plan = selected_full_plan(later, 2, len("later"))
        apply_prepared_aggregate_replay(archive, later_plan, {later: later_session}, acquired_at_ms=0)

        semantic_head = archive._conn.execute(
            """SELECT accepted_raw_id, accepted_frontier_kind, accepted_frontier
               FROM raw_revision_heads WHERE logical_source_key = 'codex-session:session'"""
        ).fetchone()
        assert semantic_head is not None
        assert tuple(semantic_head) == (later, "semantic", 3)

        for label, rejected_session, error in (
            ("older", parsed(("m0", "zero"), ("m1", "one")), "older accepted frontier"),
            (
                "conflict",
                parsed(("m0", "zero"), ("m1", "one"), ("m2", "different")),
                "conflicting accepted head",
            ),
        ):
            before = durable_index_state(archive)
            generation = 3 if label == "older" else 4
            rejected_raw = write_full(archive, label, generation)
            rejected_plan = selected_full_plan(rejected_raw, generation, len(label))
            with pytest.raises(RuntimeError, match=error):
                apply_prepared_aggregate_replay(
                    archive,
                    rejected_plan,
                    {rejected_raw: rejected_session},
                    acquired_at_ms=0,
                )
            assert durable_index_state(archive) == before
            assert archive._ensure_source_conn().execute(
                "SELECT parsed_at_ms FROM raw_sessions WHERE raw_id = ?", (rejected_raw,)
            ).fetchone() == (None,)


def _parsed_session(*messages: tuple[str, str]) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="session",
        messages=[
            ParsedMessage(provider_message_id=message_id, role=Role.USER, text=text) for message_id, text in messages
        ],
    )


def _write_quarantined_member(archive: ArchiveStore, label: str, session: ParsedSession) -> str:
    """A capture-style raw: no revision envelope, default quarantined authority."""
    raw_id = archive.write_raw_payload(
        provider=Provider.CODEX,
        payload=label.encode(),
        source_path=f"{label}.json",
        canonical_source_path=f"{label}.json",
        acquired_at_ms=1,
    )
    publish_membership_census(
        archive, raw_id, [session], parser_fingerprint="test-parser", censused_at_ms=1, revision_authority=None
    )
    return raw_id


def _write_chain_full(archive: ArchiveStore, label: str, generation: int) -> str:
    raw_id = archive.write_raw_payload(
        provider=Provider.CODEX,
        payload=label.encode(),
        source_path="session.json",
        canonical_source_path="session.json",
        acquired_at_ms=generation,
    )
    archive.bind_raw_revision(
        raw_id,
        RawRevisionEnvelope(
            "codex-session:session",
            RawRevisionKind.FULL,
            f"revision-{label}",
            generation,
            authority=RawRevisionAuthority.BYTE_PROVEN,
        ),
    )
    return raw_id


def _apply_membership_head(archive: ArchiveStore, raw_id: str, session: ParsedSession) -> None:
    _publish_membership(
        archive,
        "codex-session:session",
        MembershipClassification((raw_id,), (), ()),
        {raw_id: session},
        {raw_id: session_revision_projection(session)},
        acquired_at_ms=0,
    )


def test_batched_membership_success_supersedes_deferred_cas_evidence(tmp_path: Path) -> None:
    """The positive commit-batch route must expire CAS retry authority too."""
    bootstrap_archive_root(tmp_path)
    session = _parsed_session(("m0", "batched success"))
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = _write_quarantined_member(archive, "batched-cas", session)
        archive.record_raw_failure_evidence(
            raw_id,
            provider=Provider.CODEX,
            source_path="batched-cas.json",
            source_index=0,
            acquired_at_ms=2,
            kind=RawFailureEvidenceKind.DEFERRED_CAS_FRONTIER,
        )
        _publish_membership(
            archive,
            "codex-session:session",
            MembershipClassification((raw_id,), (), ()),
            {raw_id: session},
            {raw_id: session_revision_projection(session)},
            acquired_at_ms=3,
        )

        artifact = (
            archive._ensure_source_conn()
            .execute(
                "SELECT artifact_kind FROM raw_artifacts WHERE raw_id = ? AND source_path = ?",
                (raw_id, "batched-cas.json"),
            )
            .fetchone()
        )

    assert artifact == (RawFailureEvidenceKind.TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER.value,)


def test_retained_index_cas_failure_persists_evidence_with_first_failure_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retained-raw CAS failure cannot commit an untyped state first."""
    bootstrap_archive_root(tmp_path)
    session = _parsed_session(("m0", "retained CAS failure"))
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = _write_quarantined_member(archive, "retained-cas-failure", session)

        def raise_conflict(*_args: object, **_kwargs: object) -> None:
            raise archive_revision_governance.MembershipReplayConflictError("retained membership conflict")

        monkeypatch.setattr(archive_revision_governance, "_write_parsed_precedence_result", raise_conflict)
        with pytest.raises(archive_revision_governance.MembershipReplayConflictError):
            write_prepared_retained_session(archive, session, raw_id=raw_id, revision_authoritative=True)

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        assert source_conn.execute("SELECT parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (
            "MembershipReplayConflictError: retained membership conflict",
        )
        assert source_conn.execute(
            "SELECT artifact_kind, support_status, parse_as_session FROM raw_artifacts "
            "WHERE raw_id = ? ORDER BY artifact_id DESC LIMIT 1",
            (raw_id,),
        ).fetchone() == ("deferred_cas_frontier", "partial_decode", 1)


def _head_row(archive: ArchiveStore) -> tuple[object, ...] | None:
    row = archive._conn.execute(
        """SELECT accepted_raw_id, accepted_frontier_kind, accepted_frontier
           FROM raw_revision_heads WHERE logical_source_key = 'codex-session:session'"""
    ).fetchone()
    return None if row is None else tuple(row)


def test_chain_replay_supersedes_equal_frontier_quarantined_membership_head(tmp_path: Path) -> None:
    """Capture-vs-export head collision (the v42 rebuild crash): a byte-proven

    chain full at an EQUAL semantic frontier with different content must take
    the head from a quarantined membership (browser-capture) raw instead of
    the CAS rejecting the whole replay.
    """
    from polylogue.operations.raw_observation_derivation import raw_observation_frame
    from polylogue.sources.revision_backfill import record_session_enrichment_binding, session_enrichment_evidence_key
    from polylogue.storage.derived.raw import RawObservationDerivation

    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        capture_session = _parsed_session(("m0", "zero"), ("m1", "capture flavour"))
        capture = _write_quarantined_member(archive, "capture", capture_session)
        _apply_membership_head(archive, capture, capture_session)
        assert _head_row(archive) == (capture, "semantic", 2)

        export_session = _parsed_session(("m0", "zero"), ("m1", "export flavour"))
        export = _write_chain_full(archive, "export", 2)
        plan = plan_revision_replay([_candidate(export, RawRevisionKind.FULL, 2, size=len("export"))])
        session_id, applied = apply_prepared_revision_replay(archive, plan, {export: export_session}, acquired_at_ms=0)

        assert applied == (export,)
        assert _head_row(archive) == (export, "semantic", 2)
        stored = archive._conn.execute(
            "SELECT content_hash FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        assert stored is not None
        # The retained writer binds each Codex session to the enrichment
        # evidence it read (#5643); an unbound session reads as stale and is
        # derived again. This direct replay stands in for that writer.
        current_key = session_enrichment_evidence_key(
            provider=Provider.CODEX,
            source_path="session.json",
            native_id="session",
            index_conn=archive._conn,
            source_conn=archive._ensure_source_conn(),
            blob_root=tmp_path / "blob",
        )
        record_session_enrichment_binding(
            archive._conn, session_id=session_id, carried_key=current_key, current_key=current_key
        )
        archive.commit()
    assert (
        run_on_convergence_owner(
            tmp_path,
            "test.revision.capture-inspect",
            lambda compute: RawObservationDerivation(tmp_path, compute_adapter=compute).inspect(
                raw_observation_frame(tmp_path), (capture,)
            )[capture],
        )
        == "valid"
    )


def test_chain_replay_supersedes_quarantined_membership_head_even_when_capture_has_more_units(tmp_path: Path) -> None:
    """Chain evidence wins unconditionally: a scalar frontier cannot prove a

    capture is a content-superset, so even a capture with MORE semantic units
    hands the head to chain-governed evidence (the capture raw stays in the
    source tier; re-adoption needs a real prefix-dominance proof).
    """
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        capture_session = _parsed_session(("m0", "zero"), ("m1", "one"), ("m2", "two"))
        capture = _write_quarantined_member(archive, "capture", capture_session)
        _apply_membership_head(archive, capture, capture_session)
        assert _head_row(archive) == (capture, "semantic", 3)

        export_session = _parsed_session(("m0", "zero"), ("m1", "one"))
        export = _write_chain_full(archive, "export", 2)
        plan = plan_revision_replay([_candidate(export, RawRevisionKind.FULL, 2, size=len("export"))])
        session_id, applied = apply_prepared_revision_replay(archive, plan, {export: export_session}, acquired_at_ms=0)

        assert applied == (export,)
        assert _head_row(archive) == (export, "semantic", 2)
        stored = archive._conn.execute(
            "SELECT content_hash FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        assert stored is not None


def test_membership_replay_yields_to_chain_governed_head(tmp_path: Path) -> None:
    """Reverse arrival order: the chain head exists first; an equal-frontier

    quarantined capture cohort must yield (superseded receipts, memberships
    terminally decided) instead of raising 'cannot retire an unrelated
    accepted head'.
    """
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        export_session = _parsed_session(("m0", "zero"), ("m1", "export flavour"))
        export = _write_chain_full(archive, "export", 1)
        plan = plan_revision_replay([_candidate(export, RawRevisionKind.FULL, 1, size=len("export"))])
        apply_prepared_revision_replay(archive, plan, {export: export_session}, acquired_at_ms=0)
        # A chain-first head is byte-kind: its frontier is never comparable to
        # a capture's semantic frontier, so the capture must always yield.
        assert _head_row(archive) == (export, "byte", 6)

        capture_session = _parsed_session(("m0", "zero"), ("m1", "capture flavour"))
        capture = _write_quarantined_member(archive, "capture", capture_session)
        result = _publish_membership(
            archive,
            "codex-session:session",
            MembershipClassification((capture,), (), ()),
            {capture: capture_session},
            {capture: session_revision_projection(capture_session)},
            acquired_at_ms=0,
        )

        assert result == "codex-session:session"
        assert _head_row(archive) == (export, "byte", 6)
        receipts = archive._conn.execute(
            """SELECT decision, detail FROM raw_revision_applications
               WHERE raw_id = ? AND logical_source_key = 'codex-session:session'""",
            (capture,),
        ).fetchall()
        assert [str(row[0]) for row in receipts] == ["superseded"]
        assert f"superseded_by_chain_governed_head:{export}" in str(receipts[0][1])
        membership = (
            archive._ensure_source_conn()
            .execute(
                "SELECT decision, revision_authority FROM raw_session_memberships WHERE raw_id = ?",
                (capture,),
            )
            .fetchone()
        )
        assert membership is not None and tuple(membership) == ("superseded_equivalent", "byte_proven")
        stored = archive._conn.execute(
            "SELECT content_hash FROM sessions WHERE session_id = 'codex-session:session'"
        ).fetchone()
        assert stored is not None


def test_membership_replay_yields_when_resumed_cohort_head_masks_byte_session(tmp_path: Path) -> None:
    """An interrupted membership pass can install its quarantined raw as the
    provisional head before re-indexing the session. The retained session's
    foreign byte-governed raw still wins; replay must receipt the membership
    as superseded instead of raising the unrelated-head guard."""
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        export_session = _parsed_session(("m0", "zero"), ("m1", "export flavour"))
        export = _write_chain_full(archive, "export", 1)
        plan = plan_revision_replay([_candidate(export, RawRevisionKind.FULL, 1, size=len("export"))])
        apply_prepared_revision_replay(archive, plan, {export: export_session}, acquired_at_ms=0)

        capture_session = _parsed_session(("m0", "zero"), ("m1", "capture flavour"))
        capture = _write_quarantined_member(archive, "capture", capture_session)
        archive._conn.execute(
            "UPDATE raw_revision_heads SET accepted_raw_id = ?, accepted_frontier_kind = 'semantic', accepted_frontier = 2 WHERE logical_source_key = 'codex-session:session'",
            (capture,),
        )

        _publish_membership(
            archive,
            "codex-session:session",
            MembershipClassification((capture,), (), ()),
            {capture: capture_session},
            {capture: session_revision_projection(capture_session)},
            acquired_at_ms=0,
        )

        assert _head_row(archive) == (export, "byte", len("export"))
        receipt = archive._conn.execute(
            "SELECT decision, detail FROM raw_revision_applications WHERE raw_id = ?",
            (capture,),
        ).fetchone()
        assert receipt is not None
        assert tuple(receipt) == ("superseded", f"membership:superseded_by_chain_governed_head:{export}")


def test_membership_replay_yields_to_semantic_chain_head_even_when_capture_has_more_units(tmp_path: Path) -> None:
    """A capture cohort with more semantic units still yields to a
    chain-governed semantic head: unit counts are not a dominance proof."""
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        capture1_session = _parsed_session(("m0", "zero"))
        capture1 = _write_quarantined_member(archive, "capture1", capture1_session)
        _apply_membership_head(archive, capture1, capture1_session)
        export_session = _parsed_session(("m0", "zero"))
        export = _write_chain_full(archive, "export", 2)
        plan = plan_revision_replay([_candidate(export, RawRevisionKind.FULL, 2, size=len("export"))])
        apply_prepared_revision_replay(archive, plan, {export: export_session}, acquired_at_ms=0)
        assert _head_row(archive) == (export, "semantic", 1)

        capture2_session = _parsed_session(("m0", "zero"), ("m1", "the conversation continued"))
        capture2 = _write_quarantined_member(archive, "capture2", capture2_session)
        revisions = [
            MembershipRevision(capture1, session_revision_projection(capture1_session)),
            MembershipRevision(capture2, session_revision_projection(capture2_session)),
        ]
        classification = classify_membership_revisions(revisions)
        assert capture2 in classification.accepted_raw_ids
        _publish_membership(
            archive,
            "codex-session:session",
            classification,
            {capture1: capture1_session, capture2: capture2_session},
            {
                capture1: session_revision_projection(capture1_session),
                capture2: session_revision_projection(capture2_session),
            },
            acquired_at_ms=0,
        )

        assert _head_row(archive) == (export, "semantic", 1)
        receipts = archive._conn.execute(
            """SELECT decision, detail FROM raw_revision_applications
               WHERE raw_id = ? AND logical_source_key = 'codex-session:session'""",
            (capture2,),
        ).fetchall()
        assert [str(row[0]) for row in receipts] == ["superseded"]
        assert f"superseded_by_chain_governed_head:{export}" in str(receipts[0][1])
        stored = archive._conn.execute(
            "SELECT content_hash FROM sessions WHERE session_id = 'codex-session:session'"
        ).fetchone()
        assert stored is not None
        assert bytes(stored[0]).hex() == session_content_hash(export_session)


def test_append_replay_reindexes_the_whole_composed_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An accepted append replays the chain's composed session, never a tail.

    Every replay route writes one composed session for the whole accepted
    chain, so the stored projection is the chain's reduction (the composed
    coverage law is ``test_claude_full_append_replay_persists_reduced_coverage_and_receipts``).
    """
    bootstrap_archive_root(tmp_path)

    def parsed(*messages: tuple[str, str]) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="session",
            messages=[
                ParsedMessage(provider_message_id=message_id, role=Role.USER, text=text)
                for message_id, text in messages
            ],
        )

    indexed_writes: list[tuple[str, tuple[str, ...]]] = []
    # polylogue-1r9c: _index_parsed_for_retained_raw's real implementation
    # moved to revision_governance.py, and apply_raw_revision_replay (also in
    # that module) calls it as a direct module-internal function reference,
    # not through `self.` dynamic dispatch -- so the spy must patch the
    # revision_governance module attribute, not the ArchiveStore delegator
    # method (which only intercepts *external* callers).
    original = archive_revision_governance._index_parsed_for_retained_raw

    def spy(
        store: archive_revision_governance.RawRevisionGovernanceHost,
        session: ParsedSession,
        *,
        raw_id: str,
        **kwargs: object,
    ) -> object:
        indexed_writes.append((raw_id, tuple(message.provider_message_id or "" for message in session.messages)))
        return original(store, session, raw_id=raw_id, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(archive_revision_governance, "_index_parsed_for_retained_raw", spy)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        baseline = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"a" * 10,
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            baseline,
            RawRevisionEnvelope(
                "codex-session:session", RawRevisionKind.FULL, "full-0", 0, authority=RawRevisionAuthority.BYTE_PROVEN
            ),
        )
        plan0 = publish_fixture_byte_classification(archive, "codex-session:session")
        apply_prepared_revision_replay(archive, plan0, {baseline: parsed(("m0", "zero"))}, acquired_at_ms=0)
        indexed_writes.clear()

        append_one = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"b" * 5,
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=2,
        )
        archive.bind_raw_revision(
            append_one,
            RawRevisionEnvelope(
                "codex-session:session",
                RawRevisionKind.APPEND,
                append_source_revision("full-0", hashlib.sha256(b"b" * 5).hexdigest()),
                1,
                predecessor_source_revision="full-0",
                predecessor_raw_id=baseline,
                baseline_raw_id=baseline,
                append_start_offset=10,
                append_end_offset=15,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        plan1 = publish_fixture_byte_classification(archive, "codex-session:session")
        apply_prepared_revision_replay(
            archive,
            plan1,
            {baseline: parsed(("m0", "zero")), append_one: parsed(("m1", "one"))},
            acquired_at_ms=0,
        )
        # The whole chain's composed content is written, not the new tail.
        assert indexed_writes == [(append_one, ("m0", "m1"))]


def test_accepted_chain_indexes_one_composed_session_not_one_per_chunk(tmp_path: Path) -> None:
    """polylogue-rkdej: an accepted full-plus-append chain persists the composed
    session exactly once.

    ``apply_raw_revision_replay`` composes the chain with
    ``merge_parsed_session_chunks`` to derive ``sessions.content_hash``. If it
    then hands each ``parsed_by_raw_id`` chunk to the writer separately, the
    index describes the stream schedule instead of the session: each chunk's
    ``claude_parse_coverage`` -- a complete-input summary -- lands as its own
    row, so the persisted read model disagrees with the content hash the same
    call just stored.

    Anti-vacuity: indexing per chunk (the pre-fix loop) leaves two
    ``claude_parse_coverage`` rows carrying the chunk-local counts 1 and 2
    rather than one row carrying the composed total 3, and turns this test red.
    """
    bootstrap_archive_root(tmp_path)

    def parsed(*, message_id: str, text: str, seen: int) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CLAUDE_CODE,
            provider_session_id="chat",
            messages=[ParsedMessage(provider_message_id=message_id, role=Role.USER, text=text)],
            session_events=[
                ParsedSessionEvent(
                    event_type="claude_parse_coverage",
                    payload={"sidecar_seen": {"user": seen}},
                )
            ],
        )

    baseline_chunk = parsed(message_id="m0", text="zero", seen=1)
    append_chunk = parsed(message_id="m1", text="one", seen=2)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        baseline = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=b"a" * 10,
            source_path="chat.jsonl",
            canonical_source_path="chat.jsonl",
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            baseline,
            RawRevisionEnvelope(
                "claude-code-session:chat",
                RawRevisionKind.FULL,
                "full-0",
                0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        append_one = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=b"b" * 5,
            source_path="chat.jsonl",
            canonical_source_path="chat.jsonl",
            source_index=-1,
            acquired_at_ms=2,
        )
        archive.bind_raw_revision(
            append_one,
            RawRevisionEnvelope(
                "claude-code-session:chat",
                RawRevisionKind.APPEND,
                append_source_revision("full-0", hashlib.sha256(b"b" * 5).hexdigest()),
                1,
                predecessor_source_revision="full-0",
                predecessor_raw_id=baseline,
                baseline_raw_id=baseline,
                append_start_offset=10,
                append_end_offset=15,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )

        plan = publish_fixture_byte_classification(archive, "claude-code-session:chat")
        assert plan.accepted_raw_ids == (baseline, append_one)
        session_id, _ = apply_prepared_revision_replay(
            archive,
            plan,
            {baseline: baseline_chunk, append_one: append_chunk},
            acquired_at_ms=0,
        )

        composed = merge_parsed_session_chunks([baseline_chunk, append_chunk])
        assert len(composed) == 1

        messages = [
            row[0]
            for row in archive._conn.execute(
                "SELECT native_id FROM messages WHERE session_id = ? ORDER BY position",
                (session_id,),
            ).fetchall()
        ]
        assert messages == ["m0", "m1"]

        coverage = [
            json.loads(row[0])
            for row in archive._conn.execute(
                """SELECT payload_json FROM session_events
                   WHERE session_id = ? AND event_type = 'claude_parse_coverage'
                   ORDER BY position""",
                (session_id,),
            ).fetchall()
        ]
        assert coverage == [composed[0].session_events[-1].payload]
        assert coverage[0]["sidecar_seen"] == {"user": 3}

        stored_hash = archive._conn.execute(
            "SELECT content_hash FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert stored_hash is not None
        assert bytes(stored_hash[0]).hex() == session_content_hash(composed[0])


def test_terminal_failure_carrier_survives_ordinary_reclassification(tmp_path: Path) -> None:
    """An ordinary path classification may not take back a terminal failure carrier.

    polylogue-lqo6a: a raw refused with a typed terminal outcome owns its
    ``raw_artifacts`` row; ``_terminal_artifact_paths`` reads that carrier to
    settle the path for the raw-frontier gate. Re-observing the same
    coordinate re-derives an ordinary path classification, and overwriting
    the carrier with it leaves the path neither terminal nor headed.

    Anti-vacuity: dropping ``terminal_carrier_overwrite_predicate`` from
    either upsert lets the ordinary row land, and the final assertions read
    ``coordinator_session_stream``/``parse_as_session = 1``.
    """
    from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceArtifact, upsert_raw_artifact

    source_path = "projects/-home-user/summary-only.jsonl"
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=b'{"type":"summary","summary":"only a summary"}\n',
            source_path=source_path,
            canonical_source_path=source_path,
            acquired_at_ms=1,
        )
        archive.record_raw_failure_evidence(
            raw_id,
            provider=Provider.CLAUDE_CODE,
            source_path=source_path,
            source_index=0,
            acquired_at_ms=1,
            kind=RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE,
        )
        archive.mark_raw_parse_failed(
            raw_id,
            provider=Provider.CLAUDE_CODE,
            error=ValueError("parsed raw payload produced no sessions with positive conversational evidence"),
            preserve_existing_failure_evidence=True,
        )
        with sqlite3.connect(tmp_path / "source.db") as probe:
            carrier_id, carrier_kind = probe.execute(
                "SELECT artifact_id, artifact_kind FROM raw_artifacts WHERE raw_id = ?",
                (raw_id,),
            ).fetchone()
        assert carrier_id.startswith("raw-failure:")
        assert carrier_kind == RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE.value

    # The ordinary Claude path rule re-observing the exact same carrier id. The
    # archive's own Source handle admits only its declared producers, so the
    # upsert law runs on a fixture connection.
    with sqlite3.connect(tmp_path / "source.db") as fixture_source:
        upsert_raw_artifact(
            fixture_source,
            raw_id,
            ArchiveSourceArtifact(
                artifact_id=carrier_id,
                origin="claude-code-session",
                source_path=source_path,
                source_index=0,
                artifact_kind="coordinator_session_stream",
                support_status="supported_parseable",
                classification_reason="OriginSpec Claude artifact rule: coordinator_invocation_stream",
                parse_as_session=True,
                schema_eligible=True,
                first_observed_at_ms=2,
                last_observed_at_ms=2,
            ),
        )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        kind, parse_as_session = conn.execute(
            "SELECT artifact_kind, parse_as_session FROM raw_artifacts WHERE artifact_id = ?",
            (carrier_id,),
        ).fetchone()
    assert kind == RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE.value
    assert parse_as_session == 0


def _headless_ambiguous_cohort(
    archive: ArchiveStore,
) -> tuple[MembershipClassification, dict[str, ParsedSession]]:
    """Seed a cohort whose members all stay ambiguous (an incomplete cohort)."""

    def session_with(text: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="session",
            messages=[ParsedMessage(provider_message_id="m0", role=Role.USER, text=text)],
        )

    branch_a = session_with("alpha")
    branch_b = session_with("beta")

    def add_member(raw_id: str, session: ParsedSession) -> MembershipRevision:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=raw_id.encode(),
            source_path=f"{raw_id}.jsonl",
            canonical_source_path=f"{raw_id}.jsonl",
            acquired_at_ms=1,
            raw_id=raw_id,
        )
        publish_membership_census(
            archive, raw_id, [session], parser_fingerprint="test-parser", censused_at_ms=1, revision_authority=None
        )
        return MembershipRevision(raw_id, session_revision_projection(session))

    members = [
        add_member("branch-a", branch_a),
        add_member("branch-a-dup", branch_a),
        add_member("branch-b", branch_b),
    ]
    classification = classify_membership_revisions(members, existing_accepted_raw_id="branch-a")
    assert classification.accepted_raw_ids == ()
    session_by_raw = {"branch-a": branch_a, "branch-a-dup": branch_a, "branch-b": branch_b}
    return classification, session_by_raw


def _membership_authority(conn: sqlite3.Connection) -> list[tuple[str, str | None, str | None]]:
    return conn.execute(
        """
        SELECT raw_id, decision, revision_authority
        FROM raw_session_memberships
        WHERE logical_source_key = 'codex-session:session'
        ORDER BY raw_id
        """
    ).fetchall()


def test_incomplete_cohort_publishes_ambiguous_decisions_with_its_parse_correction(tmp_path: Path) -> None:
    """An incomplete cohort's Source outcome lands as one publication.

    Production dependency: ``prepare_membership_classification_source`` stages
    the ambiguous decisions together with the parse-state correction on the
    original seal; ``publish_prepared_revision_source`` commits them together
    (polylogue-upua6 invariant on the canonical route).

    Anti-vacuity: dropping the correction from the staged Source mutation
    leaves ``parsed_at_ms`` set, and recording decisions without their
    authority leaves them NULL.
    """
    bootstrap_archive_root(tmp_path)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        classification, session_by_raw = _headless_ambiguous_cohort(archive)
        source_conn = archive._ensure_source_conn()
        source_conn.commit()
        undecided = _membership_authority(source_conn)
        assert all(decision is None for _raw, decision, _authority in undecided)

        session_id = _publish_membership(
            archive,
            "codex-session:session",
            classification,
            session_by_raw,
            {raw_id: session_revision_projection(s) for raw_id, s in session_by_raw.items()},
            acquired_at_ms=1,
        )

    assert session_id is None
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert _membership_authority(source) == [
            ("branch-a", "ambiguous", "quarantined"),
            ("branch-a-dup", "ambiguous", "quarantined"),
            ("branch-b", "ambiguous", "quarantined"),
        ]
        assert source.execute(
            "SELECT count(*) FROM raw_sessions"
            " WHERE parsed_at_ms IS NOT NULL"
            " AND raw_id IN ('branch-a', 'branch-a-dup', 'branch-b')"
        ).fetchone() == (0,)


def test_incomplete_cohort_correction_failure_publishes_no_decision(tmp_path: Path) -> None:
    """A refused correction leaves the cohort's Source authority undecided.

    Production dependency: the staged Source mutation applies decisions and
    the parse-state correction in one transaction on the dedicated writer
    (polylogue-upua6 invariant on the canonical route).

    Anti-vacuity: publishing decisions in a separate commit before the
    correction leaves decided rows behind when the correction is refused.
    """
    bootstrap_archive_root(tmp_path)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        classification, session_by_raw = _headless_ambiguous_cohort(archive)
        with independent_source_connection(archive) as source:
            source.execute(
                """
                CREATE TRIGGER reject_incomplete_cohort_correction
                BEFORE UPDATE OF parsed_at_ms ON raw_sessions
                WHEN NEW.parsed_at_ms IS NULL
                BEGIN
                    SELECT RAISE(ABORT, 'correction refused');
                END
                """
            )
            undecided = _membership_authority(source)
        assert all(decision is None for _raw, decision, _authority in undecided)

        with pytest.raises(sqlite3.IntegrityError, match="correction refused"):
            _publish_membership(
                archive,
                "codex-session:session",
                classification,
                session_by_raw,
                {raw_id: session_revision_projection(s) for raw_id, s in session_by_raw.items()},
                acquired_at_ms=1,
            )

    with sqlite3.connect(tmp_path / "source.db") as source:
        assert _membership_authority(source) == undecided
        source.execute("DROP TRIGGER reject_incomplete_cohort_correction")


@pytest.mark.parametrize("later", [b"divergent", b"a", b"abcd", b"abcdef"])
def test_full_byte_classifier_preserves_original_source_anchor(later: bytes) -> None:
    original = b"abcd"
    payloads = {"anchor": original, "later": later}
    rows = [
        ("anchor", "anchor-hash", len(original), "byte_proven", "prior", "root", 7),
        ("later", "later-hash" if later != original else "anchor-hash", len(later), "quarantined", None, None, 0),
    ]
    updates = {
        row[0]: row[1:]
        for row in archive_revision_governance._classify_full_revision_byte_inputs(
            rows, lambda raw_id, _blob_hash: BytesIO(payloads[raw_id])
        )
    }
    assert updates["anchor"] == ("byte_proven", "prior", "root", 7)
    if later == original:
        assert updates["later"] == ("byte_proven", None, "root", 7)
    elif later.startswith(original):
        assert updates["later"] == ("byte_proven", "anchor", "root", 8)
    else:
        assert updates["later"] == ("quarantined", None, None, 0)


def test_full_byte_classifier_preserves_proved_chain_when_late_fork_arrives() -> None:
    payloads = {"base": b"a", "head": b"ab", "fork": b"ac"}
    rows = [
        ("base", "base-hash", 1, "byte_proven", None, "base", 0),
        ("head", "head-hash", 2, "byte_proven", "base", "base", 1),
        ("fork", "fork-hash", 2, "quarantined", None, None, 0),
    ]
    updates = archive_revision_governance._classify_full_revision_byte_inputs(
        rows, lambda raw_id, _blob_hash: BytesIO(payloads[raw_id])
    )
    assert updates == (
        ("base", "byte_proven", None, "base", 0),
        ("head", "byte_proven", "base", "base", 1),
        ("fork", "quarantined", None, None, 0),
    )


def test_full_byte_classifier_quarantines_late_interior_prefix_without_rebinding_head() -> None:
    payloads = {"base": b"a", "head": b"abcd", "late": b"ab"}
    rows = [
        ("base", "base-hash", 1, "byte_proven", None, "base", 0),
        ("head", "head-hash", 4, "byte_proven", "base", "base", 1),
        ("late", "late-hash", 2, "quarantined", None, None, 0),
    ]
    assert archive_revision_governance._classify_full_revision_byte_inputs(
        rows, lambda raw_id, _blob_hash: BytesIO(payloads[raw_id])
    ) == (
        ("base", "byte_proven", None, "base", 0),
        ("head", "byte_proven", "base", "base", 1),
        ("late", "quarantined", None, None, 0),
    )


def test_full_byte_classifier_initial_chain_duplicate_inherits_generation() -> None:
    payloads = {"base": b"a", "middle": b"ab", "duplicate": b"ab", "head": b"abc"}
    rows = [
        (
            raw_id,
            "middle-hash" if raw_id in {"middle", "duplicate"} else raw_id + "-hash",
            len(payload),
            "quarantined",
            None,
            None,
            0,
        )
        for raw_id, payload in payloads.items()
    ]
    actual = {
        row[0]: row[1:]
        for row in archive_revision_governance._classify_full_revision_byte_inputs(
            rows, lambda raw_id, _blob_hash: BytesIO(payloads[raw_id])
        )
    }
    assert actual["base"] == ("byte_proven", None, "base", 0)
    assert actual["head"][0] == "byte_proven"
    assert actual["head"][2:] == ("base", 2)
    assert actual["middle"][2:] == ("base", 1)
    assert actual["duplicate"][2:] == ("base", 1)


def test_full_byte_classifier_preserves_asserted_baseline_without_extension_grant() -> None:
    payloads = {"baseline": b"a", "later": b"ab"}
    rows = [
        ("baseline", "baseline-hash", 1, "asserted", None, None, 0),
        ("later", "later-hash", 2, "quarantined", None, None, 0),
    ]
    assert archive_revision_governance._classify_full_revision_byte_inputs(
        rows, lambda raw_id, _blob_hash: BytesIO(payloads[raw_id])
    ) == (
        ("baseline", "asserted", None, None, 0),
        ("later", "quarantined", None, None, 0),
    )
