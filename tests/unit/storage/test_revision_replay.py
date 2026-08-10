from __future__ import annotations

import hashlib
import json
import sqlite3
from itertools import permutations
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.revision_authority import (
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
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession
from polylogue.storage.raw_authority import RAW_AUTHORITY_PARSER_FINGERPRINT, parser_census_logical_keys
from polylogue.storage.sqlite.archive_tiers import revision_governance as archive_revision_governance
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


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
        logical_source_key="codex:session",
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
    initialize_active_archive_root(tmp_path)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"live-receipt"}}\n',
            source_path="live/codex.jsonl",
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            raw_id,
            RawRevisionEnvelope(
                "codex:live-receipt",
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


def test_membership_receipt_excludes_post_parse_pending_identity(tmp_path: Path) -> None:
    """A parser-derived membership receipt cannot retain its provisional raw key."""
    initialize_active_archive_root(tmp_path)
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
            acquired_at_ms=1,
            post_parse=True,
        )
        archive.replace_raw_membership_census(
            raw_id,
            [session],
            parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT,
            censused_at_ms=1,
        )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        receipt = conn.execute(
            "SELECT logical_keys_json FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
        ).fetchone()

    assert receipt is not None
    assert parser_census_logical_keys(receipt[0]) == ("codex-session:post-parse-receipt",)


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
    initialize_active_archive_root(tmp_path)
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
                acquired_at_ms=1,
                raw_id=raw_id,
            )
            archive.replace_raw_membership_census(
                raw_id,
                [session],
                parser_fingerprint="test-parser",
                censused_at_ms=1,
            )
            return MembershipRevision(raw_id, projection)

        members = [add_member("representative-b"), add_member("equivalent-z")]
        first = classify_membership_revisions(members)
        assert first.accepted_raw_ids == ("equivalent-z",)
        archive.apply_raw_membership_classification(
            "codex:session",
            first,
            {member.raw_id: session for member in members},
            {member.raw_id: projection for member in members},
            acquired_at_ms=1,
        )

        members.append(add_member("accepted-a"))
        second = classify_membership_revisions(members)
        assert second.accepted_raw_ids == ("accepted-a",)
        archive.apply_raw_membership_classification(
            "codex:session",
            second,
            {member.raw_id: session for member in members},
            {member.raw_id: projection for member in members},
            acquired_at_ms=2,
        )

        head = archive._conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'codex:session'"
        ).fetchone()
        assert head is not None and tuple(head) == ("accepted-a",)
        application_rows = archive._conn.execute(
            """
            SELECT raw_id, decision, accepted_raw_id
            FROM raw_revision_applications
            WHERE logical_source_key = 'codex:session'
            ORDER BY raw_id, decision
            """
        ).fetchall()
        assert [tuple(row) for row in application_rows] == [
            ("accepted-a", "selected_baseline", "accepted-a"),
            ("equivalent-z", "selected_baseline", "equivalent-z"),
            ("equivalent-z", "superseded", "accepted-a"),
            ("representative-b", "superseded", "equivalent-z"),
        ]
        matching_receipts = archive._conn.execute(
            """
            SELECT COUNT(*) FROM raw_revision_heads AS h
            JOIN raw_revision_applications AS a
              ON a.logical_source_key = h.logical_source_key
             AND a.accepted_raw_id = h.accepted_raw_id
             AND a.accepted_content_hash = h.accepted_content_hash
            WHERE h.logical_source_key = 'codex:session'
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
    initialize_active_archive_root(tmp_path)

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
                acquired_at_ms=1,
                raw_id=raw_id,
            )
            archive.replace_raw_membership_census(
                raw_id,
                [session],
                parser_fingerprint="test-parser",
                censused_at_ms=1,
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
        archive.apply_raw_membership_classification(
            "codex:session",
            classification,
            session_by_raw,
            {raw_id: session_revision_projection(session) for raw_id, session in session_by_raw.items()},
            acquired_at_ms=1,
        )

        head = archive._conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'codex:session'"
        ).fetchone()
        assert head is None

        membership_rows = (
            archive._ensure_source_conn()
            .execute(
                """
            SELECT raw_id, decision, revision_authority
            FROM raw_session_memberships
            WHERE logical_source_key = 'codex:session'
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

    ``revision_governance.classify_raw_revision_cohort`` writes a duplicate
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
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        append_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"suffix",
            source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            append_raw_id,
            RawRevisionEnvelope(
                "codex:session",
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
            acquired_at_ms=2,
        )
        archive.bind_raw_revision(
            baseline_raw_id,
            RawRevisionEnvelope(
                "codex:session",
                RawRevisionKind.FULL,
                "revision-base",
                0,
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )

        plan = archive.classify_raw_revision_cohort_for_live_watch("codex:session")

    assert {item.raw_id: item.decision for item in plan.applications} == {
        baseline_raw_id: ApplicationDecision.SELECTED_BASELINE,
        append_raw_id: ApplicationDecision.APPLIED_APPEND,
    }


def _write_full_raw(archive: ArchiveStore, *, raw_id: str, payload: bytes, acquired_at_ms: int) -> str:
    written_id = archive.write_raw_payload(
        provider=Provider.CODEX,
        payload=payload,
        source_path="session.jsonl",
        acquired_at_ms=acquired_at_ms,
        raw_id=raw_id,
    )
    archive.bind_raw_revision(
        written_id,
        RawRevisionEnvelope("codex:session", RawRevisionKind.FULL, f"revision-{raw_id}", 0),
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
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        base = _write_full_raw(archive, raw_id="raw-000-base", payload=b"a" * 10, acquired_at_ms=1)
        mid = _write_full_raw(archive, raw_id="raw-010-mid", payload=b"a" * 10 + b"b" * 10, acquired_at_ms=2)
        head = _write_full_raw(
            archive, raw_id="raw-020-head", payload=b"a" * 10 + b"b" * 10 + b"c" * 10, acquired_at_ms=3
        )
        mid_duplicate = _write_full_raw(
            archive, raw_id="raw-011-mid-dup", payload=b"a" * 10 + b"b" * 10, acquired_at_ms=4
        )

        archive.classify_raw_revision_cohort_for_live_watch("codex:session")

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
    initialize_active_archive_root(tmp_path)
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

        archive.classify_raw_revision_cohort_for_live_watch("codex:session")

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
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        baseline = _write_full_raw(archive, raw_id="raw-a-baseline", payload=b"hello world", acquired_at_ms=1)
        duplicate = _write_full_raw(archive, raw_id="raw-b-duplicate", payload=b"hello world", acquired_at_ms=2)

        plan = archive.classify_raw_revision_cohort_for_live_watch("codex:session")

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
            archive.replace_raw_membership_census(
                baseline,
                [],
                parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT,
                censused_at_ms=0,
                detail="test-duplicate-guard",
                retire_full_revision_governance=True,
            )
        archive.rollback()


def test_real_append_chain_folds_segmentation_distinct_full_snapshot(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)

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
            provider=Provider.CODEX, payload=b"a" * 10, source_path="session.jsonl", acquired_at_ms=1
        )
        archive.bind_raw_revision(
            baseline,
            RawRevisionEnvelope(
                "codex:session", RawRevisionKind.FULL, "full-0", 0, authority=RawRevisionAuthority.BYTE_PROVEN
            ),
        )
        append_one = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"b" * 5,
            source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=2,
        )
        archive.bind_raw_revision(
            append_one,
            RawRevisionEnvelope(
                "codex:session",
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
            source_index=-1,
            acquired_at_ms=3,
        )
        archive.bind_raw_revision(
            append_two,
            RawRevisionEnvelope(
                "codex:session",
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
        append_plan = archive.classify_raw_revision_cohort_for_live_watch("codex:session")
        archive.apply_raw_revision_replay(
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
            acquired_at_ms=4,
        )
        archive.bind_raw_revision(
            folded,
            RawRevisionEnvelope(
                "codex:session", RawRevisionKind.FULL, "full-folded", 3, authority=RawRevisionAuthority.BYTE_PROVEN
            ),
        )
        folded_plan = archive.classify_raw_revision_cohort_for_live_watch("codex:session")
        folded_session = parsed(("full-0", "zero"), ("full-1", "one"), ("full-2", "two"))
        before_hash = archive._conn.execute(
            "SELECT accepted_content_hash FROM raw_revision_heads WHERE logical_source_key = ?", ("codex:session",)
        ).fetchone()
        assert before_hash is not None
        assert bytes(before_hash[0]) != bytes.fromhex(session_content_hash(folded_session))
        archive.apply_raw_revision_replay(
            folded_plan,
            {folded: folded_session},
            acquired_at_ms=0,
        )

        head = archive._conn.execute(
            "SELECT accepted_raw_id, accepted_frontier FROM raw_revision_heads WHERE logical_source_key = ?",
            ("codex:session",),
        ).fetchone()
        assert head is not None
        assert tuple(head) == (folded, 20)


def test_isolated_later_raw_does_not_override_known_ambiguous_cohort(tmp_path: Path) -> None:
    """polylogue-52l2: a raw discovered for a logical identity that already
    has quarantined/ambiguous siblings must not be accepted as an
    unambiguous singleton byte-proven baseline.

    This mirrors the LIVE incremental watcher's own call sequence
    (``sources/live/batch.py``): ``bind_raw_revision`` then
    ``classify_raw_revision_cohort`` directly, with no census-phase
    re-derivation or connected-component re-expansion in between (those only
    happen in the offline ``backfill_historical_revision_evidence`` path,
    which is why this bug does not reproduce through that entry point).

    ``classify_raw_revision_cohort`` only ever queries
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
    initialize_active_archive_root(tmp_path)

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
            provider=Provider.CHATGPT, payload=b"aaa-left", source_path="a.json", acquired_at_ms=1
        )
        archive.bind_raw_revision(
            raw_a,
            RawRevisionEnvelope(
                "chatgpt:s1", RawRevisionKind.FULL, raw_a, 0, authority=RawRevisionAuthority.QUARANTINED
            ),
        )
        raw_b = archive.write_raw_payload(
            provider=Provider.CHATGPT, payload=b"bbb-right", source_path="b.json", acquired_at_ms=2
        )
        archive.bind_raw_revision(
            raw_b,
            RawRevisionEnvelope(
                "chatgpt:s1", RawRevisionKind.FULL, raw_b, 0, authority=RawRevisionAuthority.QUARANTINED
            ),
        )

        first_plan = archive.classify_raw_revision_cohort_for_live_watch("chatgpt:s1")
        assert first_plan.accepted_raw_ids == ()

        # Both siblings genuinely disagree (no byte-prefix relation) --
        # exactly what the backfill caller does when a cohort is decided
        # ambiguous: move it to membership governance so parsed-content
        # prefix rules can still arbitrate it later.
        for raw_id, session in (
            (raw_a, parsed_solo("s1", "base", "left")),
            (raw_b, parsed_solo("s1", "base", "right")),
        ):
            archive.replace_raw_membership_census(
                raw_id,
                [session],
                parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT,
                censused_at_ms=0,
                detail="historical non-prefix full revision governance",
                retire_full_revision_governance=True,
            )

        # A THIRD raw for the same logical identity, discovered afterward.
        raw_c = archive.write_raw_payload(
            provider=Provider.CHATGPT, payload=b"ccc-solo", source_path="c.json", acquired_at_ms=3
        )
        archive.bind_raw_revision(
            raw_c,
            RawRevisionEnvelope(
                "chatgpt:s1", RawRevisionKind.FULL, raw_c, 0, authority=RawRevisionAuthority.QUARANTINED
            ),
        )
        second_plan = archive.classify_raw_revision_cohort_for_live_watch("chatgpt:s1")

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
    initialize_active_archive_root(tmp_path)

    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="s1",
        messages=[ParsedMessage(provider_message_id="s1-0", role=Role.USER, text="left")],
    )

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT, payload=b"aaa-left", source_path="a.json", acquired_at_ms=1
        )
        # Durable evidence that this raw's identity was already judged
        # ambiguous -- the shape ``replace_raw_membership_census`` /
        # ``apply_raw_membership_classification`` leave behind for a
        # genuinely divergent cohort (reproduced directly here so the test
        # isolates the WRITE-PATH guard from the classifier that produces
        # this state).
        source_conn = archive._ensure_source_conn()
        with source_conn:
            source_conn.execute(
                """
                INSERT INTO raw_session_memberships (
                    raw_id, logical_source_key, provider_session_id,
                    source_revision, normalized_content_hash, message_count,
                    decision, decided_at_ms
                ) VALUES (?, 'chatgpt:s1', 's1', ?, ?, 1, 'ambiguous', 1)
                """,
                (raw_id, raw_id, bytes.fromhex(raw_id)),
            )

        returned_raw_id, session_id = archive.write_parsed_for_retained_raw(
            session,
            raw_id=raw_id,
            source_path="a.json",
            acquired_at_ms=2,
        )

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
    initialize_active_archive_root(tmp_path)

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
            provider=Provider.CHATGPT, payload=b"two-sessions", source_path="bundle.json", acquired_at_ms=1
        )
        source_conn = archive._ensure_source_conn()
        with source_conn:
            # One raw, two memberships, arbitrated differently -- the live shape.
            source_conn.execute(
                """
                INSERT INTO raw_session_memberships (
                    raw_id, logical_source_key, provider_session_id,
                    source_revision, normalized_content_hash, message_count,
                    decision, decided_at_ms
                ) VALUES (?, 'chatgpt:s-ambiguous', 's-ambiguous', ?, ?, 1, 'ambiguous', 1)
                """,
                (raw_id, raw_id, bytes.fromhex(raw_id)),
            )
            source_conn.execute(
                """
                INSERT INTO raw_session_memberships (
                    raw_id, logical_source_key, provider_session_id,
                    source_revision, normalized_content_hash, message_count,
                    decision, decided_at_ms
                ) VALUES (?, 'chatgpt:s-settled', 's-settled', ?, ?, 1, 'applied', 1)
                """,
                (raw_id, raw_id + "-b", bytes.fromhex(raw_id)),
            )

        _, ambiguous_session_id = archive.write_parsed_for_retained_raw(
            ambiguous_session, raw_id=raw_id, source_path="bundle.json", acquired_at_ms=2
        )
        _, settled_session_id = archive.write_parsed_for_retained_raw(
            settled_session, raw_id=raw_id, source_path="bundle.json", acquired_at_ms=3
        )

    with sqlite3.connect(tmp_path / "index.db") as conn:
        # The ambiguous membership is still refused ...
        assert conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id = ?", (ambiguous_session_id,)
        ).fetchone() == (0,)
        # ... and its settled sibling on the same raw is not collateral damage.
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (settled_session_id,)).fetchone() == (
            1,
        )


def test_isolated_later_raw_does_not_override_cohort_retired_under_legacy_detail_string(
    tmp_path: Path,
) -> None:
    """polylogue-hm2f: the 52l2 guard must also recognize legacy-detail retirements.

    Durable ``raw_membership_census`` rows written by ``sources/live/batch.py``
    BEFORE #3234 carry ``detail="cross-route full revision governance"`` (the
    pre-fix literal at that call site) instead of the shared
    ``HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL`` marker the #3234 guard query
    matches. This is a byte-for-byte mirror of
    ``test_isolated_later_raw_does_not_override_known_ambiguous_cohort``
    except the two siblings are retired under that legacy literal directly
    (as durable pre-#3234 rows would already be on disk) instead of the
    current shared constant -- proving the guard's widened ``detail IN (...)``
    match, not just its original single-literal match.
    """
    initialize_active_archive_root(tmp_path)

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
            provider=Provider.CHATGPT, payload=b"aaa-left", source_path="a.json", acquired_at_ms=1
        )
        archive.bind_raw_revision(
            raw_a,
            RawRevisionEnvelope(
                "chatgpt:s1", RawRevisionKind.FULL, raw_a, 0, authority=RawRevisionAuthority.QUARANTINED
            ),
        )
        raw_b = archive.write_raw_payload(
            provider=Provider.CHATGPT, payload=b"bbb-right", source_path="b.json", acquired_at_ms=2
        )
        archive.bind_raw_revision(
            raw_b,
            RawRevisionEnvelope(
                "chatgpt:s1", RawRevisionKind.FULL, raw_b, 0, authority=RawRevisionAuthority.QUARANTINED
            ),
        )

        first_plan = archive.classify_raw_revision_cohort_for_live_watch("chatgpt:s1")
        assert first_plan.accepted_raw_ids == ()

        # Retire both siblings under the LEGACY pre-#3234 literal, not the
        # current shared constant -- this is what a durable row written
        # before #3234 actually contains on disk.
        for raw_id, session in (
            (raw_a, parsed_solo("s1", "base", "left")),
            (raw_b, parsed_solo("s1", "base", "right")),
        ):
            archive.replace_raw_membership_census(
                raw_id,
                [session],
                parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT,
                censused_at_ms=0,
                detail="cross-route full revision governance",
                retire_full_revision_governance=True,
            )

        # A THIRD raw for the same logical identity, discovered afterward.
        raw_c = archive.write_raw_payload(
            provider=Provider.CHATGPT, payload=b"ccc-solo", source_path="c.json", acquired_at_ms=3
        )
        archive.bind_raw_revision(
            raw_c,
            RawRevisionEnvelope(
                "chatgpt:s1", RawRevisionKind.FULL, raw_c, 0, authority=RawRevisionAuthority.QUARANTINED
            ),
        )
        second_plan = archive.classify_raw_revision_cohort_for_live_watch("chatgpt:s1")

    # Same assertion as the shared-constant test: the isolated raw must not
    # be promoted alone against siblings retired under the legacy literal.
    assert second_plan.accepted_raw_ids == ()

    # Anti-vacuity: removing the legacy literal from the tuple this guard
    # matches against must make this exact test fail. Assert the tuple still
    # names it explicitly, so a future edit that drops it is caught here too.
    from polylogue.archive.revision_authority import RETIRED_FULL_REVISION_GOVERNANCE_DETAILS

    assert "cross-route full revision governance" in RETIRED_FULL_REVISION_GOVERNANCE_DETAILS


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
    query), so ``classify_raw_revision_cohort`` evaluates each key as a
    trivial one-member chain and unconditionally accepts BOTH as
    independent byte-proven singleton baselines -- silently splitting one
    physical document into two sessions that then race on the shared
    ``(origin, native_id)`` upsert (arbitrary last-writer-wins), instead of
    ever being compared against each other.
    """
    initialize_active_archive_root(tmp_path)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_enriched = archive.write_raw_payload(
            provider=Provider.GEMINI, payload=b"enriched-bytes", source_path="doc.json", acquired_at_ms=1
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
            provider=Provider.GEMINI, payload=b"bare-bytes", source_path="doc.json", acquired_at_ms=2
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

        enriched_plan = archive.classify_raw_revision_cohort_for_rebuild_repair("gemini:doc")
        bare_plan = archive.classify_raw_revision_cohort_for_rebuild_repair("gemini:doc-0")

    # Neither key's lone member may be promoted alone: a same-source_path
    # sibling under a different key means this identity is genuinely
    # contested, not a clean singleton chain.
    assert enriched_plan.accepted_raw_ids == ()
    assert bare_plan.accepted_raw_ids == ()


def test_real_single_append_chain_folds_segmentation_distinct_full_snapshot(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)

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
            provider=Provider.CODEX, payload=baseline_payload, source_path="session.jsonl", acquired_at_ms=1
        )
        archive.bind_raw_revision(
            baseline,
            RawRevisionEnvelope(
                "codex:session", RawRevisionKind.FULL, "base", 0, authority=RawRevisionAuthority.BYTE_PROVEN
            ),
        )
        append = archive.write_raw_payload(
            provider=Provider.CODEX, payload=tail, source_path="session.jsonl", source_index=-1, acquired_at_ms=2
        )
        append_revision = append_source_revision("base", hashlib.sha256(tail).hexdigest())
        archive.bind_raw_revision(
            append,
            RawRevisionEnvelope(
                "codex:session",
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
        archive.apply_raw_revision_replay(
            archive.raw_revision_replay_plan("codex:session"),
            {baseline: baseline_session, append: append_session},
            acquired_at_ms=0,
        )
        folded = archive.write_raw_payload(
            provider=Provider.CODEX, payload=folded_payload, source_path="session.jsonl", acquired_at_ms=3
        )
        archive.bind_raw_revision(
            folded,
            RawRevisionEnvelope(
                "codex:session", RawRevisionKind.FULL, "folded", 2, authority=RawRevisionAuthority.BYTE_PROVEN
            ),
        )
        before_hash = archive._conn.execute(
            "SELECT accepted_content_hash FROM raw_revision_heads WHERE logical_source_key = ?", ("codex:session",)
        ).fetchone()
        assert before_hash is not None
        assert bytes(before_hash[0]) != bytes.fromhex(session_content_hash(folded_session))
        archive.apply_raw_revision_replay(
            archive.raw_revision_replay_plan("codex:session"), {folded: folded_session}, acquired_at_ms=0
        )
        assert archive.raw_revision_head_raw_id("codex:session") == folded


@pytest.mark.parametrize("mutation", ["tail", "gap", "overlap", "predecessor", "baseline", "missing", "divergent"])
def test_real_append_fold_proof_mutations_roll_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    initialize_active_archive_root(tmp_path)

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
                "SELECT event_id, source_message_id, event_type, summary, payload_json FROM session_events ORDER BY event_id"
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
            provider=Provider.CODEX, payload=baseline_payload, source_path="session.jsonl", acquired_at_ms=1
        )
        archive.bind_raw_revision(
            baseline,
            RawRevisionEnvelope(
                "codex:session", RawRevisionKind.FULL, "base", 0, authority=RawRevisionAuthority.BYTE_PROVEN
            ),
        )
        append = archive.write_raw_payload(
            provider=Provider.CODEX, payload=tail, source_path="session.jsonl", source_index=-1, acquired_at_ms=2
        )
        append_revision = append_source_revision("base", hashlib.sha256(tail).hexdigest())
        archive.bind_raw_revision(
            append,
            RawRevisionEnvelope(
                "codex:session",
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
        chain = archive.raw_revision_replay_plan("codex:session")
        archive.apply_raw_revision_replay(chain, {baseline: baseline_session, append: append_session}, acquired_at_ms=0)
        folded_payload = candidate_payload
        if mutation in {"baseline", "divergent"}:
            folded_payload = (b"X" if mutation == "baseline" else baseline_payload[:5] + b"X") + folded_payload[
                1 if mutation == "baseline" else 6 :
            ]
        folded = archive.write_raw_payload(
            provider=Provider.CODEX, payload=folded_payload, source_path="session.jsonl", acquired_at_ms=3
        )
        archive.bind_raw_revision(
            folded,
            RawRevisionEnvelope(
                "codex:session", RawRevisionKind.FULL, "folded", 2, authority=RawRevisionAuthority.BYTE_PROVEN
            ),
        )
        source = archive._ensure_source_conn()
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
            original = archive.raw_revision_material

            def mutated_material(raw_id: str) -> tuple[Provider, bytes, str, RawRevisionKind]:
                provider, payload, source_path, kind = original(raw_id)
                return (
                    (provider, b"Z" * len(tail), source_path, kind)
                    if raw_id == append
                    else (provider, payload, source_path, kind)
                )

            monkeypatch.setattr(
                archive,
                "raw_revision_material",
                mutated_material,
            )
        source.commit()
        before = state(archive)
        assert before["blocks"]
        assert before["session_events"]
        assert before["attachments"]
        assert before["fts_needle"]
        assert not before["fts_candidate"]
        plan = archive.raw_revision_replay_plan("codex:session")
        with pytest.raises(RuntimeError, match="conflicting accepted head"):
            archive.apply_raw_revision_replay(plan, {folded: folded_session}, acquired_at_ms=0)
        assert state(archive) == before


def test_full_replay_preserves_semantic_head_and_rolls_back_regressions(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)

    def parsed(*messages: tuple[str, str]) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="session",
            messages=[
                ParsedMessage(provider_message_id=message_id, role=Role.USER, text=text)
                for message_id, text in messages
            ],
        )

    def write_full(archive: ArchiveStore, label: str, generation: int) -> str:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=label.encode(),
            source_path="session.json",
            acquired_at_ms=generation,
        )
        archive.bind_raw_revision(
            raw_id,
            RawRevisionEnvelope(
                "codex:session",
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
                   FROM raw_revision_heads WHERE logical_source_key = 'codex:session'"""
            ).fetchone(),
            archive._conn.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone(),
        )

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        base_session = parsed(("m0", "zero"))
        base = write_full(archive, "base", 1)
        archive.apply_raw_membership_classification(
            "codex:session",
            MembershipClassification((base,), (), ()),
            {base: base_session},
            {base: session_revision_projection(base_session)},
            acquired_at_ms=0,
        )

        later_session = parsed(("m0", "zero"), ("m1", "one"), ("m2", "two"))
        later = write_full(archive, "later", 2)
        later_plan = selected_full_plan(later, 2, len("later"))
        archive.apply_raw_revision_replay(later_plan, {later: later_session}, acquired_at_ms=0)

        semantic_head = archive._conn.execute(
            """SELECT accepted_raw_id, accepted_frontier_kind, accepted_frontier
               FROM raw_revision_heads WHERE logical_source_key = 'codex:session'"""
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
                archive.apply_raw_revision_replay(
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
        acquired_at_ms=1,
    )
    archive.replace_raw_membership_census(
        raw_id,
        [session],
        parser_fingerprint="test-parser",
        censused_at_ms=1,
    )
    return raw_id


def _write_chain_full(archive: ArchiveStore, label: str, generation: int) -> str:
    raw_id = archive.write_raw_payload(
        provider=Provider.CODEX,
        payload=label.encode(),
        source_path="session.json",
        acquired_at_ms=generation,
    )
    archive.bind_raw_revision(
        raw_id,
        RawRevisionEnvelope(
            "codex:session",
            RawRevisionKind.FULL,
            f"revision-{label}",
            generation,
            authority=RawRevisionAuthority.BYTE_PROVEN,
        ),
    )
    return raw_id


def _apply_membership_head(archive: ArchiveStore, raw_id: str, session: ParsedSession) -> None:
    archive.apply_raw_membership_classification(
        "codex:session",
        MembershipClassification((raw_id,), (), ()),
        {raw_id: session},
        {raw_id: session_revision_projection(session)},
        acquired_at_ms=0,
    )


def test_batched_membership_success_supersedes_deferred_cas_evidence(tmp_path: Path) -> None:
    """The positive commit-batch route must expire CAS retry authority too."""
    initialize_active_archive_root(tmp_path)
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
        archive.apply_raw_membership_classification(
            "codex:session",
            MembershipClassification((raw_id,), (), ()),
            {raw_id: session},
            {raw_id: session_revision_projection(session)},
            acquired_at_ms=3,
            manage_transaction=False,
        )
        archive.commit()

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
    initialize_active_archive_root(tmp_path)
    session = _parsed_session(("m0", "retained CAS failure"))
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = _write_quarantined_member(archive, "retained-cas-failure", session)

        def raise_conflict(*_args: object, **_kwargs: object) -> None:
            raise archive_revision_governance.MembershipReplayConflictError("retained membership conflict")

        monkeypatch.setattr(archive_revision_governance, "_write_parsed_precedence_result", raise_conflict)
        with pytest.raises(archive_revision_governance.MembershipReplayConflictError):
            archive._index_parsed_for_retained_raw(
                session,
                raw_id=raw_id,
                source_index=0,
                stage_timings_s=None,
                stage_timing_prefix="test",
                manage_transaction=False,
                preacquired_attachment_blobs={},
                finalize_raw_parse=False,
                revision_authoritative=True,
            )

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
           FROM raw_revision_heads WHERE logical_source_key = 'codex:session'"""
    ).fetchone()
    return None if row is None else tuple(row)


def test_chain_replay_supersedes_equal_frontier_quarantined_membership_head(tmp_path: Path) -> None:
    """Capture-vs-export head collision (the v42 rebuild crash): a byte-proven

    chain full at an EQUAL semantic frontier with different content must take
    the head from a quarantined membership (browser-capture) raw instead of
    the CAS rejecting the whole replay.
    """
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        capture_session = _parsed_session(("m0", "zero"), ("m1", "capture flavour"))
        capture = _write_quarantined_member(archive, "capture", capture_session)
        _apply_membership_head(archive, capture, capture_session)
        assert _head_row(archive) == (capture, "semantic", 2)

        export_session = _parsed_session(("m0", "zero"), ("m1", "export flavour"))
        export = _write_chain_full(archive, "export", 2)
        plan = plan_revision_replay([_candidate(export, RawRevisionKind.FULL, 2, size=len("export"))])
        session_id, applied = archive.apply_raw_revision_replay(plan, {export: export_session}, acquired_at_ms=0)

        assert applied == (export,)
        assert _head_row(archive) == (export, "semantic", 2)
        stored = archive._conn.execute(
            "SELECT content_hash FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        assert stored is not None


def test_chain_replay_supersedes_quarantined_membership_head_even_when_capture_has_more_units(tmp_path: Path) -> None:
    """Chain evidence wins unconditionally: a scalar frontier cannot prove a

    capture is a content-superset, so even a capture with MORE semantic units
    hands the head to chain-governed evidence (the capture raw stays in the
    source tier; re-adoption needs a real prefix-dominance proof).
    """
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        capture_session = _parsed_session(("m0", "zero"), ("m1", "one"), ("m2", "two"))
        capture = _write_quarantined_member(archive, "capture", capture_session)
        _apply_membership_head(archive, capture, capture_session)
        assert _head_row(archive) == (capture, "semantic", 3)

        export_session = _parsed_session(("m0", "zero"), ("m1", "one"))
        export = _write_chain_full(archive, "export", 2)
        plan = plan_revision_replay([_candidate(export, RawRevisionKind.FULL, 2, size=len("export"))])
        session_id, applied = archive.apply_raw_revision_replay(plan, {export: export_session}, acquired_at_ms=0)

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
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        export_session = _parsed_session(("m0", "zero"), ("m1", "export flavour"))
        export = _write_chain_full(archive, "export", 1)
        plan = plan_revision_replay([_candidate(export, RawRevisionKind.FULL, 1, size=len("export"))])
        archive.apply_raw_revision_replay(plan, {export: export_session}, acquired_at_ms=0)
        # A chain-first head is byte-kind: its frontier is never comparable to
        # a capture's semantic frontier, so the capture must always yield.
        assert _head_row(archive) == (export, "byte", 6)

        capture_session = _parsed_session(("m0", "zero"), ("m1", "capture flavour"))
        capture = _write_quarantined_member(archive, "capture", capture_session)
        result = archive.apply_raw_membership_classification(
            "codex:session",
            MembershipClassification((capture,), (), ()),
            {capture: capture_session},
            {capture: session_revision_projection(capture_session)},
            acquired_at_ms=0,
        )

        assert result == "codex-session:session"
        assert _head_row(archive) == (export, "byte", 6)
        receipts = archive._conn.execute(
            """SELECT decision, detail FROM raw_revision_applications
               WHERE raw_id = ? AND logical_source_key = 'codex:session'""",
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
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        export_session = _parsed_session(("m0", "zero"), ("m1", "export flavour"))
        export = _write_chain_full(archive, "export", 1)
        plan = plan_revision_replay([_candidate(export, RawRevisionKind.FULL, 1, size=len("export"))])
        archive.apply_raw_revision_replay(plan, {export: export_session}, acquired_at_ms=0)

        capture_session = _parsed_session(("m0", "zero"), ("m1", "capture flavour"))
        capture = _write_quarantined_member(archive, "capture", capture_session)
        archive._conn.execute(
            "UPDATE raw_revision_heads SET accepted_raw_id = ?, accepted_frontier_kind = 'semantic', accepted_frontier = 2 WHERE logical_source_key = 'codex:session'",
            (capture,),
        )

        archive.apply_raw_membership_classification(
            "codex:session",
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
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        capture1_session = _parsed_session(("m0", "zero"))
        capture1 = _write_quarantined_member(archive, "capture1", capture1_session)
        _apply_membership_head(archive, capture1, capture1_session)
        export_session = _parsed_session(("m0", "zero"))
        export = _write_chain_full(archive, "export", 2)
        plan = plan_revision_replay([_candidate(export, RawRevisionKind.FULL, 2, size=len("export"))])
        archive.apply_raw_revision_replay(plan, {export: export_session}, acquired_at_ms=0)
        assert _head_row(archive) == (export, "semantic", 1)

        capture2_session = _parsed_session(("m0", "zero"), ("m1", "the conversation continued"))
        capture2 = _write_quarantined_member(archive, "capture2", capture2_session)
        revisions = [
            MembershipRevision(capture1, session_revision_projection(capture1_session)),
            MembershipRevision(capture2, session_revision_projection(capture2_session)),
        ]
        classification = classify_membership_revisions(revisions)
        assert capture2 in classification.accepted_raw_ids
        archive.apply_raw_membership_classification(
            "codex:session",
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
               WHERE raw_id = ? AND logical_source_key = 'codex:session'""",
            (capture2,),
        ).fetchall()
        assert [str(row[0]) for row in receipts] == ["superseded"]
        assert f"superseded_by_chain_governed_head:{export}" in str(receipts[0][1])
        stored = archive._conn.execute(
            "SELECT content_hash FROM sessions WHERE session_id = 'codex-session:session'"
        ).fetchone()
        assert stored is not None
        assert bytes(stored[0]).hex() == session_content_hash(export_session)


def test_skip_already_applied_indexes_only_new_tail_of_append_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-de2a: the live watcher's per-append replay must not re-index
    every historical append on every new one.

    Simulates the exact live-watcher shape: baseline arrives, then two more
    appends arrive one at a time, each replaying the WHOLE accumulated chain
    (as the real ``append_ingest.py`` hot path always does -- it always
    passes ``parsed_by_raw_id`` for the entire ``plan.accepted_raw_ids``).
    With ``skip_already_applied=True``, only the newly accepted tail raw_id
    should reach ``_index_parsed_for_retained_raw`` on the 2nd and 3rd calls
    -- not the whole chain again. Without the fix, every call re-indexes
    every position, an O(n) cost per append that made the daemon's writer
    gate hold for minutes to hours as a session's append count grew
    (confirmed root cause; see ``apply_raw_revision_replay``'s
    ``skip_already_applied`` docstring).
    """
    initialize_active_archive_root(tmp_path)

    def parsed(*messages: tuple[str, str]) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="session",
            messages=[
                ParsedMessage(provider_message_id=message_id, role=Role.USER, text=text)
                for message_id, text in messages
            ],
        )

    indexed_raw_ids: list[str] = []
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
        indexed_raw_ids.append(raw_id)
        return original(store, session, raw_id=raw_id, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(archive_revision_governance, "_index_parsed_for_retained_raw", spy)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        baseline = archive.write_raw_payload(
            provider=Provider.CODEX, payload=b"a" * 10, source_path="session.jsonl", acquired_at_ms=1
        )
        archive.bind_raw_revision(
            baseline,
            RawRevisionEnvelope(
                "codex:session", RawRevisionKind.FULL, "full-0", 0, authority=RawRevisionAuthority.BYTE_PROVEN
            ),
        )
        plan0 = archive.classify_raw_revision_cohort_for_live_watch("codex:session")
        archive.apply_raw_revision_replay(
            plan0, {baseline: parsed(("m0", "zero"))}, acquired_at_ms=0, skip_already_applied=True
        )
        assert indexed_raw_ids == [baseline]
        indexed_raw_ids.clear()

        append_one = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"b" * 5,
            source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=2,
        )
        archive.bind_raw_revision(
            append_one,
            RawRevisionEnvelope(
                "codex:session",
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
        plan1 = archive.classify_raw_revision_cohort_for_live_watch("codex:session")
        assert plan1.accepted_raw_ids == (baseline, append_one)
        # The real live-watcher hot path always reparses+passes the FULL
        # accepted chain (see append_ingest.py), not just the new tail.
        archive.apply_raw_revision_replay(
            plan1,
            {baseline: parsed(("m0", "zero")), append_one: parsed(("m1", "one"))},
            acquired_at_ms=0,
            skip_already_applied=True,
        )
        # Only the NEW tail position was actually indexed -- the baseline
        # was already durably written by the first call above.
        assert indexed_raw_ids == [append_one]
        indexed_raw_ids.clear()

        append_two = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"c" * 5,
            source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=3,
        )
        archive.bind_raw_revision(
            append_two,
            RawRevisionEnvelope(
                "codex:session",
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
        plan2 = archive.classify_raw_revision_cohort_for_live_watch("codex:session")
        assert plan2.accepted_raw_ids == (baseline, append_one, append_two)
        archive.apply_raw_revision_replay(
            plan2,
            {
                baseline: parsed(("m0", "zero")),
                append_one: parsed(("m1", "one")),
                append_two: parsed(("m2", "two")),
            },
            acquired_at_ms=0,
            skip_already_applied=True,
        )
        # Again: only the newest position, not the two already-applied ones.
        assert indexed_raw_ids == [append_two]
        indexed_raw_ids.clear()

        # Correctness is unaffected by skipping the already-applied writes:
        # every message from every accepted position is still present.
        rows = archive._conn.execute(
            "SELECT block_type, search_text FROM blocks JOIN messages USING (message_id)"
            " WHERE messages.session_id = 'codex-session:session' ORDER BY messages.position"
        ).fetchall()
        texts = [str(row[1]) for row in rows]
        assert any("zero" in text for text in texts)
        assert any("one" in text for text in texts)
        assert any("two" in text for text in texts)

        head = archive._conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'codex:session'"
        ).fetchone()
        assert head is not None
        assert head[0] == append_two


def test_skip_already_applied_default_false_still_reindexes_whole_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backfill/restore/membership callers do not pass ``skip_already_applied``
    and must keep today's full self-healing re-apply of every historical
    position -- only the live-append hot path opts into the fast tail-only
    mode.
    """
    initialize_active_archive_root(tmp_path)

    def parsed(*messages: tuple[str, str]) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="session",
            messages=[
                ParsedMessage(provider_message_id=message_id, role=Role.USER, text=text)
                for message_id, text in messages
            ],
        )

    indexed_raw_ids: list[str] = []
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
        indexed_raw_ids.append(raw_id)
        return original(store, session, raw_id=raw_id, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(archive_revision_governance, "_index_parsed_for_retained_raw", spy)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        baseline = archive.write_raw_payload(
            provider=Provider.CODEX, payload=b"a" * 10, source_path="session.jsonl", acquired_at_ms=1
        )
        archive.bind_raw_revision(
            baseline,
            RawRevisionEnvelope(
                "codex:session", RawRevisionKind.FULL, "full-0", 0, authority=RawRevisionAuthority.BYTE_PROVEN
            ),
        )
        plan0 = archive.classify_raw_revision_cohort_for_live_watch("codex:session")
        archive.apply_raw_revision_replay(plan0, {baseline: parsed(("m0", "zero"))}, acquired_at_ms=0)
        indexed_raw_ids.clear()

        append_one = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"b" * 5,
            source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=2,
        )
        archive.bind_raw_revision(
            append_one,
            RawRevisionEnvelope(
                "codex:session",
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
        plan1 = archive.classify_raw_revision_cohort_for_live_watch("codex:session")
        archive.apply_raw_revision_replay(
            plan1,
            {baseline: parsed(("m0", "zero")), append_one: parsed(("m1", "one"))},
            acquired_at_ms=0,
        )
        # Default (no skip): the whole chain is re-indexed, exactly as before
        # this fix -- pinning that non-opted-in callers are unaffected.
        assert indexed_raw_ids == [baseline, append_one]
