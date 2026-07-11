from __future__ import annotations

from itertools import permutations
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.archive.revision_replay import (
    ApplicationDecision,
    RevisionCandidate,
    RevisionReplayPlan,
    plan_revision_replay,
)
from polylogue.archive.session_revision_membership import MembershipClassification
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_revision_projection
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
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

        plan = archive.classify_raw_revision_cohort("codex:session")

    assert {item.raw_id: item.decision for item in plan.applications} == {
        baseline_raw_id: ApplicationDecision.SELECTED_BASELINE,
        append_raw_id: ApplicationDecision.APPLIED_APPEND,
    }


def test_real_append_chain_accepts_equivalent_same_end_full(tmp_path: Path) -> None:
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
            RawRevisionEnvelope("codex:session", RawRevisionKind.FULL, "full-0", 0),
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
                "append-1",
                0,
                predecessor_source_revision="full-0",
                append_start_offset=10,
                append_end_offset=15,
                authority=RawRevisionAuthority.QUARANTINED,
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
                "append-2",
                0,
                predecessor_source_revision="append-1",
                append_start_offset=15,
                append_end_offset=20,
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )
        append_plan = archive.classify_raw_revision_cohort("codex:session")
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
            provider=Provider.CODEX, payload=b"a" * 20, source_path="session.jsonl", acquired_at_ms=4
        )
        archive.bind_raw_revision(
            folded,
            RawRevisionEnvelope("codex:session", RawRevisionKind.FULL, "full-folded", 0),
        )
        folded_plan = archive.classify_raw_revision_cohort("codex:session")
        archive.apply_raw_revision_replay(
            folded_plan,
            {folded: parsed(("m0", "zero"), ("m1", "one"), ("m2", "two"))},
            acquired_at_ms=0,
        )

        head = archive._conn.execute(
            "SELECT accepted_raw_id, accepted_frontier FROM raw_revision_heads WHERE logical_source_key = ?",
            ("codex:session",),
        ).fetchone()
        assert head is not None
        assert tuple(head) == (folded, 20)


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
