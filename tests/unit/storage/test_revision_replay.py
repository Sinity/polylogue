from __future__ import annotations

import hashlib
import json
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
from polylogue.pipeline.ids import session_content_hash, session_revision_projection
from polylogue.sources.dispatch import merge_parsed_session_chunks, parse_stream_payload
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
        folded_plan = archive.classify_raw_revision_cohort("codex:session")
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


def test_real_single_append_chain_folds_segmentation_distinct_full_snapshot(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        baseline_payload, tail = _codex_fold_payloads()
        baseline_session = _parse_codex_jsonl(baseline_payload)
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
            "head": archive._conn.execute(
                "SELECT accepted_raw_id, accepted_content_hash, accepted_frontier FROM raw_revision_heads"
            ).fetchall(),
            "receipts": archive._conn.execute(
                "SELECT decision_id FROM raw_revision_applications ORDER BY decision_id"
            ).fetchall(),
        }

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        baseline_payload, tail = _codex_fold_payloads()
        baseline_session = _parse_codex_jsonl(baseline_payload)
        append_session = _parse_codex_jsonl(tail)
        folded_session = _parse_codex_jsonl(baseline_payload + tail)
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
        folded_payload = baseline_payload + tail
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
        assert before["fts_needle"]
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
