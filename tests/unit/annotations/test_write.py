"""Roundtrip tests for the schema-validated annotation write path (polylogue-rxdo.7).

Covers the bead's acceptance-criteria shape end to end at the single-row
Python-API level: schema-validated candidate write -> query back via a typed
``value.<path>`` DSL predicate -> judge accept -> active row visible. The
JSONL/batch import surface itself is deferred (see module docstrings in
``polylogue/annotations/``); this exercises the atomic operation that surface
would loop over.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.annotations.schema import AnnotationField, AnnotationSchema
from polylogue.annotations.write import (
    AnnotationValidationError,
    assertion_id_for_schema_annotation,
    upsert_annotation_assertion,
)
from polylogue.archive.query.expression import parse_unit_source_expression
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import (
    assertion_id_for_annotation,
    judge_assertion_candidate,
    read_assertion_envelope,
    upsert_annotation,
)
from tests.infra.storage_records import SessionBuilder


def _value_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _delegation_tone_schema(**overrides: object) -> AnnotationSchema:
    defaults: dict[str, object] = {
        "schema_id": "test.delegation-tone",
        "version": 1,
        "title": "Test delegation tone",
        "fields": (
            AnnotationField(name="score", value_type="integer", minimum=1, maximum=5),
            AnnotationField(name="status", value_type="enum", enum_values=("approved", "rejected")),
            AnnotationField(name="abstain", value_type="boolean", required=False),
        ),
        "target_ref_kinds": ("session",),
        "abstain_field": "abstain",
        "evidence_policy": "required",
    }
    defaults.update(overrides)
    return AnnotationSchema(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def user_conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    user_db = tmp_path / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    conn = sqlite3.connect(user_db)
    yield conn
    conn.close()


class TestUpsertAnnotationAssertion:
    def test_valid_row_lands_candidate_with_schema_stamp(self, user_conn: sqlite3.Connection) -> None:
        schema = _delegation_tone_schema()
        envelope = upsert_annotation_assertion(
            user_conn,
            schema=schema,
            target_ref="session:codex-session:demo",
            value={"score": 5, "status": "approved"},
            row_key="row-1",
            evidence_refs=["session:codex-session:demo"],
            author_ref="agent:labeler",
            author_kind="agent",
            now_ms=1_000,
        )
        user_conn.commit()

        assert envelope.status == "candidate"
        assert envelope.context_policy.get("inject") is False
        value = _value_dict(envelope.value)
        assert value["_schema"] == "test.delegation-tone@v1"
        assert value["score"] == 5
        assert value["status"] == "approved"
        assert envelope.kind == "annotation"

    def test_invalid_row_raises_and_writes_nothing(self, user_conn: sqlite3.Connection) -> None:
        schema = _delegation_tone_schema()
        with pytest.raises(AnnotationValidationError) as excinfo:
            upsert_annotation_assertion(
                user_conn,
                schema=schema,
                target_ref="session:codex-session:demo",
                value={"score": 99, "status": "approved"},
                row_key="row-bad",
                evidence_refs=["session:codex-session:demo"],
                author_ref="agent:labeler",
                author_kind="agent",
            )
        assert any("score" in error for error in excinfo.value.errors)

        assertion_id = assertion_id_for_schema_annotation(
            schema_qualified_id=schema.qualified_id,
            target_ref="session:codex-session:demo",
            author_ref="agent:labeler",
            row_key="row-bad",
        )
        assert read_assertion_envelope(user_conn, assertion_id) is None

    def test_missing_required_evidence_rejected(self, user_conn: sqlite3.Connection) -> None:
        schema = _delegation_tone_schema()
        with pytest.raises(AnnotationValidationError) as excinfo:
            upsert_annotation_assertion(
                user_conn,
                schema=schema,
                target_ref="session:codex-session:demo",
                value={"score": 4, "status": "approved"},
                row_key="row-no-evidence",
                evidence_refs=(),
                author_ref="agent:labeler",
                author_kind="agent",
            )
        assert any("evidence" in error for error in excinfo.value.errors)

    def test_wrong_target_kind_rejected(self, user_conn: sqlite3.Connection) -> None:
        schema = _delegation_tone_schema()
        with pytest.raises(AnnotationValidationError) as excinfo:
            upsert_annotation_assertion(
                user_conn,
                schema=schema,
                target_ref="block:codex-session:demo:0",
                value={"score": 4, "status": "approved"},
                row_key="row-wrong-kind",
                evidence_refs=["session:codex-session:demo"],
                author_ref="agent:labeler",
                author_kind="agent",
            )
        assert any("target_ref_kinds" in error for error in excinfo.value.errors)

    def test_abstain_row_skips_required_fields(self, user_conn: sqlite3.Connection) -> None:
        schema = _delegation_tone_schema()
        envelope = upsert_annotation_assertion(
            user_conn,
            schema=schema,
            target_ref="session:codex-session:demo",
            value={"abstain": True},
            row_key="row-abstain",
            evidence_refs=["session:codex-session:demo"],
            author_ref="agent:labeler",
            author_kind="agent",
        )
        assert envelope.status == "candidate"
        assert _value_dict(envelope.value)["abstain"] is True

    def test_user_authored_row_is_not_forced_candidate(self, user_conn: sqlite3.Connection) -> None:
        # 37t.15 chokepoint sanity: this write helper does not itself request a
        # status, so a genuinely user-authored row is free to land active if
        # the caller also supplies status explicitly downstream. Default
        # author_kind here is "agent" -- switching to "user" exercises the
        # chokepoint's other branch without this helper hardcoding either.
        schema = _delegation_tone_schema()
        envelope = upsert_annotation_assertion(
            user_conn,
            schema=schema,
            target_ref="session:codex-session:demo",
            value={"score": 3, "status": "approved"},
            row_key="row-user",
            evidence_refs=["session:codex-session:demo"],
            author_ref="user:operator",
            author_kind="user",
        )
        # upsert_assertion defaults status to "active" when the caller omits
        # it and author_kind == "user" (the trusted-author default), so this
        # documents that the write helper itself makes no candidacy promise
        # -- promotion policy is entirely upsert_assertion's chokepoint.
        assert envelope.status == "active"

    def test_deterministic_id_distinct_from_freeform_annotation_helper(self) -> None:
        schema = _delegation_tone_schema()
        schema_annotation_id = assertion_id_for_schema_annotation(
            schema_qualified_id=schema.qualified_id,
            target_ref="session:codex-session:demo",
            author_ref="agent:labeler",
            row_key="row-1",
        )
        freeform_annotation_id = assertion_id_for_annotation("row-1")
        assert schema_annotation_id != freeform_annotation_id


class TestAnnotationRoundtripWithQueryAndJudge:
    def test_import_query_judge_roundtrip(self, tmp_path: Path) -> None:
        """Bead AC roundtrip shape: import candidates, query via typed value
        predicate, judge one active -- at the single-row Python-API level."""

        index_db = tmp_path / "index.db"
        (
            SessionBuilder(index_db, "demo")
            .provider("codex")
            .git_repository_url("polylogue")
            .title("delegation session")
            .add_message("m1", role="assistant", text="dispatching work")
            .save()
        )
        user_db = tmp_path / "user.db"
        initialize_archive_database(user_db, ArchiveTier.USER)

        schema = _delegation_tone_schema()
        rows: list[dict[str, object]] = [
            {"score": 5, "status": "approved"},
            {"score": 4, "status": "approved"},
            {"score": 2, "status": "rejected"},
            {"score": 1, "status": "rejected"},
            {"abstain": True},
        ]
        target_ref = "session:codex-session:demo"
        envelopes = []
        with sqlite3.connect(user_db) as conn:
            for index, row_value in enumerate(rows):
                envelopes.append(
                    upsert_annotation_assertion(
                        conn,
                        schema=schema,
                        target_ref=target_ref,
                        value=row_value,
                        row_key=f"row-{index}",
                        evidence_refs=[target_ref],
                        author_ref="agent:labeler-1",
                        author_kind="agent",
                        now_ms=1_000 + index,
                    )
                )
            conn.commit()
            assert all(envelope.status == "candidate" for envelope in envelopes)

        with ArchiveStore.open_existing(tmp_path) as archive:
            high_score_source = parse_unit_source_expression("assertions where kind:annotation AND value.score:>=4")
            assert high_score_source is not None
            high_score_rows = archive.query_assertions(high_score_source.predicate, limit=100)
            assert {row.assertion_id for row in high_score_rows} == {
                envelopes[0].assertion_id,
                envelopes[1].assertion_id,
            }

            approved_source = parse_unit_source_expression("assertions where kind:annotation AND value.status:approved")
            assert approved_source is not None
            approved_rows = archive.query_assertions(approved_source.predicate, limit=100)
            assert len(approved_rows) == 2

        with sqlite3.connect(user_db) as conn:
            judged = judge_assertion_candidate(
                conn,
                candidate_ref=f"assertion:{envelopes[0].assertion_id}",
                decision="accept",
                actor_ref="user:operator",
                reason="clear high-quality label",
            )
            conn.commit()

        resulting_assertion = judged.resulting_assertion
        assert resulting_assertion is not None
        assert resulting_assertion.status == "active"
        assert _value_dict(resulting_assertion.value)["score"] == 5

        with ArchiveStore.open_existing(tmp_path) as archive:
            active_source = parse_unit_source_expression(
                "assertions where kind:annotation AND status:active AND value.score:>=4"
            )
            assert active_source is not None
            active_rows = archive.query_assertions(active_source.predicate, limit=100)
            assert [row.assertion_id for row in active_rows] == [resulting_assertion.assertion_id]


def test_upsert_annotation_freeform_note_still_works_independently_of_schema_path(tmp_path: Path) -> None:
    """Guard against the two 'annotation' concepts (freeform note vs schema-
    validated label) colliding on write path or identity."""

    user_db = tmp_path / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    with sqlite3.connect(user_db) as conn:
        envelope = upsert_annotation(conn, "session", "codex-session:demo", "a plain operator note")
        conn.commit()
    assert envelope.body == "a plain operator note"
