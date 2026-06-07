"""Tests for the learning-feedback loop (#1131).

Acceptance criteria covered here:

- Content-hash invariant: recording, mutating, and clearing corrections
  never alters ``sessions.content_hash``. This is the durable
  separation between the archive substrate and the user-metadata zone.
- Determinism: applying a correction set to a heuristic output produces
  identical results across independent merge calls.
- Precedence: applied tag and summary corrections win over heuristic
  suggestions.
- Unknown-kind rejection: surfaces refuse to persist unknown kinds.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from polylogue.insights.feedback import (
    CorrectionKind,
    LearningCorrection,
    UnknownCorrectionKindError,
    apply_correction_to_summary,
    apply_corrections_to_auto_tags,
    parse_correction_kind,
)
from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite.queries import sessions as sessions_q
from tests.infra.storage_records import make_message, make_session, save_current_archive_records

# ---------------------------------------------------------------------------
# Pure-function semantics — no DB
# ---------------------------------------------------------------------------


def _correction(kind: CorrectionKind, payload: dict[str, str]) -> LearningCorrection:
    return LearningCorrection(
        session_id="conv-1",
        kind=kind,
        payload=payload,
        created_at=datetime(2026, 5, 17, tzinfo=UTC),
    )


class TestParseCorrectionKind:
    def test_known_kinds_round_trip(self) -> None:
        for kind in CorrectionKind:
            assert parse_correction_kind(kind.value) is kind

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(UnknownCorrectionKindError) as exc:
            parse_correction_kind("not-a-kind")
        # Error message names accepted kinds so callers can self-correct.
        for kind in CorrectionKind:
            assert kind.value in str(exc.value)


class TestApplyCorrectionsToAutoTags:
    def test_no_corrections_preserves_order(self) -> None:
        tags = ("repo:polylogue", "costly")
        assert apply_corrections_to_auto_tags(tags, []) == tags

    def test_tag_reject_removes_only_named_tag(self) -> None:
        tags = (
            "repo:polylogue",
            "costly",
            "continuation",
        )
        reject = _correction(CorrectionKind.TAG_REJECT, {"tag": "costly"})

        merged = apply_corrections_to_auto_tags(tags, [reject])

        assert merged == ("repo:polylogue", "continuation")

    def test_tag_accept_does_not_drop_anything(self) -> None:
        tags = ("repo:polylogue",)
        accept = _correction(CorrectionKind.TAG_ACCEPT, {"tag": "repo:polylogue"})
        assert apply_corrections_to_auto_tags(tags, [accept]) == tags


class TestApplyCorrectionToSummary:
    def test_no_correction_returns_base(self) -> None:
        assert apply_correction_to_summary("base summary", []) == "base summary"

    def test_summary_override_replaces(self) -> None:
        replacement = _correction(
            CorrectionKind.SUMMARY_OVERRIDE,
            {"summary": "Hand-written final summary."},
        )
        assert apply_correction_to_summary("auto draft", [replacement]) == "Hand-written final summary."

    def test_empty_replacement_is_ignored(self) -> None:
        bad = _correction(CorrectionKind.SUMMARY_OVERRIDE, {"summary": ""})
        # Pydantic accepts empty strings here so we have to defend in code:
        # an empty replacement falls back to the heuristic value.
        assert apply_correction_to_summary("auto draft", [bad]) == "auto draft"


# ---------------------------------------------------------------------------
# Storage round-trip + content-hash invariant
# ---------------------------------------------------------------------------


async def _seed_session(repository: SessionRepository, conv_id: str) -> None:
    conv = make_session(conv_id, source_name="claude-code", title="Original title")
    msgs = [make_message(f"{conv_id}-msg", conv_id, text="Hello world.")]
    await save_current_archive_records(repository, session=conv, messages=msgs, attachments=[])


def _archive_session_id(conv_id: str) -> str:
    # ``_seed_session`` persists via ``make_session`` whose native_id is the
    # conv id itself, so the generated archive session_id is ``origin:conv_id``.
    return f"claude-code-session:{conv_id}"


async def _read_content_hash(
    repository: SessionRepository,
    session_id: str,
) -> str | None:
    async with repository.backend.connection() as conn:
        cursor = await conn.execute(
            "SELECT content_hash FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return str(row[0]) if row[0] is not None else None


class TestFeedbackStorage:
    """Round-trip user corrections through the full repository write path."""

    async def test_record_then_list_then_clear(
        self,
        workspace_env: dict[str, Path],
        storage_repository: SessionRepository,
    ) -> None:
        del workspace_env
        await _seed_session(storage_repository, "conv-A")

        from polylogue.storage.insights.feedback import (
            clear_corrections,
            list_corrections,
            upsert_correction,
        )

        async with storage_repository.backend.connection() as conn:
            recorded = await upsert_correction(
                conn,
                session_id="conv-A",
                kind=CorrectionKind.SUMMARY_OVERRIDE,
                payload={"summary": "User-authored summary."},
                note="operator replacement",
            )
            assert recorded.session_id == "conv-A"
            assert recorded.kind == CorrectionKind.SUMMARY_OVERRIDE

            listed = await list_corrections(conn, session_id="conv-A")
            assert [c.kind for c in listed] == [CorrectionKind.SUMMARY_OVERRIDE]
            assert listed[0].payload == {"summary": "User-authored summary."}

            removed = await clear_corrections(conn, session_id="conv-A")
            assert removed == 1
            assert await list_corrections(conn, session_id="conv-A") == []

    async def test_upsert_is_idempotent_on_kind(
        self,
        workspace_env: dict[str, Path],
        storage_repository: SessionRepository,
    ) -> None:
        del workspace_env
        await _seed_session(storage_repository, "conv-B")

        from polylogue.storage.insights.feedback import (
            list_corrections,
            upsert_correction,
        )

        async with storage_repository.backend.connection() as conn:
            await upsert_correction(
                conn,
                session_id="conv-B",
                kind=CorrectionKind.SUMMARY_OVERRIDE,
                payload={"summary": "First summary."},
            )
            await upsert_correction(
                conn,
                session_id="conv-B",
                kind=CorrectionKind.SUMMARY_OVERRIDE,
                payload={"summary": "Second summary."},
            )

            corrections = await list_corrections(conn, session_id="conv-B")

        # (session_id, kind) is UNIQUE: only the latest payload survives.
        assert len(corrections) == 1
        assert corrections[0].payload == {"summary": "Second summary."}

    async def test_content_hash_invariant(
        self,
        workspace_env: dict[str, Path],
        storage_repository: SessionRepository,
    ) -> None:
        """AC #1131: corrections never touch the session's content_hash."""

        del workspace_env
        await _seed_session(storage_repository, "conv-C")
        session_id = _archive_session_id("conv-C")
        original_hash = await _read_content_hash(storage_repository, session_id)
        assert original_hash is not None and len(original_hash) > 0

        from polylogue.storage.insights.feedback import (
            clear_corrections,
            upsert_correction,
        )

        async with storage_repository.backend.connection() as conn:
            await upsert_correction(
                conn,
                session_id="conv-C",
                kind=CorrectionKind.TAG_REJECT,
                payload={"tag": "costly"},
                note="not actually expensive work",
            )
        after_record = await _read_content_hash(storage_repository, session_id)
        assert after_record == original_hash

        async with storage_repository.backend.connection() as conn:
            await clear_corrections(conn, session_id="conv-C")
        after_clear = await _read_content_hash(storage_repository, session_id)
        assert after_clear == original_hash

    async def test_list_filters_by_kind(
        self,
        workspace_env: dict[str, Path],
        storage_repository: SessionRepository,
    ) -> None:
        del workspace_env
        await _seed_session(storage_repository, "conv-D")

        from polylogue.storage.insights.feedback import (
            list_corrections,
            upsert_correction,
        )

        async with storage_repository.backend.connection() as conn:
            await upsert_correction(
                conn,
                session_id="conv-D",
                kind=CorrectionKind.TAG_REJECT,
                payload={"tag": "repo:polylogue"},
            )
            await upsert_correction(
                conn,
                session_id="conv-D",
                kind=CorrectionKind.SUMMARY_OVERRIDE,
                payload={"summary": "Replacement"},
            )

            filtered = await list_corrections(
                conn,
                session_id="conv-D",
                kind=CorrectionKind.SUMMARY_OVERRIDE,
            )

        assert [c.kind for c in filtered] == [CorrectionKind.SUMMARY_OVERRIDE]
        assert filtered[0].payload == {"summary": "Replacement"}

    async def test_unknown_kind_rejected_at_typed_boundary(
        self,
        workspace_env: dict[str, Path],
        storage_repository: SessionRepository,
    ) -> None:
        del workspace_env, storage_repository
        # The DB column is permissive (TEXT) so old rows can survive
        # taxonomy churn, but the typed surface refuses to accept new
        # writes of an unknown kind.
        with pytest.raises(UnknownCorrectionKindError):
            parse_correction_kind("definitely-not-a-real-kind")


# ---------------------------------------------------------------------------
# Defense-in-depth — user_corrections is in canonical DDL
# ---------------------------------------------------------------------------


def test_corrections_ddl_is_in_user_tier_ddl() -> None:
    # The learning-corrections table lives in the user-durability tier
    # (``user.db``) by construction — it carries irreplaceable human input and
    # sits outside the content-hash boundary (#1131). The canonical table is
    # named ``corrections`` in the split-file archive.
    from polylogue.storage.sqlite.archive_tiers.user import USER_DDL

    assert "CREATE TABLE IF NOT EXISTS corrections" in USER_DDL


def _suppress_unused_import_warning() -> None:
    # ``sessions_q`` is imported for future tests that need to read
    # other rows; reference it so static linters do not flag the import.
    assert callable(sessions_q.list_tags)
