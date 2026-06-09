"""Tests for #839 AC #2: backfill pre-#839 message_type rows."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.storage.message_type_backfill import (
    count_messages_by_type_sync,
    count_unclassified_message_type_sync,
)
from polylogue.storage.repair import repair_message_type_backfill
from tests.infra.archive_scenarios import open_index_db
from tests.infra.storage_records import SessionBuilder, db_setup

# Native ``messages.native_id`` keeps the provider-native message id verbatim
# (the builder seeds ``provider_message_id = <message_id>``), so tests query
# the native tree by ``native_id`` rather than the generated
# ``<session_id>:<native_id>`` ``message_id``.


def _make_db_with_messages(db_path: Path) -> str:
    """Create a database with test messages, return conv_id."""
    conv_id = "conv-backfill-1"
    builder = SessionBuilder(db_path, conv_id)
    builder.provider("chatgpt").title("Backfill Test")

    # Context markers (#839 context set)
    builder.add_message(
        message_id="ctx-1",
        role="user",
        text="<environment_context>\n<cwd>/x</cwd>\n</environment_context>",
        message_type="message",
    )
    builder.add_message(
        message_id="ctx-2",
        role="user",
        text="<system-reminder>ignored</system-reminder>\n\n<system>\nYou are an agent\n</system>",
        message_type="message",
    )
    builder.add_message(
        message_id="ctx-3",
        role="user",
        text="Base directory for this skill: /tmp/skill\n\n# Instructions",
        message_type="message",
    )

    # Protocol markers (#839 protocol set)
    builder.add_message(
        message_id="proto-1",
        role="user",
        text="<bash-input>ls</bash-input>",
        message_type="message",
    )
    builder.add_message(
        message_id="proto-2",
        role="user",
        text="<task-notification><status>completed</status></task-notification>",
        message_type="message",
    )

    # Plain messages (should stay 'message')
    builder.add_message(
        message_id="plain-1",
        role="user",
        text="Hello, how are you?",
        message_type="message",
    )
    builder.add_message(
        message_id="plain-2",
        role="assistant",
        text="I'm doing well, thanks!",
        message_type="message",
    )

    # Already-classified messages (should not change)
    builder.add_message(
        message_id="already-done",
        role="user",
        text="<environment_context>\n<more/>\n</environment_context>",
        message_type="context",
    )

    builder.save()
    return conv_id


def _make_config(workspace_env: dict[str, Path], db_path: Path) -> Config:
    """Create a minimal Config for tests."""
    return Config(
        archive_root=Path(workspace_env["archive_root"]),
        render_root=Path(workspace_env["archive_root"]),
        sources=[],
        db_path=db_path,
    )


class TestMessageTypeBackfill:
    """Integration tests for the #839 message_type backfill repair."""

    def test_backfill_classes_context_messages(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Messages matching context markers get message_type = 'context'."""
        db_path = db_setup(workspace_env)
        _make_db_with_messages(db_path)

        cfg = _make_config(workspace_env, db_path)

        result = repair_message_type_backfill(cfg, dry_run=False)
        assert result.success, result.detail
        assert result.repaired_count > 0

        with open_index_db(db_path) as conn:
            # Context messages classified
            for mid in ("ctx-1", "ctx-2", "ctx-3"):
                row = conn.execute(
                    "SELECT message_type FROM messages WHERE native_id = ?",
                    (f"{mid}",),
                ).fetchone()
                assert row is not None, f"Missing message {mid}"
                assert row[0] == "context", f"{mid}: expected context, got {row[0]}"

            # Protocol messages classified
            for mid in ("proto-1", "proto-2"):
                row = conn.execute(
                    "SELECT message_type FROM messages WHERE native_id = ?",
                    (f"{mid}",),
                ).fetchone()
                assert row is not None, f"Missing message {mid}"
                assert row[0] == "protocol", f"{mid}: expected protocol, got {row[0]}"

            # Plain messages unchanged
            for mid in ("plain-1", "plain-2"):
                row = conn.execute(
                    "SELECT message_type FROM messages WHERE native_id = ?",
                    (f"{mid}",),
                ).fetchone()
                assert row is not None, f"Missing message {mid}"
                assert row[0] == "message", f"{mid}: expected message, got {row[0]}"

            # Already-classified message unchanged
            row = conn.execute(
                "SELECT message_type FROM messages WHERE native_id = ?",
                ("already-done",),
            ).fetchone()
            assert row is not None
            assert row[0] == "context", f"already-classified: expected context, got {row[0]}"

    def test_backfill_is_idempotent(self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
        """A second run of the backfill is a no-op."""
        db_path = db_setup(workspace_env)
        _make_db_with_messages(db_path)

        cfg = _make_config(workspace_env, db_path)

        # First run
        result1 = repair_message_type_backfill(cfg, dry_run=False)
        assert result1.success
        first_count = result1.repaired_count
        assert first_count > 0

        # Second run: should be a no-op
        result2 = repair_message_type_backfill(cfg, dry_run=False)
        assert result2.success
        assert result2.repaired_count == 0, f"Second run should be no-op, but repaired {result2.repaired_count}"

    def test_backfill_matches_pipeline_classification(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backfill uses the same classify_text_message_type as new-ingest pipeline."""
        from polylogue.archive.message.artifacts import classify_text_message_type
        from polylogue.archive.message.types import MessageType

        test_cases = [
            ("<environment_context>\n<cwd>/x</cwd>\n</environment_context>", MessageType.CONTEXT),
            (
                "<system-reminder>ignored</system-reminder>\n\n<system>\nYou are an agent\n</system>",
                MessageType.CONTEXT,
            ),
            ("Base directory for this skill: /tmp/skill\n\n# body", MessageType.CONTEXT),
            ("<bash-input>ls</bash-input>", MessageType.PROTOCOL),
            ("<task-notification><status>done</status></task-notification>", MessageType.PROTOCOL),
            (
                "Caveat: The messages below were generated by the user while running local commands.",
                MessageType.PROTOCOL,
            ),
            ("Hello, how are you?", None),
            ("I'm an assistant response.", None),
            ("", None),
            (None, None),
        ]

        for text, expected in test_cases:
            result = classify_text_message_type(text)
            assert result == expected, f"classify({text!r}) = {result}, expected {expected}"

    def test_dry_run_does_not_mutate(self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
        """Dry-run reports count but does not change any rows."""
        db_path = db_setup(workspace_env)
        _make_db_with_messages(db_path)

        cfg = _make_config(workspace_env, db_path)

        result = repair_message_type_backfill(cfg, dry_run=True)
        assert result.success
        assert result.repaired_count > 0, "Dry-run should count candidate rows"

        # Verify no rows were mutated
        with open_index_db(db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE message_type IN ('context', 'protocol')"
            ).fetchone()[0]
            assert count == 1, f"Only the pre-existing 'context' row should exist, found {count}"

    def test_before_after_artifact_class_counts(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """#839 AC #6: before/after counts move from default ``message`` into
        ``context`` and ``protocol`` exactly by the rows the classifier
        matches, with the backfill ``repaired_count`` matching the diff.

        Pre-fix the repair would over-report because it summed
        ``conn.total_changes`` (cumulative across the connection) on each
        UPDATE batch instead of the per-statement ``rowcount``.
        """
        db_path = db_setup(workspace_env)
        _make_db_with_messages(db_path)

        cfg = _make_config(workspace_env, db_path)

        with open_index_db(db_path) as conn:
            before = count_messages_by_type_sync(conn)
            preview = count_unclassified_message_type_sync(conn)
        # Fixture has 3 context-marker rows + 2 protocol-marker rows + 2
        # plain dialogue + 1 already-classified ``context``.
        assert before == {"message": 7, "context": 1}
        # Preview now counts only classifier-positive candidates, not
        # every default ``message`` row.
        assert preview == 5

        result = repair_message_type_backfill(cfg, dry_run=False)
        assert result.success, result.detail
        assert result.repaired_count == 5, (
            f"repaired_count should equal classifier-positive rows, got {result.repaired_count}"
        )

        with open_index_db(db_path) as conn:
            after = count_messages_by_type_sync(conn)
        assert after == {"message": 2, "context": 4, "protocol": 2}
        assert after["context"] - before["context"] == 3
        assert after.get("protocol", 0) - before.get("protocol", 0) == 2
        assert before["message"] - after["message"] == 5

    def test_user_role_with_non_prose_semantic_type(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """#839 AC #5: provider role ``user`` but stored semantic type is
        not ordinary user prose. After backfill, these rows are flagged
        as context/protocol by the persisted ``message_type`` and the
        Message runtime helpers (``is_context_dump``/``is_protocol_artifact``)
        agree, even though ``role == 'user'`` would otherwise look like
        plain dialogue.
        """
        from polylogue.archive.message.model_runtime import MessageRuntimeMixin
        from polylogue.archive.message.types import MessageType

        db_path = db_setup(workspace_env)
        _make_db_with_messages(db_path)
        cfg = _make_config(workspace_env, db_path)

        repair_message_type_backfill(cfg, dry_run=False)

        with open_index_db(db_path) as conn:
            rows = conn.execute(
                "SELECT native_id, role, message_type FROM messages WHERE native_id IN ('ctx-1', 'proto-1', 'plain-1')"
            ).fetchall()
        by_id = {row[0]: (row[1], row[2]) for row in rows}
        assert by_id["ctx-1"] == ("user", "context")
        assert by_id["proto-1"] == ("user", "protocol")
        assert by_id["plain-1"] == ("user", "message")

        # The runtime mixin reads from persisted message_type only.
        class _MsgStub(MessageRuntimeMixin):
            def __init__(self, mt: MessageType) -> None:
                self.message_type = mt

        ctx_msg = _MsgStub(MessageType.CONTEXT)
        proto_msg = _MsgStub(MessageType.PROTOCOL)
        plain_msg = _MsgStub(MessageType.MESSAGE)
        assert ctx_msg.is_context_dump is True
        assert ctx_msg.is_protocol_artifact is False
        assert proto_msg.is_protocol_artifact is True
        assert proto_msg.is_context_dump is False
        assert plain_msg.is_context_dump is False
        assert plain_msg.is_protocol_artifact is False
