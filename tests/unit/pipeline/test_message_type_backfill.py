"""Tests for #839 AC #2: backfill pre-#839 message_type rows."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.storage.repair import repair_message_type_backfill
from polylogue.storage.sqlite.connection import connection_context
from tests.infra.storage_records import ConversationBuilder


def _make_db_with_messages(db_path: Path) -> str:
    """Create a database with test messages, return conv_id."""
    conv_id = "conv-backfill-1"
    builder = ConversationBuilder(db_path, conv_id)
    builder.title("Backfill Test")

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
        db_path = Path(workspace_env["archive_root"]) / "polylogue.db"
        _make_db_with_messages(db_path)

        cfg = _make_config(workspace_env, db_path)
        monkeypatch.setattr("polylogue.paths.db_path", lambda: db_path)

        result = repair_message_type_backfill(cfg, dry_run=False)
        assert result.success, result.detail
        assert result.repaired_count > 0

        with connection_context(db_path) as conn:
            # Context messages classified
            for mid in ("ctx-1", "ctx-2", "ctx-3"):
                row = conn.execute(
                    "SELECT message_type FROM messages WHERE message_id = ?",
                    (f"{mid}",),
                ).fetchone()
                assert row is not None, f"Missing message {mid}"
                assert row[0] == "context", f"{mid}: expected context, got {row[0]}"

            # Protocol messages classified
            for mid in ("proto-1", "proto-2"):
                row = conn.execute(
                    "SELECT message_type FROM messages WHERE message_id = ?",
                    (f"{mid}",),
                ).fetchone()
                assert row is not None, f"Missing message {mid}"
                assert row[0] == "protocol", f"{mid}: expected protocol, got {row[0]}"

            # Plain messages unchanged
            for mid in ("plain-1", "plain-2"):
                row = conn.execute(
                    "SELECT message_type FROM messages WHERE message_id = ?",
                    (f"{mid}",),
                ).fetchone()
                assert row is not None, f"Missing message {mid}"
                assert row[0] == "message", f"{mid}: expected message, got {row[0]}"

            # Already-classified message unchanged
            row = conn.execute(
                "SELECT message_type FROM messages WHERE message_id = ?",
                ("already-done",),
            ).fetchone()
            assert row is not None
            assert row[0] == "context", f"already-classified: expected context, got {row[0]}"

    def test_backfill_is_idempotent(self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
        """A second run of the backfill is a no-op."""
        db_path = Path(workspace_env["archive_root"]) / "polylogue.db"
        _make_db_with_messages(db_path)

        cfg = _make_config(workspace_env, db_path)
        monkeypatch.setattr("polylogue.paths.db_path", lambda: db_path)

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
        db_path = Path(workspace_env["archive_root"]) / "polylogue.db"
        _make_db_with_messages(db_path)

        cfg = _make_config(workspace_env, db_path)
        monkeypatch.setattr("polylogue.paths.db_path", lambda: db_path)

        result = repair_message_type_backfill(cfg, dry_run=True)
        assert result.success
        assert result.repaired_count > 0, "Dry-run should count candidate rows"

        # Verify no rows were mutated
        with connection_context(db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE message_type IN ('context', 'protocol')"
            ).fetchone()[0]
            assert count == 1, f"Only the pre-existing 'context' row should exist, found {count}"
