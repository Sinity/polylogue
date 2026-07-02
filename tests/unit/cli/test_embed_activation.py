"""Tests for the embedding activation onboarding flow (#1217).

Covers the new ``polylogue ops embed`` group:

* ``enable`` writes the ``[embedding]`` section into the user TOML and refuses
  to proceed when ``sqlite-vec`` or the Voyage API key is missing.
* ``preflight`` reports cost estimates without contacting the provider.
* ``disable`` flips the flag but does not drop embeddings.
* ``backfill`` honours the cost cap and the ``--yes`` non-interactive switch.
* ``--lexical`` / ``--semantic`` desugar correctly at the root request layer.

Native ``auto``→``hybrid`` retrieval elevation is covered by
``TestHybridAutoElevation`` below (#1743).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.embed import (
    _splice_embedding_section,
    embed_command,
)
from polylogue.cli.root_request import RootModeRequest
from polylogue.storage.embeddings.preflight import (
    PreflightReport,
    effective_cost_cap,
    message_window_for_cost,
    read_pending_message_count,
)

# ---------------------------------------------------------------------------
# Splicer
# ---------------------------------------------------------------------------


class TestSpliceEmbeddingSection:
    def test_inserts_when_missing(self) -> None:
        existing = '[archive]\nroot = "/tmp/x"\n'
        out = _splice_embedding_section(existing, enabled=True, voyage_api_key="pa-test")
        assert "[embedding]" in out
        assert "enabled = true" in out
        assert 'voyage_api_key = "pa-test"' in out
        # Existing section untouched.
        assert "[archive]" in out

    def test_replaces_existing_section(self) -> None:
        existing = (
            '[archive]\nroot = "/tmp/x"\n\n'
            '[embedding]\nenabled = false\nvoyage_api_key = "old"\n\n'
            '[logging]\nlevel = "INFO"\n'
        )
        out = _splice_embedding_section(existing, enabled=True, voyage_api_key="new")
        assert "enabled = true" in out
        assert "old" not in out
        assert 'voyage_api_key = "new"' in out
        # The trailing section must survive intact.
        assert "[logging]" in out

    def test_disable_keeps_key_when_provided(self) -> None:
        existing = '[embedding]\nenabled = true\nvoyage_api_key = "k"\n'
        out = _splice_embedding_section(existing, enabled=False, voyage_api_key="k")
        assert "enabled = false" in out
        assert "voyage_api_key" in out

    def test_disable_drops_key_when_none(self) -> None:
        existing = '[embedding]\nenabled = true\nvoyage_api_key = "k"\n'
        out = _splice_embedding_section(existing, enabled=False, voyage_api_key=None)
        assert "enabled = false" in out
        assert "voyage_api_key" not in out


# ---------------------------------------------------------------------------
# CLI smoke tests with stub provider
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def stub_env(tmp_path: Path) -> Any:
    env = MagicMock()
    env.ui = MagicMock()
    env.ui.console = MagicMock()
    env.ui.plain = True
    env.config = MagicMock()
    env.config.db_path = tmp_path / "archive.db"
    env.config.index_config = MagicMock(voyage_api_key=None)
    env.repository = MagicMock()
    env.repository.backend = MagicMock(db_path=tmp_path / "archive.db")
    return env


def _patch_preflight(report: PreflightReport) -> Any:
    return patch("polylogue.cli.commands.embed._build_preflight_report", return_value=report)


def _make_report(**kwargs: Any) -> PreflightReport:
    defaults: dict[str, Any] = {
        "total_sessions": 10,
        "pending_sessions": 4,
        "pending_messages": 100,
        "estimated_tokens": 50000,
        "estimated_cost_usd": 0.005,
        "model": "voyage-4",
        "dimension": 1024,
        "cost_cap_usd": 5.0,
    }
    defaults.update(kwargs)
    return PreflightReport(**defaults)


class TestEnableCommand:
    def test_refuses_without_sqlite_vec(self, cli_runner: CliRunner, stub_env: Any) -> None:
        with patch("polylogue.cli.commands.embed._check_sqlite_vec_available", return_value=(False, "missing")):
            result = cli_runner.invoke(embed_command, ["enable"], obj=stub_env)
        assert result.exit_code != 0
        assert "missing" in (
            result.output + result.stderr_bytes.decode() if hasattr(result, "stderr_bytes") else result.output
        )

    def test_refuses_without_voyage_key(
        self, cli_runner: CliRunner, stub_env: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        with patch("polylogue.cli.commands.embed._check_sqlite_vec_available", return_value=(True, None)):
            result = cli_runner.invoke(embed_command, ["enable"], obj=stub_env)
        assert result.exit_code != 0

    def test_writes_config_on_yes(
        self, cli_runner: CliRunner, stub_env: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "polylogue.toml"
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(target))
        report = _make_report()
        with (
            patch("polylogue.cli.commands.embed._check_sqlite_vec_available", return_value=(True, None)),
            _patch_preflight(report),
        ):
            result = cli_runner.invoke(embed_command, ["enable", "--voyage-api-key", "pa-test", "--yes"], obj=stub_env)
        assert result.exit_code == 0, result.output
        assert target.exists()
        body = target.read_text()
        assert "[embedding]" in body
        assert "enabled = true" in body
        assert "pa-test" in body

    def test_no_store_key_omits_secret(
        self, cli_runner: CliRunner, stub_env: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "polylogue.toml"
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(target))
        report = _make_report()
        with (
            patch("polylogue.cli.commands.embed._check_sqlite_vec_available", return_value=(True, None)),
            _patch_preflight(report),
        ):
            result = cli_runner.invoke(
                embed_command,
                ["enable", "--voyage-api-key", "pa-test", "--yes", "--no-store-key"],
                obj=stub_env,
            )
        assert result.exit_code == 0, result.output
        body = target.read_text()
        assert "enabled = true" in body
        assert "pa-test" not in body


class TestPreflightCommand:
    def test_cost_window_translates_to_message_window(self) -> None:
        assert message_window_for_cost(0.10) == 2000

    def test_run_cost_cap_takes_the_lower_positive_bound(self) -> None:
        assert effective_cost_cap(5.0, 0.10) == 0.10
        assert effective_cost_cap(0.0, 0.10) == 0.10
        assert effective_cost_cap(5.0, None) == 5.0

    def test_preflight_count_bypasses_schema_version_gate_for_readiness(self, tmp_path: Path) -> None:
        db_path = tmp_path / "archive.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
            conn.execute("CREATE TABLE messages (message_id TEXT PRIMARY KEY, session_id TEXT)")
            conn.execute("INSERT INTO sessions (session_id) VALUES ('conv-1')")
            conn.execute("INSERT INTO messages (message_id, session_id) VALUES ('msg-1', 'conv-1')")
            conn.execute("PRAGMA user_version = 9")

        assert read_pending_message_count(db_path) == (1, 1, 1)

    def test_preflight_count_uses_active_archive_with_index_anchor(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.archive.message.roles import Role
        from polylogue.core.enums import BlockType, MaterialOrigin, Provider
        from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        archive_root = tmp_path / "archive"
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
        with ArchiveStore(archive_root) as archive:
            archive.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="preflight-v1",
                    title="Embedding preflight v1",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            text="This human-authored preflight message is long enough to embed.",
                            blocks=[
                                ParsedContentBlock(
                                    type=BlockType.TEXT,
                                    text="This human-authored preflight message is long enough to embed.",
                                )
                            ],
                            material_origin=MaterialOrigin.HUMAN_AUTHORED,
                        )
                    ],
                )
            )

        db_anchor = tmp_path / "data" / "polylogue" / "custom.sqlite"
        db_anchor.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_anchor) as conn:
            conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
            conn.execute("CREATE TABLE messages (message_id TEXT PRIMARY KEY, session_id TEXT)")
            conn.execute("INSERT INTO sessions VALUES ('unsupported-extra')")
            conn.executemany(
                "INSERT INTO messages VALUES (?, 'unsupported-extra')",
                [("unsupported-msg-1",), ("unsupported-msg-2",), ("unsupported-msg-3",)],
            )
            conn.commit()
        assert read_pending_message_count(db_anchor) == (1, 1, 1)

    def test_preflight_count_honors_archive_window_limits(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.archive.message.roles import Role
        from polylogue.core.enums import BlockType, MaterialOrigin, Provider
        from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        archive_root = tmp_path / "archive"
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
        with ArchiveStore(archive_root) as archive:
            for index in range(3):
                archive.write_parsed(
                    ParsedSession(
                        source_name=Provider.CODEX,
                        provider_session_id=f"preflight-window-{index}",
                        title=f"Embedding preflight window {index}",
                        messages=[
                            ParsedMessage(
                                provider_message_id="m1",
                                role=Role.USER,
                                text=f"This human-authored preflight window message {index} is long enough to embed.",
                                blocks=[
                                    ParsedContentBlock(
                                        type=BlockType.TEXT,
                                        text=f"This human-authored preflight window message {index} is long enough to embed.",
                                    )
                                ],
                                material_origin=MaterialOrigin.HUMAN_AUTHORED,
                            )
                        ],
                    )
                )

        db_anchor = tmp_path / "data" / "polylogue" / "custom.sqlite"
        assert read_pending_message_count(db_anchor, max_sessions=2) == (3, 2, 2)
        assert read_pending_message_count(db_anchor, max_messages=1) == (3, 1, 1)

    def test_preflight_does_not_touch_provider(self, cli_runner: CliRunner, stub_env: Any) -> None:
        report = _make_report(pending_sessions=4, estimated_cost_usd=0.42)
        with _patch_preflight(report):
            result = cli_runner.invoke(embed_command, ["preflight"], obj=stub_env)
        assert result.exit_code == 0, result.output

    def test_preflight_passes_bounded_window_options(self, cli_runner: CliRunner, stub_env: Any) -> None:
        report = _make_report(pending_sessions=2, pending_messages=7)
        fake_preflight = MagicMock(return_value=report)
        with patch("polylogue.cli.commands.embed._build_preflight_report", fake_preflight):
            result = cli_runner.invoke(
                embed_command,
                [
                    "preflight",
                    "--max-sessions",
                    "2",
                    "--max-messages",
                    "7",
                    "--max-cost-usd",
                    "0.10",
                    "--min-messages",
                    "5",
                ],
                obj=stub_env,
            )

        assert result.exit_code == 0, result.output
        assert fake_preflight.call_args.kwargs["max_sessions"] == 2
        assert fake_preflight.call_args.kwargs["max_messages"] == 7
        assert fake_preflight.call_args.kwargs["max_cost_usd"] == 0.10
        assert fake_preflight.call_args.kwargs["min_messages"] == 5

    def test_preflight_json_emits_machine_readable_window_plan(self, cli_runner: CliRunner, stub_env: Any) -> None:
        report = _make_report(
            pending_sessions=2,
            pending_messages=2000,
            estimated_tokens=1_000_000,
            estimated_cost_usd=0.10,
            windowed=True,
            max_sessions=3,
            max_messages=2000,
            max_cost_usd=0.10,
            min_messages=5,
        )
        with _patch_preflight(report):
            result = cli_runner.invoke(embed_command, ["preflight", "--format", "json"], obj=stub_env)

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["pending_sessions"] == 2
        assert payload["pending_messages"] == 2000
        assert payload["estimated_cost_usd"] == 0.10
        assert payload["monthly_cost_cap_usd"] == 5.0
        assert payload["effective_cost_cap_usd"] == 0.10
        assert payload["pricing"]["approximate"] is True
        assert payload["min_messages"] == 5
        assert payload["backfill_args"] == [
            "ops",
            "embed",
            "backfill",
            "--yes",
            "--max-sessions",
            "3",
            "--max-messages",
            "2000",
            "--max-cost-usd",
            "0.1",
            "--min-messages",
            "5",
        ]
        assert (
            payload["backfill_command"]
            == "polylogue ops embed backfill --yes --max-sessions 3 --max-messages 2000 --max-cost-usd 0.1 --min-messages 5"
        )

    def test_preflight_json_omits_backfill_command_when_backlog_empty(
        self, cli_runner: CliRunner, stub_env: Any
    ) -> None:
        report = _make_report(pending_sessions=0, pending_messages=0, estimated_tokens=0, estimated_cost_usd=0.0)
        with _patch_preflight(report):
            result = cli_runner.invoke(embed_command, ["preflight", "--format", "json"], obj=stub_env)

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["backfill_args"] is None
        assert payload["backfill_command"] is None


class TestDisableCommand:
    def test_disable_writes_disabled_flag(
        self, cli_runner: CliRunner, stub_env: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "polylogue.toml"
        target.write_text('[embedding]\nenabled = true\nvoyage_api_key = "k"\n', encoding="utf-8")
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(target))
        stub_env.config.index_config = MagicMock(voyage_api_key="k")
        result = cli_runner.invoke(embed_command, ["disable"], obj=stub_env)
        assert result.exit_code == 0, result.output
        body = target.read_text()
        assert "enabled = false" in body
        assert "voyage_api_key" in body


class TestBackfillCommand:
    def test_backfill_requires_key(self, cli_runner: CliRunner, stub_env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        report = _make_report()
        with _patch_preflight(report):
            result = cli_runner.invoke(embed_command, ["backfill", "--yes"], obj=stub_env)
        assert result.exit_code != 0

    def test_backfill_runs_against_stub_provider(
        self,
        cli_runner: CliRunner,
        stub_env: Any,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
        report = _make_report(pending_sessions=2, pending_messages=4)
        fake_provider = MagicMock()
        fake_provider.upsert = MagicMock()
        from polylogue.storage.embeddings.materialization import (
            EmbedSessionOutcome,
            PendingSession,
        )

        index_db = tmp_path / "index.db"
        sqlite3.connect(index_db).close()
        pending = [
            PendingSession(session_id="conv-1", title="A", message_count=2),
            PendingSession(session_id="conv-2", title="B", message_count=2),
        ]
        outcomes = [
            EmbedSessionOutcome(status="embedded", session_id="conv-1", embedded_message_count=2),
            EmbedSessionOutcome(status="embedded", session_id="conv-2", embedded_message_count=2),
        ]
        with (
            _patch_preflight(report),
            patch("polylogue.cli.commands.embed._active_archive_index_path", return_value=index_db),
            patch(
                "polylogue.storage.search_providers.create_vector_provider",
                return_value=fake_provider,
            ),
            patch(
                "polylogue.storage.embeddings.materialization.select_pending_archive_session_window",
                return_value=pending,
            ),
            patch(
                "polylogue.storage.embeddings.materialization.embed_archive_session_sync",
                side_effect=outcomes,
            ),
        ):
            result = cli_runner.invoke(embed_command, ["backfill", "--yes"], obj=stub_env)
        assert result.exit_code == 0, result.output
        assert "Embedded 2" in result.output

    def test_backfill_passes_bounded_window_options(
        self,
        cli_runner: CliRunner,
        stub_env: Any,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
        report = _make_report(max_messages=2000)
        fake_select = MagicMock(return_value=[])
        fake_preflight = MagicMock(return_value=report)
        index_db = tmp_path / "index.db"
        sqlite3.connect(index_db).close()
        with (
            patch("polylogue.cli.commands.embed._build_preflight_report", fake_preflight),
            patch("polylogue.cli.commands.embed._active_archive_index_path", return_value=index_db),
            patch("polylogue.storage.search_providers.create_vector_provider", return_value=MagicMock()),
            patch(
                "polylogue.storage.embeddings.materialization.select_pending_archive_session_window",
                fake_select,
            ),
        ):
            result = cli_runner.invoke(
                embed_command,
                ["backfill", "--yes", "--max-sessions", "3", "--max-messages", "7000", "--max-cost-usd", "0.10"],
                obj=stub_env,
            )

        assert result.exit_code == 0, result.output
        assert fake_preflight.call_args.kwargs["max_sessions"] == 3
        assert fake_preflight.call_args.kwargs["max_messages"] == 7000
        assert fake_preflight.call_args.kwargs["max_cost_usd"] == 0.10
        assert fake_select.call_args.kwargs["max_sessions"] == 3
        assert fake_select.call_args.kwargs["max_messages"] == 2000

    def test_backfill_routes_archive_to_materializer(
        self,
        cli_runner: CliRunner,
        stub_env: Any,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
        from polylogue.storage.embeddings.materialization import (
            EmbedSessionOutcome,
            PendingSession,
        )

        index_db = tmp_path / "index.db"
        sqlite3.connect(index_db).close()
        pending = [PendingSession(session_id="codex-session:v1", title="v1", message_count=2)]
        fake_provider = MagicMock()
        fake_embed = MagicMock(
            return_value=EmbedSessionOutcome(
                status="embedded",
                session_id="codex-session:v1",
                embedded_message_count=2,
            )
        )
        with (
            _patch_preflight(_make_report(pending_sessions=1, pending_messages=2, max_messages=2)),
            patch("polylogue.cli.commands.embed._active_archive_index_path", return_value=index_db),
            patch("polylogue.storage.search_providers.create_vector_provider", return_value=fake_provider),
            patch(
                "polylogue.storage.embeddings.materialization.select_pending_archive_session_window",
                return_value=pending,
            ) as fake_select,
            patch("polylogue.storage.embeddings.materialization.embed_archive_session_sync", fake_embed),
            patch("polylogue.storage.embeddings.materialization.iter_pending_sessions") as old_iter,
        ):
            result = cli_runner.invoke(embed_command, ["backfill", "--yes"], obj=stub_env)

        assert result.exit_code == 0, result.output
        assert "Embedded 1" in result.output
        assert fake_select.call_args.kwargs["max_messages"] == 2
        fake_embed.assert_called_once_with(index_db, fake_provider, "codex-session:v1")
        old_iter.assert_not_called()

    def test_backfill_json_outputs_structured_result(
        self,
        cli_runner: CliRunner,
        stub_env: Any,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
        from polylogue.storage.embeddings.materialization import (
            EmbedSessionOutcome,
            PendingSession,
        )

        index_db = tmp_path / "index.db"
        sqlite3.connect(index_db).close()
        pending = [PendingSession(session_id="codex-session:v1", title="v1", message_count=2)]
        with (
            _patch_preflight(_make_report(pending_sessions=1, pending_messages=2, max_messages=2)),
            patch("polylogue.cli.commands.embed._active_archive_index_path", return_value=index_db),
            patch("polylogue.storage.search_providers.create_vector_provider", return_value=MagicMock()),
            patch(
                "polylogue.storage.embeddings.materialization.select_pending_archive_session_window",
                return_value=pending,
            ),
            patch(
                "polylogue.storage.embeddings.materialization.embed_archive_session_sync",
                return_value=EmbedSessionOutcome(
                    status="embedded",
                    session_id="codex-session:v1",
                    embedded_message_count=2,
                ),
            ),
        ):
            result = cli_runner.invoke(embed_command, ["backfill", "--yes", "--format", "json"], obj=stub_env)

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["status"] == "complete"
        assert payload["embedded_sessions"] == 1
        assert payload["skipped_sessions"] == 0
        assert payload["error_count"] == 0
        assert payload["candidate_sessions"] == 1
        assert payload["processed_sessions"] == 1
        assert payload["estimated_cost_usd"] == 0.0001
        assert payload["preflight"]["pending_messages"] == 2
        assert payload["sessions"] == [
            {
                "embedded_message_count": 2,
                "error": None,
                "estimated_cost_usd": 0.0001,
                "index": 1,
                "session_id": "codex-session:v1",
                "status": "embedded",
                "title": "v1",
                "total": 1,
            }
        ]
        from polylogue.storage.sqlite.archive_tiers.ops_write import list_embedding_catchup_runs

        with sqlite3.connect(tmp_path / "ops.db") as conn:
            runs = list_embedding_catchup_runs(conn)
        assert len(runs) == 1
        assert runs[0].status == "completed"
        assert runs[0].scanned_sessions == 1
        assert runs[0].embedded_sessions == 1
        assert runs[0].skipped_sessions == 0
        assert runs[0].embedded_messages == 2
        assert runs[0].error_count == 0
        assert runs[0].estimated_cost_usd == 0.0001

    def test_backfill_json_requires_yes_for_clean_stdout(
        self,
        cli_runner: CliRunner,
        stub_env: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")

        result = cli_runner.invoke(embed_command, ["backfill", "--format", "json"], obj=stub_env)

        assert result.exit_code != 0
        assert "requires --yes" in result.output

    def test_backfill_stop_after_seconds_stops_before_next_session(
        self,
        cli_runner: CliRunner,
        stub_env: Any,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
        from polylogue.storage.embeddings.materialization import (
            EmbedSessionOutcome,
            PendingSession,
        )

        index_db = tmp_path / "index.db"
        sqlite3.connect(index_db).close()
        fake_embed = MagicMock(
            side_effect=[
                EmbedSessionOutcome(status="embedded", session_id="conv-1", embedded_message_count=1),
                EmbedSessionOutcome(status="embedded", session_id="conv-2", embedded_message_count=1),
            ]
        )
        pending = [
            PendingSession(session_id="conv-1", title="A", message_count=1),
            PendingSession(session_id="conv-2", title="B", message_count=1),
        ]
        with (
            _patch_preflight(_make_report()),
            patch("polylogue.cli.commands.embed._active_archive_index_path", return_value=index_db),
            patch("polylogue.storage.search_providers.create_vector_provider", return_value=MagicMock()),
            patch(
                "polylogue.storage.embeddings.materialization.select_pending_archive_session_window",
                return_value=pending,
            ),
            patch("polylogue.storage.embeddings.materialization.embed_archive_session_sync", fake_embed),
            patch("polylogue.cli.commands.embed.time.monotonic", side_effect=[0.0, 0.0, 2.0]),
        ):
            result = cli_runner.invoke(
                embed_command,
                ["backfill", "--yes", "--stop-after-seconds", "1"],
                obj=stub_env,
            )

        assert result.exit_code == 0, result.output
        assert fake_embed.call_count == 1
        assert "Stopped early: time limit reached" in result.output

    def test_backfill_max_errors_stops_after_provider_error(
        self,
        cli_runner: CliRunner,
        stub_env: Any,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
        from polylogue.storage.embeddings.materialization import (
            EmbedSessionOutcome,
            PendingSession,
        )

        index_db = tmp_path / "index.db"
        sqlite3.connect(index_db).close()
        fake_embed = MagicMock(
            side_effect=[
                EmbedSessionOutcome(status="error", session_id="conv-1", error="provider 429"),
                EmbedSessionOutcome(status="embedded", session_id="conv-2", embedded_message_count=1),
            ]
        )
        pending = [
            PendingSession(session_id="conv-1", title="A", message_count=1),
            PendingSession(session_id="conv-2", title="B", message_count=1),
        ]
        with (
            _patch_preflight(_make_report()),
            patch("polylogue.cli.commands.embed._active_archive_index_path", return_value=index_db),
            patch("polylogue.storage.search_providers.create_vector_provider", return_value=MagicMock()),
            patch(
                "polylogue.storage.embeddings.materialization.select_pending_archive_session_window",
                return_value=pending,
            ),
            patch("polylogue.storage.embeddings.materialization.embed_archive_session_sync", fake_embed),
        ):
            result = cli_runner.invoke(
                embed_command,
                ["backfill", "--yes", "--max-errors", "1"],
                obj=stub_env,
            )

        assert result.exit_code == 0, result.output
        assert fake_embed.call_count == 1
        assert "Stopped early: max errors reached" in result.output

    def test_backfill_run_cost_cap_stops_before_provider_call(
        self,
        cli_runner: CliRunner,
        stub_env: Any,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
        from polylogue.storage.embeddings.materialization import (
            EmbedSessionOutcome,
            PendingSession,
        )

        index_db = tmp_path / "index.db"
        sqlite3.connect(index_db).close()
        fake_embed = MagicMock(
            side_effect=[
                EmbedSessionOutcome(status="embedded", session_id="conv-1", embedded_message_count=2),
                EmbedSessionOutcome(status="embedded", session_id="conv-2", embedded_message_count=2),
            ]
        )
        pending = [
            PendingSession(session_id="conv-1", title="A", message_count=2),
            PendingSession(session_id="conv-2", title="B", message_count=2),
        ]
        with (
            _patch_preflight(_make_report(max_cost_usd=0.00005)),
            patch("polylogue.cli.commands.embed._active_archive_index_path", return_value=index_db),
            patch("polylogue.storage.search_providers.create_vector_provider", return_value=MagicMock()),
            patch(
                "polylogue.storage.embeddings.materialization.select_pending_archive_session_window",
                return_value=pending,
            ),
            patch("polylogue.storage.embeddings.materialization.embed_archive_session_sync", fake_embed),
        ):
            result = cli_runner.invoke(
                embed_command,
                ["backfill", "--yes", "--max-cost-usd", "0.00005"],
                obj=stub_env,
            )

        assert result.exit_code == 0, result.output
        assert fake_embed.call_count == 0
        assert "Stopped early: cost cap would be exceeded" in result.output


# ---------------------------------------------------------------------------
# --lexical / --semantic shortcuts
# ---------------------------------------------------------------------------


class TestLexicalSemanticShortcuts:
    def test_lexical_forces_dialogue_lane(self) -> None:
        req = RootModeRequest.from_params({"query": ("foo", "bar"), "lexical": True})
        spec = req.query_spec()
        assert spec.retrieval_lane == "dialogue"

    def test_semantic_promotes_query_to_similar_text(self) -> None:
        req = RootModeRequest.from_params({"query": ("explain this", "code"), "semantic": True})
        spec = req.query_spec()
        assert spec.similar_text == "explain this code"
        assert spec.query_terms == ()

    def test_semantic_promotes_click_context_query_terms(self) -> None:
        import click

        ctx = click.Context(click.Command("polylogue"))
        ctx.params = {"semantic": True, "similar_text": None}
        ctx.meta["polylogue_query_terms"] = ("explain this", "code")

        spec = RootModeRequest.from_context(ctx).query_spec()

        assert spec.similar_text == "explain this code"
        assert spec.query_terms == ()

    def test_default_keeps_query_terms_unchanged(self) -> None:
        req = RootModeRequest.from_params({"query": ("foo",)})
        spec = req.query_spec()
        assert spec.retrieval_lane == "auto"
        assert spec.similar_text is None
        assert spec.query_terms == ("foo",)


# ---------------------------------------------------------------------------
# Native auto -> hybrid retrieval elevation (#1743)
# ---------------------------------------------------------------------------


def _seed_searchable_archive(archive_root: Path) -> None:
    from tests.infra.storage_records import SessionBuilder

    (
        SessionBuilder(archive_root / "index.db", "elev-1")
        .provider("chatgpt")
        .title("Python Error Handling")
        .created_at("2026-04-01T09:00:00+00:00")
        .updated_at("2026-04-01T09:10:00+00:00")
        .add_message("m1", role="user", text="How to handle exceptions in Python?")
        .add_message("m2", role="assistant", text="Use try-except blocks for Python error handling.")
        .save()
    )


def _seed_embeddings_meta(archive_root: Path, *, needs_reindex: int) -> None:
    """Write a single ``message_embeddings_meta`` row into ``embeddings.db``.

    The freshness predicate reads this regular table (not the vec0 virtual
    table), so the embedding extension is not required to drive elevation.
    """
    conn = sqlite3.connect(archive_root / "embeddings.db")
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS message_embeddings_meta (
                message_id      TEXT PRIMARY KEY,
                model           TEXT NOT NULL,
                dimension       INTEGER NOT NULL,
                content_hash    BLOB NOT NULL,
                embedded_at_ms  INTEGER,
                needs_reindex   INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.execute(
            "INSERT INTO message_embeddings_meta VALUES (?, ?, ?, ?, ?, ?)",
            ("elev-1:m1", "voyage-4", 1024, b"\x00" * 32, 1, needs_reindex),
        )
        conn.commit()
    finally:
        conn.close()


def _run_native_search(archive_root: Path, state_dir: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    from polylogue.cli import cli

    result = CliRunner().invoke(cli, ["--plain", "find", "Python", "-f", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload, dict)
    return payload


class TestHybridAutoElevationPredicate:
    """The freshness gate that decides whether auto may elevate."""

    def test_missing_embeddings_db_is_not_ready(self, tmp_path: Path) -> None:
        from polylogue.cli.archive_query import _archive_embeddings_retrieval_ready

        assert _archive_embeddings_retrieval_ready(tmp_path / "embeddings.db") is False

    def test_empty_meta_is_not_ready(self, tmp_path: Path) -> None:
        from polylogue.cli.archive_query import _archive_embeddings_retrieval_ready

        conn = sqlite3.connect(tmp_path / "embeddings.db")
        conn.executescript(
            "CREATE TABLE message_embeddings_meta (message_id TEXT PRIMARY KEY, model TEXT, "
            "dimension INTEGER, content_hash BLOB, embedded_at_ms INTEGER, needs_reindex INTEGER);"
        )
        conn.commit()
        conn.close()
        assert _archive_embeddings_retrieval_ready(tmp_path / "embeddings.db") is False

    def test_fresh_embeddings_are_ready(self, tmp_path: Path) -> None:
        from polylogue.cli.archive_query import _archive_embeddings_retrieval_ready

        _seed_embeddings_meta(tmp_path, needs_reindex=0)
        assert _archive_embeddings_retrieval_ready(tmp_path / "embeddings.db") is True

    def test_stale_embeddings_are_not_ready(self, tmp_path: Path) -> None:
        from polylogue.cli.archive_query import _archive_embeddings_retrieval_ready

        _seed_embeddings_meta(tmp_path, needs_reindex=1)
        assert _archive_embeddings_retrieval_ready(tmp_path / "embeddings.db") is False


class TestHybridAutoElevation:
    """End-to-end native search elevation behavior driven via CliRunner."""

    def test_auto_without_embeddings_stays_dialogue(
        self, cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        archive_root = cli_workspace["archive_root"]
        _seed_searchable_archive(archive_root)

        payload = _run_native_search(archive_root, cli_workspace["state_dir"], monkeypatch)

        assert payload["retrieval_lane"] == "dialogue"
        assert payload["items"]

    def test_auto_with_fresh_embeddings_elevates_to_hybrid(
        self, cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        archive_root = cli_workspace["archive_root"]
        _seed_searchable_archive(archive_root)
        _seed_embeddings_meta(archive_root, needs_reindex=0)

        fake_provider = MagicMock()
        fake_provider.query = MagicMock(return_value=[])
        with patch("polylogue.cli.archive_query.create_vector_provider", return_value=fake_provider):
            payload = _run_native_search(archive_root, cli_workspace["state_dir"], monkeypatch)

        assert payload["retrieval_lane"] == "hybrid"
        # Hybrid fuses the lexical leg with the (empty) vector leg, so lexical
        # matches still surface even though the stub provider returned nothing.
        assert payload["items"]

    def test_auto_with_stale_embeddings_stays_dialogue(
        self, cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        archive_root = cli_workspace["archive_root"]
        _seed_searchable_archive(archive_root)
        _seed_embeddings_meta(archive_root, needs_reindex=1)

        fake_provider = MagicMock()
        fake_provider.query = MagicMock(return_value=[])
        with patch("polylogue.cli.archive_query.create_vector_provider", return_value=fake_provider):
            payload = _run_native_search(archive_root, cli_workspace["state_dir"], monkeypatch)

        assert payload["retrieval_lane"] == "dialogue"

    def test_lexical_flag_stays_dialogue_with_fresh_embeddings(
        self, cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        archive_root = cli_workspace["archive_root"]
        _seed_searchable_archive(archive_root)
        _seed_embeddings_meta(archive_root, needs_reindex=0)

        monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_dir"]))
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        from polylogue.cli import cli

        fake_provider = MagicMock()
        fake_provider.query = MagicMock(return_value=[])
        with patch("polylogue.cli.archive_query.create_vector_provider", return_value=fake_provider):
            result = CliRunner().invoke(cli, ["--plain", "--lexical", "find", "Python", "-f", "json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["retrieval_lane"] == "dialogue"
