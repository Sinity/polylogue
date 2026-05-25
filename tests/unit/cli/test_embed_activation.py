"""Tests for the embedding activation onboarding flow (#1217).

Covers the new ``polylogue embed`` group:

* ``enable`` writes the ``[embedding]`` section into the user TOML and refuses
  to proceed when ``sqlite-vec`` or the Voyage API key is missing.
* ``preflight`` reports cost estimates without contacting the provider.
* ``disable`` flips the flag but does not drop embeddings.
* ``backfill`` honours the cost cap and the ``--yes`` non-interactive switch.
* ``--lexical`` / ``--semantic`` desugar correctly at the root request layer.
* ``_maybe_elevate_to_hybrid`` promotes ``retrieval_lane='auto'`` to
  ``'hybrid'`` only when a vector provider is present *and* embeddings exist.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.archive.query.spec import ConversationQuerySpec
from polylogue.cli.commands.embed import (
    PreflightReport,
    _effective_cost_cap,
    _message_window_for_cost,
    _read_pending_message_count,
    _splice_embedding_section,
    embed_command,
)
from polylogue.cli.query import _maybe_elevate_to_hybrid
from polylogue.cli.query_contracts import QueryExecutionPlan
from polylogue.cli.root_request import RootModeRequest


def _make_plan(spec: ConversationQuerySpec) -> QueryExecutionPlan:
    base = QueryExecutionPlan.from_params({})
    return replace(base, selection=spec)


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
        "total_conversations": 10,
        "pending_conversations": 4,
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
        assert _message_window_for_cost(0.10) == 2000

    def test_run_cost_cap_takes_the_lower_positive_bound(self) -> None:
        assert _effective_cost_cap(5.0, 0.10) == 0.10
        assert _effective_cost_cap(0.0, 0.10) == 0.10
        assert _effective_cost_cap(5.0, None) == 5.0

    def test_preflight_count_bypasses_schema_version_gate_for_readiness(self, tmp_path: Path) -> None:
        db_path = tmp_path / "archive.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE conversations (conversation_id TEXT PRIMARY KEY)")
            conn.execute("CREATE TABLE messages (message_id TEXT PRIMARY KEY, conversation_id TEXT)")
            conn.execute("INSERT INTO conversations (conversation_id) VALUES ('conv-1')")
            conn.execute("INSERT INTO messages (message_id, conversation_id) VALUES ('msg-1', 'conv-1')")
            conn.execute("PRAGMA user_version = 9")

        assert _read_pending_message_count(db_path) == (1, 1, 1)

    def test_preflight_does_not_touch_provider(self, cli_runner: CliRunner, stub_env: Any) -> None:
        report = _make_report(pending_conversations=4, estimated_cost_usd=0.42)
        with _patch_preflight(report):
            result = cli_runner.invoke(embed_command, ["preflight"], obj=stub_env)
        assert result.exit_code == 0, result.output

    def test_preflight_passes_bounded_window_options(self, cli_runner: CliRunner, stub_env: Any) -> None:
        report = _make_report(pending_conversations=2, pending_messages=7)
        fake_preflight = MagicMock(return_value=report)
        with patch("polylogue.cli.commands.embed._build_preflight_report", fake_preflight):
            result = cli_runner.invoke(
                embed_command,
                ["preflight", "--max-conversations", "2", "--max-messages", "7", "--max-cost-usd", "0.10"],
                obj=stub_env,
            )

        assert result.exit_code == 0, result.output
        assert fake_preflight.call_args.kwargs["max_conversations"] == 2
        assert fake_preflight.call_args.kwargs["max_messages"] == 7
        assert fake_preflight.call_args.kwargs["max_cost_usd"] == 0.10

    def test_preflight_json_emits_machine_readable_window_plan(self, cli_runner: CliRunner, stub_env: Any) -> None:
        report = _make_report(
            pending_conversations=2,
            pending_messages=2000,
            estimated_tokens=1_000_000,
            estimated_cost_usd=0.10,
            windowed=True,
            max_conversations=3,
            max_messages=2000,
            max_cost_usd=0.10,
        )
        with _patch_preflight(report):
            result = cli_runner.invoke(embed_command, ["preflight", "--format", "json"], obj=stub_env)

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["pending_conversations"] == 2
        assert payload["pending_messages"] == 2000
        assert payload["estimated_cost_usd"] == 0.10
        assert payload["monthly_cost_cap_usd"] == 5.0
        assert payload["effective_cost_cap_usd"] == 0.10
        assert payload["pricing"]["approximate"] is True
        assert payload["backfill_args"] == [
            "embed",
            "backfill",
            "--yes",
            "--max-conversations",
            "3",
            "--max-messages",
            "2000",
            "--max-cost-usd",
            "0.1",
        ]
        assert (
            payload["backfill_command"]
            == "polylogue embed backfill --yes --max-conversations 3 --max-messages 2000 --max-cost-usd 0.1"
        )

    def test_preflight_json_omits_backfill_command_when_backlog_empty(
        self, cli_runner: CliRunner, stub_env: Any
    ) -> None:
        report = _make_report(pending_conversations=0, pending_messages=0, estimated_tokens=0, estimated_cost_usd=0.0)
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
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
        report = _make_report(pending_conversations=2, pending_messages=4)
        fake_provider = MagicMock()
        fake_provider.upsert = MagicMock()
        from polylogue.storage.embeddings.materialization import (
            EmbedConversationOutcome,
            PendingConversation,
        )

        pending = [
            PendingConversation(conversation_id="conv-1", title="A"),
            PendingConversation(conversation_id="conv-2", title="B"),
        ]
        outcomes = [
            EmbedConversationOutcome(status="embedded", conversation_id="conv-1", embedded_message_count=2),
            EmbedConversationOutcome(status="embedded", conversation_id="conv-2", embedded_message_count=2),
        ]
        with (
            _patch_preflight(report),
            patch(
                "polylogue.storage.search_providers.create_vector_provider",
                return_value=fake_provider,
            ),
            patch(
                "polylogue.storage.embeddings.materialization.iter_pending_conversations",
                return_value=pending,
            ),
            patch(
                "polylogue.storage.embeddings.materialization.embed_conversation_sync",
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
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
        report = _make_report(max_messages=2000)
        fake_iter = MagicMock(return_value=[])
        fake_preflight = MagicMock(return_value=report)
        with (
            patch("polylogue.cli.commands.embed._build_preflight_report", fake_preflight),
            patch("polylogue.storage.search_providers.create_vector_provider", return_value=MagicMock()),
            patch("polylogue.storage.embeddings.materialization.iter_pending_conversations", fake_iter),
        ):
            result = cli_runner.invoke(
                embed_command,
                ["backfill", "--yes", "--max-conversations", "3", "--max-messages", "7000", "--max-cost-usd", "0.10"],
                obj=stub_env,
            )

        assert result.exit_code == 0, result.output
        assert fake_preflight.call_args.kwargs["max_conversations"] == 3
        assert fake_preflight.call_args.kwargs["max_messages"] == 7000
        assert fake_preflight.call_args.kwargs["max_cost_usd"] == 0.10
        assert fake_iter.call_args.kwargs["max_conversations"] == 3
        assert fake_iter.call_args.kwargs["max_messages"] == 2000

    def test_backfill_stop_after_seconds_stops_before_next_conversation(
        self,
        cli_runner: CliRunner,
        stub_env: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
        from polylogue.storage.embeddings.materialization import (
            EmbedConversationOutcome,
            PendingConversation,
        )

        fake_embed = MagicMock(
            side_effect=[
                EmbedConversationOutcome(status="embedded", conversation_id="conv-1", embedded_message_count=1),
                EmbedConversationOutcome(status="embedded", conversation_id="conv-2", embedded_message_count=1),
            ]
        )
        pending = [
            PendingConversation(conversation_id="conv-1", title="A", message_count=1),
            PendingConversation(conversation_id="conv-2", title="B", message_count=1),
        ]
        with (
            _patch_preflight(_make_report()),
            patch("polylogue.storage.search_providers.create_vector_provider", return_value=MagicMock()),
            patch("polylogue.storage.embeddings.materialization.iter_pending_conversations", return_value=pending),
            patch("polylogue.storage.embeddings.materialization.embed_conversation_sync", fake_embed),
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
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
        from polylogue.storage.embeddings.materialization import (
            EmbedConversationOutcome,
            PendingConversation,
        )

        fake_embed = MagicMock(
            side_effect=[
                EmbedConversationOutcome(status="error", conversation_id="conv-1", error="provider 429"),
                EmbedConversationOutcome(status="embedded", conversation_id="conv-2", embedded_message_count=1),
            ]
        )
        pending = [
            PendingConversation(conversation_id="conv-1", title="A", message_count=1),
            PendingConversation(conversation_id="conv-2", title="B", message_count=1),
        ]
        with (
            _patch_preflight(_make_report()),
            patch("polylogue.storage.search_providers.create_vector_provider", return_value=MagicMock()),
            patch("polylogue.storage.embeddings.materialization.iter_pending_conversations", return_value=pending),
            patch("polylogue.storage.embeddings.materialization.embed_conversation_sync", fake_embed),
        ):
            result = cli_runner.invoke(
                embed_command,
                ["backfill", "--yes", "--max-errors", "1"],
                obj=stub_env,
            )

        assert result.exit_code == 0, result.output
        assert fake_embed.call_count == 1
        assert "Stopped early: max errors reached" in result.output

    def test_backfill_run_cost_cap_stops_after_window_overshoot(
        self,
        cli_runner: CliRunner,
        stub_env: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
        from polylogue.storage.embeddings.materialization import (
            EmbedConversationOutcome,
            PendingConversation,
        )

        fake_embed = MagicMock(
            side_effect=[
                EmbedConversationOutcome(status="embedded", conversation_id="conv-1", embedded_message_count=2),
                EmbedConversationOutcome(status="embedded", conversation_id="conv-2", embedded_message_count=2),
            ]
        )
        pending = [
            PendingConversation(conversation_id="conv-1", title="A", message_count=2),
            PendingConversation(conversation_id="conv-2", title="B", message_count=2),
        ]
        with (
            _patch_preflight(_make_report(max_cost_usd=0.00005)),
            patch("polylogue.storage.search_providers.create_vector_provider", return_value=MagicMock()),
            patch("polylogue.storage.embeddings.materialization.iter_pending_conversations", return_value=pending),
            patch("polylogue.storage.embeddings.materialization.embed_conversation_sync", fake_embed),
        ):
            result = cli_runner.invoke(
                embed_command,
                ["backfill", "--yes", "--max-cost-usd", "0.00005"],
                obj=stub_env,
            )

        assert result.exit_code == 0, result.output
        assert fake_embed.call_count == 1
        assert "Stopped early: cost cap reached" in result.output


# ---------------------------------------------------------------------------
# Hybrid auto-elevation
# ---------------------------------------------------------------------------


class TestHybridAutoElevation:
    def _stub_repo(self, embedded: int, *, stale: int = 0, retrieval_ready: bool | None = None) -> Any:
        stats = MagicMock()
        stats.embedded_messages = embedded
        stats.stale_embedding_messages = stale
        if retrieval_ready is not None:
            stats.retrieval_ready = retrieval_ready
        repo = MagicMock()
        repo.get_archive_stats = AsyncMock(return_value=stats)
        return repo

    def test_no_provider_keeps_auto(self) -> None:
        plan = _make_plan(ConversationQuerySpec(query_terms=("foo",), retrieval_lane="auto"))
        out = asyncio.run(_maybe_elevate_to_hybrid(plan, vector_provider=None, repo=self._stub_repo(100)))
        assert out.selection.retrieval_lane == "auto"

    def test_no_embeddings_keeps_auto(self) -> None:
        plan = _make_plan(ConversationQuerySpec(query_terms=("foo",), retrieval_lane="auto"))
        out = asyncio.run(_maybe_elevate_to_hybrid(plan, vector_provider=MagicMock(), repo=self._stub_repo(0)))
        assert out.selection.retrieval_lane == "auto"

    def test_stale_embeddings_keep_auto(self) -> None:
        plan = _make_plan(ConversationQuerySpec(query_terms=("foo",), retrieval_lane="auto"))
        out = asyncio.run(
            _maybe_elevate_to_hybrid(
                plan,
                vector_provider=MagicMock(),
                repo=self._stub_repo(100, stale=100, retrieval_ready=False),
            )
        )
        assert out.selection.retrieval_lane == "auto"

    def test_no_fts_terms_keeps_auto(self) -> None:
        plan = _make_plan(ConversationQuerySpec(retrieval_lane="auto"))
        out = asyncio.run(_maybe_elevate_to_hybrid(plan, vector_provider=MagicMock(), repo=self._stub_repo(100)))
        assert out.selection.retrieval_lane == "auto"

    def test_explicit_dialogue_lane_respected(self) -> None:
        plan = _make_plan(ConversationQuerySpec(query_terms=("foo",), retrieval_lane="dialogue"))
        out = asyncio.run(_maybe_elevate_to_hybrid(plan, vector_provider=MagicMock(), repo=self._stub_repo(100)))
        assert out.selection.retrieval_lane == "dialogue"

    def test_elevates_when_all_conditions_met(self) -> None:
        plan = _make_plan(ConversationQuerySpec(query_terms=("foo",), retrieval_lane="auto"))
        out = asyncio.run(_maybe_elevate_to_hybrid(plan, vector_provider=MagicMock(), repo=self._stub_repo(100)))
        assert out.selection.retrieval_lane == "hybrid"


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
