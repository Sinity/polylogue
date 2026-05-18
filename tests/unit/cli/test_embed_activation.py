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
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.archive.query.spec import ConversationQuerySpec
from polylogue.cli.commands.embed import (
    PreflightReport,
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
    def test_preflight_does_not_touch_provider(self, cli_runner: CliRunner, stub_env: Any) -> None:
        report = _make_report(pending_conversations=4, estimated_cost_usd=0.42)
        with _patch_preflight(report):
            result = cli_runner.invoke(embed_command, ["preflight"], obj=stub_env)
        assert result.exit_code == 0, result.output


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


# ---------------------------------------------------------------------------
# Hybrid auto-elevation
# ---------------------------------------------------------------------------


class TestHybridAutoElevation:
    def _stub_repo(self, embedded: int) -> Any:
        stats = MagicMock()
        stats.embedded_messages = embedded
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

    def test_default_keeps_query_terms_unchanged(self) -> None:
        req = RootModeRequest.from_params({"query": ("foo",)})
        spec = req.query_spec()
        assert spec.retrieval_lane == "auto"
        assert spec.similar_text is None
        assert spec.query_terms == ("foo",)
