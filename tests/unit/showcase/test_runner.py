"""Unit tests for showcase workspace seeding behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from polylogue.showcase.runner import ShowcaseRunner


async def _fake_run_sources(**_kwargs):  # pragma: no cover - test helper
    return None


def _write_min_fixture(fixture_dir: Path, provider: str = "chatgpt") -> None:
    provider_dir = fixture_dir / provider
    provider_dir.mkdir(parents=True, exist_ok=True)
    (provider_dir / "sample.json").write_text("{}")


class TestShowcaseRunnerSeeding:
    """Covers synthetic seed behavior in ShowcaseRunner."""

    def test_seed_workspace_generates_synthetic_fixtures(self, tmp_path):
        runner = ShowcaseRunner(synthetic_count=7)
        workspace = tmp_path / "workspace"

        def _fake_generate(fixtures_root: Path, *, count: int) -> None:
            assert count == 7
            _write_min_fixture(fixtures_root, provider="codex")

        with patch.object(runner, "_generate_synthetic_fixtures", side_effect=_fake_generate) as mock_generate:
            with patch("polylogue.pipeline.runner.run_sources", new_callable=AsyncMock, side_effect=_fake_run_sources):
                runner._seed_workspace(workspace)

        assert mock_generate.call_count == 1
        assert (workspace / "data" / "polylogue" / "inbox" / "codex" / "sample.json").exists()

    def test_generate_synthetic_fixtures_uses_showcase_style(self, tmp_path):
        runner = ShowcaseRunner(synthetic_count=1)
        fixture_root = tmp_path / "fixtures"

        fake_corpus = MagicMock()
        fake_corpus.wire_format.encoding = "json"
        fake_corpus.generate.return_value = [b"{}"]

        with patch(
            "polylogue.schemas.synthetic.SyntheticCorpus.available_providers",
            return_value=["chatgpt"],
        ):
            with patch(
                "polylogue.schemas.synthetic.SyntheticCorpus.for_provider",
                return_value=fake_corpus,
            ):
                runner._generate_synthetic_fixtures(fixture_root, count=1)

        fake_corpus.generate.assert_called_once()
        assert fake_corpus.generate.call_args.kwargs["style"] == "showcase"


class TestShowcaseRunnerWorkspaceEnv:
    """Covers pre-configured workspace_env behavior."""

    def test_workspace_env_skips_seeding(self, tmp_path):
        """When workspace_env is provided, _seed_workspace should not be called."""
        env_vars = {
            "HOME": str(tmp_path / "home"),
            "XDG_DATA_HOME": str(tmp_path / "data"),
            "POLYLOGUE_ARCHIVE_ROOT": str(tmp_path / "archive"),
            "POLYLOGUE_RENDER_ROOT": str(tmp_path / "render"),
            "POLYLOGUE_FORCE_PLAIN": "1",
        }
        runner = ShowcaseRunner(
            workspace_env=env_vars,
            output_dir=tmp_path / "output",
            tier_filter=0,  # limit exercises
        )

        with patch.object(runner, "_seed_workspace") as mock_seed:
            with patch.object(runner, "_select_exercises", return_value=[]):
                runner.run()

        mock_seed.assert_not_called()
        assert runner._env_vars == env_vars

    def test_no_workspace_env_seeds_normally(self, tmp_path):
        """Without workspace_env, _seed_workspace should be called."""
        runner = ShowcaseRunner(
            output_dir=tmp_path / "output",
            tier_filter=0,
        )

        with patch.object(runner, "_seed_workspace") as mock_seed:
            with patch.object(runner, "_select_exercises", return_value=[]):
                runner.run()

        mock_seed.assert_called_once()
