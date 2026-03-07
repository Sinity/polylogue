"""Unit tests for showcase workspace seeding behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from polylogue.showcase.runner import ShowcaseRunner


async def _fake_run_sources(**_kwargs):  # pragma: no cover - test helper
    return None


def _write_min_fixture(fixture_dir: Path, provider: str = "chatgpt") -> None:
    provider_dir = fixture_dir / provider
    provider_dir.mkdir(parents=True, exist_ok=True)
    (provider_dir / "sample.json").write_text("{}")


class TestShowcaseRunnerSeeding:
    """Covers fixture vs synthetic seed selection in ShowcaseRunner."""

    def test_seed_workspace_uses_packaged_fixture_path_by_default(self, tmp_path):
        runner = ShowcaseRunner(showcase_data="fixtures")
        workspace = tmp_path / "workspace"

        def _fake_copy(fixtures_root: Path) -> None:
            _write_min_fixture(fixtures_root, provider="chatgpt")

        with patch.object(runner, "_copy_fixtures", side_effect=_fake_copy) as mock_copy:
            with patch.object(runner, "_generate_synthetic_fixtures") as mock_generate:
                with patch("polylogue.pipeline.runner.run_sources", side_effect=_fake_run_sources):
                    runner._seed_workspace(workspace)

        assert mock_copy.call_count == 1
        assert mock_generate.call_count == 0
        assert (workspace / "data" / "polylogue" / "inbox" / "chatgpt" / "sample.json").exists()

    def test_seed_workspace_uses_synthetic_seed_when_configured(self, tmp_path):
        runner = ShowcaseRunner(showcase_data="synthetic", synthetic_count=7)
        workspace = tmp_path / "workspace"

        def _fake_generate(fixtures_root: Path, *, count: int) -> None:
            assert count == 7
            _write_min_fixture(fixtures_root, provider="codex")

        with patch.object(runner, "_copy_fixtures") as mock_copy:
            with patch.object(runner, "_generate_synthetic_fixtures", side_effect=_fake_generate) as mock_generate:
                with patch("polylogue.pipeline.runner.run_sources", side_effect=_fake_run_sources):
                    runner._seed_workspace(workspace)

        assert mock_copy.call_count == 0
        assert mock_generate.call_count == 1
        assert (workspace / "data" / "polylogue" / "inbox" / "codex" / "sample.json").exists()

    def test_generate_synthetic_fixtures_uses_showcase_style(self, tmp_path):
        runner = ShowcaseRunner(showcase_data="synthetic", synthetic_count=1)
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
