"""Unit tests for showcase workspace seeding behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from polylogue.scenarios import CorpusSpec, polylogue_execution
from polylogue.showcase.cli_boundary import ShowcaseCliResult
from polylogue.showcase.corpus_requests import showcase_corpus_request
from polylogue.showcase.exercises import Exercise
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
        runner = ShowcaseRunner(corpus_request=showcase_corpus_request(count=7))
        workspace = tmp_path / "workspace"

        def _fake_generate(fixtures_root: Path, *, request) -> None:
            assert request.count == 7
            assert request.style == "showcase"
            _write_min_fixture(fixtures_root, provider="codex")

        with patch.object(runner, "_generate_synthetic_fixtures", side_effect=_fake_generate) as mock_generate:
            with patch("polylogue.pipeline.runner.run_sources", new_callable=AsyncMock, side_effect=_fake_run_sources):
                runner._seed_workspace(workspace, exercises=[])

        assert mock_generate.call_count == 1
        assert (workspace / "data" / "polylogue" / "inbox" / "codex" / "sample.json").exists()

    def test_runner_seeds_workspace_with_selected_exercises(self, tmp_path):
        runner = ShowcaseRunner(output_dir=tmp_path / "output")
        selected = [
            Exercise(
                name="help-main",
                group="structural",
                description="Main help",
                execution=polylogue_execution("--help"),
                corpus_specs=(CorpusSpec.for_provider("chatgpt", count=2),),
            )
        ]

        with patch.object(runner, "_select_exercises", return_value=selected):
            with patch.object(runner, "_seed_workspace") as mock_seed:
                runner.run()

        mock_seed.assert_called_once()
        assert mock_seed.call_args.kwargs["exercises"] == selected

    def test_generate_synthetic_fixtures_uses_showcase_style(self, tmp_path):
        request = showcase_corpus_request(count=1)
        runner = ShowcaseRunner(corpus_request=request)
        fixture_root = tmp_path / "fixtures"

        fake_corpus = MagicMock()
        fake_corpus.wire_format.encoding = "json"
        fake_corpus.generate_batch.return_value = SimpleNamespace(
            artifacts=[SimpleNamespace(raw_bytes=b"{}")],
            report=SimpleNamespace(generated_count=1, element_kind="conversation_document"),
        )

        with patch(
            "polylogue.schemas.synthetic.SyntheticCorpus.available_providers",
            return_value=["chatgpt"],
        ):
            with patch(
                "polylogue.schemas.synthetic.SyntheticCorpus.from_spec",
                return_value=fake_corpus,
            ):
                with patch(
                    "polylogue.schemas.synthetic.SyntheticCorpus.generate_batch_for_spec",
                    return_value=fake_corpus.generate_batch.return_value,
                ) as mock_generate_batch_for_spec:
                    runner._generate_synthetic_fixtures(fixture_root, request=request)

        mock_generate_batch_for_spec.assert_called_once()
        spec = mock_generate_batch_for_spec.call_args.args[0]
        assert isinstance(spec, CorpusSpec)
        assert spec.style == "showcase"


class TestShowcaseRunnerWorkspaceEnv:
    """Covers pre-configured workspace_env behavior."""

    def test_workspace_env_skips_seeding(self, tmp_path):
        """When workspace_env is provided, _seed_workspace should not be called."""
        env_vars = {
            "HOME": str(tmp_path / "home"),
            "XDG_DATA_HOME": str(tmp_path / "data"),
            "POLYLOGUE_ARCHIVE_ROOT": str(tmp_path / "archive"),
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
        selected = [
            Exercise(
                name="help-main",
                group="structural",
                description="Main help",
                execution=polylogue_execution("--help"),
            )
        ]

        with patch.object(runner, "_seed_workspace") as mock_seed:
            with patch.object(runner, "_select_exercises", return_value=selected):
                runner.run()

        mock_seed.assert_called_once()
        assert mock_seed.call_args.kwargs["exercises"] == selected


class TestShowcaseRunnerExecution:
    def test_run_exercise_uses_cli_boundary(self, tmp_path):
        runner = ShowcaseRunner(
            output_dir=tmp_path / "output",
            workspace_env={"POLYLOGUE_ARCHIVE_ROOT": str(tmp_path / "archive")},
        )
        exercise = Exercise(
            name="help-main",
            group="structural",
            description="Main help",
            execution=polylogue_execution("--help"),
        )

        with patch(
            "polylogue.showcase.runner.invoke_showcase_cli",
            return_value=ShowcaseCliResult(exit_code=0, stdout="polylogue\n", stderr=""),
        ) as mock_invoke:
            result = runner._run_exercise(exercise)

        assert result.passed is True
        assert result.exit_code == 0
        mock_invoke.assert_called_once()
        assert mock_invoke.call_args.args[0] == ["--plain", "--help"]
