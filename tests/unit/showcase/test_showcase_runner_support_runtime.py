# mypy: disable-error-code="arg-type"

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from polylogue.scenarios import AssertionSpec, CorpusRequest, CorpusSpec, polylogue_execution
from polylogue.showcase.exercise_models import Exercise
from polylogue.showcase.showcase_runner_support import (
    _merge_exercise_corpus_specs,
    generate_showcase_fixtures,
    run_exercise,
    seed_workspace,
    seed_workspace_with,
    select_exercises,
    validate_exercise_output,
)


def _make_exercise(
    name: str,
    *,
    env: str = "any",
    writes: bool = False,
    tier: int = 1,
    depends_on: str | None = None,
    corpus_specs: tuple[CorpusSpec, ...] = (),
    assertion: AssertionSpec | None = None,
    timeout_s: float = 60.0,
) -> Exercise:
    return Exercise(
        name=name,
        group="query-read",
        description=name,
        execution=polylogue_execution("stats"),
        env=env,
        writes=writes,
        tier=tier,
        depends_on=depends_on,
        corpus_specs=corpus_specs,
        assertion=assertion or AssertionSpec(),
        timeout_s=timeout_s,
    )


def test_select_exercises_filters_seeded_mode() -> None:
    base = _make_exercise("base")
    after = _make_exercise("after", depends_on="base")

    with patch(
        "polylogue.showcase.showcase_runner_support.EXERCISES",
        [
            _make_exercise("live-only", env="live"),
            base,
            after,
            _make_exercise("seeded-only", env="seeded", tier=2),
        ],
    ):
        selected = select_exercises(live=False, tier_filter=None, extra_exercises=[])

    assert [exercise.name for exercise in selected] == ["base", "after", "seeded-only"]


def test_select_exercises_filters_live_mode_and_writes() -> None:
    with patch(
        "polylogue.showcase.showcase_runner_support.EXERCISES",
        [
            _make_exercise("seeded-only", env="seeded"),
            _make_exercise("write", writes=True),
            _make_exercise("live", env="live", tier=2),
            _make_exercise("any", tier=2),
        ],
    ):
        selected = select_exercises(live=True, tier_filter=2, extra_exercises=[])

    assert [exercise.name for exercise in selected] == ["live", "any"]


def test_seed_workspace_uses_default_showcase_request(tmp_path: Path) -> None:
    workspace = SimpleNamespace(env_vars={"POLYLOGUE_ARCHIVE_ROOT": str(tmp_path / "archive")})

    with (
        patch(
            "polylogue.showcase.showcase_runner_support.create_verification_workspace", return_value=workspace
        ) as mock_create,
        patch(
            "polylogue.showcase.showcase_runner_support.build_synthetic_corpus_scenarios", return_value=("scenario",)
        ) as mock_build,
        patch("polylogue.showcase.showcase_runner_support.seed_workspace_from_scenarios") as mock_seed,
    ):
        env = seed_workspace(tmp_path / "workspace")

    mock_create.assert_called_once_with(tmp_path / "workspace")
    mock_build.assert_called_once()
    assert isinstance(mock_build.call_args.kwargs["request"], CorpusRequest)
    mock_seed.assert_called_once_with(workspace, corpus_scenarios=("scenario",))
    assert env == workspace.env_vars


def test_seed_workspace_with_uses_fixture_generation_when_no_compiled_specs(tmp_path: Path) -> None:
    workspace = SimpleNamespace(
        env_vars={"POLYLOGUE_ARCHIVE_ROOT": str(tmp_path / "archive")},
        fixture_dir=tmp_path / "fixtures",
    )
    called: dict[str, object] = {}

    def _generate_fixtures(fixture_dir: Path, request: CorpusRequest) -> None:
        called["fixture_dir"] = fixture_dir
        called["request"] = request

    with (
        patch("polylogue.showcase.showcase_runner_support.create_verification_workspace", return_value=workspace),
        patch("polylogue.showcase.showcase_runner_support.run_pipeline_for_fixture_workspace") as mock_run,
    ):
        env = seed_workspace_with(
            tmp_path / "workspace",
            generate_fixtures=_generate_fixtures,
        )

    assert called["fixture_dir"] == workspace.fixture_dir
    assert isinstance(called["request"], CorpusRequest)
    mock_run.assert_called_once_with(workspace)
    assert env == workspace.env_vars


def test_seed_workspace_with_prefers_corpus_specs_over_generated_fixtures(tmp_path: Path) -> None:
    workspace = SimpleNamespace(
        env_vars={"POLYLOGUE_ARCHIVE_ROOT": str(tmp_path / "archive")},
        fixture_dir=tmp_path / "fixtures",
    )
    exercise = _make_exercise(
        "seeded",
        corpus_specs=(CorpusSpec.for_provider("chatgpt", count=2, messages_min=4, messages_max=4),),
    )

    with (
        patch("polylogue.showcase.showcase_runner_support.create_verification_workspace", return_value=workspace),
        patch("polylogue.showcase.showcase_runner_support.seed_workspace_from_specs") as mock_seed,
        patch("polylogue.showcase.showcase_runner_support.run_pipeline_for_fixture_workspace") as mock_run,
    ):
        env = seed_workspace_with(
            tmp_path / "workspace",
            exercises=(exercise,),
            generate_fixtures=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not generate")),
        )

    mock_seed.assert_called_once()
    mock_run.assert_not_called()
    assert env == workspace.env_vars


def test_merge_exercise_corpus_specs_deduplicates_payloads() -> None:
    repeated = CorpusSpec.for_provider("chatgpt", count=2, messages_min=4, messages_max=4)
    unique = CorpusSpec.for_provider("codex", count=1, messages_min=4, messages_max=4)

    merged = _merge_exercise_corpus_specs(
        (
            _make_exercise("one", corpus_specs=(repeated, unique)),
            _make_exercise("two", corpus_specs=(repeated,)),
        )
    )

    assert merged == (repeated, unique)


def test_generate_showcase_fixtures_uses_default_request(tmp_path: Path) -> None:
    with patch("polylogue.showcase.showcase_runner_support.generate_synthetic_fixtures") as mock_generate:
        generate_showcase_fixtures(tmp_path / "fixtures")

    mock_generate.assert_called_once()
    assert mock_generate.call_args.args[0] == tmp_path / "fixtures"
    assert isinstance(mock_generate.call_args.kwargs["request"], CorpusRequest)


def test_run_exercise_passes_plain_env_and_validates_output() -> None:
    exercise = _make_exercise("json", assertion=AssertionSpec(stdout_contains=("ok",)))
    calls: dict[str, object] = {}

    def _invoke(
        execution: object, *, env: dict[str, str] | None, cwd: Path | None = None, timeout: float = 60.0
    ) -> SimpleNamespace:
        calls["execution"] = execution
        calls["env"] = env
        calls["cwd"] = cwd
        calls["timeout"] = timeout
        return SimpleNamespace(output="ok output", exit_code=0)

    result = run_exercise(exercise, env_vars={"EXISTING": "1"}, invoke_showcase_cli_fn=_invoke)

    assert result.passed is True
    assert result.error is None
    assert calls["execution"] == exercise.execution
    assert calls["env"] == {"EXISTING": "1", "POLYLOGUE_FORCE_PLAIN": "1"}
    assert calls["timeout"] == exercise.timeout_s


def test_run_exercise_handles_timeout() -> None:
    exercise = _make_exercise("timeout", timeout_s=5.0)

    def _invoke(*_args: object, **_kwargs: object) -> SimpleNamespace:
        raise subprocess.TimeoutExpired("polylogue", 5.0)

    result = run_exercise(exercise, env_vars={}, invoke_showcase_cli_fn=_invoke)

    assert result.passed is False
    assert result.exit_code == -1
    assert result.error == "timed out after 5s"


def test_run_exercise_handles_invoker_crash() -> None:
    exercise = _make_exercise("crash")

    def _invoke(*_args: object, **_kwargs: object) -> SimpleNamespace:
        raise RuntimeError("boom")

    result = run_exercise(exercise, env_vars={}, invoke_showcase_cli_fn=_invoke)

    assert result.passed is False
    assert result.exit_code == -1
    assert result.error == "invoke crashed: boom"


def test_validate_exercise_output_delegates_to_assertion() -> None:
    exercise = _make_exercise("assert", assertion=AssertionSpec(stdout_contains=("ok",)))

    assert validate_exercise_output(exercise, "ok output", 0) is None
    assert validate_exercise_output(exercise, "bad output", 0) == "output missing 'ok'"
