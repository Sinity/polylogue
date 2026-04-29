from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from devtools import lab_corpus, lab_scenario
from polylogue.scenarios import CorpusProfile, CorpusSpec
from polylogue.showcase.qa_runner import QAResult


def test_lab_corpus_seed_env_only_creates_database(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = lab_corpus.main(["seed", "--env-only", "-o", str(tmp_path), "-n", "1", "-p", "chatgpt"])

    assert result == 0
    output = capsys.readouterr().out
    assert f'export HOME="{tmp_path / "home"}"' in output
    assert f'export XDG_CONFIG_HOME="{tmp_path / "config"}"' in output
    assert f'export XDG_CACHE_HOME="{tmp_path / "cache"}"' in output
    assert (tmp_path / "data" / "polylogue" / "polylogue.db").exists()


def test_lab_corpus_inferred_generation_uses_unique_prefixes_per_same_provider_spec(
    tmp_path: Path,
) -> None:
    inferred_specs = (
        CorpusSpec(
            provider="chatgpt",
            package_version="v1",
            count=1,
            messages_min=4,
            messages_max=4,
            seed=7,
            profile=CorpusProfile(family_ids=("cluster-a",)),
        ),
        CorpusSpec(
            provider="chatgpt",
            package_version="v1",
            count=1,
            messages_min=4,
            messages_max=4,
            seed=8,
            profile=CorpusProfile(family_ids=("cluster-b",)),
        ),
    )

    with patch(
        "polylogue.schemas.operator.inference.list_inferred_corpus_specs",
        return_value=inferred_specs,
    ):
        result = lab_corpus.main(
            ["generate", "--corpus-source", "inferred", "-o", str(tmp_path), "-n", "1", "-p", "chatgpt"]
        )

    assert result == 0
    provider_dir = tmp_path / "chatgpt"
    assert (provider_dir / "sample-v1-cluster-a-00.json").exists()
    assert (provider_dir / "sample-v1-cluster-b-00.json").exists()


def test_lab_corpus_inferred_generation_fails_when_no_specs_match(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch(
        "polylogue.schemas.operator.inference.list_inferred_corpus_specs",
        return_value=(),
    ):
        result = lab_corpus.main(["generate", "--corpus-source", "inferred", "-o", str(tmp_path), "-p", "chatgpt"])

    assert result == 2
    assert "No corpus scenarios matched" in capsys.readouterr().err


def test_lab_scenario_archive_smoke_runs_exercises_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    qa_result = QAResult(
        audit_skipped=True,
        proof_skipped=True,
        exercises_skipped=True,
        invariants_skipped=True,
    )

    with patch("devtools.lab_scenario.run_qa_session", return_value=qa_result) as mock_run:
        result = lab_scenario.main(["run", "archive-smoke", "--tier", "0", "--json"])

    assert result == 0
    request = mock_run.call_args.args[0]
    assert request.skip_audit is True
    assert request.skip_proof is True
    assert request.skip_exercises is False
    assert request.skip_invariants is True
    assert request.tier_filter == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall_status"] == "ok"


def test_lab_scenario_live_archive_smoke_uses_active_archive() -> None:
    qa_result = QAResult(
        audit_skipped=True,
        proof_skipped=True,
        exercises_skipped=True,
        invariants_skipped=True,
    )

    with patch("devtools.lab_scenario.run_qa_session", return_value=qa_result) as mock_run:
        result = lab_scenario.main(["run", "archive-smoke", "--live", "--tier", "0"])

    assert result == 0
    request = mock_run.call_args.args[0]
    assert request.live is True
    assert request.fresh is False
    assert request.ingest is False


def test_lab_scenario_verify_baselines_delegates_to_baseline_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[bool] = []

    def fake_verify_showcase_baselines(*, update: bool) -> int:
        captured.append(update)
        return 0

    monkeypatch.setattr(lab_scenario, "verify_showcase_baselines", fake_verify_showcase_baselines)

    assert lab_scenario.main(["verify-baselines", "--update"]) == 0
    assert captured == [True]
