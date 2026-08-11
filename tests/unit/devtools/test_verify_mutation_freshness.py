"""Behavior tests for mutation-campaign artifact freshness."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from devtools import mutmut_campaign, verify_mutation_freshness


def _write_artifact(
    repo_root: Path,
    campaign: str,
    *,
    created_at: datetime,
    counts: dict[str, int] | None = None,
) -> Path:
    artifact_dir = repo_root / ".local" / "mutation-campaigns" / campaign
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{created_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(
        json.dumps(
            {
                "campaign": campaign,
                "created_at": created_at.isoformat(),
                "counts": counts or {"killed": 7, "survived": 3},
            }
        )
    )
    return path


def _use_catalog(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    monkeypatch.setattr(
        verify_mutation_freshness,
        "build_mutation_entries",
        lambda: tuple(SimpleNamespace(name=name) for name in names),
    )


def test_default_artifact_paths_uses_timestamped_layout() -> None:
    when = datetime(2026, 5, 19, 12, 34, 56, tzinfo=UTC)
    json_path, md_path = mutmut_campaign.default_artifact_paths("filters", when)
    assert json_path.as_posix() == ".local/mutation-campaigns/filters/20260519T123456Z.json"
    assert md_path.as_posix() == ".local/mutation-campaigns/filters/20260519T123456Z.md"


@pytest.mark.parametrize(
    ("age_days", "expected"),
    [(2, "fresh"), (90, "stale")],
)
def test_assess_campaign_uses_real_artifact_age(tmp_path: Path, age_days: int, expected: str) -> None:
    now = datetime(2026, 5, 19, tzinfo=UTC)
    _write_artifact(tmp_path, "filters", created_at=now - timedelta(days=age_days))
    result = verify_mutation_freshness.assess_campaign(
        "filters",
        repo_root=tmp_path,
        now=now,
        freshness_days=60,
    )
    assert result.state == expected
    assert result.kill_rate == pytest.approx(0.7)


def test_assess_campaign_reports_missing_artifact(tmp_path: Path) -> None:
    result = verify_mutation_freshness.assess_campaign(
        "filters",
        repo_root=tmp_path,
        now=datetime(2026, 5, 19, tzinfo=UTC),
        freshness_days=60,
    )
    assert result.state == "missing"
    assert result.artifact_count == 0


def test_orphan_artifact_detection(tmp_path: Path) -> None:
    _write_artifact(tmp_path, "ghost-campaign", created_at=datetime(2026, 5, 19, tzinfo=UTC))
    assert verify_mutation_freshness._orphan_artifact_names(tmp_path, ["filters"]) == ["ghost-campaign"]


def test_main_uses_executable_catalog_and_soft_missing_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _use_catalog(monkeypatch, "filters")
    monkeypatch.setattr(verify_mutation_freshness, "ROOT", tmp_path)
    rc = verify_mutation_freshness.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "missing: 1" in out
    assert "blocking=False" in out


def test_strict_fails_when_catalog_campaign_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _use_catalog(monkeypatch, "filters")
    monkeypatch.setattr(verify_mutation_freshness, "ROOT", tmp_path)
    assert verify_mutation_freshness.main(["--strict"]) == 1
    assert "[BLOCK] missing artifact: filters" in capsys.readouterr().out


def test_kill_rate_gate_reads_real_campaign_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _use_catalog(monkeypatch, "filters")
    _write_artifact(tmp_path, "filters", created_at=datetime(2099, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(verify_mutation_freshness, "ROOT", tmp_path)
    rc = verify_mutation_freshness.main(["--enforce-kill-rate", "--default-min-kill-rate", "0.9"])
    assert rc == 1
    assert "kill rate below threshold: filters" in capsys.readouterr().out


def test_kill_rate_gate_ignores_campaign_without_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_catalog(monkeypatch, "filters", "models")
    _write_artifact(
        tmp_path,
        "filters",
        created_at=datetime(2099, 1, 1, tzinfo=UTC),
        counts={"killed": 19, "survived": 1},
    )
    monkeypatch.setattr(verify_mutation_freshness, "ROOT", tmp_path)
    assert verify_mutation_freshness.main(["--enforce-kill-rate"]) == 0
