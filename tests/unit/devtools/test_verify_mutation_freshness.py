"""Tests for ``devtools verify-mutation-freshness`` (#1304)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from devtools import mutmut_campaign, verify_mutation_freshness


def _write_artifact(
    repo_root: Path,
    campaign: str,
    *,
    created_at: datetime,
    counts: dict[str, int] | None = None,
    name: str | None = None,
) -> Path:
    artifact_dir = repo_root / ".local" / "mutation-campaigns" / campaign
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    path = artifact_dir / f"{name or stamp}.json"
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


def _write_manifest(path: Path, entries: list[dict[str, object]]) -> None:
    import yaml

    path.write_text(
        yaml.safe_dump(
            {"description": "test", "mutation_campaigns": entries},
            sort_keys=False,
        )
    )


def test_default_artifact_paths_uses_timestamped_layout() -> None:
    when = datetime(2026, 5, 19, 12, 34, 56, tzinfo=UTC)
    json_path, md_path = mutmut_campaign.default_artifact_paths("filters", when)
    assert json_path.as_posix() == ".local/mutation-campaigns/filters/20260519T123456Z.json"
    assert md_path.as_posix() == ".local/mutation-campaigns/filters/20260519T123456Z.md"


def test_assess_campaign_fresh(tmp_path: Path) -> None:
    now = datetime(2026, 5, 19, tzinfo=UTC)
    _write_artifact(tmp_path, "filters", created_at=now - timedelta(days=2))
    result = verify_mutation_freshness.assess_campaign(
        {"name": "filters", "status": "active", "freshness_days": 60},
        repo_root=tmp_path,
        now=now,
        default_freshness_days=60,
    )
    assert result.state == "fresh"
    assert result.kill_rate == pytest.approx(0.7)
    assert result.newest_age_days is not None and result.newest_age_days < 3


def test_assess_campaign_stale(tmp_path: Path) -> None:
    now = datetime(2026, 5, 19, tzinfo=UTC)
    _write_artifact(tmp_path, "filters", created_at=now - timedelta(days=90))
    result = verify_mutation_freshness.assess_campaign(
        {"name": "filters", "status": "active", "freshness_days": 60},
        repo_root=tmp_path,
        now=now,
        default_freshness_days=60,
    )
    assert result.state == "stale"
    assert result.newest_age_days is not None and result.newest_age_days > 60


def test_assess_campaign_missing(tmp_path: Path) -> None:
    now = datetime(2026, 5, 19, tzinfo=UTC)
    result = verify_mutation_freshness.assess_campaign(
        {"name": "filters", "status": "active"},
        repo_root=tmp_path,
        now=now,
        default_freshness_days=60,
    )
    assert result.state == "missing"
    assert result.artifact_count == 0


def test_assess_campaign_inactive_skipped(tmp_path: Path) -> None:
    now = datetime(2026, 5, 19, tzinfo=UTC)
    result = verify_mutation_freshness.assess_campaign(
        {"name": "filters", "status": "archived", "freshness_days": 60},
        repo_root=tmp_path,
        now=now,
        default_freshness_days=60,
    )
    assert result.state == "inactive"


def test_orphan_artifact_detection(tmp_path: Path) -> None:
    now = datetime(2026, 5, 19, tzinfo=UTC)
    _write_artifact(tmp_path, "ghost-campaign", created_at=now)
    orphans = verify_mutation_freshness._orphan_artifact_names(tmp_path, ["filters"])
    assert orphans == ["ghost-campaign"]


def test_main_soft_default_exits_zero_when_all_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "campaign-coverage.yaml"
    _write_manifest(
        manifest,
        [{"name": "filters", "status": "active", "freshness_days": 60}],
    )
    monkeypatch.setattr(verify_mutation_freshness, "ROOT", tmp_path)
    rc = verify_mutation_freshness.main(["--yaml", str(manifest)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "missing: 1" in out
    assert "blocking=False" in out


def test_main_strict_fails_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "campaign-coverage.yaml"
    _write_manifest(
        manifest,
        [{"name": "filters", "status": "active", "freshness_days": 60}],
    )
    monkeypatch.setattr(verify_mutation_freshness, "ROOT", tmp_path)
    rc = verify_mutation_freshness.main(["--yaml", str(manifest), "--strict"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "[BLOCK] missing artifact: filters" in out
    assert "blocking=True" in out


def test_main_strict_passes_when_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "campaign-coverage.yaml"
    _write_manifest(
        manifest,
        [{"name": "filters", "status": "active", "freshness_days": 60}],
    )
    # Use a deliberately-future timestamp so the artifact reads as fresh
    # regardless of when the test runs, without touching the host clock.
    _write_artifact(tmp_path, "filters", created_at=datetime(2099, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(verify_mutation_freshness, "ROOT", tmp_path)
    rc = verify_mutation_freshness.main(["--yaml", str(manifest), "--strict"])
    assert rc == 0


def test_assess_campaign_carries_min_kill_rate_threshold(tmp_path: Path) -> None:
    """A fresh campaign records its threshold (per-entry overrides the default)."""
    now = datetime(2026, 5, 19, tzinfo=UTC)
    _write_artifact(tmp_path, "filters", created_at=now - timedelta(days=1))
    result = verify_mutation_freshness.assess_campaign(
        {"name": "filters", "status": "active", "min_kill_rate": 0.8},
        repo_root=tmp_path,
        now=now,
        default_freshness_days=60,
        default_min_kill_rate=0.5,
    )
    assert result.min_kill_rate == pytest.approx(0.8)  # per-entry override wins
    assert result.kill_rate == pytest.approx(0.7)


def test_enforce_kill_rate_blocks_fresh_campaign_below_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--enforce-kill-rate fails when a fresh campaign is below its floor."""
    manifest = tmp_path / "campaign-coverage.yaml"
    _write_manifest(
        manifest,
        [{"name": "filters", "status": "active", "min_kill_rate": 0.9}],
    )
    # kill rate 7/(7+3) = 0.70, below the 0.90 floor.
    _write_artifact(tmp_path, "filters", created_at=datetime(2099, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(verify_mutation_freshness, "ROOT", tmp_path)
    rc = verify_mutation_freshness.main(["--yaml", str(manifest), "--enforce-kill-rate"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "[BLOCK] kill rate below threshold: filters" in out
    assert "blocking=True" in out


def test_enforce_kill_rate_passes_when_above_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--enforce-kill-rate passes when the fresh campaign clears its floor."""
    manifest = tmp_path / "campaign-coverage.yaml"
    _write_manifest(
        manifest,
        [{"name": "filters", "status": "active", "min_kill_rate": 0.5}],
    )
    _write_artifact(tmp_path, "filters", created_at=datetime(2099, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(verify_mutation_freshness, "ROOT", tmp_path)
    rc = verify_mutation_freshness.main(["--yaml", str(manifest), "--enforce-kill-rate"])
    assert rc == 0


def test_enforce_kill_rate_ignores_missing_campaigns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A campaign with no artifact is missing, not below-threshold — not blocked.

    This is the CI contract: a fresh checkout has artifacts only for the
    just-run group, so --enforce-kill-rate (without --strict) must not flag the
    artifact-less campaigns.
    """
    manifest = tmp_path / "campaign-coverage.yaml"
    _write_manifest(
        manifest,
        [
            {"name": "filters", "status": "active", "min_kill_rate": 0.9},
            {"name": "fts5", "status": "active", "min_kill_rate": 0.9},
        ],
    )
    # Only filters has an artifact, and it clears the floor.
    _write_artifact(
        tmp_path,
        "filters",
        created_at=datetime(2099, 1, 1, tzinfo=UTC),
        counts={"killed": 19, "survived": 1},
    )
    monkeypatch.setattr(verify_mutation_freshness, "ROOT", tmp_path)
    rc = verify_mutation_freshness.main(["--yaml", str(manifest), "--enforce-kill-rate"])
    assert rc == 0


def test_enforce_kill_rate_uses_manifest_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A campaign without its own min_kill_rate inherits the manifest default."""
    import yaml

    manifest = tmp_path / "campaign-coverage.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "description": "test",
                "default_min_kill_rate": 0.9,
                "mutation_campaigns": [{"name": "filters", "status": "active"}],
            },
            sort_keys=False,
        )
    )
    _write_artifact(tmp_path, "filters", created_at=datetime(2099, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(verify_mutation_freshness, "ROOT", tmp_path)
    rc = verify_mutation_freshness.main(["--yaml", str(manifest), "--enforce-kill-rate"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "kill rate below threshold: filters" in out


def test_committed_manifest_lint_soft_is_clean() -> None:
    """The committed manifest passes the lint in soft mode."""
    rc = verify_mutation_freshness.main([])
    assert rc == 0
