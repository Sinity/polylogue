from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.context.configuration_evidence import (
    ConfigurationArtifactVersion,
    artifact_from_bytes,
    compare_cohorts,
    git_artifact_history,
    join_invocations,
    resolve_context,
)


def _artifact(path: str, payload: bytes, start: int, end: int | None = None) -> ConfigurationArtifactVersion:
    return artifact_from_bytes(
        kind="instruction",
        path=path,
        payload=payload,
        owner="operator",
        repository="repo",
        observed_from_ms=start,
        observed_until_ms=end,
    )


def test_revisions_are_content_addressed_and_same_actor_can_change_context() -> None:
    old = _artifact("CLAUDE.md", b"old", 0, 10)
    new = _artifact("CLAUDE.md", b"new", 10)
    assert old.content_hash != new.content_hash
    assert resolve_context((old, new), at_ms=5).status == "exact"
    assert resolve_context((old, new), at_ms=15).status == "exact"
    assert resolve_context((old, new), at_ms=5).context != resolve_context((old, new), at_ms=15).context


def test_gaps_and_overlaps_are_explicit() -> None:
    first = _artifact("CLAUDE.md", b"one", 0, 10)
    second = _artifact("CLAUDE.md", b"two", 5, 15)
    overlap = resolve_context((first, second), at_ms=7)
    assert overlap.status == "overlap"
    assert overlap.overlapping_paths == ("CLAUDE.md",)
    gap = resolve_context((first,), at_ms=20, expected_paths=("CLAUDE.md", "settings.json"))
    assert gap.status == "gap"
    assert gap.missing_paths == ("CLAUDE.md", "settings.json")


def test_structural_invocations_join_only_unique_declaring_revision() -> None:
    declaration = _artifact("review", b"skill body", 0, 10)
    joined = join_invocations((("review", "instruction", 4), ("missing", "instruction", 4)), (declaration,))
    assert joined[0].declaration == declaration
    assert joined[1].declaration is None


def test_efficacy_requires_honest_limits() -> None:
    report = compare_cohorts(
        cohort="with-skill",
        compared_cohort="without-skill",
        outcome="completed",
        confounds=("task mix",),
        coverage="12 sessions",
        judgment_authority="operator judgment",
    )
    assert report.confounds == ("task mix",)
    with pytest.raises(ValueError):
        compare_cohorts(
            cohort="a", compared_cohort="b", outcome="x", confounds=(), coverage="12", judgment_authority="human"
        )


def test_git_history_uses_committed_bytes(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    path = tmp_path / "CLAUDE.md"
    path.write_bytes(b"committed")
    subprocess.run(["git", "add", "CLAUDE.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)
    path.write_bytes(b"uncommitted")
    history = git_artifact_history(tmp_path, "CLAUDE.md", owner="operator", kind="instruction")
    assert len(history) == 1
    assert (
        history[0].content_hash
        == artifact_from_bytes(
            kind="instruction",
            path="CLAUDE.md",
            payload=b"committed",
            owner="operator",
            repository=str(tmp_path),
            observed_from_ms=0,
        ).content_hash
    )
