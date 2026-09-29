from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.context.configuration_evidence import (
    ConfigurationArtifactVersion,
    ConfigurationObservation,
    EfficacyComparison,
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
        EfficacyComparison("a", "b", "x", (), "", "")
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


def test_git_history_preserves_same_second_revisions_as_ambiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    monkeypatch.setenv("GIT_AUTHOR_DATE", "2026-01-01T00:00:00+0000")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2026-01-01T00:00:00+0000")
    path = tmp_path / "CLAUDE.md"
    for payload in (b"first", b"second"):
        path.write_bytes(payload)
        subprocess.run(["git", "add", "CLAUDE.md"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", payload.decode()], cwd=tmp_path, check=True)

    history = git_artifact_history(tmp_path, "CLAUDE.md", owner="operator", kind="instruction")

    # Both snapshots stay ambiguous inside their second; the final commit-order
    # state is published from the next second on.
    assert len(history) == 3
    assert history[-1].observed_from_ms == 1_767_225_601_000 and history[-1].observed_until_ms is None
    assert resolve_context(history, at_ms=1_767_225_600_000).status == "overlap"
    assert resolve_context(history, at_ms=1_767_225_601_000).artifacts[0].content_hash == history[-1].content_hash


def test_capture_and_git_history_hash_symlink_target_name(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "target").write_text("target contents")
    (tmp_path / "AGENTS.md").symlink_to("target")
    subprocess.run(["git", "add", "AGENTS.md", "target"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "symlink"], cwd=tmp_path, check=True)
    from polylogue.context.configuration_evidence import capture_path

    live = capture_path(
        tmp_path / "AGENTS.md", kind="instruction", owner="operator", repository=None, observed_from_ms=1
    )
    historic = git_artifact_history(tmp_path, "AGENTS.md", owner="operator", kind="instruction")[0]
    assert live.content_hash == historic.content_hash


def test_observation_unknowns_and_artifact_kind_affect_context_identity() -> None:
    instruction = artifact_from_bytes(
        kind="instruction", path="same", payload=b"x", owner="o", repository=None, observed_from_ms=0
    )
    hook = artifact_from_bytes(kind="hook", path="same", payload=b"x", owner="o", repository=None, observed_from_ms=0)
    exact = resolve_context((instruction,), at_ms=1)
    unknown = resolve_context(ConfigurationObservation((instruction,), ("mcp_profile",)), at_ms=1)
    other_kind = resolve_context((hook,), at_ms=1)
    assert exact.context != unknown.context
    assert unknown.context is not None
    assert unknown.status == "partial" and not unknown.context.is_complete
    assert exact.context != other_kind.context


def test_invocations_join_by_explicit_declaration_name_mapping() -> None:
    declaration = artifact_from_bytes(
        kind="skill",
        path=".claude/skills/review/SKILL.md",
        payload=b"review",
        owner="o",
        repository=None,
        observed_from_ms=0,
    )
    joined = join_invocations((("review", "skill", 1),), (declaration,), declaration_names={"review": declaration.path})
    assert joined[0].declaration == declaration


def test_git_deletion_closes_the_prior_revision(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    path = tmp_path / "CLAUDE.md"
    path.write_text("old")
    subprocess.run(["git", "add", "CLAUDE.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "add"], cwd=tmp_path, check=True)
    path.unlink()
    subprocess.run(["git", "add", "-u"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "delete"], cwd=tmp_path, check=True)
    history = git_artifact_history(tmp_path, "CLAUDE.md", owner="o", kind="instruction")
    assert len(history) == 1 and history[0].observed_until_ms is not None


def test_an_unmapped_invocation_still_matches_by_its_own_name() -> None:
    """A partial declaration mapping does not make every artifact of the kind match.

    Anti-vacuity: default the lookup to the candidate's own path and the
    unmapped ``deploy`` skill joins the sole active review declaration.
    """
    review = artifact_from_bytes(
        kind="skill", path="review/SKILL.md", payload=b"review", owner="o", repository=None, observed_from_ms=0
    )
    joined = join_invocations(
        (("deploy", "skill", 1), ("review", "skill", 1)),
        (review,),
        declaration_names={"review": review.path},
    )
    assert joined[0].declaration is None
    assert joined[1].declaration == review


@pytest.mark.parametrize("same_second", [False, True])
def test_git_history_reads_pre_rename_revisions_under_their_historical_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_second: bool
) -> None:
    """Revisions from before a rename are history, not deletions.

    Anti-vacuity: read every followed commit under the final name and the
    pre-rename commits fail ``git show``, become deletion stamps, and leave
    the original bytes out of the history; in one timestamp second the
    stamps also produced an inverted interval.
    """
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    if same_second:
        monkeypatch.setenv("GIT_AUTHOR_DATE", "2026-01-01T00:00:00+0000")
        monkeypatch.setenv("GIT_COMMITTER_DATE", "2026-01-01T00:00:00+0000")
    (tmp_path / "old").mkdir()
    (tmp_path / "old/SKILL.md").write_bytes(b"original")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "add"], cwd=tmp_path, check=True)
    (tmp_path / "review").mkdir()
    subprocess.run(["git", "mv", "old/SKILL.md", "review/SKILL.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "rename"], cwd=tmp_path, check=True)
    (tmp_path / "review/SKILL.md").write_bytes(b"revised")
    subprocess.run(["git", "commit", "-qam", "revise"], cwd=tmp_path, check=True)

    history = git_artifact_history(tmp_path, "review/SKILL.md", owner="o", kind="skill")

    original = artifact_from_bytes(
        kind="skill",
        path="review/SKILL.md",
        payload=b"original",
        owner="o",
        repository=str(tmp_path),
        observed_from_ms=0,
    )
    assert original.content_hash in {revision.content_hash for revision in history}
    assert {revision.path for revision in history} == {"review/SKILL.md"}
    assert all(
        revision.observed_until_ms is None or revision.observed_until_ms > revision.observed_from_ms
        for revision in history
    )


def test_git_history_frames_paths_that_look_like_metadata(tmp_path: Path) -> None:
    """A path whose bytes resemble a log marker is still read as a path.

    Anti-vacuity: recognize records by a text prefix such as ``commit:`` and
    the path token is taken for a commit hash, so neither revision is read.
    """
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    for payload in (b"first", b"second"):
        (tmp_path / "commit:notes").write_bytes(payload)
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", payload.decode()], cwd=tmp_path, check=True)

    history = git_artifact_history(tmp_path, "commit:notes", owner="o", kind="instruction")

    expected = {
        artifact_from_bytes(
            kind="instruction",
            path="commit:notes",
            payload=payload,
            owner="o",
            repository=str(tmp_path),
            observed_from_ms=0,
        ).content_hash
        for payload in (b"first", b"second")
    }
    assert {revision.content_hash for revision in history} == expected
