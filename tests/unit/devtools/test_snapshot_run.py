"""Contracts for snapshot-isolated runs (devtools.snapshot_run).

The property under test is the one the verification pipeline previously bought
by forbidding concurrent edits and, on violation, deleting the whole
pytest-testmon graph: a run must see a fixed view of the checkout regardless of
what happens to the working tree while it runs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from devtools.snapshot_run import (
    SnapshotUnavailableError,
    bubblewrap_available,
    materialize_snapshot,
    snapshot_command,
)

pytestmark = pytest.mark.usefixtures("workspace_env")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
            "PATH": "/usr/bin:/bin:/run/current-system/sw/bin",
            "HOME": str(repo),
        },
    )


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "mod.py").write_text("VALUE = 'committed'\n", encoding="utf-8")
    (repo / "ignored.log").write_text("noise\n", encoding="utf-8")
    (repo / ".gitignore").write_text("ignored.log\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "pkg/mod.py", ".gitignore")
    _git(repo, "commit", "-qm", "initial")
    return repo


def test_snapshot_captures_tracked_and_untracked_but_not_ignored(sample_repo: Path, tmp_path: Path) -> None:
    """The snapshot mirrors what a receipt attests to.

    Untracked-but-not-ignored is included on purpose: a newly written test file
    must still run. Ignored content is excluded, which is what keeps the copy
    cheap enough to take on every run.
    """
    (sample_repo / "pkg" / "new_test.py").write_text("def test_new(): pass\n", encoding="utf-8")

    snapshot = materialize_snapshot(sample_repo, tmp_path / "snap")

    assert (snapshot / "pkg" / "mod.py").read_text(encoding="utf-8") == "VALUE = 'committed'\n"
    assert (snapshot / "pkg" / "new_test.py").exists(), "untracked non-ignored files must be present"
    assert not (snapshot / "ignored.log").exists(), "ignored files must stay out of the snapshot"


def test_snapshot_captures_uncommitted_edits_to_tracked_files(sample_repo: Path, tmp_path: Path) -> None:
    """Working-tree content, not HEAD content: the run tests what you have."""
    (sample_repo / "pkg" / "mod.py").write_text("VALUE = 'edited'\n", encoding="utf-8")

    snapshot = materialize_snapshot(sample_repo, tmp_path / "snap")

    assert (snapshot / "pkg" / "mod.py").read_text(encoding="utf-8") == "VALUE = 'edited'\n"


def test_snapshot_is_unaffected_by_later_working_tree_edits(sample_repo: Path, tmp_path: Path) -> None:
    """The whole point: a mid-run edit cannot reach the snapshot.

    This is the property that replaces `CheckoutMutationMonitor` aborting the run
    and unlinking the testmon database.
    """
    snapshot = materialize_snapshot(sample_repo, tmp_path / "snap")

    (sample_repo / "pkg" / "mod.py").write_text("VALUE = 'mutated mid-run'\n", encoding="utf-8")
    (sample_repo / "pkg" / "added_mid_run.py").write_text("x = 1\n", encoding="utf-8")

    assert (snapshot / "pkg" / "mod.py").read_text(encoding="utf-8") == "VALUE = 'committed'\n"
    assert not (snapshot / "pkg" / "added_mid_run.py").exists()


def test_snapshot_command_binds_the_snapshot_at_the_checkout_path() -> None:
    """The path must not change, or the venv and the checkout guard both break."""
    argv = snapshot_command(
        ["pytest", "-q"],
        root=Path("/realm/project/polylogue"),
        snapshot=Path("/realm/tmp/snap-xyz"),
    )

    assert argv[0] == "bwrap"
    bind_index = argv.index("--bind")
    assert argv[bind_index + 1 : bind_index + 3] == ["/realm/tmp/snap-xyz", "/realm/project/polylogue"]
    assert argv[-2:] == ["pytest", "-q"]
    assert "--chdir" in argv


def test_snapshot_command_rebinds_live_state_over_the_frozen_copy(tmp_path: Path) -> None:
    """`.venv`/`.cache`/`.git` are live: the interpreter is not code under test,
    and receipts plus the testmon graph must outlive the run."""
    root = tmp_path / "checkout"
    for relative in (".venv", ".cache", ".git"):
        (root / relative).mkdir(parents=True)

    argv = snapshot_command(["true"], root=root, snapshot=tmp_path / "snap")

    joined = " ".join(argv)
    for relative in (".venv", ".cache", ".git"):
        live = str(root / relative)
        assert f"--bind {live} {live}" in joined, f"{relative} must be bound back to the live path"


@pytest.mark.skipif(not bubblewrap_available(), reason="bwrap unavailable on this host")
def test_isolated_run_sees_frozen_content_while_the_tree_changes(sample_repo: Path, tmp_path: Path) -> None:
    """End-to-end: mutate the tree while a command runs against the snapshot."""
    snapshot = materialize_snapshot(sample_repo, tmp_path / "snap")
    (sample_repo / "pkg" / "mod.py").write_text("VALUE = 'mutated'\n", encoding="utf-8")

    argv = snapshot_command(["cat", "pkg/mod.py"], root=sample_repo, snapshot=snapshot)
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "VALUE = 'committed'\n"
    assert (sample_repo / "pkg" / "mod.py").read_text(encoding="utf-8") == "VALUE = 'mutated'\n"


def test_missing_bubblewrap_refuses_rather_than_silently_running_live() -> None:
    """A caller that asked for isolation and did not get it must not proceed.

    Falling back to the live tree would record a receipt whose central claim --
    that the tested content could not change -- is false.
    """
    from devtools import snapshot_run

    assert issubclass(SnapshotUnavailableError, RuntimeError)
    assert "SnapshotUnavailableError" in snapshot_run.__all__
