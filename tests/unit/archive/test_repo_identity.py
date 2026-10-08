"""Tests for repository-root and repo-name normalization."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from polylogue.archive.actions.actions import Action
from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.roles import Role
from polylogue.archive.models import Message, Session
from polylogue.archive.session import attribution as attribution_module
from polylogue.archive.session.attribution import extract_attribution, extract_attribution_from_actions
from polylogue.archive.session.repo_identity import (
    normalize_repo_name,
    normalize_repo_names,
    normalize_repo_path,
    normalize_repo_paths,
    repo_relative_path,
)
from polylogue.archive.session.session_profile import SessionProfile, build_session_profile
from polylogue.archive.viewport.viewports import ToolCategory
from polylogue.core.enums import Origin
from polylogue.core.types import SessionId

REPO_ROOT = Path(__file__).resolve().parents[3]
README_PATH = REPO_ROOT / "README.md"


def _make_repo(tmp_path: Path, name: str) -> Path:
    repo_root = tmp_path / name
    (repo_root / ".git").mkdir(parents=True)
    return repo_root


def test_repo_identity_normalization_filters_noise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    sinnix_repo = _make_repo(tmp_path, "sinnix")
    polylogue_repo = _make_repo(tmp_path, "polylogue")

    assert normalize_repo_name(str(sinnix_repo)) == "sinnix"
    assert normalize_repo_name(f"{sinnix_repo}#switch") is None
    assert normalize_repo_name(str(polylogue_repo / "README.md")) == "polylogue"
    assert normalize_repo_name("https://github.com/Sinity/sinex.git") == "sinex"
    assert normalize_repo_names(["sinex"]) == ("sinex",)
    assert normalize_repo_name("\\S+") is None
    assert normalize_repo_name("README.md") is None
    assert normalize_repo_name(".snapshots/root") is None
    assert normalize_repo_names(
        [
            "https://github.com/Sinity/polylogue.git",
            "git@github.com:Sinity/sinex.git",
            "\\S+",
            "README.md",
        ],
        repo_paths=[str(sinnix_repo)],
    ) == ("polylogue", "sinex", "sinnix")
    assert normalize_repo_path(f"{sinnix_repo}#nixosConfigurations.sinnix-prime") is None
    assert normalize_repo_paths(
        [
            str(polylogue_repo / "README.md"),
            str(sinnix_repo),
            str(tmp_path / "README.md"),
        ]
    ) == (str(polylogue_repo), str(sinnix_repo))


def test_attribution_does_not_probe_archived_automount_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    original_resolve = Path.resolve

    def fail_on_automount(self: Path, strict: bool = False) -> Path:
        if str(self).startswith("/mnt/"):
            raise AssertionError(f"attempted to resolve archived automount path: {self}")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", fail_on_automount)
    action = Action(
        action_id="action-automount-path",
        message_id="msg-automount-path",
        timestamp=datetime(2026, 5, 16, 12, 0, tzinfo=timezone.utc),
        sequence_index=0,
        kind=ToolCategory.FILE_READ,
        tool_name="Read",
        tool_id=None,
        origin=Origin.CLAUDE_CODE_SESSION,
        affected_paths=("/mnt/pendrv/chatlog/claude_code/project/src/main.py",),
        cwd_path="/mnt/pendrv/chatlog/claude_code/project",
        branch_names=(),
        command=None,
        query=None,
        url=None,
        output_text=None,
        search_text="automount path",
        raw={},
    )

    attribution = extract_attribution_from_actions([action])

    assert attribution.file_paths_touched == ("/mnt/pendrv/chatlog/claude_code/project/src/main.py",)
    assert attribution.cwd_paths == ("/mnt/pendrv/chatlog/claude_code/project",)
    assert attribution.repo_paths == ()
    assert attribution.repo_names == ()
    assert attribution.languages_detected == ("python",)


def test_repo_identity_normalization_canonicalizes_absolute_parent_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    sinnix_repo = _make_repo(tmp_path, "sinnix")
    noisy_repo_path = f"/../../{sinnix_repo.relative_to(Path('/'))}"
    noisy_file_path = f"{noisy_repo_path}/README.md"

    assert normalize_repo_path(noisy_repo_path) == str(sinnix_repo)
    assert normalize_repo_path(noisy_file_path) == str(sinnix_repo)
    assert normalize_repo_name(noisy_file_path) == "sinnix"


def test_repo_identity_normalization_honours_git_ceiling_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The upward walk stops before examining a ``GIT_CEILING_DIRECTORIES`` entry.

    Anti-vacuity: deleting the ceiling check in ``_find_git_root`` makes the
    first assertion return the outer repo instead of ``None``.
    """
    outer_repo = _make_repo(tmp_path, "outer")
    inner_file = outer_repo / "inner" / "file"
    inner_file.parent.mkdir(parents=True)
    inner_file.touch()

    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(outer_repo))
    assert normalize_repo_path(str(inner_file)) is None
    assert normalize_repo_name(str(inner_file)) is None

    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert normalize_repo_path(str(inner_file)) == str(outer_repo)
    assert normalize_repo_name(str(inner_file)) == "outer"


def test_repo_identity_checks_starting_root_before_its_ceiling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _make_repo(tmp_path, "repo")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(repo))
    assert normalize_repo_path(str(repo)) == str(repo)


def test_repo_identity_normalization_ignores_unreadable_git_admin_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    unreadable_git_dir = Path("/boot/.git")
    original_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if self in {unreadable_git_dir, unreadable_git_dir / ".git"}:
            raise PermissionError("denied")
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)

    assert normalize_repo_path(str(unreadable_git_dir)) is None
    assert normalize_repo_name(str(unreadable_git_dir)) is None
    assert normalize_repo_paths([str(unreadable_git_dir)]) == ()


def test_repo_identity_normalization_ignores_transcript_and_state_git_repos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    transcript_repo = tmp_path / ".claude" / "projects"
    (transcript_repo / ".git").mkdir(parents=True)
    config_transcript_repo = tmp_path / ".config" / "claude" / "projects"
    (config_transcript_repo / ".git").mkdir(parents=True)
    state_repo = tmp_path / ".local" / "state" / "sinex" / "blob-repository"
    (state_repo / ".git").mkdir(parents=True)

    assert normalize_repo_path(str(transcript_repo)) is None
    assert normalize_repo_name(str(transcript_repo)) is None
    assert normalize_repo_path(str(config_transcript_repo)) is None
    assert normalize_repo_name(str(config_transcript_repo)) is None
    assert normalize_repo_path(str(state_repo)) is None
    assert normalize_repo_name(str(state_repo)) is None
    assert normalize_repo_names(repo_paths=[str(transcript_repo), str(config_transcript_repo), str(state_repo)]) == ()


def test_session_profile_from_dict_preserves_explicit_repo_names_and_normalizes_repo_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    polylogue_repo = _make_repo(tmp_path, "polylogue")
    sinnix_repo = _make_repo(tmp_path, "sinnix")

    profile = SessionProfile.from_dict(
        {
            "session_id": "conv-normalize-profile",
            "origin": "claude-code-session",
            "repo_paths": [
                str(polylogue_repo / "README.md"),
                str(sinnix_repo),
                str(tmp_path / "README.md"),
            ],
            "repo_names": ["polylogue", "sinnix"],
            "work_events": [],
            "phases": [],
            "tool_categories": {},
            "tags": [],
            "auto_tags": [],
        }
    )

    assert profile.repo_paths == (str(polylogue_repo), str(sinnix_repo))
    assert profile.repo_names == ("polylogue", "sinnix")


def test_build_session_profile_normalizes_repo_roots_from_workdirs_and_tool_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    sinnix_repo = _make_repo(tmp_path, "sinnix")

    session = Session(
        id=SessionId("conv-normalize-build"),
        origin=Origin.CLAUDE_CODE_SESSION,
        title="Normalization",
        created_at=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 24, 10, 5, tzinfo=timezone.utc),
        working_directories=(str(REPO_ROOT),),
        messages=MessageCollection(
            messages=[
                Message(
                    id="u1",
                    role=Role.USER,
                    origin=Origin.CLAUDE_CODE_SESSION,
                    text="Compare the current repo with the system repo.",
                    timestamp=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
                ),
                Message(
                    id="a1",
                    role=Role.ASSISTANT,
                    origin=Origin.CLAUDE_CODE_SESSION,
                    text="Inspecting those paths.",
                    timestamp=datetime(2026, 3, 24, 10, 1, tzinfo=timezone.utc),
                    blocks=[
                        {
                            "type": "tool_use",
                            "tool_name": "Read",
                            "tool_input": {"file_path": str(sinnix_repo / "README.md")},
                        }
                    ],
                ),
            ]
        ),
    )

    profile = build_session_profile(session)

    assert sorted(profile.repo_paths) == sorted([str(REPO_ROOT), str(sinnix_repo)])
    assert sorted(profile.repo_names) == sorted([REPO_ROOT.name, "sinnix"])


def test_extract_attribution_preserves_repo_name_from_provider_git_remote() -> None:
    session = Session(
        id=SessionId("conv-provider-git-remote"),
        origin=Origin.CODEX_SESSION,
        title="Provider Git Remote",
        created_at=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 24, 10, 5, tzinfo=timezone.utc),
        git_branch="master",
        git_repository_url="git@github.com:Sinity/sinex.git",
        messages=MessageCollection(
            messages=[
                Message(
                    id="u1",
                    role=Role.USER,
                    origin=Origin.CODEX_SESSION,
                    text="Continue work on the branch and summarize the status.",
                    timestamp=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
                )
            ]
        ),
    )

    attribution = extract_attribution(session)

    assert attribution.repo_names == ("sinex",)
    assert attribution.branch_names == ("master",)
    assert attribution.languages_detected == ()


def test_extract_attribution_ignores_configured_claude_transcript_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    transcript_repo = tmp_path / ".config" / "claude" / "projects"
    (transcript_repo / ".git").mkdir(parents=True)
    work_repo = _make_repo(tmp_path, "sinnix")

    session = Session(
        id=SessionId("conv-ignore-transcript-repo"),
        origin=Origin.CLAUDE_CODE_SESSION,
        title="Transcript repo noise",
        created_at=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 24, 10, 5, tzinfo=timezone.utc),
        working_directories=(str(transcript_repo),),
        messages=MessageCollection(
            messages=[
                Message(
                    id="u1",
                    role=Role.USER,
                    origin=Origin.CLAUDE_CODE_SESSION,
                    text="Inspect the live repo state.",
                    timestamp=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
                ),
                Message(
                    id="a1",
                    role=Role.ASSISTANT,
                    origin=Origin.CLAUDE_CODE_SESSION,
                    text="Inspecting.",
                    timestamp=datetime(2026, 3, 24, 10, 1, tzinfo=timezone.utc),
                    blocks=[
                        {
                            "type": "tool_use",
                            "tool_name": "Read",
                            "tool_input": {"file_path": f"{work_repo}/README.md"},
                        }
                    ],
                ),
            ]
        ),
    )

    attribution = extract_attribution(session)

    assert attribution.repo_paths == (str(work_repo),)
    assert attribution.repo_names == ("sinnix",)


def test_extract_attribution_filters_transcript_temp_and_snapshot_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    work_repo = _make_repo(tmp_path, "sinnix")
    repo_tool_result = work_repo / "tool-results" / "parser.py"
    repo_tool_result.parent.mkdir()
    repo_tool_result.touch()
    repo_tool_result.unlink()
    system_file = Path("/etc/systemd/system/sinex-gateway.service")
    action = Action(
        action_id="action-noise-filter",
        message_id="msg-noise-filter",
        timestamp=datetime(2026, 4, 12, 15, 0, tzinfo=timezone.utc),
        sequence_index=0,
        kind=ToolCategory.FILE_READ,
        tool_name="Read",
        tool_id=None,
        origin=Origin.CLAUDE_CODE_SESSION,
        affected_paths=(
            str(work_repo / "README.md"),
            str(repo_tool_result),
            str(work_repo / ".claude" / "settings.json"),
            ".snapshot/",
            ".snapshots/root",
            ".btrfs/snapshot",
            str(Path.home() / ".claude" / "settings.local.json"),
            str(Path.home() / ".config" / "claude" / "projects" / "foo" / "tool-results" / "out.txt"),
            "/tmp/claude-1000/foo/tasks/bar.output",
            "/realm/.snapshot/realm.latest",
            "/nix/store/abcd1234-unit-script-sinex-gateway/bin/sinex-gateway",
            str(system_file),
        ),
        cwd_path=None,
        branch_names=(),
        command=None,
        query=None,
        url=None,
        output_text=None,
        search_text="noise filter",
        raw={},
    )

    attribution = extract_attribution_from_actions([action])

    assert sorted(attribution.file_paths_touched) == sorted(
        [
            str(system_file),
            str(work_repo / "README.md"),
            str(repo_tool_result),
        ]
    )
    assert sorted(attribution.repo_paths) == sorted([str(work_repo)])
    assert sorted(attribution.repo_names) == sorted(["sinnix"])


def test_attribution_does_not_let_broad_repo_ancestor_claim_agent_spool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attribution_module, "_repo_root_from_path", lambda _path: "/tmp")
    action = Action(
        action_id="action-broad-temp-repo",
        message_id="msg-broad-temp-repo",
        timestamp=datetime(2026, 4, 12, 15, 0, tzinfo=timezone.utc),
        sequence_index=0,
        kind=ToolCategory.FILE_READ,
        tool_name="Read",
        tool_id=None,
        origin=Origin.CLAUDE_CODE_SESSION,
        affected_paths=("/tmp/claude-1000/foo/tasks/bar.py",),
        cwd_path=None,
        branch_names=(),
        command=None,
        query=None,
        url=None,
        output_text=None,
        search_text="noise filter",
        raw={},
    )

    attribution = extract_attribution_from_actions([action])

    assert attribution.file_paths_touched == ()
    assert attribution.languages_detected == ()


def test_attribution_preserves_repo_directory_with_agent_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attribution_module, "_repo_root_from_path", lambda _path: "/tmp/project")
    deleted_path = "/tmp/project/codex-client/deleted.py"
    action = Action(
        action_id="action-agent-prefix-repo-directory",
        message_id="msg-agent-prefix-repo-directory",
        timestamp=datetime(2026, 4, 12, 15, 0, tzinfo=timezone.utc),
        sequence_index=0,
        kind=ToolCategory.FILE_WRITE,
        tool_name="Write",
        tool_id=None,
        origin=Origin.CODEX_SESSION,
        affected_paths=(deleted_path,),
        cwd_path=None,
        branch_names=(),
        command=None,
        query=None,
        url=None,
        output_text=None,
        search_text="deleted repository path",
        raw={},
    )

    attribution = extract_attribution_from_actions([action])

    assert attribution.file_paths_touched == (deleted_path,)
    assert attribution.languages_detected == ("python",)


def test_attribution_preserves_nested_numeric_agent_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent-like names are noise only when they are direct temporary roots."""
    monkeypatch.setattr(attribution_module, "_repo_root_from_path", lambda _path: "/tmp/project")
    deleted_path = "/tmp/project/claude-3/parser.py"
    action = Action(
        action_id="action-nested-agent-prefix-repo-directory",
        message_id="msg-nested-agent-prefix-repo-directory",
        timestamp=datetime(2026, 4, 12, 15, 0, tzinfo=timezone.utc),
        sequence_index=0,
        kind=ToolCategory.FILE_WRITE,
        tool_name="Write",
        tool_id=None,
        origin=Origin.CLAUDE_CODE_SESSION,
        affected_paths=(deleted_path,),
        cwd_path=None,
        branch_names=(),
        command=None,
        query=None,
        url=None,
        output_text=None,
        search_text="deleted repository path",
        raw={},
    )

    attribution = extract_attribution_from_actions([action])

    assert attribution.file_paths_touched == (deleted_path,)
    assert attribution.languages_detected == ("python",)


def test_extract_attribution_does_not_infer_r_from_dialogue_text() -> None:
    session = Session(
        id=SessionId("conv-dialogue-r-noise"),
        origin=Origin.CODEX_SESSION,
        title="Dialogue R Noise",
        created_at=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 24, 10, 5, tzinfo=timezone.utc),
        messages=MessageCollection(
            messages=[
                Message(
                    id="u1",
                    role=Role.USER,
                    origin=Origin.CODEX_SESSION,
                    text="Keep the variable r untouched and summarize the branch status.",
                    timestamp=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
                )
            ]
        ),
    )

    attribution = extract_attribution(session)

    assert attribution.languages_detected == ()


def test_session_profile_preserves_repo_names() -> None:
    profile = SessionProfile.from_dict(
        {
            "session_id": "conv-day-normalize",
            "origin": "claude-code-session",
            "created_at": "2026-03-24T10:00:00+00:00",
            "updated_at": "2026-03-24T10:05:00+00:00",
            "canonical_session_date": "2026-03-24",
            "repo_paths": [str(README_PATH)],
            "repo_names": ["polylogue"],
            "work_events": [],
            "phases": [],
            "tool_categories": {},
            "tags": [],
            "auto_tags": [],
        }
    )

    assert profile.repo_names == ("polylogue",)
    assert profile.to_dict()["repo_names"] == ["polylogue"]


@pytest.mark.parametrize(
    "component", ["project copy", "project#copy", "project(copy)", "project;copy", "project copy "]
)
def test_structured_cwd_never_resolves_to_a_delimiter_prefix_neighbor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, component: str
) -> None:
    from polylogue.storage.sqlite.archive_tiers.write import _discovered_repo_root_path

    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    neighbor = tmp_path / "project"
    root = tmp_path / component
    for repo in (neighbor, root):
        subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    cwd = root / "src"
    cwd.mkdir()
    literal = str(cwd)
    attribution = extract_attribution_from_actions([], working_directories=[literal])
    assert attribution.cwd_paths == (literal,)
    assert attribution.repo_paths == (str(root),)
    assert attribution.repo_names == (component,)
    action = Action(
        action_id="neutral-cwd-action",
        message_id="neutral-message",
        timestamp=datetime(2026, 4, 12, 15, 0, tzinfo=timezone.utc),
        sequence_index=0,
        kind=ToolCategory.FILE_READ,
        tool_name="Read",
        tool_id=None,
        origin=Origin.CLAUDE_CODE_SESSION,
        affected_paths=(),
        cwd_path=literal,
        branch_names=(),
        command=None,
        query=None,
        url=None,
        output_text=None,
        search_text="",
        raw={},
    )
    action_attribution = extract_attribution_from_actions([action])
    assert action_attribution.cwd_paths == (literal,)
    assert action_attribution.repo_paths == (str(root),)
    assert normalize_repo_path(literal) == str(root)
    assert normalize_repo_name(literal) == component
    assert normalize_repo_path(str(root)) == str(root)
    # file:// keeps URL grammar: a raw # starts its fragment, whereas
    # the structured local path above treats # as a literal filename byte.
    uri_root = neighbor if "#" in component else root
    assert normalize_repo_path(f"file://localhost{literal}") == str(uri_root)
    assert _discovered_repo_root_path(literal) == str(root)


def test_repo_discovery_refreshes_absence_and_enclosing_root_without_cache_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.archive_tiers.write import _discovered_repo_root_path

    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    outer = tmp_path / "outer"
    inner = outer / "inner"
    cwd = inner / "src"
    cwd.mkdir(parents=True)
    literal = str(cwd)
    assert normalize_repo_path(literal) is None
    assert normalize_repo_name(literal) is None
    subprocess.run(["git", "init", "--quiet", str(outer)], check=True)
    assert normalize_repo_path(literal) == str(outer)
    assert normalize_repo_name(literal) == "outer"
    assert _discovered_repo_root_path(literal) == str(outer)
    subprocess.run(["git", "init", "--quiet", str(inner)], check=True)
    assert normalize_repo_path(literal) == str(inner)
    assert normalize_repo_name(literal) == "inner"
    assert _discovered_repo_root_path(literal) == str(inner)
    assert extract_attribution_from_actions([], working_directories=[literal]).repo_paths == (str(inner),)
    # Removing the nested marker must expose the outer repository again.
    (inner / ".git").rename(inner / "retired-git")
    assert normalize_repo_path(literal) == str(outer)
    assert normalize_repo_name(literal) == "outer"


def test_literal_worktree_path_preserves_linked_git_marker_and_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Neutral",
            "-c",
            "user.email=neutral@example.invalid",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "neutral",
        ],
        check=True,
    )
    linked = tmp_path / "project linked"
    subprocess.run(["git", "-C", str(root), "worktree", "add", "--quiet", "--detach", str(linked)], check=True)
    cwd = linked / "src"
    cwd.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert (linked / ".git").is_file()
    assert normalize_repo_path(str(cwd)) == str(linked)
    assert normalize_repo_name(str(cwd)) == "project linked"
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(linked))
    assert normalize_repo_path(str(cwd)) is None
    assert normalize_repo_path(str(linked)) == str(linked)


@pytest.mark.parametrize(
    ("path", "expected"),
    [("/neutral/project copy /file ", "file "), ("/neutral/project copy/file ", "/neutral/project copy/file ")],
)
def test_repo_relative_literal_paths_preserve_blank_and_neighbor(path: str, expected: str) -> None:
    assert repo_relative_path(path, "/neutral/project copy ") == expected


def test_build_session_profile_round_trips_literal_basename_without_phantom_neighbor(tmp_path: Path) -> None:
    root = tmp_path / "project "
    neighbor = tmp_path / "project"
    for repo in (root, neighbor):
        subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    session = Session(
        id=SessionId("neutral-literal-profile"),
        origin=Origin.CODEX_SESSION,
        title="neutral",
        created_at=datetime(2026, 4, 12, tzinfo=timezone.utc),
        updated_at=datetime(2026, 4, 12, tzinfo=timezone.utc),
        working_directories=(str(root),),
        messages=MessageCollection(messages=[]),
    )
    profile = build_session_profile(session)
    assert profile.repo_paths == (str(root),)
    assert profile.repo_names == ("project ",)
    restored = SessionProfile.from_dict(profile.to_dict())
    assert restored.repo_names == profile.repo_names
    assert restored.repo_paths == profile.repo_paths
    assert normalize_repo_names(["project ", " explicit "], repo_paths=[str(root)]) == ("explicit", "project ")
