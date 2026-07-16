"""Tests for repository-root and repo-name normalization."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from polylogue.archive.actions.actions import Action
from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.roles import Role
from polylogue.archive.models import Message, Session
from polylogue.archive.session.attribution import extract_attribution, extract_attribution_from_actions
from polylogue.archive.session.repo_identity import (
    normalize_repo_name,
    normalize_repo_names,
    normalize_repo_path,
    normalize_repo_paths,
)
from polylogue.archive.session.session_profile import SessionProfile, build_session_profile
from polylogue.archive.session.session_summaries import summarize_day
from polylogue.archive.viewport.viewports import ToolCategory
from polylogue.core.enums import Origin
from polylogue.core.types import SessionId
from polylogue.insights.archive_models import DaySessionSummaryPayload
from polylogue.insights.archive_summaries import aggregate_day_session_summary_insights
from polylogue.storage.runtime import DaySessionSummaryRecord

REPO_ROOT = Path(__file__).resolve().parents[3]
README_PATH = REPO_ROOT / "README.md"


def _make_repo(tmp_path: Path, name: str) -> Path:
    repo_root = tmp_path / name
    (repo_root / ".git").mkdir(parents=True)
    return repo_root


def test_repo_identity_normalization_filters_noise(tmp_path: Path) -> None:
    sinnix_repo = _make_repo(tmp_path, "sinnix")
    polylogue_repo = _make_repo(tmp_path, "polylogue")

    assert normalize_repo_name(f"{sinnix_repo}#switch") == "sinnix"
    assert normalize_repo_name(f"{polylogue_repo}/README.md`\\n\\nPass") == "polylogue"
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
        repo_paths=[f"{sinnix_repo}#switch"],
    ) == ("polylogue", "sinex", "sinnix")
    assert normalize_repo_path(f"{sinnix_repo}#nixosConfigurations.sinnix-prime") == str(sinnix_repo)
    assert normalize_repo_paths(
        [
            f"{polylogue_repo}/README.md`\\n\\nPass",
            f"{sinnix_repo}#switch",
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


def test_repo_identity_normalization_canonicalizes_absolute_parent_traversal(tmp_path: Path) -> None:
    sinnix_repo = _make_repo(tmp_path, "sinnix")
    noisy_repo_path = f"/../../{sinnix_repo.relative_to(Path('/'))}"
    noisy_file_path = f"{noisy_repo_path}/README.md"

    assert normalize_repo_path(noisy_repo_path) == str(sinnix_repo)
    assert normalize_repo_path(noisy_file_path) == str(sinnix_repo)
    assert normalize_repo_name(noisy_file_path) == "sinnix"


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


def test_repo_identity_normalization_ignores_transcript_and_state_git_repos(tmp_path: Path) -> None:
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


def test_session_profile_from_dict_preserves_explicit_repo_names_and_normalizes_repo_paths(tmp_path: Path) -> None:
    polylogue_repo = _make_repo(tmp_path, "polylogue")
    sinnix_repo = _make_repo(tmp_path, "sinnix")

    profile = SessionProfile.from_dict(
        {
            "session_id": "conv-normalize-profile",
            "origin": "claude-code-session",
            "repo_paths": [
                f"{polylogue_repo}/README.md`\\n\\nPass",
                f"{sinnix_repo}#switch",
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


def test_build_session_profile_normalizes_repo_roots_from_workdirs_and_tool_paths(tmp_path: Path) -> None:
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
                            "tool_input": {"file_path": f"{sinnix_repo}#nixosConfigurations.sinnix-prime"},
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


def test_extract_attribution_ignores_configured_claude_transcript_repo(tmp_path: Path) -> None:
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


def test_extract_attribution_filters_transcript_temp_and_snapshot_paths(tmp_path: Path) -> None:
    work_repo = _make_repo(tmp_path, "sinnix")
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
        ]
    )
    assert sorted(attribution.repo_paths) == sorted([str(work_repo)])
    assert sorted(attribution.repo_names) == sorted(["sinnix"])


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


def test_day_summary_and_aggregate_products_preserve_repo_names() -> None:
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

    summary = summarize_day([profile], date(2026, 3, 24))
    assert summary.repos_active == ("polylogue",)

    product = aggregate_day_session_summary_insights(
        [
            DaySessionSummaryRecord(
                day="2026-03-24",
                source_name="claude-code",
                materialized_at="2026-03-24T10:10:00+00:00",
                work_event_breakdown={},
                repos_active=("polylogue",),
                payload=DaySessionSummaryPayload.model_validate(summary.to_dict()),
                search_text="claude-code",
            )
        ]
    )[0]

    assert product.summary.repos_active == ("polylogue",)
