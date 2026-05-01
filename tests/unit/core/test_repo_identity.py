"""Tests for repository-root and repo-name normalization."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from polylogue.archive.action_event.action_events import ActionEvent
from polylogue.archive.conversation.attribution import extract_attribution, extract_attribution_from_action_events
from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.session.session_profile import SessionProfile, build_session_profile
from polylogue.archive.session.session_summaries import summarize_day
from polylogue.lib.models import Conversation, Message
from polylogue.lib.repo_identity import (
    normalize_repo_name,
    normalize_repo_names,
    normalize_repo_path,
    normalize_repo_paths,
)
from polylogue.lib.roles import Role
from polylogue.lib.viewport.viewports import ToolCategory
from polylogue.products.archive_models import DaySessionSummaryPayload
from polylogue.products.archive_summaries import aggregate_day_session_summary_products
from polylogue.storage.runtime import DaySessionSummaryRecord
from polylogue.types import ConversationId, Provider

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
            "conversation_id": "conv-normalize-profile",
            "provider": "claude-code",
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

    conversation = Conversation(
        id=ConversationId("conv-normalize-build"),
        provider=Provider.CLAUDE_CODE,
        title="Normalization",
        created_at=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 24, 10, 5, tzinfo=timezone.utc),
        provider_meta={"working_directories": [str(REPO_ROOT)]},
        messages=MessageCollection(
            messages=[
                Message(
                    id="u1",
                    role=Role.USER,
                    provider=Provider.CLAUDE_CODE,
                    text="Compare the current repo with the system repo.",
                    timestamp=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
                ),
                Message(
                    id="a1",
                    role=Role.ASSISTANT,
                    provider=Provider.CLAUDE_CODE,
                    text="Inspecting those paths.",
                    timestamp=datetime(2026, 3, 24, 10, 1, tzinfo=timezone.utc),
                    content_blocks=[
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

    profile = build_session_profile(conversation)

    assert profile.repo_paths == (str(REPO_ROOT), str(sinnix_repo))
    assert profile.repo_names == (REPO_ROOT.name, "sinnix")


def test_extract_attribution_preserves_repo_name_from_provider_git_remote() -> None:
    conversation = Conversation(
        id=ConversationId("conv-provider-git-remote"),
        provider=Provider.CODEX,
        title="Provider Git Remote",
        created_at=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 24, 10, 5, tzinfo=timezone.utc),
        provider_meta={"git": {"branch": "master", "repository_url": "git@github.com:Sinity/sinex.git"}},
        messages=MessageCollection(
            messages=[
                Message(
                    id="u1",
                    role=Role.USER,
                    provider=Provider.CODEX,
                    text="Continue work on the branch and summarize the status.",
                    timestamp=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
                )
            ]
        ),
    )

    attribution = extract_attribution(conversation)

    assert attribution.repo_names == ("sinex",)
    assert attribution.branch_names == ("master",)
    assert attribution.languages_detected == ()


def test_extract_attribution_ignores_configured_claude_transcript_repo(tmp_path: Path) -> None:
    transcript_repo = tmp_path / ".config" / "claude" / "projects"
    (transcript_repo / ".git").mkdir(parents=True)
    work_repo = _make_repo(tmp_path, "sinnix")

    conversation = Conversation(
        id=ConversationId("conv-ignore-transcript-repo"),
        provider=Provider.CLAUDE_CODE,
        title="Transcript repo noise",
        created_at=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 24, 10, 5, tzinfo=timezone.utc),
        provider_meta={"working_directories": [str(transcript_repo)]},
        messages=MessageCollection(
            messages=[
                Message(
                    id="u1",
                    role=Role.USER,
                    provider=Provider.CLAUDE_CODE,
                    text="Inspect the live repo state.",
                    timestamp=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
                ),
                Message(
                    id="a1",
                    role=Role.ASSISTANT,
                    provider=Provider.CLAUDE_CODE,
                    text="Inspecting.",
                    timestamp=datetime(2026, 3, 24, 10, 1, tzinfo=timezone.utc),
                    content_blocks=[
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

    attribution = extract_attribution(conversation)

    assert attribution.repo_paths == (str(work_repo),)
    assert attribution.repo_names == ("sinnix",)


def test_extract_attribution_filters_transcript_temp_and_snapshot_paths(tmp_path: Path) -> None:
    work_repo = _make_repo(tmp_path, "sinnix")
    system_file = Path("/etc/systemd/system/sinex-gateway.service")
    action = ActionEvent(
        event_id="evt-noise-filter",
        message_id="msg-noise-filter",
        timestamp=datetime(2026, 4, 12, 15, 0, tzinfo=timezone.utc),
        sequence_index=0,
        kind=ToolCategory.FILE_READ,
        tool_name="Read",
        tool_id=None,
        provider=Provider.CLAUDE_CODE,
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

    attribution = extract_attribution_from_action_events([action])

    assert attribution.file_paths_touched == (
        str(system_file),
        str(work_repo / "README.md"),
    )
    assert attribution.repo_paths == (str(work_repo),)
    assert attribution.repo_names == ("sinnix",)


def test_extract_attribution_does_not_infer_r_from_dialogue_text() -> None:
    conversation = Conversation(
        id=ConversationId("conv-dialogue-r-noise"),
        provider=Provider.CODEX,
        title="Dialogue R Noise",
        created_at=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 24, 10, 5, tzinfo=timezone.utc),
        messages=MessageCollection(
            messages=[
                Message(
                    id="u1",
                    role=Role.USER,
                    provider=Provider.CODEX,
                    text="Keep the variable r untouched and summarize the branch status.",
                    timestamp=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
                )
            ]
        ),
    )

    attribution = extract_attribution(conversation)

    assert attribution.languages_detected == ()


def test_day_summary_and_aggregate_products_preserve_repo_names() -> None:
    profile = SessionProfile.from_dict(
        {
            "conversation_id": "conv-day-normalize",
            "provider": "claude-code",
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

    product = aggregate_day_session_summary_products(
        [
            DaySessionSummaryRecord(
                day="2026-03-24",
                provider_name="claude-code",
                materialized_at="2026-03-24T10:10:00+00:00",
                work_event_breakdown={},
                repos_active=("polylogue",),
                payload=DaySessionSummaryPayload.model_validate(summary.to_dict()),
                search_text="claude-code",
            )
        ]
    )[0]

    assert product.summary.repos_active == ("polylogue",)
