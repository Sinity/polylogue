"""Tests for repository-root and repo-name normalization."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from polylogue.archive_product_summaries import aggregate_day_session_summary_products
from polylogue.lib.attribution import extract_attribution
from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Conversation, Message
from polylogue.lib.repo_identity import (
    normalize_repo_name,
    normalize_repo_names,
    normalize_repo_path,
    normalize_repo_paths,
)
from polylogue.lib.session_profile import SessionProfile, build_session_profile
from polylogue.lib.session_summaries import summarize_day
from polylogue.storage.store import DaySessionSummaryRecord

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
    assert normalize_repo_name("\\S+") is None
    assert normalize_repo_name("README.md") is None
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
        id="conv-normalize-build",
        provider="claude-code",
        title="Normalization",
        created_at=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 24, 10, 5, tzinfo=timezone.utc),
        provider_meta={"working_directories": [str(REPO_ROOT)]},
        messages=MessageCollection(
            messages=[
                Message(
                    id="u1",
                    role="user",
                    provider="claude-code",
                    text="Compare the current repo with the system repo.",
                    timestamp=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
                ),
                Message(
                    id="a1",
                    role="assistant",
                    provider="claude-code",
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
    assert profile.repo_names == ("polylogue", "sinnix")


def test_extract_attribution_preserves_repo_name_from_provider_git_remote() -> None:
    conversation = Conversation(
        id="conv-provider-git-remote",
        provider="codex",
        title="Provider Git Remote",
        created_at=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 24, 10, 5, tzinfo=timezone.utc),
        provider_meta={"git": {"branch": "master", "repository_url": "git@github.com:Sinity/sinex.git"}},
        messages=MessageCollection(
            messages=[
                Message(
                    id="u1",
                    role="user",
                    provider="codex",
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
                payload=summary.to_dict(),
                search_text="claude-code",
            )
        ]
    )[0]

    assert product.summary["repos_active"] == ["polylogue"]
