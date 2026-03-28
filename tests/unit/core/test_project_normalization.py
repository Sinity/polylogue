"""Tests for canonical workspace project normalization."""

from __future__ import annotations

from datetime import date, datetime, timezone

from polylogue.archive_product_summaries import aggregate_day_session_summary_products
from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Conversation, Message
from polylogue.lib.project_normalization import (
    normalize_project_name,
    normalize_project_names,
    normalize_repo_path,
    normalize_repo_paths,
)
from polylogue.lib.session_profile import SessionProfile, build_session_profile
from polylogue.lib.session_summaries import summarize_day
from polylogue.storage.store import DaySessionSummaryRecord


def test_workspace_project_normalization_filters_noise() -> None:
    assert normalize_project_name("/realm/project/sinnix#switch") == "sinnix"
    assert normalize_project_name("/realm/project/polylogue`\\n\\nPass") == "polylogue"
    assert normalize_project_name("sinex)") == "sinex"
    assert normalize_project_name("\\S+") is None
    assert normalize_project_name("README.md") is None
    assert normalize_project_names(
        ["polylogue`\\n\\nPass", "\\S+", "README.md", "sinex)"],
        repo_paths=["/realm/project/sinnix#switch"],
    ) == ("polylogue", "sinex", "sinnix")
    assert normalize_repo_path("/realm/project/sinnix#nixosConfigurations.sinnix-prime") == "/realm/project/sinnix"
    assert normalize_repo_paths(
        [
            "/realm/project/polylogue`\\n\\nPass",
            "/realm/project/sinnix#switch",
            "/realm/project/README.md",
        ]
    ) == ("/realm/project/polylogue", "/realm/project/sinnix")


def test_session_profile_from_dict_normalizes_stale_projects_and_repo_paths() -> None:
    profile = SessionProfile.from_dict(
        {
            "conversation_id": "conv-normalize-profile",
            "provider": "claude-code",
            "repo_paths": [
                "/realm/project/polylogue`\\n\\nPass",
                "/realm/project/sinnix#switch",
                "/realm/project/README.md",
            ],
            "canonical_projects": [
                "polylogue`\\n\\nPass",
                "sinnix#switch",
                "\\S+",
                "README.md",
            ],
            "work_events": [],
            "phases": [],
            "tool_categories": {},
            "tags": [],
            "auto_tags": [],
        }
    )

    assert profile.repo_paths == ("/realm/project/polylogue", "/realm/project/sinnix")
    assert profile.canonical_projects == ("polylogue", "sinnix")


def test_build_session_profile_normalizes_workspace_path_noise() -> None:
    conversation = Conversation(
        id="conv-normalize-build",
        provider="claude-code",
        title="Normalization",
        created_at=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 24, 10, 5, tzinfo=timezone.utc),
        messages=MessageCollection(
            messages=[
                Message(
                    id="u1",
                    role="user",
                    provider="claude-code",
                    text=(
                        "Compare /realm/project/polylogue`\\n\\nPass "
                        "with /realm/project/sinnix#switch and ignore /realm/project/README.md."
                    ),
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
                            "tool_input": {"file_path": "/realm/project/sinnix#nixosConfigurations.sinnix-prime"},
                        }
                    ],
                ),
            ]
        ),
    )

    profile = build_session_profile(conversation)

    assert profile.repo_paths == ("/realm/project/polylogue", "/realm/project/sinnix")
    assert profile.canonical_projects == ("polylogue", "sinnix")


def test_day_summary_and_aggregate_products_normalize_stale_project_payloads() -> None:
    profile = SessionProfile.from_dict(
        {
            "conversation_id": "conv-day-normalize",
            "provider": "claude-code",
            "created_at": "2026-03-24T10:00:00+00:00",
            "updated_at": "2026-03-24T10:05:00+00:00",
            "canonical_session_date": "2026-03-24",
            "repo_paths": ["/realm/project/polylogue`\\n\\nPass"],
            "canonical_projects": ["polylogue`\\n\\nPass", "\\S+", "README.md"],
            "work_events": [],
            "phases": [],
            "tool_categories": {},
            "tags": [],
            "auto_tags": [],
        }
    )

    summary = summarize_day([profile], date(2026, 3, 24))
    assert summary.projects_active == ("polylogue",)

    product = aggregate_day_session_summary_products(
        [
            DaySessionSummaryRecord(
                day="2026-03-24",
                provider_name="claude-code",
                materialized_at="2026-03-24T10:10:00+00:00",
                work_event_breakdown={},
                projects_active=("polylogue`\\n\\nPass", "\\S+", "README.md"),
                payload=summary.to_dict(),
                search_text="claude-code",
            )
        ]
    )[0]

    assert product.summary["projects_active"] == ["polylogue"]
