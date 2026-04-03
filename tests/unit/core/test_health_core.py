"""Unit contracts for health reporting (consolidated module)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    ("name", "status_name", "detail", "expected"),
    [
        ("test", "OK", "All good", {"name": "test", "status": "ok", "count": 0, "detail": "All good", "breakdown": {}}),
        ("database", "ERROR", "Connection failed", {"name": "database", "status": "error", "count": 0, "detail": "Connection failed", "breakdown": {}}),
        ("archive", "WARNING", "Directory missing", {"name": "archive", "status": "warning", "count": 0, "detail": "Directory missing", "breakdown": {}}),
    ],
)
def test_health_check_dataclass_contract(name: str, status_name: str, detail: str, expected: dict[str, object]) -> None:
    from polylogue.health import HealthCheck, VerifyStatus

    check = HealthCheck(name=name, status=getattr(VerifyStatus, status_name), summary=detail)
    assert {
        "name": check.name,
        "status": check.status.value,
        "count": check.count,
        "detail": check.summary,
        "breakdown": check.breakdown,
    } == expected


@pytest.mark.parametrize(
    ("deep", "expected_checks"),
    [
        (
            False,
            {
                "config",
                "archive_root",
                "render_root",
                "database",
                "index",
                "orphaned_messages",
                "orphaned_content_blocks",
                "action_event_read_model",
                "action_event_fts",
                "fts_sync",
                "transcript_embeddings",
                "transcript_embedding_freshness",
                "retrieval_evidence",
                "retrieval_inference",
                "retrieval_enrichment",
                "session_profile_enrichment_fts",
                "schemas_coverage",
                "schemas_freshness",
            },
        ),
        (
            True,
            {
                "config",
                "archive_root",
                "render_root",
                "database",
                "sqlite_integrity",
                "index",
                "orphaned_messages",
                "orphaned_content_blocks",
                "action_event_read_model",
                "action_event_fts",
                "fts_sync",
                "transcript_embeddings",
                "transcript_embedding_freshness",
                "retrieval_evidence",
                "retrieval_inference",
                "retrieval_enrichment",
                "session_profile_enrichment_fts",
                "schemas_coverage",
                "schemas_freshness",
            },
        ),
    ],
)
def test_run_health_core_contract(cli_workspace, deep: bool, expected_checks: set[str]) -> None:
    from polylogue.config import get_config
    from polylogue.health import run_archive_health

    report = run_archive_health(get_config(), deep=deep)
    names = {check.name for check in report.checks}
    assert report.timestamp > 0
    assert expected_checks.issubset(names)
    assert sum(report.summary.values()) == len(report.checks)


@pytest.mark.parametrize(
    ("embedded_conversations", "embedded_messages", "pending_conversations", "expected_status", "expected_text"),
    [
        (0, 0, 0, "warning", "Transcript embeddings pending"),
        (2, 50, 1, "warning", "Transcript embeddings pending"),
        (3, 90, 0, "ok", "Transcript embeddings ready"),
    ],
)
def test_run_health_embedding_status_contract(
    cli_workspace,
    embedded_conversations: int,
    embedded_messages: int,
    pending_conversations: int,
    expected_status: str,
    expected_text: str,
) -> None:
    from polylogue.config import get_config
    from polylogue.health import run_archive_health
    from polylogue.maintenance_models import DerivedModelStatus
    from tests.infra.storage_records import ConversationBuilder

    ConversationBuilder(cli_workspace["db_path"], "health-embed-1").add_message(text="hello").save()
    ConversationBuilder(cli_workspace["db_path"], "health-embed-2").add_message(text="hello").save()
    ConversationBuilder(cli_workspace["db_path"], "health-embed-3").add_message(text="hello").save()

    embeddings_ready = (
        embedded_conversations == 3 and pending_conversations == 0
    )
    expected_detail = (
        f"Transcript embeddings ready ({embedded_conversations:,}/3 conversations, {embedded_messages:,} messages)"
        if embeddings_ready
        else (
            f"Transcript embeddings pending ({embedded_conversations:,}/3 conversations, "
            f"pending {pending_conversations:,}, stale 0, missing provenance 0)"
        )
    )
    with patch(
        "polylogue.storage.derived_status.collect_derived_model_statuses_sync",
        return_value={
            "messages_fts": DerivedModelStatus(
                name="messages_fts",
                ready=True,
                detail="Messages FTS ready",
                source_rows=3,
                materialized_rows=3,
            ),
            "action_events": DerivedModelStatus(
                name="action_events",
                ready=True,
                detail="Action events ready",
                source_documents=3,
                materialized_documents=3,
                materialized_rows=0,
            ),
            "action_events_fts": DerivedModelStatus(
                name="action_events_fts",
                ready=True,
                detail="Action-event FTS ready",
                source_rows=0,
                materialized_rows=0,
            ),
            "transcript_embeddings": DerivedModelStatus(
                name="transcript_embeddings",
                ready=embeddings_ready,
                detail=expected_detail,
                source_documents=3,
                materialized_documents=embedded_conversations,
                materialized_rows=embedded_messages,
                pending_documents=pending_conversations,
            ),
            "retrieval_evidence": DerivedModelStatus(
                name="retrieval_evidence",
                ready=True,
                detail="Evidence retrieval ready",
                source_rows=3,
                materialized_rows=3,
            ),
            "retrieval_inference": DerivedModelStatus(
                name="retrieval_inference",
                ready=True,
                detail="Inference retrieval ready",
                source_rows=3,
                materialized_rows=3,
            ),
            "retrieval_enrichment": DerivedModelStatus(
                name="retrieval_enrichment",
                ready=True,
                detail="Enrichment retrieval ready",
                source_rows=3,
                materialized_rows=3,
            ),
            "session_profile_enrichment_fts": DerivedModelStatus(
                name="session_profile_enrichment_fts",
                ready=True,
                detail="Session-profile enrichment FTS ready",
                source_rows=3,
                materialized_rows=3,
            ),
        },
    ):
        report = run_archive_health(get_config())

    check = next(c for c in report.checks if c.name == "transcript_embeddings")
    assert check.status.value == expected_status
    assert expected_text in check.summary


@pytest.mark.parametrize(
    ("path_name", "missing"),
    [("archive_root", True), ("render_root", True), ("archive_root", False), ("render_root", False)],
)
def test_run_health_path_contracts(tmp_path: Path, path_name: str, missing: bool) -> None:
    from polylogue.config import Config, Source
    from polylogue.health import VerifyStatus, run_archive_health

    archive_root = tmp_path / "archive"
    render_root = tmp_path / "render"
    if path_name != "archive_root" or not missing:
        archive_root.mkdir(parents=True, exist_ok=True)
    if path_name != "render_root" or not missing:
        render_root.mkdir(parents=True, exist_ok=True)

    report = run_archive_health(
        Config(archive_root=archive_root, render_root=render_root, sources=[Source(name="test", path=tmp_path)])
    )
    check = next(c for c in report.checks if c.name == path_name)
    assert check.status == (VerifyStatus.WARNING if missing else VerifyStatus.OK)


def test_run_health_includes_source_checks(cli_workspace) -> None:
    from polylogue.config import get_config
    from polylogue.health import run_archive_health

    config = get_config()
    report = run_archive_health(config)
    source_checks = [check for check in report.checks if check.name.startswith("source:")]
    assert len(source_checks) >= len(config.sources)


@pytest.mark.parametrize("deep", [False, True])
def test_get_health_contract(cli_workspace, deep: bool) -> None:
    from polylogue.config import get_config
    from polylogue.health import get_health

    config = get_config()
    first = get_health(config, deep=deep)
    assert first.timestamp > 0
    # use_cached is accepted but ignored (no cache layer)
    second = get_health(config, use_cached=True)
    assert second.timestamp > 0


def test_cached_health_summary_returns_live_status(tmp_path: Path) -> None:
    from polylogue.health import cached_health_summary

    result = cached_health_summary(tmp_path)
    # With a valid DB, returns "OK (N conversations)"; without, returns "unavailable (...)"
    assert isinstance(result, str)
    assert "OK" in result or "unavailable" in result or "schema" in result


def test_verify_status_contract() -> None:
    from polylogue.health import HEALTH_TTL_SECONDS, VerifyStatus

    assert str(VerifyStatus.OK) == "ok"
    assert str(VerifyStatus.WARNING) == "warning"
    assert str(VerifyStatus.ERROR) == "error"
    assert 60 <= HEALTH_TTL_SECONDS <= 3600
