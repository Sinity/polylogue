from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from polylogue.pipeline.services.planning_backlog import collect_parse_backlog, collect_validation_backlog
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.raw_ingest_artifacts import (
    RawIngestArtifactState,
    parse_backlog_query_spec,
    validation_backlog_query_spec,
)
from polylogue.storage.store import RawConversationRecord
from polylogue.types import ValidationStatus


def test_raw_ingest_artifact_state_classifies_quarantine_and_backlogs() -> None:
    passed_unparsed = RawIngestArtifactState(validation_status=ValidationStatus.PASSED)
    failed_unparsed = RawIngestArtifactState(validation_status=ValidationStatus.FAILED)
    unvalidated_parsed = RawIngestArtifactState(parsed_at="2026-04-13T00:00:00Z")

    assert passed_unparsed.needs_validation_backlog() is False
    assert passed_unparsed.needs_parse_backlog() is True
    assert passed_unparsed.quarantined is False

    assert failed_unparsed.needs_validation_backlog() is False
    assert failed_unparsed.needs_parse_backlog() is False
    assert failed_unparsed.quarantined is True

    assert unvalidated_parsed.needs_validation_backlog() is False
    assert unvalidated_parsed.needs_parse_backlog() is False
    assert unvalidated_parsed.needs_validation_backlog(force_reparse=True) is True
    assert unvalidated_parsed.needs_parse_backlog(force_reparse=True) is True


def test_raw_ingest_backlog_query_specs_match_force_reparse_contract() -> None:
    ordinary_validate = validation_backlog_query_spec()
    ordinary_parse = parse_backlog_query_spec()
    forced_validate = validation_backlog_query_spec(force_reparse=True)
    forced_parse = parse_backlog_query_spec(force_reparse=True)

    assert ordinary_validate.require_unparsed is True
    assert ordinary_validate.require_unvalidated is True
    assert ordinary_validate.validation_statuses is None

    assert ordinary_parse.require_unparsed is True
    assert ordinary_parse.validation_statuses == ("passed", "skipped")

    assert forced_validate.require_unparsed is False
    assert forced_validate.require_unvalidated is True
    assert forced_validate.validation_statuses is None

    assert forced_parse.require_unparsed is False
    assert forced_parse.validation_statuses is None


@pytest.mark.asyncio
async def test_collect_backlogs_match_shared_raw_ingest_state(tmp_path: Path) -> None:
    backend = SQLiteBackend(db_path=tmp_path / "test.db")
    try:
        source_name = "inbox-a"
        raw_ids = (
            "raw-passed-unparsed",
            "raw-skipped-unparsed",
            "raw-failed-unparsed",
            "raw-passed-parsed",
            "raw-unvalidated-unparsed",
            "raw-unvalidated-parsed",
        )

        for raw_id in raw_ids:
            await backend.save_raw_conversation(
                RawConversationRecord(
                    raw_id=raw_id,
                    provider_name="chatgpt",
                    source_name=source_name,
                    source_path=f"/tmp/{raw_id}.json",
                    blob_size=len(b'{"id":"x"}'),
                    acquired_at=datetime.now(tz=timezone.utc).isoformat(),
                )
            )

        await backend.mark_raw_validated("raw-passed-unparsed", status="passed", provider="chatgpt", mode="strict")
        await backend.mark_raw_validated("raw-skipped-unparsed", status="skipped", provider="chatgpt", mode="strict")
        await backend.mark_raw_validated("raw-failed-unparsed", status="failed", provider="chatgpt", mode="strict")
        await backend.mark_raw_validated("raw-passed-parsed", status="passed", provider="chatgpt", mode="strict")
        await backend.mark_raw_parsed("raw-passed-parsed", payload_provider="chatgpt")
        await backend.mark_raw_parsed("raw-unvalidated-parsed", payload_provider="chatgpt")

        states = await backend.get_raw_conversation_states(list(raw_ids))

        ordinary_validation = set(
            await collect_validation_backlog(
                backend,
                source_names=[source_name],
                force_reparse=False,
            )
        )
        ordinary_parse = set(
            await collect_parse_backlog(
                backend,
                source_names=[source_name],
                force_reparse=False,
            )
        )
        forced_validation = set(
            await collect_validation_backlog(
                backend,
                source_names=[source_name],
                force_reparse=True,
            )
        )
        forced_parse = set(
            await collect_parse_backlog(
                backend,
                source_names=[source_name],
                force_reparse=True,
            )
        )

        expected_ordinary_validation = {
            raw_id
            for raw_id in raw_ids
            if RawIngestArtifactState.from_state(states[raw_id]).needs_validation_backlog(force_reparse=False)
        }
        expected_ordinary_parse = {
            raw_id
            for raw_id in raw_ids
            if RawIngestArtifactState.from_state(states[raw_id]).needs_parse_backlog(force_reparse=False)
        }
        expected_forced_validation = {
            raw_id
            for raw_id in raw_ids
            if RawIngestArtifactState.from_state(states[raw_id]).needs_validation_backlog(force_reparse=True)
        }
        expected_forced_parse = {
            raw_id
            for raw_id in raw_ids
            if RawIngestArtifactState.from_state(states[raw_id]).needs_parse_backlog(force_reparse=True)
        }

        assert ordinary_validation == expected_ordinary_validation == {"raw-unvalidated-unparsed"}
        assert ordinary_parse == expected_ordinary_parse == {"raw-passed-unparsed", "raw-skipped-unparsed"}
        assert forced_validation == expected_forced_validation == {
            "raw-unvalidated-parsed",
            "raw-unvalidated-unparsed",
        }
        assert forced_parse == expected_forced_parse == set(raw_ids)
    finally:
        await backend.close()
