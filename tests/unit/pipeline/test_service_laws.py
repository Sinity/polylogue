"""Law-based contracts for pipeline services."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import HealthCheck, given, settings

from polylogue.config import Source
from polylogue.pipeline.services.acquisition import AcquisitionService
from polylogue.pipeline.services.parsing import ParseResult, ParsingService
from polylogue.pipeline.services.validation import ValidationService
from polylogue.sources.parsers.base import RawConversationData
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from tests.infra.strategies import (
    acquisition_input_batch_strategy,
    build_acquisition_raw_bytes,
    build_validation_payload,
    expected_parse_merge_totals,
    expected_validation_contract,
    parse_merge_events_strategy,
    validation_case_strategy,
)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(acquisition_input_batch_strategy())
async def test_acquisition_law_counts_unique_raws_and_uses_provider_fallback(batch) -> None:
    """Acquisition should store each unique raw payload once and preserve fallback provider naming."""
    with TemporaryDirectory() as tempdir:
        backend = SQLiteBackend(db_path=Path(tempdir) / "acquire.db")
        source_name = "generated-source"

        raw_items = [
            RawConversationData(
                raw_bytes=build_acquisition_raw_bytes(spec),
                source_path=f"/tmp/{index}.json",
                source_index=index,
                provider_hint=spec.provider_hint,
            )
            for index, spec in enumerate(batch)
        ]

        expected_first_provider: dict[str, str] = {}
        for spec in batch:
            expected_first_provider.setdefault(spec.payload_id, spec.provider_hint or source_name)

        try:
            with patch("polylogue.pipeline.services.acquisition.iter_source_raw_data", return_value=iter(raw_items)):
                result = await AcquisitionService(backend=backend).acquire_sources(
                    [Source(name=source_name, path=Path("/tmp/inbox"))]
                )

            unique_payloads = list(dict.fromkeys(spec.payload_id for spec in batch))
            assert result.counts["acquired"] == len(unique_payloads)
            assert result.counts["skipped"] == len(batch) - len(unique_payloads)
            assert len(result.raw_ids) == len(unique_payloads)

            for raw_id in result.raw_ids:
                stored = await backend.get_raw_conversation(raw_id)
                assert stored is not None
                payload_id = json.loads(stored.raw_content)["id"]
                assert stored.provider_name == expected_first_provider[payload_id]
                assert stored.payload_provider is None
        finally:
            await backend.close()


@settings(max_examples=30, deadline=None)
@given(validation_case_strategy())
async def test_validation_law_matches_mode_and_payload_contract(case) -> None:
    """Validation mode, malformed JSONL, and schema verdicts must produce one stable persisted contract."""
    from polylogue.schemas import ValidationResult

    raw_content, provider_name, source_path = build_validation_payload(case)
    raw_record = MagicMock(
        raw_id="raw-1",
        raw_content=raw_content,
        provider_name=provider_name,
        source_path=source_path,
        payload_provider=None,
    )
    backend = MagicMock()
    backend.get_raw_conversations_batch = AsyncMock(return_value=[raw_record])
    backend.mark_raw_validated = AsyncMock()
    backend.mark_raw_parsed = AsyncMock()

    class _SyntheticValidator:
        provider = provider_name

        def __init__(self) -> None:
            self.max_samples_seen = "unset"

        def validation_samples(self, payload, max_samples=None):
            self.max_samples_seen = max_samples
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
            return [payload]

        def validate(self, _sample):
            return ValidationResult(
                is_valid=case.invalid_sample_count == 0,
                errors=["schema error"] if case.invalid_sample_count else [],
            )

    validator = _SyntheticValidator()

    with patch(
        "polylogue.schemas.validator.SchemaValidator.for_provider",
        return_value=validator,
    ):
        with patch.dict("os.environ", {"POLYLOGUE_SCHEMA_VALIDATION": case.mode}, clear=False):
            result = await ValidationService(backend=backend).validate_raw_ids(raw_ids=["raw-1"])

    expected = expected_validation_contract(case)
    if expected["validation_samples_called"]:
        assert validator.max_samples_seen is None
    else:
        assert validator.max_samples_seen == "unset"
    assert result.counts["invalid"] == expected["invalid_count"]
    assert result.parseable_raw_ids == (["raw-1"] if expected["parseable"] else [])
    assert result.invalid_raw_ids == ([] if expected["parseable"] else ["raw-1"])

    mark_validated = backend.mark_raw_validated.await_args.kwargs
    assert mark_validated["status"] == expected["status"]
    if expected["mark_raw_parsed"]:
        backend.mark_raw_parsed.assert_awaited_once()
    else:
        backend.mark_raw_parsed.assert_not_awaited()


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(parse_merge_events_strategy())
async def test_parse_result_merge_law_accumulates_counts_and_processed_ids(events) -> None:
    """ParseResult.merge_result should be componentwise additive with processed-id union semantics."""
    result = ParseResult()
    for event in events:
        await result.merge_result(
            conversation_id=event.conversation_id,
            result_counts=event.result_counts,
            content_changed=event.content_changed,
        )

    expected = expected_parse_merge_totals(events)
    assert result.counts == expected["counts"]
    assert result.changed_counts == expected["changed_counts"]
    assert result.processed_ids == expected["processed_ids"]


async def test_parse_raw_record_contract_updates_payload_provider_and_dispatches_once(tmp_path: Path) -> None:
    """_parse_raw_record should classify once, persist payload_provider on the record, and dispatch that provider."""
    repository = MagicMock()
    repository.backend = MagicMock()
    service = ParsingService(repository=repository, archive_root=tmp_path / "archive", config=MagicMock())
    raw_record = MagicMock(
        raw_content=b'{"id":"conv-1"}',
        source_path="/tmp/conversation.json",
        provider_name="chatgpt",
        payload_provider="",
        raw_id="raw-1",
    )
    envelope = MagicMock(provider="gemini", payload={"id": "parsed"})

    with patch("polylogue.pipeline.services.parsing.build_raw_payload_envelope", return_value=envelope) as mock_envelope:
        with patch("polylogue.pipeline.services.parsing.parse_payload", return_value=["parsed"]) as mock_parse:
            result = await service._parse_raw_record(raw_record)

    mock_envelope.assert_called_once_with(
        raw_record.raw_content,
        source_path=raw_record.source_path,
        fallback_provider=raw_record.provider_name,
        payload_provider=None,
    )
    mock_parse.assert_called_once_with("gemini", {"id": "parsed"}, "raw-1")
    assert raw_record.payload_provider == "gemini"
    assert result == ["parsed"]
