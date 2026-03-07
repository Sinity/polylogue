"""Tests for pipeline service classes.

Consolidated from:
- test_pipeline_services.py (service initialization tests)
- test_pipeline_services_acquisition.py (AcquisitionService tests)
- test_pipeline_services_parsing.py (ParsingService tests)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.config import Config, Source
from polylogue.pipeline.services import IndexService, IngestPlan
from polylogue.pipeline.services.acquisition import AcquireResult, AcquisitionService
from polylogue.pipeline.services.parsing import ParseResult, ParsingService
from polylogue.pipeline.services.planning import PlanningService
from polylogue.pipeline.services.validation import ValidationService
from polylogue.sources.parsers.base import RawConversationData
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import PlanResult, RawConversationRecord

# ============================================================================
# IndexService Tests
# ============================================================================


class TestPipelineIndexService:
    """Tests for IndexService."""

    def test_initialization(self, tmp_path: Path):
        """IndexService should initialize with config."""
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        service = IndexService(config)
        assert service.config is config

    async def test_update_index_empty_list(self, tmp_path: Path, workspace_env):
        """IndexService should handle empty conversation list."""
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        service = IndexService(config, backend=backend)
        assert await service.update_index([]) is True

    async def test_get_index_status_when_no_index(self, tmp_path: Path):
        """IndexService should return status when index doesn't exist."""
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        service = IndexService(config)
        status = await service.get_index_status()
        assert "exists" in status and "count" in status


# ============================================================================
# AcquireResult Tests
# ============================================================================


class TestAcquireResult:
    """Tests for AcquireResult."""

    @pytest.mark.parametrize("count_field", ["acquired", "skipped", "errors"])
    def test_counts_initialized_to_zero(self, count_field):
        """All count fields start at zero."""
        assert AcquireResult().counts[count_field] == 0



# ============================================================================
# AcquisitionService Tests
# ============================================================================


class TestAcquisitionServiceAcquireSources:
    """Tests for AcquisitionService.acquire_sources method."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        return SQLiteBackend(db_path=tmp_path / "test.db")

    def test_init_sets_backend(self, backend: SQLiteBackend):
        """Backend is stored on init."""
        service = AcquisitionService(backend=backend)
        assert service.backend is backend

    async def test_acquire_empty_sources(self, backend: SQLiteBackend):
        """Empty sources list returns empty result."""
        result = await AcquisitionService(backend=backend).acquire_sources([])
        assert all(result.counts[k] == 0 for k in ["acquired", "skipped", "errors"]) and result.raw_ids == []

    @pytest.mark.parametrize(
        "num_convos,expected_acquired,expected_skipped,raw_content",
        [
            (1, 1, 0, b'{"id": "test-conv", "messages": []}'),
            (3, 3, 0, b'{"id": "conv-0"}'),
            (2, 1, 1, b'{"id": "same-conv"}'),
        ],
    )
    @patch("polylogue.pipeline.services.acquisition.iter_source_raw_data")
    async def test_acquire_conversations(
        self,
        mock_iter,
        backend: SQLiteBackend,
        num_convos: int,
        expected_acquired: int,
        expected_skipped: int,
        raw_content: bytes,
    ):
        """Acquire conversations: single, multiple, or duplicates."""
        if num_convos == 1:
            raw_data = RawConversationData(
                raw_bytes=raw_content,
                source_path="/tmp/test.json",
                source_index=0,
                provider_hint="chatgpt",
            )
            mock_iter.return_value = iter([raw_data])
        elif num_convos == 3:
            convos = [
                RawConversationData(
                    raw_bytes=f'{{"id": "conv-{i}"}}'.encode(),
                    source_path="/tmp/test.json",
                    source_index=i,
                    provider_hint="chatgpt",
                )
                for i in range(3)
            ]
            mock_iter.return_value = iter(convos)
        else:  # num_convos == 2
            raw_data = RawConversationData(
                raw_bytes=raw_content,
                source_path="/tmp/test.json",
                source_index=0,
                provider_hint="chatgpt",
            )
            mock_iter.return_value = iter([raw_data, raw_data])

        service = AcquisitionService(backend=backend)
        source = Source(name="test-source", path=Path("/tmp/inbox"))
        result = await service.acquire_sources([source])

        assert result.counts["acquired"] == expected_acquired
        assert result.counts["skipped"] == expected_skipped
        assert len(result.raw_ids) == expected_acquired

        if expected_acquired > 0:
            stored = await backend.get_raw_conversation(result.raw_ids[0])
            assert stored is not None
            assert stored.raw_content == (raw_content if num_convos != 3 else b'{"id": "conv-0"}')
            assert stored.provider_name == "chatgpt"

    @pytest.mark.parametrize(
        "provider_hint,source_name,expected_provider",
        [("chatgpt", "test-source", "chatgpt"), (None, "my-inbox", "my-inbox")],
    )
    @patch("polylogue.pipeline.services.acquisition.iter_source_raw_data")
    async def test_acquire_provider_fallback(
        self,
        mock_iter,
        backend: SQLiteBackend,
        provider_hint: str | None,
        source_name: str,
        expected_provider: str,
    ):
        """Source name is used as fallback provider when provider_hint is None."""
        raw_data = RawConversationData(
            raw_bytes=b'{"id": "test"}',
            source_path="/tmp/test.json",
            source_index=0,
            provider_hint=provider_hint,
        )
        mock_iter.return_value = iter([raw_data])
        service = AcquisitionService(backend=backend)
        source = Source(name=source_name, path=Path("/tmp/inbox"))
        result = await service.acquire_sources([source])
        stored = await backend.get_raw_conversation(result.raw_ids[0])
        assert stored is not None
        assert stored.provider_name == expected_provider

    @patch("polylogue.pipeline.services.acquisition.iter_source_raw_data")
    async def test_progress_callback_called(self, mock_iter, backend: SQLiteBackend):
        """Progress callback is invoked for each conversation."""
        raw_data = RawConversationData(
            raw_bytes=b'{"id": "test"}',
            source_path="/tmp/test.json",
            source_index=0,
        )
        mock_iter.return_value = iter([raw_data])
        backend.get_known_source_mtimes = AsyncMock(return_value={})
        callback = MagicMock()
        service = AcquisitionService(backend=backend)
        source = Source(name="test-source", path=Path("/tmp/inbox"))
        await service.acquire_sources([source], progress_callback=callback)
        callback.assert_any_call(1, desc="Acquiring [test-source]")
        assert mock_iter.call_args is not None
        kwargs = mock_iter.call_args.kwargs
        assert kwargs.get("known_mtimes") is not None

    @pytest.mark.parametrize("error_scenario", ["iteration_error", "none_raw_data"])
    @patch("polylogue.pipeline.services.acquisition.iter_source_raw_data")
    async def test_acquire_handles_errors(self, mock_iter, backend: SQLiteBackend, error_scenario: str):
        """Errors during source iteration are counted."""
        if error_scenario == "iteration_error":
            mock_iter.side_effect = ValueError("File not found")
        else:
            mock_iter.return_value = iter([None])
        service = AcquisitionService(backend=backend)
        source = Source(name="test-source", path=Path("/tmp/inbox"))
        result = await service.acquire_sources([source])
        assert result.counts["errors"] == 1
        assert result.counts["acquired"] == 0


class TestAcquisitionServiceIntegration:
    """Integration tests for AcquisitionService with real files."""

    def _make_conv(self, cid: str, title: str, time: int, msg: str) -> dict:
        return {"id": cid, "title": title, "create_time": time, "update_time": time+100, "mapping": {
            "root": {"id": "root", "message": None, "children": ["msg1"]},
            "msg1": {"id": "msg1", "message": {"id": "msg1", "author": {"role": "user"},
                "content": {"content_type": "text", "parts": [msg]}, "create_time": time+50},
                "parent": "root", "children": []}}}

    async def test_acquire_real_chatgpt_file(self, tmp_path: Path):
        """Acquire from a real ChatGPT conversations.json file."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        (inbox / "conversations.json").write_text(json.dumps([self._make_conv("conv-1", "Test Chat", 1700000000, "Hello")]))
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        result = await AcquisitionService(backend=backend).acquire_sources([Source(name="chatgpt-inbox", path=inbox)])
        assert result.counts["acquired"] == 1 and result.counts["errors"] == 0 and len(result.raw_ids) == 1
        stored = await backend.get_raw_conversation(result.raw_ids[0])
        data = json.loads(stored.raw_content)
        assert stored.provider_name == "chatgpt"
        assert isinstance(data, list)
        assert data[0]["id"] == "conv-1" and data[0]["title"] == "Test Chat"

    async def test_acquire_multiple_json_files(self, tmp_path: Path):
        """Acquire stores one raw payload per file (not per conversation in a bundle)."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        convs = [self._make_conv("conv-1", "Chat 1", 1700000000, "Hello"),
                 self._make_conv("conv-2", "Chat 2", 1700000200, "World")]
        (inbox / "conversations.json").write_text(json.dumps(convs))
        result = await AcquisitionService(backend=SQLiteBackend(db_path=tmp_path / "test.db")).acquire_sources([Source(name="chatgpt-export", path=inbox)])
        assert result.counts["acquired"] == 1 and len(result.raw_ids) == 1


# ============================================================================
# ValidationService Tests
# ============================================================================


class TestValidationService:
    """Tests for raw payload schema validation stage."""

    def test_validation_default_mode_is_strict(self, monkeypatch):
        """Unset env should default validation mode to strict."""
        monkeypatch.delenv("POLYLOGUE_SCHEMA_VALIDATION", raising=False)
        service = ValidationService(backend=MagicMock())
        assert service._schema_validation_mode() == "strict"

    async def test_validation_off_mode_skips_processing(self, monkeypatch):
        """off mode should skip schema checks but persist skipped validation state."""
        backend = MagicMock()
        backend.get_raw_conversations_batch = AsyncMock()
        backend.mark_raw_validated = AsyncMock()
        service = ValidationService(backend=backend)
        monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "off")

        result = await service.validate_raw_ids(raw_ids=["raw-1", "raw-2"])

        assert result.parseable_raw_ids == ["raw-1", "raw-2"]
        backend.get_raw_conversations_batch.assert_not_called()
        assert backend.mark_raw_validated.await_count == 2

    async def test_validation_strict_blocks_invalid_payloads(self, monkeypatch):
        """strict mode should block invalid payloads and persist parse_error."""
        from polylogue.schemas import ValidationResult

        raw_record = MagicMock(
            raw_id="raw-1",
            raw_content=b'{"id": 1}',
            provider_name="chatgpt",
            source_path="/tmp/conversations.json",
        )
        backend = MagicMock()
        backend.get_raw_conversations_batch = AsyncMock(return_value=[raw_record])
        backend.mark_raw_validated = AsyncMock()
        backend.mark_raw_parsed = AsyncMock()

        class _AlwaysInvalidValidator:
            provider = "chatgpt"

            def validation_samples(self, payload, max_samples=16):
                return [payload]

            def validate(self, _sample):
                return ValidationResult(
                    is_valid=False,
                    errors=["id: 1 is not of type 'string'"],
                )

        monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "strict")
        monkeypatch.setattr(
            "polylogue.schemas.validator.SchemaValidator.for_provider",
            lambda _provider: _AlwaysInvalidValidator(),
        )

        result = await ValidationService(backend=backend).validate_raw_ids(raw_ids=["raw-1"])

        assert result.counts["invalid"] == 1
        assert result.parseable_raw_ids == []
        backend.mark_raw_validated.assert_awaited_once()
        backend.mark_raw_parsed.assert_called_once()

    async def test_validation_advisory_reports_invalid_but_keeps_parseable(self, monkeypatch):
        """advisory mode should report invalid payloads without blocking parse."""
        from polylogue.schemas import ValidationResult

        raw_record = MagicMock(
            raw_id="raw-1",
            raw_content=b'{"id": 1}',
            provider_name="chatgpt",
            source_path="/tmp/conversations.json",
        )
        backend = MagicMock()
        backend.get_raw_conversations_batch = AsyncMock(return_value=[raw_record])
        backend.mark_raw_validated = AsyncMock()
        backend.mark_raw_parsed = AsyncMock()

        class _AlwaysInvalidValidator:
            provider = "chatgpt"

            def validation_samples(self, payload, max_samples=16):
                return [payload]

            def validate(self, _sample):
                return ValidationResult(
                    is_valid=False,
                    errors=["id: 1 is not of type 'string'"],
                )

        monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "advisory")
        monkeypatch.setattr(
            "polylogue.schemas.validator.SchemaValidator.for_provider",
            lambda _provider: _AlwaysInvalidValidator(),
        )

        result = await ValidationService(backend=backend).validate_raw_ids(raw_ids=["raw-1"])

        assert result.counts["invalid"] == 1
        assert result.parseable_raw_ids == ["raw-1"]
        backend.mark_raw_validated.assert_awaited_once()
        kwargs = backend.mark_raw_validated.await_args.kwargs
        assert kwargs["status"] == "passed"
        backend.mark_raw_parsed.assert_not_called()

    async def test_validation_max_samples_all_uses_all_record_samples(self, monkeypatch):
        """`POLYLOGUE_SCHEMA_VALIDATION_MAX_SAMPLES=all` validates all JSONL dict records."""
        from polylogue.schemas import ValidationResult

        raw_record = MagicMock(
            raw_id="raw-1",
            raw_content=(
                b'{"type":"session_meta"}\n'
                b'{"type":"response_item","payload":{"type":"message"}}\n'
                b'{"record_type":"state"}'
            ),
            provider_name="codex",
            source_path="/tmp/session.jsonl",
        )
        backend = MagicMock()
        backend.get_raw_conversations_batch = AsyncMock(return_value=[raw_record])
        backend.mark_raw_validated = AsyncMock()
        backend.mark_raw_parsed = AsyncMock()

        class _CapturingValidator:
            provider = "codex"

            def __init__(self):
                self.max_samples_seen = None

            def validation_samples(self, payload, max_samples=16):
                self.max_samples_seen = max_samples
                return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else [payload]

            def validate(self, _sample):
                return ValidationResult(is_valid=True)

        capturing = _CapturingValidator()
        monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION_MAX_SAMPLES", "all")
        monkeypatch.setattr(
            "polylogue.schemas.validator.SchemaValidator.for_provider",
            lambda _provider: capturing,
        )

        result = await ValidationService(backend=backend).validate_raw_ids(raw_ids=["raw-1"])

        assert result.parseable_raw_ids == ["raw-1"]
        assert capturing.max_samples_seen == 3

    async def test_validation_strict_fails_on_malformed_jsonl_lines(self, monkeypatch):
        """Strict mode should fail payloads that contain malformed JSONL lines."""
        from polylogue.schemas import ValidationResult

        raw_record = MagicMock(
            raw_id="raw-1",
            raw_content=(
                b'{"type":"session_meta"}\n'
                b'not json at all\n'
                b'{"type":"response_item","payload":{"type":"message"}}'
            ),
            provider_name="codex",
            source_path="/tmp/session.jsonl",
        )
        backend = MagicMock()
        backend.get_raw_conversations_batch = AsyncMock(return_value=[raw_record])
        backend.mark_raw_validated = AsyncMock()
        backend.mark_raw_parsed = AsyncMock()

        class _AlwaysValidValidator:
            provider = "codex"

            def validation_samples(self, payload, max_samples=16):
                return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else [payload]

            def validate(self, _sample):
                return ValidationResult(is_valid=True)

        monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "strict")
        monkeypatch.setattr(
            "polylogue.schemas.validator.SchemaValidator.for_provider",
            lambda _provider: _AlwaysValidValidator(),
        )

        result = await ValidationService(backend=backend).validate_raw_ids(raw_ids=["raw-1"])

        assert result.counts["invalid"] == 1
        assert result.parseable_raw_ids == []
        kwargs = backend.mark_raw_validated.await_args.kwargs
        assert kwargs["status"] == "failed"
        assert "Malformed JSONL lines" in (kwargs.get("error") or "")
        backend.mark_raw_parsed.assert_awaited_once()

    async def test_validation_advisory_allows_malformed_jsonl_lines(self, monkeypatch):
        """Advisory mode should keep malformed JSONL payloads parseable."""
        from polylogue.schemas import ValidationResult

        raw_record = MagicMock(
            raw_id="raw-1",
            raw_content=(
                b'{"type":"session_meta"}\n'
                b'not json at all\n'
                b'{"type":"response_item","payload":{"type":"message"}}'
            ),
            provider_name="codex",
            source_path="/tmp/session.jsonl",
        )
        backend = MagicMock()
        backend.get_raw_conversations_batch = AsyncMock(return_value=[raw_record])
        backend.mark_raw_validated = AsyncMock()
        backend.mark_raw_parsed = AsyncMock()

        class _AlwaysValidValidator:
            provider = "codex"

            def validation_samples(self, payload, max_samples=16):
                return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else [payload]

            def validate(self, _sample):
                return ValidationResult(is_valid=True)

        monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "advisory")
        monkeypatch.setattr(
            "polylogue.schemas.validator.SchemaValidator.for_provider",
            lambda _provider: _AlwaysValidValidator(),
        )

        result = await ValidationService(backend=backend).validate_raw_ids(raw_ids=["raw-1"])

        assert result.parseable_raw_ids == ["raw-1"]
        kwargs = backend.mark_raw_validated.await_args.kwargs
        assert kwargs["status"] == "passed"
        backend.mark_raw_parsed.assert_not_called()


# ============================================================================
# ParseResult Tests
# ============================================================================


class TestParseResultInit:
    """Tests for ParseResult initialization."""

    @pytest.mark.parametrize("count_field", ["conversations", "messages", "attachments", "skipped_conversations", "skipped_messages", "skipped_attachments"])
    def test_counts_initialized_to_zero(self, count_field):
        """All count fields start at zero."""
        assert ParseResult().counts[count_field] == 0

    @pytest.mark.parametrize("changed_count_field", ["conversations", "messages", "attachments"])
    def test_changed_counts_initialized_to_zero(self, changed_count_field):
        """All changed_counts fields start at zero."""
        assert ParseResult().changed_counts[changed_count_field] == 0



class TestParseResultMerge:
    """Tests for ParseResult.merge_result method."""

    @pytest.mark.parametrize("conv_id,result_counts,content_changed,exp_c,exp_m,exp_a,exp_s,in_p", [
        ("conv1", {"conversations": 1, "messages": 5, "attachments": 2, "skipped_conversations": 0, "skipped_messages": 1, "skipped_attachments": 0}, True, 1, 5, 2, 1, True),
        ("conv1", {"conversations": 1, "messages": 2, "attachments": 0, "skipped_conversations": 0, "skipped_messages": 0, "skipped_attachments": 0}, True, 1, 2, 0, 0, True),
        ("conv1", {"conversations": 0, "messages": 0, "attachments": 0, "skipped_conversations": 1, "skipped_messages": 0, "skipped_attachments": 0}, False, 0, 0, 0, 0, False),
        ("conv-123", {"conversations": 1, "messages": 1, "attachments": 0, "skipped_conversations": 0, "skipped_messages": 0, "skipped_attachments": 0}, True, 1, 1, 0, 0, True),
        ("conv-456", {"conversations": 1, "messages": 5, "attachments": 0, "skipped_conversations": 0, "skipped_messages": 0, "skipped_attachments": 0}, False, 1, 5, 0, 0, True),
        ("conv-skipped", {"conversations": 0, "messages": 0, "attachments": 0, "skipped_conversations": 1, "skipped_messages": 5, "skipped_attachments": 0}, False, 0, 0, 0, 5, False),
    ])
    async def test_merge_result_scenarios(self, conv_id: str, result_counts: dict, content_changed: bool, exp_c: int, exp_m: int, exp_a: int, exp_s: int, in_p: bool):
        """Parametrized merge scenarios."""
        result = ParseResult()
        await result.merge_result(conversation_id=conv_id, result_counts=result_counts, content_changed=content_changed)
        assert result.counts["conversations"] == exp_c and result.counts["messages"] == exp_m
        assert result.counts["attachments"] == exp_a and result.counts["skipped_messages"] == exp_s
        assert (conv_id in result.processed_ids) == in_p

    async def test_merge_multiple_conversations(self):
        """Multiple merges accumulate correctly."""
        result = ParseResult()
        await result.merge_result(
            "conv1",
            {
                "conversations": 1,
                "messages": 3,
                "attachments": 1,
                "skipped_conversations": 0,
                "skipped_messages": 0,
                "skipped_attachments": 0,
            },
            content_changed=True,
        )
        await result.merge_result(
            "conv2",
            {
                "conversations": 1,
                "messages": 7,
                "attachments": 0,
                "skipped_conversations": 0,
                "skipped_messages": 2,
                "skipped_attachments": 0,
            },
            content_changed=True,
        )
        assert result.counts["conversations"] == 2
        assert result.counts["messages"] == 10
        assert result.counts["attachments"] == 1
        assert result.counts["skipped_messages"] == 2

    async def test_merge_thread_safe(self):
        """Concurrent merges don't cause race conditions."""
        result = ParseResult()

        async def merge_batch(start_id: int) -> None:
            for i in range(100):
                await result.merge_result(
                    f"conv-{start_id}-{i}",
                    {
                        "conversations": 1,
                        "messages": 2,
                        "attachments": 1,
                        "skipped_conversations": 0,
                        "skipped_messages": 0,
                        "skipped_attachments": 0,
                    },
                    content_changed=True,
                )

        tasks = [merge_batch(i) for i in range(5)]
        await asyncio.gather(*tasks)

        assert result.counts["conversations"] == 500
        assert result.counts["messages"] == 1000
        assert result.counts["attachments"] == 500
        assert len(result.processed_ids) == 500


# ============================================================================
# ParsingService Tests
# ============================================================================


class TestParsingServiceInit:
    """Tests for ParsingService initialization."""

    @pytest.mark.parametrize(
        "attr_name",
        ["repository", "archive_root", "config"],
    )
    def test_init_attributes(self, attr_name: str):
        """Service initialization sets required attributes."""
        mock_repo = MagicMock()
        mock_config = MagicMock(spec=Config)
        archive_path = Path("/tmp/archive")

        service = ParsingService(
            repository=mock_repo,
            archive_root=archive_path,
            config=mock_config,
        )

        if attr_name == "repository":
            assert service.repository is mock_repo
        elif attr_name == "archive_root":
            assert service.archive_root == archive_path
        elif attr_name == "config":
            assert service.config is mock_config


class TestParsingServiceParseSources:
    """Tests for ParsingService.parse_sources method.

    parse_sources() orchestrates:
    1. ACQUIRE stage via AcquisitionService.acquire_sources()
    2. VALIDATE stage via ValidationService.validate_raw_ids()
    3. PARSE stage via self.parse_from_raw()

    These tests mock the stage boundaries to verify orchestration logic.
    """

    async def test_ingest_empty_sources_returns_empty_result(self):
        """Empty sources list returns empty ParseResult via an empty canonical plan."""
        mock_repo = MagicMock()
        mock_backend = MagicMock()
        mock_repo.backend = mock_backend
        mock_config = MagicMock(spec=Config)

        service = ParsingService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        empty_plan = IngestPlan(
            summary=PlanResult(timestamp=0, stage="all", counts={}, sources=[], cursors={}),
            store_records=[],
            validate_records=[],
            parse_ready_raw_ids=[],
        )
        with patch(
            "polylogue.pipeline.services.planning.PlanningService.build_plan",
            new=AsyncMock(return_value=empty_plan),
        ) as mock_build_plan:
            with patch(
                "polylogue.pipeline.services.acquisition.AcquisitionService.store_records",
                new=AsyncMock(return_value=AcquireResult()),
            ) as mock_store_records:
                with patch.object(service, "parse_from_raw", new_callable=AsyncMock) as mock_parse:
                    result = await service.parse_sources([])

        mock_build_plan.assert_awaited_once()
        mock_store_records.assert_awaited_once_with([])
        mock_parse.assert_not_called()

        assert result.counts["conversations"] == 0
        assert result.counts["messages"] == 0
        assert len(result.processed_ids) == 0

    async def test_ingest_calls_acquire_then_parse(self):
        """parse_sources executes the planner, stores raws, validates, then parses."""
        mock_repo = MagicMock()
        mock_backend = MagicMock()
        mock_repo.backend = mock_backend
        mock_config = MagicMock(spec=Config)

        service = ParsingService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        planned_records = [
            RawConversationRecord(
                raw_id="raw-1",
                provider_name="chatgpt",
                source_name="test-source",
                source_path="/tmp/a.json",
                raw_content=b'{"id":"a"}',
                acquired_at=datetime.now(tz=timezone.utc).isoformat(),
            ),
            RawConversationRecord(
                raw_id="raw-2",
                provider_name="chatgpt",
                source_name="test-source",
                source_path="/tmp/b.json",
                raw_content=b'{"id":"b"}',
                acquired_at=datetime.now(tz=timezone.utc).isoformat(),
            ),
        ]
        plan = IngestPlan(
            summary=PlanResult(
                timestamp=0,
                stage="all",
                counts={"scan": 2, "store_raw": 2, "validate": 2, "parse": 2},
                sources=["test-source"],
                cursors={},
            ),
            store_records=planned_records,
            validate_records=planned_records,
            parse_ready_raw_ids=[],
        )
        acquire_result = AcquireResult()
        acquire_result.raw_ids = ["raw-1", "raw-2"]
        acquire_result.counts["acquired"] = 2
        validation_result = MagicMock()
        validation_result.parseable_raw_ids = ["raw-1", "raw-2"]

        with patch(
            "polylogue.pipeline.services.planning.PlanningService.build_plan",
            new=AsyncMock(return_value=plan),
        ) as mock_build_plan:
            with patch(
                "polylogue.pipeline.services.acquisition.AcquisitionService.store_records",
                new=AsyncMock(return_value=acquire_result),
            ) as mock_store_records:
                with patch(
                    "polylogue.pipeline.services.validation.ValidationService.validate_raw_ids",
                    new=AsyncMock(return_value=validation_result),
                ) as mock_validate:
                    mock_parse_result = ParseResult()
                    mock_parse_result.counts["conversations"] = 2
                    mock_parse_result.counts["messages"] = 5
                    mock_parse_result.processed_ids = {"conv-1", "conv-2"}
                    with patch.object(
                        service, "parse_from_raw", new_callable=AsyncMock, return_value=mock_parse_result
                    ) as mock_parse:
                        source = Source(name="test-source", path=Path("/tmp/inbox"))
                        result = await service.parse_sources([source])

        mock_build_plan.assert_awaited_once()
        mock_store_records.assert_awaited_once_with(planned_records)
        mock_validate.assert_awaited_once_with(
            raw_ids=["raw-1", "raw-2"],
            progress_callback=None,
        )
        mock_parse.assert_awaited_once_with(
            raw_ids=["raw-1", "raw-2"],
            progress_callback=None,
        )

        assert result.counts["conversations"] == 2
        assert result.counts["messages"] == 5
        assert result.processed_ids == {"conv-1", "conv-2"}

    async def test_ingest_skips_parse_when_nothing_acquired(self):
        """If the canonical plan has nothing parseable, parse stage is skipped."""
        mock_repo = MagicMock()
        mock_backend = MagicMock()
        mock_repo.backend = mock_backend
        mock_config = MagicMock(spec=Config)

        service = ParsingService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        plan = IngestPlan(
            summary=PlanResult(timestamp=0, stage="all", counts={}, sources=["test-source"], cursors={}),
            store_records=[],
            validate_records=[],
            parse_ready_raw_ids=[],
        )
        acquire_result = AcquireResult()
        acquire_result.counts["skipped"] = 5
        with patch(
            "polylogue.pipeline.services.planning.PlanningService.build_plan",
            new=AsyncMock(return_value=plan),
        ):
            with patch(
                "polylogue.pipeline.services.acquisition.AcquisitionService.store_records",
                new=AsyncMock(return_value=acquire_result),
            ):
                with patch.object(service, "parse_from_raw", new_callable=AsyncMock) as mock_parse:
                    source = Source(name="test-source", path=Path("/tmp/inbox"))
                    result = await service.parse_sources([source])

        mock_parse.assert_not_called()

        assert result.counts["conversations"] == 0

    async def test_progress_callback_passed_to_both_stages(self):
        """Progress callback is forwarded to planning, validation, and parse stages."""
        mock_repo = MagicMock()
        mock_backend = MagicMock()
        mock_repo.backend = mock_backend
        mock_config = MagicMock(spec=Config)

        service = ParsingService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        callback = MagicMock()
        planned_record = RawConversationRecord(
            raw_id="raw-1",
            provider_name="chatgpt",
            source_name="test-source",
            source_path="/tmp/a.json",
            raw_content=b'{"id":"a"}',
            acquired_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        plan = IngestPlan(
            summary=PlanResult(
                timestamp=0,
                stage="all",
                counts={"scan": 1, "store_raw": 1, "validate": 1, "parse": 1},
                sources=["test-source"],
                cursors={},
            ),
            store_records=[planned_record],
            validate_records=[planned_record],
            parse_ready_raw_ids=[],
        )
        acquire_result = AcquireResult()
        acquire_result.raw_ids = ["raw-1"]
        validation_result = MagicMock()
        validation_result.parseable_raw_ids = ["raw-1"]

        with patch(
            "polylogue.pipeline.services.planning.PlanningService.build_plan",
            new=AsyncMock(return_value=plan),
        ) as mock_build_plan:
            with patch(
                "polylogue.pipeline.services.acquisition.AcquisitionService.store_records",
                new=AsyncMock(return_value=acquire_result),
            ):
                with patch(
                    "polylogue.pipeline.services.validation.ValidationService.validate_raw_ids",
                    new=AsyncMock(return_value=validation_result),
                ) as mock_validate:
                    mock_parse_result = ParseResult()
                    with patch.object(
                        service, "parse_from_raw", new_callable=AsyncMock, return_value=mock_parse_result
                    ) as mock_parse:
                        source = Source(name="test-source", path=Path("/tmp/inbox"))
                        await service.parse_sources([source], progress_callback=callback)

        build_kwargs = mock_build_plan.await_args.kwargs
        assert build_kwargs["progress_callback"] is callback
        mock_validate.assert_awaited_once_with(
            raw_ids=["raw-1"],
            progress_callback=callback,
        )
        mock_parse.assert_awaited_once_with(
            raw_ids=["raw-1"],
            progress_callback=callback,
        )

    async def test_backend_not_initialized_raises(self):
        """RuntimeError raised if repository backend is None."""
        mock_repo = MagicMock()
        mock_repo.backend = None
        mock_config = MagicMock(spec=Config)

        service = ParsingService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        source = Source(name="test-source", path=Path("/tmp/inbox"))

        with pytest.raises(RuntimeError, match="backend is not initialized"):
            await service.parse_sources([source])

    async def test_planning_includes_scoped_validation_backlog(self, tmp_path: Path):
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        planner = PlanningService(backend=backend, config=config)
        source_dir = tmp_path / "inbox-a"
        source_dir.mkdir()

        await backend.save_raw_conversation(
            RawConversationRecord(
                raw_id="raw-scoped",
                provider_name="chatgpt",
                source_name="inbox-a",
                source_path="/tmp/a.json",
                raw_content=b'{"id":"a"}',
                acquired_at=datetime.now(tz=timezone.utc).isoformat(),
            )
        )
        await backend.save_raw_conversation(
            RawConversationRecord(
                raw_id="raw-legacy-provider",
                provider_name="inbox-a",
                source_name=None,
                source_path="/tmp/legacy.json",
                raw_content=b'{"id":"legacy"}',
                acquired_at=datetime.now(tz=timezone.utc).isoformat(),
            )
        )
        await backend.save_raw_conversation(
            RawConversationRecord(
                raw_id="raw-other",
                provider_name="chatgpt",
                source_name="inbox-b",
                source_path="/tmp/b.json",
                raw_content=b'{"id":"b"}',
                acquired_at=datetime.now(tz=timezone.utc).isoformat(),
            )
        )

        plan = await planner.build_plan(
            sources=[Source(name="inbox-a", path=source_dir)],
            stage="validate",
        )

        assert plan.summary.counts["validate"] == 2
        assert plan.summary.details["backlog_validate"] == 2
        assert {record.raw_id for record in plan.validate_records} == {"raw-scoped", "raw-legacy-provider"}

    async def test_planning_includes_only_parseable_backlog_statuses(self, tmp_path: Path):
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        planner = PlanningService(backend=backend, config=config)
        source_dir = tmp_path / "inbox-a"
        source_dir.mkdir()

        for raw_id, status in (
            ("raw-passed", "passed"),
            ("raw-skipped", "skipped"),
            ("raw-failed", "failed"),
        ):
            await backend.save_raw_conversation(
                RawConversationRecord(
                    raw_id=raw_id,
                    provider_name="chatgpt",
                    source_name="inbox-a",
                    source_path=f"/tmp/{raw_id}.json",
                    raw_content=b'{"id":"x"}',
                    acquired_at=datetime.now(tz=timezone.utc).isoformat(),
                )
            )
            await backend.mark_raw_validated(
                raw_id,
                status=status,
                provider="chatgpt",
                mode="strict",
            )

        plan = await planner.build_plan(
            sources=[Source(name="inbox-a", path=source_dir)],
            stage="parse",
        )

        assert plan.summary.counts["parse"] == 2
        assert plan.summary.details["backlog_parse"] == 2
        assert set(plan.parse_ready_raw_ids) == {"raw-passed", "raw-skipped"}


# ============================================================================
# ParsingService Integration Tests
# ============================================================================


class TestParsingServiceIntegration:
    """Integration tests for ParsingService with real database."""

    def _conv_json(self, cid: str, title: str, msg: str) -> dict:
        return {"id": cid, "title": title, "create_time": 1700000000, "update_time": 1700000100,
            "mapping": {"root": {"id": "root", "message": None, "children": ["msg1"]},
            "msg1": {"id": "msg1", "message": {"id": "msg1", "author": {"role": "user"},
                "content": {"content_type": "text", "parts": [msg]}, "create_time": 1700000050},
                "parent": "root", "children": []}}}

    async def test_ingest_with_real_database(self, cli_workspace, tmp_path, monkeypatch):
        """Full ingestion flow with real database."""
        monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "off")
        inbox = cli_workspace["inbox_dir"]
        (inbox / "conversations.json").write_text(json.dumps([self._conv_json("test-conv-1", "Test Conversation", "Hello, world!")]))
        backend = SQLiteBackend(db_path=cli_workspace["db_path"])
        config = Config(archive_root=cli_workspace["archive_root"], render_root=cli_workspace["render_root"],
                       sources=[Source(name="test-inbox", path=inbox)])
        result = await ParsingService(repository=ConversationRepository(backend=backend),
                                 archive_root=cli_workspace["archive_root"], config=config).parse_sources(config.sources)
        assert result.counts["conversations"] >= 1 and len(result.processed_ids) >= 1

    async def test_parse_from_raw_parses_stored_conversations(self, cli_workspace, tmp_path):
        """Full acquire -> parse flow using database-driven testing."""
        inbox = cli_workspace["inbox_dir"]
        (inbox / "conversations.json").write_text(json.dumps([self._conv_json("test-conv-raw", "Test Raw Conversation", "Hello from raw!")]))
        backend = SQLiteBackend(db_path=cli_workspace["db_path"])
        config = Config(archive_root=cli_workspace["archive_root"], render_root=cli_workspace["render_root"],
                       sources=[Source(name="test-inbox", path=inbox)])
        acquire_result = await AcquisitionService(backend=backend).acquire_sources(config.sources)
        raw_ids = acquire_result.raw_ids
        assert len(raw_ids) == 1
        parse_result = await ParsingService(repository=ConversationRepository(backend=backend),
                                       archive_root=cli_workspace["archive_root"], config=config).parse_from_raw(raw_ids=raw_ids)
        assert parse_result.counts["conversations"] >= 1 and len(parse_result.processed_ids) >= 1
        async with backend._get_connection() as conn:
            cursor = await conn.execute("SELECT raw_id FROM conversations WHERE conversation_id = ?",
                             (list(parse_result.processed_ids)[0],))
            row = await cursor.fetchone()
        assert row is not None and row["raw_id"] == raw_ids[0]
