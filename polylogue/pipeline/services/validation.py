"""Schema validation service for raw conversation payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.pipeline.services.validation_flow import (
    evaluate_raw_records as _evaluate_raw_records,
)
from polylogue.pipeline.services.validation_flow import (
    schema_validation_mode,
    validation_progress_desc,
)
from polylogue.pipeline.services.validation_flow import (
    validate_raw_ids as _validate_raw_ids,
)
from polylogue.pipeline.stage_models import ValidateResult
from polylogue.protocols import ProgressCallback
from polylogue.storage.store import RawConversationRecord
from polylogue.types import ValidationMode

if TYPE_CHECKING:
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository

__all__ = ["ValidationService", "ValidateResult"]


class ValidationService:
    """Validate raw payloads against provider schemas."""

    SCHEMA_VALIDATION_MODE_ENV = "POLYLOGUE_SCHEMA_VALIDATION"
    SCHEMA_VALIDATION_DEFAULT = ValidationMode.STRICT
    SCHEMA_VALIDATION_MODES = frozenset(ValidationMode)

    # Keep batches aligned with parse batching.
    RAW_BATCH_SIZE = 50

    def __init__(self, backend: SQLiteBackend):
        self.backend = backend
        from polylogue.storage.repository import ConversationRepository

        self.repository: ConversationRepository = ConversationRepository(backend=backend)

    def _schema_validation_mode(self) -> ValidationMode:
        return schema_validation_mode(
            env_var=self.SCHEMA_VALIDATION_MODE_ENV,
            default=self.SCHEMA_VALIDATION_DEFAULT,
        )

    def _validation_progress_desc(self, processed: int, total: int) -> str:
        return validation_progress_desc(processed, total)

    async def validate_raw_ids(
        self,
        *,
        raw_ids: list[str],
        progress_callback: ProgressCallback | None = None,
        persist: bool = True,
    ) -> ValidateResult:
        return await _validate_raw_ids(
            repository=self.repository,
            raw_ids=raw_ids,
            progress_callback=progress_callback,
            persist=persist,
            validation_mode=self._schema_validation_mode(),
            raw_batch_size=self.RAW_BATCH_SIZE,
        )

    async def evaluate_raw_records(
        self,
        *,
        raw_records: list[RawConversationRecord],
        progress_callback: ProgressCallback | None = None,
        persist: bool = False,
        mode: ValidationMode | None = None,
        progress_total: int | None = None,
        progress_offset: int = 0,
    ) -> ValidateResult:
        return await _evaluate_raw_records(
            repository=self.repository,
            raw_records=raw_records,
            progress_callback=progress_callback,
            persist=persist,
            mode=mode or self._schema_validation_mode(),
            progress_total=progress_total,
            progress_offset=progress_offset,
        )
