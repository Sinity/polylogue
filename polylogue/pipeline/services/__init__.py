"""Pipeline services package.

Service classes that handle specific pipeline operations:
- ParsingService: Conversation parsing from sources
- IndexService: Full-text and vector search indexing
- RenderService: Markdown and HTML rendering
- AcquisitionService: Raw data acquisition from sources
- ValidationService: Schema validation for raw payloads
- PlanningService: Canonical preview/runtime ingest planning
- IngestState: Typed acquire/validate/parse state transitions
"""

from __future__ import annotations

from .acquisition import AcquireResult, AcquisitionService
from .ingest_state import IngestPhase, IngestState
from .indexing import IndexService
from .planning import IngestPlan, PlanningService
from .parsing import ParseResult, ParsingService
from .rendering import RenderResult, RenderService
from .validation import ValidateResult, ValidationService

__all__ = [
    "AcquireResult",
    "AcquisitionService",
    "IndexService",
    "IngestPlan",
    "IngestPhase",
    "IngestState",
    "ParseResult",
    "PlanningService",
    "ParsingService",
    "RenderResult",
    "RenderService",
    "ValidateResult",
    "ValidationService",
]
