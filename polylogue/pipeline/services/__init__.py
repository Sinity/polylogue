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
from .indexing import IndexService
from .ingest_state import IngestPhase, IngestState
from .parsing import ParseResult, ParsingService
from .planning import IngestPlan, PlanningService
from .rendering import RenderResult, RenderService

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
]
