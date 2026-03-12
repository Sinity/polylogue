"""Pipeline services package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .acquisition import AcquireResult, AcquisitionService
    from .indexing import IndexService
    from .parsing import IngestPhase, IngestState, ParseResult, ParsingService
    from .planning import IngestPlan, PlanningService
    from .rendering import RenderResult, RenderService
    from .validation import ValidateResult, ValidationService


def __getattr__(name: str) -> object:
    lazy_exports = {
        "AcquireResult": ("polylogue.pipeline.services.acquisition", "AcquireResult"),
        "AcquisitionService": ("polylogue.pipeline.services.acquisition", "AcquisitionService"),
        "IndexService": ("polylogue.pipeline.services.indexing", "IndexService"),
        "IngestPlan": ("polylogue.pipeline.services.planning", "IngestPlan"),
        "IngestPhase": ("polylogue.pipeline.services.parsing", "IngestPhase"),
        "IngestState": ("polylogue.pipeline.services.parsing", "IngestState"),
        "ParseResult": ("polylogue.pipeline.services.parsing", "ParseResult"),
        "ParsingService": ("polylogue.pipeline.services.parsing", "ParsingService"),
        "PlanningService": ("polylogue.pipeline.services.planning", "PlanningService"),
        "RenderResult": ("polylogue.pipeline.services.rendering", "RenderResult"),
        "RenderService": ("polylogue.pipeline.services.rendering", "RenderService"),
        "ValidateResult": ("polylogue.pipeline.services.validation", "ValidateResult"),
        "ValidationService": ("polylogue.pipeline.services.validation", "ValidationService"),
    }
    module_spec = lazy_exports.get(name)
    if module_spec is not None:
        module_name, attr_name = module_spec
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AcquireResult",
    "AcquisitionService",
    "IndexService",
    "IngestPlan",
    "IngestPhase",
    "IngestState",
    "ParseResult",
    "ParsingService",
    "PlanningService",
    "RenderResult",
    "RenderService",
    "ValidateResult",
    "ValidationService",
]
