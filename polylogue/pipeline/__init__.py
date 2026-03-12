"""Pipeline package for data parsing, rendering, and indexing."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.pipeline.runner import latest_run, plan_sources, run_sources
    from polylogue.pipeline.services.indexing import IndexService
    from polylogue.pipeline.services.parsing import ParsingService
    from polylogue.pipeline.services.rendering import RenderService


def __getattr__(name: str) -> object:
    lazy_exports = {
        "IndexService": ("polylogue.pipeline.services.indexing", "IndexService"),
        "ParsingService": ("polylogue.pipeline.services.parsing", "ParsingService"),
        "RenderService": ("polylogue.pipeline.services.rendering", "RenderService"),
        "latest_run": ("polylogue.pipeline.runner", "latest_run"),
        "plan_sources": ("polylogue.pipeline.runner", "plan_sources"),
        "run_sources": ("polylogue.pipeline.runner", "run_sources"),
    }
    module_spec = lazy_exports.get(name)
    if module_spec is not None:
        module_name, attr_name = module_spec
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "IndexService",
    "ParsingService",
    "RenderService",
    "latest_run",
    "plan_sources",
    "run_sources",
]
