"""Pipeline package for data parsing and indexing."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.pipeline.services.indexing import IndexService
    from polylogue.pipeline.services.parsing import ParsingService


def __getattr__(name: str) -> object:
    lazy_exports = {
        "IndexService": ("polylogue.pipeline.services.indexing", "IndexService"),
        "ParsingService": ("polylogue.pipeline.services.parsing", "ParsingService"),
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
]
