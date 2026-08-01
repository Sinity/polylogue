"""Runtime-facing schema validation and harmonization API.

The public names are resolved lazily (PEP 562): ``polylogue.schemas.validator``
transitively imports the whole provider-parser universe
(``archive.raw_payload.decode`` -> ``sources.dispatch``) plus ``jsonschema``,
~0.9s of import work. Light in-package modules (``drift_sentinel``) are
imported from hot CLI paths (the root-callback drift marker, archive-tier DDL
modules); an eager package ``__init__`` taxed every one of those paths with
the full validator chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polylogue.schemas.validator import (
        SchemaValidator,
        ValidationResult,
        validate_provider_export,
    )

__all__ = [
    "SchemaValidator",
    "ValidationResult",
    "validate_provider_export",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from polylogue.schemas import validator

        return getattr(validator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
