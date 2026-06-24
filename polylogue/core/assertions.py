"""Assertion lifecycle value objects and policy normalization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from polylogue.core.json import JSONDocument, JSONValue, require_json_document, require_json_value


@dataclass(frozen=True, slots=True)
class AssertionValue:
    """JSON-compatible assertion value normalized before storage writes."""

    value: JSONValue | None

    @classmethod
    def from_raw(cls, value: object | None) -> AssertionValue:
        if value is None:
            return cls(None)
        return cls(require_json_value(value, context="assertion value"))

    def as_json_value(self) -> JSONValue | None:
        return self.value


@dataclass(frozen=True, slots=True)
class AssertionStaleness:
    """JSON-object staleness descriptor for an assertion row."""

    payload: JSONDocument

    @classmethod
    def from_raw(cls, value: Mapping[str, object] | None) -> AssertionStaleness | None:
        if value is None:
            return None
        return cls(require_json_document(dict(value), context="assertion staleness"))

    def as_json_document(self) -> JSONDocument:
        return dict(self.payload)


@dataclass(frozen=True, slots=True)
class AssertionContextPolicy:
    """Typed context-injection policy for durable assertion rows."""

    payload: JSONDocument

    @classmethod
    def default(cls) -> AssertionContextPolicy:
        return cls({"inject": False})

    @classmethod
    def from_raw(cls, value: Mapping[str, object] | None) -> AssertionContextPolicy:
        if value is None:
            return cls.default()
        payload = require_json_document(dict(value), context="assertion context_policy")
        payload.setdefault("inject", False)
        return cls(payload)

    def as_json_document(self) -> JSONDocument:
        return dict(self.payload)

    @property
    def inject(self) -> bool:
        return bool(self.payload.get("inject"))


__all__ = ["AssertionContextPolicy", "AssertionStaleness", "AssertionValue"]
