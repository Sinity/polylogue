"""Assertion lifecycle value objects and policy normalization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from polylogue.core.json import JSONDocument, JSONValue, require_json_document, require_json_value

AssertionContextTrustClass: TypeAlias = Literal["operator", "system", "quoted"]


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


def derive_assertion_context_trust(
    *,
    author_kind: object,
    author_ref: object,
    status: object,
    context_policy: object,
    source_authority: AssertionContextTrustClass,
) -> AssertionContextTrustClass:
    """Derive context trust from provenance and source authority.

    ``context_policy`` is an assertion-controlled capability cap, never a
    source of authority. Assertion prose is not eligible for ``system`` trust:
    this source can emit authenticated user guidance (``operator``) or
    explicitly quoted evidence.
    """

    provenance_authority: AssertionContextTrustClass = "quoted"
    if (
        _text_value(author_kind) == "user"
        and _text_value(author_ref).startswith("user:")
        and _text_value(status) == "active"
    ):
        provenance_authority = "operator"

    if source_authority != "operator" or provenance_authority != "operator":
        return "quoted"

    requested_trust = _requested_policy_trust(context_policy)
    if requested_trust in {None, "operator"}:
        return "operator"
    # ``system`` is intentionally unavailable to assertion prose. It remains
    # a lower authority request, so conservatively render it as quoted data.
    return "quoted"


def constrain_assertion_context_policy(
    context_policy: AssertionContextPolicy,
    *,
    author_kind: object,
    author_ref: object,
    status: object,
    source_authority: AssertionContextTrustClass = "quoted",
) -> AssertionContextPolicy:
    """Persist an unauthorized operator/system request as a quoted cap."""

    payload = context_policy.as_json_document()
    requested_trust = _requested_policy_trust(payload)
    if requested_trust is None:
        return context_policy
    if (
        derive_assertion_context_trust(
            author_kind=author_kind,
            author_ref=author_ref,
            status=status,
            context_policy=payload,
            source_authority=source_authority,
        )
        == "operator"
    ):
        return context_policy
    payload["trust_class"] = "quoted"
    return AssertionContextPolicy.from_raw(payload)


def _requested_policy_trust(context_policy: object) -> AssertionContextTrustClass | None:
    if not isinstance(context_policy, Mapping):
        return None
    requested = context_policy.get("trust_class")
    if requested == "operator":
        return cast(AssertionContextTrustClass, requested)
    if requested == "system":
        return cast(AssertionContextTrustClass, requested)
    if requested == "quoted":
        return cast(AssertionContextTrustClass, requested)
    return None


def _text_value(value: object) -> str:
    return str(getattr(value, "value", value) or "")


__all__ = [
    "AssertionContextPolicy",
    "AssertionContextTrustClass",
    "AssertionStaleness",
    "AssertionValue",
    "constrain_assertion_context_policy",
    "derive_assertion_context_trust",
]
