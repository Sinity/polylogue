"""Canonical, content-addressed identities for analysis query objects.

The public reference grammar is intentionally small and immutable:

* ``query:<sha256>`` identifies a canonical *expanded* planned query;
* ``query-run:qr_<id>`` identifies one operational execution; and
* ``result-set:<id>`` identifies a promoted durable result manifest.

``canonical_query_plan`` receives the typed plan after macro expansion.  It
normalizes every string to NFC and sorts only children of commutative AND/OR
nodes.  All other sequence order (including pipelines, ``except``, sort, and
limit) remains semantic and therefore participates in the digest unchanged.
Relative-time bounds belong to an execution record, not this plan: callers
hash the dynamic AST and record its resolved bounds on ``query_runs``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Final, TypeAlias

from polylogue.core.hashing import hash_payload
from polylogue.core.refs import ObjectRef

QUERY_REF_KIND: Final = "query"
QUERY_RUN_REF_KIND: Final = "query-run"
RESULT_SET_REF_KIND: Final = "result-set"
QUERY_RUN_ID_PREFIX: Final = "qr_"

_COMMUTATIVE_OPERATORS: Final = frozenset({"and", "or"})
_SHA256_HEX_RE: Final = re.compile(r"^[0-9a-f]{64}$")

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Mapping[str, "JsonValue"] | Sequence["JsonValue"]


def canonical_query_plan(
    planned_ast: Mapping[str, JsonValue],
    *,
    grain: str,
    lane: str,
    rank_policy: str,
    field_aliases: Mapping[str, str] | None = None,
) -> dict[str, JsonValue]:
    """Return the stable canonical payload used to identify a query.

    ``planned_ast`` must already be macro-expanded and typed by the query
    planner. ``field_aliases`` maps accepted aliases to the planner's canonical
    field token; it is applied only to values in ``field`` keys.
    """
    aliases = {_nfc(key): _nfc(value) for key, value in (field_aliases or {}).items()}
    return {
        "ast": _canonical_value(planned_ast, field_aliases=aliases),
        "grain": _nfc(grain),
        "lane": _nfc(lane),
        "rank_policy": _nfc(rank_policy),
    }


def query_hash_for_plan(
    planned_ast: Mapping[str, JsonValue],
    *,
    grain: str,
    lane: str,
    rank_policy: str,
    field_aliases: Mapping[str, str] | None = None,
) -> str:
    """Return the SHA-256 identity of an expanded planned query."""
    return hash_payload(
        canonical_query_plan(
            planned_ast,
            grain=grain,
            lane=lane,
            rank_policy=rank_policy,
            field_aliases=field_aliases,
        )
    )


def query_ref(query_hash: str) -> ObjectRef:
    """Build the registered public ref for a canonical query hash."""
    _require_sha256(query_hash, label="query hash")
    return ObjectRef(kind=QUERY_REF_KIND, object_id=query_hash)


def query_run_ref(run_id: str) -> ObjectRef:
    """Build the registered public ref for one operational query run."""
    if not run_id.startswith(QUERY_RUN_ID_PREFIX) or len(run_id) == len(QUERY_RUN_ID_PREFIX):
        raise ValueError(f"query run id must start with {QUERY_RUN_ID_PREFIX!r}")
    return ObjectRef(kind=QUERY_RUN_REF_KIND, object_id=run_id)


def result_set_ref(result_set_id: str) -> ObjectRef:
    """Build the registered public ref for one promoted result-set manifest."""
    if not result_set_id:
        raise ValueError("result set id cannot be empty")
    return ObjectRef(kind=RESULT_SET_REF_KIND, object_id=result_set_id)


def _canonical_value(value: JsonValue, *, field_aliases: Mapping[str, str]) -> JsonValue:
    if isinstance(value, str):
        return _nfc(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        normalized = {
            _nfc(str(key)): _canonical_value(item, field_aliases=field_aliases) for key, item in value.items()
        }
        field = normalized.get("field")
        if isinstance(field, str):
            normalized["field"] = field_aliases.get(field, field)
        operator = normalized.get("operator", normalized.get("op"))
        if isinstance(operator, str):
            canonical_operator = operator.casefold()
            if "operator" in normalized:
                normalized["operator"] = canonical_operator
            else:
                normalized["op"] = canonical_operator
            operator = canonical_operator
        children = normalized.get("children")
        if isinstance(operator, str) and operator in _COMMUTATIVE_OPERATORS and isinstance(children, Sequence):
            normalized["children"] = sorted(children, key=_canonical_sort_key)
        return normalized
    if isinstance(value, Sequence):
        return [_canonical_value(item, field_aliases=field_aliases) for item in value]
    raise TypeError(f"query plan contains unsupported JSON value: {type(value).__name__}")


def _canonical_sort_key(value: JsonValue) -> str:
    # hash_payload's encoding is the protocol's compact sorted-key JSON form.
    # Using it as a sort key makes commutative children deterministic too.
    return hash_payload(value)


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _require_sha256(value: str, *, label: str) -> None:
    if not _SHA256_HEX_RE.fullmatch(value):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")


__all__ = [
    "JsonValue",
    "QUERY_REF_KIND",
    "QUERY_RUN_ID_PREFIX",
    "QUERY_RUN_REF_KIND",
    "RESULT_SET_REF_KIND",
    "canonical_query_plan",
    "query_hash_for_plan",
    "query_ref",
    "query_run_ref",
    "result_set_ref",
]
