"""Producer-owned positions for live search continuation."""

from __future__ import annotations

import base64
import json
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from polylogue.core.errors import PolylogueError

if TYPE_CHECKING:
    from polylogue.archive.query.plan import SessionQueryPlan

Scalar = str | int | float | None


SEARCH_CURSOR_VERSION: Literal[2] = 2


class InvalidSearchCursorError(PolylogueError, ValueError):
    """The token does not identify this search relation."""

    http_status_code = 400


class SearchOrderValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    value: str | int | float | None
    descending: bool = False
    numeric_text: bool = False

    @field_validator("numeric_text")
    @classmethod
    def validate_numeric(cls, numeric: bool, info: ValidationInfo) -> bool:
        if numeric:
            value = info.data.get("value")
            try:
                if value is None or not Decimal(str(value)).is_finite():
                    raise ValueError("numeric cursor key must be finite")
            except InvalidOperation as exc:
                raise ValueError("numeric cursor key is invalid") from exc
        return numeric


class SearchPosition(BaseModel):
    model_config = ConfigDict(frozen=True)
    grain: Literal["block", "session"]
    identity: str
    key: tuple[SearchOrderValue, ...] = Field(min_length=1)

    def after(self, anchor: SearchPosition) -> bool:
        if self.grain != anchor.grain or len(self.key) != len(anchor.key):
            raise InvalidSearchCursorError("cursor order does not match the producer")
        for current, previous in zip(self.key, anchor.key, strict=True):
            if (current.descending, current.numeric_text) != (previous.descending, previous.numeric_text):
                raise InvalidSearchCursorError("cursor comparator does not match the producer")
            a: Scalar | Decimal = current.value
            b: Scalar | Decimal = previous.value
            if current.numeric_text:
                a = Decimal(str(a)) if a is not None else None
                b = Decimal(str(b)) if b is not None else None
            if a == b:
                continue
            # SQLite places NULL before all non-null values in ascending order.
            greater = b is None if a is not None else False
            if a is not None and b is not None:
                if isinstance(a, str) and isinstance(b, str):
                    greater = a > b
                elif isinstance(a, (int, float, Decimal)) and isinstance(b, (int, float, Decimal)):
                    greater = Decimal(str(a)) > Decimal(str(b))
                else:
                    raise InvalidSearchCursorError("cursor key type does not match the producer")
            return not greater if current.descending else greater
        return False


class SearchCursor(BaseModel):
    model_config = ConfigDict(frozen=True)
    v: Literal[2]
    position: SearchPosition
    lane: str
    query_hash: str


def search_cursor_lane_matches_request(cursor_lane: str, requested_lane: str | None) -> bool:
    return requested_lane in {None, "", "auto"} or requested_lane == cursor_lane


def encode_search_cursor(position: SearchPosition, *, lane: str, request_identity: str) -> str:
    body = SearchCursor(
        v=SEARCH_CURSOR_VERSION, position=position, lane=lane, query_hash=request_identity
    ).model_dump_json()
    return base64.urlsafe_b64encode(body.encode()).decode().rstrip("=")


def decode_search_cursor(token: str) -> SearchCursor:
    try:
        body = json.loads(base64.b64decode(token + "=" * (-len(token) % 4), altchars=b"-_", validate=True))
        cursor = SearchCursor.model_validate(body)
        if cursor.v != SEARCH_CURSOR_VERSION:
            raise ValueError("unsupported cursor version")
        return cursor
    except Exception as exc:
        raise InvalidSearchCursorError("invalid search cursor") from exc


def position(
    grain: Literal["block", "session"], identity: str, values: tuple[tuple[object, bool, bool], ...]
) -> SearchPosition:
    return SearchPosition(
        grain=grain,
        identity=identity,
        key=tuple(
            SearchOrderValue(value=cast(Scalar, value), descending=descending, numeric_text=numeric)
            for value, descending, numeric in values
        ),
    )


def plan_cursor_identity(plan: SessionQueryPlan) -> str:
    """Bind declarative selection, lane and order; omit presentation/runtime objects.

    Python callbacks are executable mutable state, not a wire query identity.
    Their reads retain offsets but do not mint a live cursor.
    """
    import hashlib
    from dataclasses import fields
    from datetime import datetime

    arguments: dict[str, object] = {}
    for field in fields(plan):
        if field.name in {"cursor", "offset", "limit", "vector_provider", "predicates"}:
            continue
        value = getattr(plan, field.name)
        if field.name == "boolean_predicate":
            value = value.to_payload() if value is not None else None
        elif isinstance(value, datetime):
            value = value.isoformat()
        arguments[field.name] = value
    encoded = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_plan_cursor(plan: SessionQueryPlan) -> SearchCursor | None:
    """Refuse incompatible tokens before optional lane traversal or archive scans."""
    if not plan.cursor:
        return None
    if plan.predicates:
        raise InvalidSearchCursorError("opaque Python callback filters do not support search cursors")
    cursor = decode_search_cursor(plan.cursor)
    if cursor.query_hash != plan_cursor_identity(plan) or not search_cursor_lane_matches_request(
        cursor.lane, plan.retrieval_lane
    ):
        raise InvalidSearchCursorError("cursor belongs to a different search request")
    if plan.latest:
        raise InvalidSearchCursorError("latest selection does not support search cursors")
    if plan.sort == "random":
        raise InvalidSearchCursorError("random search does not support cursors")
    return cursor
