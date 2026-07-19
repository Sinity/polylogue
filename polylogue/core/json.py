"""Central JSON utilities with a pluggable fast-JSON backend.

Backend selection happens once at import time, in priority order:

1. ``orjson`` -- fastest, but ships no ``cp314t`` (free-threaded Python 3.14)
   wheels and its build explicitly refuses to compile under free-threaded
   Python (polylogue-xikl phase 1 gate finding, 2026-07-19).
2. ``msgspec`` -- ships ``cp314t`` wheels since 0.20.0 (Nov 2025); used as
   the fast fallback whenever orjson is unavailable (e.g. under 3.14t, or
   any environment that installed polylogue without the ``speed`` extra).
3. stdlib ``json`` -- always available; the final fallback.

Every direct ``import orjson`` / ``import msgspec`` elsewhere in the
codebase should route through this facade instead, so backend selection,
bytes/str normalization, and decode-error unification live in one place.
See polylogue-xikl (free-threading adoption epic) and polylogue-7mtf (the
3.14t experiment that surfaced the orjson blocker).

Callers must not assume byte-for-byte output parity *across* backends for
anything beyond what this module's parameters guarantee (compact vs.
2-space-indent, sorted vs. insertion dict-key order, ASCII-safe UTF-8
encoding). Within one process the active backend is fixed at import time,
so output is self-consistent for hashing/idempotency purposes.
"""

from __future__ import annotations

import importlib
import json as _stdlib_json
from collections.abc import Callable
from decimal import Decimal
from types import ModuleType
from typing import Literal, TypeAlias, TypeGuard, cast

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONDocument: TypeAlias = dict[str, JSONValue]
JSONDocumentList: TypeAlias = list[JSONDocument]
JSONEncoder: TypeAlias = Callable[[object], object]

JSONBackend = Literal["orjson", "msgspec", "stdlib"]


class JSONDecodeError(ValueError):
    """Backend-agnostic JSON decode failure.

    Raised by :func:`loads` regardless of which backend is active, so
    callers never need to import or catch a backend-specific exception
    type (``orjson.JSONDecodeError``, ``msgspec.DecodeError``,
    ``json.JSONDecodeError``) -- catch this instead.
    """


def _try_import(name: str) -> ModuleType | None:
    """Import *name* if available, else return None.

    Deliberately NOT a plain ``try: import X as _x / except ImportError:
    _x = None`` -- across mypy environments where the optional package isn't
    installed at all, `ignore_missing_imports` makes a literal `import X`
    resolve to `Any` rather than erroring, which conflicts with an explicit
    `ModuleType | None` pre-declaration in a way that varies (and errors)
    depending on whether the package happens to be installed in whichever
    environment mypy runs in. Routing through a plain function call sidesteps
    that entirely: the return type is `ModuleType | None` in every environment.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


_orjson = _try_import("orjson")
_msgspec = _try_import("msgspec")
_msgspec_json = _try_import("msgspec.json")

if _orjson is not None:
    _BACKEND: JSONBackend = "orjson"
elif _msgspec_json is not None:
    _BACKEND = "msgspec"
else:
    _BACKEND = "stdlib"


def backend() -> JSONBackend:
    """Return the JSON backend selected at import time.

    One of ``"orjson"``, ``"msgspec"``, or ``"stdlib"``. Exposed for
    diagnostics/benchmarking; production code should not branch on this --
    the whole point of the facade is that callers don't need to know.
    """
    return _BACKEND


def is_json_value(value: object) -> TypeGuard[JSONValue]:
    """Return whether *value* is representable as JSON."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and is_json_value(item) for key, item in value.items())
    return False


def is_json_document(value: object) -> TypeGuard[JSONDocument]:
    """Return whether *value* is a JSON object with string keys."""
    return isinstance(value, dict) and all(isinstance(key, str) and is_json_value(item) for key, item in value.items())


def json_document(value: object) -> JSONDocument:
    """Coerce a value into a string-keyed JSON object when possible."""
    return value if is_json_document(value) else {}


def require_json_value(value: object, *, context: str = "JSON value") -> JSONValue:
    """Return a JSON value or raise when a producer violates the contract."""
    if is_json_value(value):
        return value
    raise TypeError(f"{context} is not JSON-compatible")


def require_json_document(value: object, *, context: str = "JSON document") -> JSONDocument:
    """Return a JSON document or raise when a producer violates the contract."""
    if is_json_document(value):
        return value
    raise TypeError(f"{context} is not a JSON object")


def json_document_list(value: object) -> JSONDocumentList:
    """Coerce a value into a list of string-keyed JSON objects."""
    if not isinstance(value, list):
        return []
    documents: JSONDocumentList = []
    for item in value:
        if is_json_document(item):
            documents.append(item)
    return documents


def normalize_json_decimal(value: object) -> object:
    """Recursively lower JSON parser Decimal values to JSON numbers."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, list):
        return [normalize_json_decimal(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_json_decimal(item) for key, item in value.items()}
    return value


def _reject_non_finite_token(token: str) -> JSONValue:
    raise ValueError(f"invalid non-finite JSON token: {token}")


def _loaded_json_value(value: object) -> JSONValue:
    if is_json_value(value):
        return value
    raise ValueError("loaded JSON payload does not satisfy the JSONValue contract")


def _default_encoder(user_default: JSONEncoder | None = None) -> JSONEncoder:
    """Create a JSON encoder that handles Decimal values.

    Raises :class:`TypeError` for anything it can't handle -- the contract
    shared by orjson's and stdlib json's ``default`` hook. msgspec's
    ``enc_hook`` wants :class:`NotImplementedError` instead; see
    :func:`_msgspec_enc_hook` for the adapter.
    """

    def _encoder(obj: object) -> object:
        if user_default is not None:
            try:
                return user_default(obj)
            except TypeError:
                pass
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    return _encoder


def _msgspec_enc_hook(encoder: JSONEncoder) -> Callable[[object], object]:
    def _hook(obj: object) -> object:
        try:
            return encoder(obj)
        except TypeError as exc:
            # msgspec's enc_hook contract signals "unsupported" via
            # NotImplementedError, not TypeError (see msgspec.json.encode docs).
            raise NotImplementedError(str(exc)) from exc

    return _hook


def _raw_loads(data: str | bytes | bytearray) -> object:
    result: object
    if _BACKEND == "orjson":
        if _orjson is None:
            raise RuntimeError("core.json backend is 'orjson' but the orjson module is unavailable")
        try:
            result = _orjson.loads(data)
        except _orjson.JSONDecodeError as exc:
            raise JSONDecodeError(str(exc)) from exc
        return result
    if _BACKEND == "msgspec":
        if _msgspec_json is None or _msgspec is None:
            raise RuntimeError("core.json backend is 'msgspec' but the msgspec module is unavailable")
        try:
            result = _msgspec_json.decode(data)
        except _msgspec.DecodeError as exc:
            raise JSONDecodeError(str(exc)) from exc
        return result
    try:
        result = _stdlib_json.loads(data, parse_constant=_reject_non_finite_token)
    except (_stdlib_json.JSONDecodeError, ValueError) as exc:
        raise JSONDecodeError(str(exc)) from exc
    return result


def _raw_dumps_bytes(obj: object, *, encoder: JSONEncoder, sort_keys: bool, indent: int | None) -> bytes | None:
    """Attempt the fast-backend encode; return ``None`` to signal a stdlib fallback."""
    if _BACKEND == "orjson":
        if _orjson is None:
            raise RuntimeError("core.json backend is 'orjson' but the orjson module is unavailable")
        option = 0
        if sort_keys:
            option |= _orjson.OPT_SORT_KEYS
        if indent == 2:
            option |= _orjson.OPT_INDENT_2
        try:
            # cast: `_orjson`'s declared type is `ModuleType | None` (for the
            # optional-import None check above), which widens attribute access
            # to `Any` -- orjson's own stub types `.dumps()` as `-> bytes`.
            return cast(bytes, _orjson.dumps(obj, default=encoder, option=option or None))
        except TypeError:
            return None
    if _BACKEND == "msgspec":
        if _msgspec_json is None:
            raise RuntimeError("core.json backend is 'msgspec' but the msgspec module is unavailable")
        # msgspec encodes decimal.Decimal natively as a JSON *string* (unlike
        # orjson/stdlib, which raise and defer to the default/enc_hook callback
        # below) -- pre-normalize so all three backends agree it's a number.
        prepared = normalize_json_decimal(obj)
        try:
            raw = _msgspec_json.encode(
                prepared,
                enc_hook=_msgspec_enc_hook(encoder),
                order="sorted" if sort_keys else None,
            )
        except (TypeError, NotImplementedError):
            return None
        if indent == 2:
            raw = _msgspec_json.format(raw, indent=2)
        # cast: same widening as the orjson branch above; msgspec.json.encode
        # is `-> bytes` and .format(bytes-like, ...) is `-> bytes` per its stub.
        return cast(bytes, raw)
    return None


def dumps_bytes(
    obj: object,
    *,
    default: JSONEncoder | None = None,
    sort_keys: bool = False,
    indent: int | None = None,
    append_newline: bool = False,
) -> bytes:
    """Dump *obj* to UTF-8 JSON bytes via the active backend.

    ``sort_keys``/``indent`` are backend-neutral semantic flags (not an
    orjson option bitmask) so behavior is identical regardless of which
    backend is active. ``indent`` only supports ``None`` (compact) or ``2``
    (pretty, 2-space) -- the only two shapes any call site in this codebase
    needs.
    """
    if indent is not None and indent != 2:
        raise ValueError(f"dumps_bytes indent must be None or 2, got {indent!r}")
    encoder = _default_encoder(default)
    payload = _raw_dumps_bytes(obj, encoder=encoder, sort_keys=sort_keys, indent=indent)
    if payload is None:
        separators = (",", ":") if indent is None else (",", ": ")
        payload = _stdlib_json.dumps(
            obj,
            default=encoder,
            sort_keys=sort_keys,
            indent=indent,
            ensure_ascii=False,
            separators=separators,
        ).encode("utf-8")
    if append_newline:
        payload += b"\n"
    return payload


def dumps(
    obj: object,
    *,
    default: JSONEncoder | None = None,
    sort_keys: bool = False,
    indent: int | None = None,
) -> str:
    """Dump object to JSON string."""
    return dumps_bytes(obj, default=default, sort_keys=sort_keys, indent=indent).decode("utf-8")


def loads(obj: str | bytes | bytearray) -> JSONValue:
    """Load object from JSON string or bytes.

    Tries the active fast backend first; falls back to a strict stdlib
    parse (still rejecting non-finite tokens) on failure, then raises
    :class:`JSONDecodeError` if both fail.
    """
    try:
        return _loaded_json_value(_raw_loads(obj))
    except JSONDecodeError as exc:
        try:
            return _loaded_json_value(_stdlib_json.loads(obj, parse_constant=_reject_non_finite_token))
        except (_stdlib_json.JSONDecodeError, ValueError):
            raise exc from None


__all__ = [
    "JSONBackend",
    "JSONDecodeError",
    "JSONDocument",
    "JSONDocumentList",
    "JSONEncoder",
    "JSONScalar",
    "JSONValue",
    "backend",
    "dumps",
    "dumps_bytes",
    "is_json_document",
    "is_json_value",
    "json_document",
    "json_document_list",
    "loads",
    "normalize_json_decimal",
    "require_json_document",
    "require_json_value",
]
