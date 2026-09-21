"""Central JSON utilities over the one codec polylogue ships with.

``msgspec`` is the codec, not an accelerator over a stdlib default. It is a
declared ``[project]`` dependency (pyproject.toml) and a member of the
program's own list in flake.nix, and `polylogue.runtime`'s
``REQUIRED_NATIVE_PACKAGES`` refuses to start without it -- so this module
imports it unconditionally and an install that dropped it fails at import
with ``ModuleNotFoundError`` rather than quietly selecting a second codec.
``orjson`` used to be tried ahead of it, but it ships no ``cp314t`` wheel and
its build refuses to compile free-threaded (polylogue-xikl phase 1 gate
finding, 2026-07-19) -- it could never load on the shipped interpreter, so it
was removed rather than kept as a dead accelerator option.

There is deliberately no stdlib *backend*. stdlib ``json`` remains in use for
two bounded jobs that are not codec selection: a second-chance decode in
:func:`loads`, and the unencodable-object path in :func:`dumps_bytes`, where
``_raw_dumps_bytes`` returns ``None`` so stdlib raises the facade's uniform
``TypeError``. Neither can produce canonical bytes for a payload msgspec
accepted, so neither is a route by which a content hash changes.

Every direct ``import msgspec`` elsewhere in the codebase should route
through this facade instead, so bytes/str normalization and decode-error
unification live in one place. See polylogue-xikl (free-threading adoption
epic) and polylogue-7mtf (the 3.14t experiment that surfaced the orjson
blocker).

Callers must not assume byte-for-byte output parity with any other JSON
writer for anything beyond what this module's parameters guarantee (compact
vs. 2-space-indent, sorted vs. insertion dict-key order, ASCII-safe UTF-8
encoding).

**Float exponent formatting** (byte-stability guarantee, and why one codec):
the facade normalizes msgspec's float-exponent output to a fixed canonical
form (msgspec omits the ``+`` sign on positive exponents -- ``1e30`` -- this
facade always writes ``1e+30``, the format previously established by orjson
and still required for existing content hashes computed under it), so
canonical/content-hash dumps (`material_protocol/v1/canonical.py`) stay
byte-identical to every archive ever written by this facade. stdlib json's
float formatter is a *larger* departure (a different decimal-vs-exponent
threshold entirely, e.g. ``1e-05`` where this facade writes ``0.00001``) that
is not reconciled and never will be -- which is exactly why there is no
stdlib backend to fall into: an install lacking msgspec must fail loudly, not
hash the same payload differently with nothing observable to say so
(`tests/unit/test_packaging_dependencies.py` pins both halves).
"""

from __future__ import annotations

import json as _stdlib_json
import re
from collections.abc import Callable
from decimal import Decimal
from typing import TypeAlias, TypeGuard, cast

# Unconditional on purpose: see the module docstring. A missing msgspec is a
# broken install, not a slower one, and must not be discovered as a differing
# content hash months later.
import msgspec
import msgspec.json

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONDocument: TypeAlias = dict[str, JSONValue]
JSONDocumentList: TypeAlias = list[JSONDocument]
JSONEncoder: TypeAlias = Callable[[object], object]


class JSONDecodeError(ValueError):
    """Backend-agnostic JSON decode failure.

    Raised by :func:`loads` regardless of which backend is active, so
    callers never need to import or catch a backend-specific exception
    type (``msgspec.DecodeError``, ``json.JSONDecodeError``) -- catch this
    instead.
    """


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
    """Recursively lower JSON parser Decimal values to JSON numbers.

    Returns *value* itself when the tree holds no ``Decimal``, so a caller
    that keeps the result does not pay for a rebuilt copy of every decoded
    record. ``ijson`` is the only decoder in the pipeline that produces
    ``Decimal``; msgspec and stdlib ``json`` never do.
    """
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, list):
        changed = False
        items: list[object] = []
        for item in value:
            lowered = normalize_json_decimal(item)
            changed = changed or lowered is not item
            items.append(lowered)
        return items if changed else value
    if isinstance(value, dict):
        changed = False
        mapping: dict[object, object] = {}
        for key, item in value.items():
            lowered = normalize_json_decimal(item)
            changed = changed or lowered is not item
            mapping[key] = lowered
        return mapping if changed else value
    return value


class _NotJSON:
    """Sentinel for a node outside JSON's own type vocabulary."""

    __slots__ = ()


_NOT_JSON = _NotJSON()


def json_document_or_none(value: object) -> JSONDocument | None:
    """Return *value* as a JSON object with Decimal lowered, else ``None``.

    One walk where ``normalize_json_decimal`` followed by
    :func:`is_json_document` walks twice and rebuilds the whole tree. The
    validation and the lowering answer the same question about the same node,
    so they are asked together; a node outside JSON's vocabulary aborts the
    walk instead of being discovered by a second pass.
    """
    if not isinstance(value, dict):
        return None
    lowered = _lower_json_value(value)
    return cast(JSONDocument, lowered) if lowered is not _NOT_JSON else None


def _lower_json_value(value: object) -> object:
    """Lower Decimal and validate in one pass; ``_NOT_JSON`` on a bad node.

    Identity-preserving: a subtree needing no change is returned as itself, so
    the common case allocates nothing.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, list):
        changed = False
        items: list[object] = []
        for item in value:
            lowered = _lower_json_value(item)
            if lowered is _NOT_JSON:
                return _NOT_JSON
            changed = changed or lowered is not item
            items.append(lowered)
        return items if changed else value
    if isinstance(value, dict):
        changed = False
        mapping: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                return _NOT_JSON
            lowered = _lower_json_value(item)
            if lowered is _NOT_JSON:
                return _NOT_JSON
            changed = changed or lowered is not item
            mapping[key] = lowered
        return mapping if changed else value
    return _NOT_JSON


# The type vocabulary JSON itself defines (plus `tuple`, which every backend
# already serializes as an array with no special-casing needed). Anything
# outside this set is a candidate for cross-backend divergence below.
_JSON_NATIVE_SCALAR: tuple[type, ...] = (type(None), bool, int, float, str)


def _prepare_for_msgspec(value: object, encoder: JSONEncoder) -> object:
    """Normalize a payload so msgspec only ever encodes JSON-native shapes.

    ``msgspec.json.encode`` has built-in support for a much wider set of
    Python types than JSON itself -- and silently encodes several of them
    in ways orjson/stdlib json *reject* outright with ``TypeError``:

    - ``decimal.Decimal`` -> JSON string (never calls ``enc_hook``), rather
      than raising so `default`/``enc_hook`` can convert it to a number.
    - ``set``/``frozenset`` -> JSON array, silently baking in whatever
      order Python's hash-randomized iteration happened to produce.
    - ``bytes``/``bytearray``/``memoryview`` -> base64-encoded JSON string.
    - ``datetime``/``date``/``time``/``timedelta``/``uuid.UUID`` -> ISO/str.

    Encoding these natively under msgspec but raising under orjson/stdlib
    would make the facade's error contract depend on which backend happens
    to be active -- exactly what this module exists to prevent (pinned by
    `test_set_raises_type_error`, `test_decimal_encodes_to_float`,
    `test_dumps_custom_handler_takes_precedence_for_decimal`: a caller's
    `default` handler must get first say, and anything still unhandled must
    raise `TypeError`, identically to orjson/stdlib).

    So every value outside JSON's own vocabulary (``None``, ``bool``,
    ``int``, ``float``, ``str``, ``list``/``tuple``, ``dict``) is routed
    through *encoder* before msgspec ever sees it -- exactly mirroring the
    orjson/stdlib `default`-hook path.
    """
    if isinstance(value, dict):
        return {key: _prepare_for_msgspec(item, encoder) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_prepare_for_msgspec(item, encoder) for item in value]
    if isinstance(value, _JSON_NATIVE_SCALAR):
        return value
    return encoder(value)


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


# Matches a bare JSON number with an exponent for the scanner below. String
# literals are skipped as opaque spans before this expression is attempted.
_MSGSPEC_EXPONENT_TOKEN_RE = re.compile(rb"-?(?:0|[1-9]\d*)(?:\.\d+)?e(-?\d+)")


def _msgspec_exponent_plus_sign(exponent: bytes) -> bytes:
    return exponent if exponent.startswith(b"-") else b"+" + exponent


def _normalize_msgspec_float_exponents(data: bytes) -> bytes:
    """Make msgspec's exponent-notation floats byte-identical to orjson's.

    msgspec and orjson agree on every float-formatting decision checked
    empirically (decimal-vs-exponent threshold, digit count, no zero-padding)
    *except* one: orjson always writes an explicit ``+`` for a positive
    exponent (``1e+30``), while msgspec omits it (``1e30``). Negative
    exponents already match byte-for-byte (both write ``1e-6``, never
    ``1e-06``). This is the one seam that matters for
    `material_protocol/v1/canonical.py`'s content-hash stability across
    backends -- without it, canonical bytes would differ purely because the
    active JSON backend changed (found via a coordinator repro on
    ``dumps_bytes({"tiny": 5e-324}, sort_keys=True)``, PR #3155 review).

    stdlib json is a different, larger departure from orjson (it picks
    decimal vs. exponent notation at a different magnitude threshold, e.g.
    ``1e-05`` for ``0.00001`` where orjson/msgspec write ``0.00001``, and
    zero-pads short exponents to 2 digits) -- reconciling that would mean
    reimplementing orjson's float formatter from scratch, not a one-line fix.
    stdlib therefore cannot produce these bytes at all, which is why msgspec
    is a declared base dependency: canonical/hash byte stability holds for
    orjson and msgspec, and a supported install always has one of them.
    """

    # Avoid a regex that has to repeatedly backtrack through large quoted
    # strings (browser captures can contain multi-megabyte base64 values).
    # Scan string literals as opaque spans, and only run the small number
    # matcher at positions outside strings.  The output is one linear copy
    # of the input, plus at most one byte per normalized exponent.
    chunks: list[bytes] = []
    cursor = 0
    index = 0
    length = len(data)
    changed = False
    while index < length:
        byte = data[index]
        if byte == 34:  # '"'
            if cursor < index:
                chunks.append(data[cursor:index])
            end = index + 1
            while end < length:
                current = data[end]
                if current == 92:  # '\\'; escaped byte cannot close a string
                    end += 2
                elif current == 34:
                    end += 1
                    break
                else:
                    end += 1
            chunks.append(data[index:end])
            index = end
            cursor = end
            continue
        if byte == 45 or 48 <= byte <= 57:  # '-' or a decimal digit
            match = _MSGSPEC_EXPONENT_TOKEN_RE.match(data, index)
            if match is not None:
                exponent = match.group(1)
                if not exponent.startswith(b"-"):
                    chunks.append(data[cursor : match.start(1)])
                    chunks.append(b"+")
                    chunks.append(exponent)
                    changed = True
                    index = match.end()
                    cursor = index
                    continue
                index = match.end()
                continue
        index += 1
    if cursor < length:
        chunks.append(data[cursor:])
    return b"".join(chunks) if changed else data


def _raw_loads(data: str | bytes | bytearray) -> object:
    try:
        return msgspec.json.decode(data)
    except (msgspec.DecodeError, UnicodeDecodeError) as exc:
        # msgspec validates JSON *structure* through its own DecodeError, but
        # invalid UTF-8 bytes embedded inside a string's content leak as a raw
        # stdlib UnicodeDecodeError instead (confirmed against msgspec
        # directly, not just through this facade). Without this, a payload
        # with a merely-malformed *value* -- not a malformed document --
        # crashes the caller instead of raising the facade's unified decode
        # error, which `loads`' second-chance stdlib parse then depends on.
        raise JSONDecodeError(str(exc)) from exc


def _raw_dumps_bytes(obj: object, *, encoder: JSONEncoder, sort_keys: bool, indent: int | None) -> bytes | None:
    """Encode with msgspec, or ``None`` when the payload is unencodable.

    ``None`` is not a backend fallback: it means msgspec refused a value the
    caller's ``default`` hook could not convert either, and :func:`dumps_bytes`
    re-runs the encode through stdlib json purely so the caller sees the one
    uniform ``TypeError`` this facade promises. A payload msgspec *accepted*
    never reaches stdlib, so canonical bytes have exactly one producer.
    """
    # msgspec encodes decimal.Decimal natively as a JSON *string* (unlike
    # stdlib, which raises and defers to the default hook) -- pre-normalize so
    # it is a number either way and a caller's custom `default` handler still
    # gets first say.
    prepared = _prepare_for_msgspec(obj, encoder)
    try:
        raw = msgspec.json.encode(
            prepared,
            enc_hook=_msgspec_enc_hook(encoder),
            order="sorted" if sort_keys else None,
        )
    except (TypeError, NotImplementedError):
        return None
    raw = _normalize_msgspec_float_exponents(raw)
    if indent == 2:
        raw = msgspec.json.format(raw, indent=2)
    # cast: msgspec.json.encode is `-> bytes` and .format(bytes-like, ...) is
    # `-> bytes` per its stub.
    return cast(bytes, raw)


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

    Decodes with msgspec, then gives a document msgspec rejected one
    second-chance strict stdlib parse (still rejecting non-finite tokens)
    before raising :class:`JSONDecodeError`. This is a leniency seam on the
    read side only -- it produces no bytes, so it cannot move a content hash.
    """
    try:
        return _loaded_json_value(_raw_loads(obj))
    except JSONDecodeError as exc:
        try:
            return _loaded_json_value(_stdlib_json.loads(obj, parse_constant=_reject_non_finite_token))
        except (_stdlib_json.JSONDecodeError, ValueError):
            raise exc from None


__all__ = [
    "JSONDecodeError",
    "JSONDocument",
    "JSONDocumentList",
    "JSONEncoder",
    "JSONScalar",
    "JSONValue",
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
