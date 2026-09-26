"""ID generation and content hashing logic for pipeline items."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence, Set
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, TypeVar, cast, overload

from polylogue.core.digest import QUERY
from polylogue.core.enums import BlockType, Origin, Provider
from polylogue.core.hashing import hash_bytes, hash_payload
from polylogue.core.json import JSONValue, dumps
from polylogue.core.message_owner import MessageOwnerAmbiguityError, MessageOwnerCoordinate
from polylogue.core.sources import origin_from_provider
from polylogue.core.text_identity import nfc
from polylogue.core.types import ContentHash, MessageId, SessionId

# ParsedMessage/ParsedSession/ParsedAttachment/ParsedContentBlock are used only
# as parameter/return type annotations below (never constructed or
# isinstance-checked here). Importing them eagerly forces the whole
# `polylogue.sources` package init -- including the Drive download subsystem
# -- onto every caller of this pure hashing/id module (polylogue-8s70: this
# was ~395ms of the raw-authority storage stack's ~670ms import cost, the
# single largest contributor). TYPE_CHECKING-only keeps static typing intact while
# deferring the real import to whichever caller actually needs `sources`.
if TYPE_CHECKING:
    from polylogue.sources import ParsedMessage, ParsedSession
    from polylogue.sources.parsers.base import ParsedAttachment, ParsedContentBlock, ParsedSessionEvent


# Content identity is an explicit partition of parser fields.  Keep transport
# evidence and provider measurements out of this identity only when an
# independent owner derives them; adding a parser field without deciding its
# identity is an error, not an accidental compatibility behavior.
_HASHED_FIELDS: dict[str, frozenset[str]] = {
    "ParsedContentBlock": frozenset(
        {
            "type",
            "text",
            "tool_name",
            "tool_id",
            "tool_input",
            "media_type",
            "metadata",
            "is_error",
            "exit_code",
            "tool_outcome",
            "outcome_unknown_reason",
            "file_edit",
            "web_constructs",
        }
    ),
    "ParsedMessage": frozenset(
        {
            "provider_message_id",
            "role",
            "text",
            "timestamp",
            "occurred_at_ms",
            "blocks",
            "message_type",
            "material_origin",
            "parent_message_provider_id",
            "model_name",
            "model_effort",
            "sender_name",
            "recipient",
            "delivery_status",
            "end_turn",
            "user_context_text",
            "paste_spans",
            "stop_reason",
            "is_aborted_mid_stream",
        }
    ),
    "ParsedSession": frozenset(
        {
            "source_name",
            "provider_session_id",
            "title",
            "session_kind",
            "created_at",
            "updated_at",
            "messages",
            "active_leaf_message_provider_id",
            "attachments",
            "session_events",
            "parent_session_provider_id",
            "branch_point_provider_message_id",
            "branch_type",
            "title_source",
            "title_ref",
            "instructions_text",
            "working_directories",
            "git_branch",
            "git_repository_url",
            "provider_project_ref",
            "team_name",
            "git_commit_hash",
            "display_name",
            "pending_drafts",
            "session_refs",
        }
    ),
}

_EXCLUDED_FIELDS: dict[str, dict[str, str]] = {
    "ParsedContentBlock": {
        "signature": "provider cryptographic signatures are re-issued on replay",
    },
    "ParsedMessage": {
        "parent_message_position": "parser-only linkage resolved to the stored parent identity",
        "owner_coordinate": "parser-only ownership evidence resolved before storage",
        "position": "parser-only occurrence coordinate used for ordering and owner resolution",
        "branch_index": "parser-only branch ordering coordinate resolved into lineage",
        "variant_index": "parser-only duplicate occurrence coordinate resolved by owner evidence",
        "is_active_path": "parser-derived path marker owned by lineage materialization",
        "is_active_leaf": "parser-derived leaf marker owned by lineage materialization",
        "input_tokens": "provider usage measurement is owned by usage/cost derivation",
        "output_tokens": "provider usage measurement is owned by usage/cost derivation",
        "cache_read_tokens": "provider usage measurement is owned by usage/cost derivation",
        "cache_write_tokens": "provider usage measurement is owned by usage/cost derivation",
        "duration_ms": "provider timing measurement is owned by usage/cost derivation",
    },
    "ParsedSession": {
        "content_hash": "parse-side validated identity carrier, not semantic session content",
        "provider_session_aliases": "parser-derived alternate session identifiers are retained as lookup metadata",
        "created_at_provenance": "timestamp authority provenance is independent metadata",
        "updated_at_provenance": "timestamp authority provenance is independent metadata",
        "unit_accounting": "parser admission accounting is validated independently before lowering",
        "reported_duration_ms": "provider timing measurement is owned by usage/cost derivation",
        "reported_cost_usd": "provider monetary measurement is owned by usage/cost derivation",
        "models_used": "provider usage summary is derived from message model fields",
        "ingest_flags": "parser quality annotations are independent session tags",
    },
}

# Message ownership has a stricter identity boundary than the complete
# semantic hash. These are the stable intrinsic fields used to distinguish
# duplicate id-less messages; ordering, path state, usage, and other semantic
# revision fields must not become owner evidence. Keeping this boundary
# separate from ``_EXCLUDED_FIELDS`` lets the complete content identity remain
# sensitive to those fields.
_OWNER_MATCH_FIELDS: dict[str, frozenset[str]] = {
    "ParsedMessage": frozenset({"role", "text", "timestamp", "blocks"}),
}


def validate_semantic_hash_partition() -> None:
    """Fail if parsed model fields are missing from the identity decision."""
    from polylogue.sources.parsers.base_models import ParsedContentBlock, ParsedMessage, ParsedSession

    models = (
        ("ParsedContentBlock", ParsedContentBlock),
        ("ParsedMessage", ParsedMessage),
        ("ParsedSession", ParsedSession),
    )
    for name, model in models:
        fields = frozenset(cast(Mapping[str, object], model.model_fields))
        hashed = _HASHED_FIELDS[name]
        excluded = frozenset(_EXCLUDED_FIELDS[name])
        if hashed & excluded or hashed | excluded != fields:
            raise AssertionError(
                f"{name} semantic hash partition drift: missing={sorted(fields - hashed - excluded)}, "
                f"duplicate={sorted(hashed & excluded)}, stale={sorted((hashed | excluded) - fields)}"
            )
    owner_fields = _OWNER_MATCH_FIELDS["ParsedMessage"]
    if not owner_fields <= _HASHED_FIELDS["ParsedMessage"] or "position" in owner_fields:
        raise AssertionError("ParsedMessage owner partition must be a position-free semantic subset")


def bound_session_content_hash(convo: ParsedSession) -> ContentHash | None:
    """Return a validated parse-side session hash, when one was carried.

    A bound hash is trusted only as a 32-byte hexadecimal digest.  The
    semantic hash function remains available for callers that need to derive
    or independently re-check identity.
    """
    value = getattr(convo, "content_hash", None)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("bound session content_hash must be hexadecimal SHA-256 text")
    try:
        digest = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("bound session content_hash must be hexadecimal SHA-256 text") from exc
    if len(digest) != 32:
        raise ValueError("bound session content_hash must be a SHA-256 digest")
    return ContentHash(digest.hex())


def _hash_field_value(value: object) -> JSONValue:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif isinstance(value, list):
        value = [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in value]
    if isinstance(value, Mapping):
        value = dict(value)
    return cast(JSONValue, _normalize_nested_for_hash(value))


def _model_hash_payload(model: object, fields: frozenset[str]) -> dict[str, JSONValue]:
    return {field: _hash_field_value(getattr(model, field)) for field in sorted(fields)}


# Sentinel values to distinguish None from empty in hash computations
_NULL_SENTINEL = "__POLYLOGUE_NULL__"
_EMPTY_SENTINEL = "__POLYLOGUE_EMPTY__"
HashScalar: TypeAlias = str | int | float | bool | None
MessageContent: TypeAlias = tuple[bytes, bytes, int]
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class SessionRevisionProjection:
    """Canonical content-only comparison value for a session revision.

    One invariant governs every field below: *a conversation is an unordered
    collection of items keyed by content-derived identity, each carrying only
    content-bearing fields -- nothing else may enter the value used to
    compare two acquisitions of it* (polylogue-aggz). Concretely:

    - Identity is never array position. ``message_contents``,
      ``attachment_contents``, and ``event_contents`` are unordered, not
      ordered tuples -- a provider's export can replay an unchanged item set
      in a different array sequence across separate export requests (proven
      for Claude.ai messages and ChatGPT ``generation_lifecycle`` events:
      same items, different sequence, every re-export), and a set has no
      order to violate (polylogue-c429, polylogue-nuec).
    - Identity is derived from content, never a provider id whose PRESENCE
      is itself unstable. ``attachment_contents``' key is
      ``(anchoring message, name, media type)`` -- never the provider's own
      attachment id, which Claude.ai does not consistently emit for the same
      attachment across export vintages (one vintage carries a real UUID,
      the other has none) -- no id-minting scheme can make a real id and a
      synthetic one collide, so the id is simply not part of identity
      (polylogue-d8al, polylogue-hith). The same reasoning selects
      ``(event type, anchoring message)`` for events, folding in the event's
      own content hash only when that pair is ambiguous within one revision
      (e.g. more than one ``chatgpt_block_metadata`` event on one message,
      one per block) -- still content-derived, never the array index.
    - Acquisition state and provider-reported measurement are not content.
      An attachment's bytes may be known or not (``attachment_contents``
      omits an identity until its bytes are read, while
      ``attachment_identities`` already knows the reference exists) --
      resolving them is evidence *growing*, not the attachment becoming a
      different one (polylogue-bu1i). ChatGPT's ``generation_lifecycle``
      event re-derives ``elapsed_duration_ms`` from the raw export's own
      timing metadata on every export request, and the value is not stable
      across requests for the SAME generation even when the transcript is
      byte-identical -- excluded from ``event_contents`` by
      ``_EVENT_CONTENT_PAYLOAD_ALLOWLIST``, an explicit per-event-type
      ALLOWLIST of content-bearing payload fields rather than a denylist of
      fields discovered volatile after the fact (three volatility axes were
      each found only after shipping: acquisition state, array order,
      provider-reported duration -- a denylist means the next provider
      quirk is silently invisible to comparison until someone notices and
      files a bead; an allowlist means a NEW field a parser adds later
      cannot silently enter identity without an explicit decision to add it).

    ``session_hash``, ``message_hashes`` (ordered), and ``event_hashes``
    (ordered, unstripped) are UNCHANGED by any of this: they still cover the
    full, order-sensitive, unstripped payload, so a real reorder, a real
    duration change, or newly-acquired bytes still change the session's
    content hash and still trigger a re-write (idempotency is a different
    question from revision *comparison*, and only the latter is
    content-only). ``attachment_identities`` retains the same unordered set
    of content-derived keys; prepared disk sessions may keep it in scratch
    storage so frontier counting does not require resident rows.
    """

    session_hash: bytes
    message_hashes: Sequence[bytes]
    # Each item carries its unordered multiplicity. Repeated timestamp-less
    # id-less messages can have the same content-derived identity and content;
    # collapsing them into a set would falsely make one and two turns equal.
    message_contents: Set[MessageContent]
    attachment_identities: Set[bytes]
    attachment_contents: Set[tuple[bytes, bytes]]
    event_hashes: Sequence[bytes]
    event_contents: Set[tuple[bytes, bytes]]
    # Timestamped id-less messages intentionally share a revision axis even
    # when their mutable content changes. Native-id axes retain strict content
    # conflict semantics in ``session_revision_membership``.
    mutable_message_identities: Set[bytes] = frozenset()
    #: ``(event_identity, anchor_free_identity)`` for events whose anchoring
    #: message is provider-remeasured rather than content. ChatGPT re-anchors
    #: a ``generation_lifecycle`` event to a different
    #: ``source_message_provider_id`` between export vintages of the SAME
    #: conversation, so its identity -- keyed on (event_type, anchoring
    #: message) -- reads as two disjoint events instead of one, and the
    #: revision axis calls an unchanged conversation a fork (polylogue-uqwd).
    #: The anchor-free identity lets ``session_revision_membership`` recognise
    #: the moved event as the same slot. Comparison-layer only: this value is
    #: never persisted and never enters ``session_hash``, so nothing here
    #: changes stored identity or requires a reparse.
    anchor_free_event_identities: Set[tuple[bytes, bytes]] = frozenset()


class _DiskRevisionStore:
    """Disposable owner for one prepared revision's projected evidence."""

    def __init__(self, parent: Path | None) -> None:
        self._scratch = tempfile.TemporaryDirectory(prefix="polylogue-revision-", dir=parent)
        self.conn = sqlite3.connect(Path(self._scratch.name) / "projection.db")
        self.conn.execute("CREATE TABLE message_hash (ordinal INTEGER PRIMARY KEY, digest BLOB NOT NULL)")
        self.conn.execute("CREATE TABLE event_hash (ordinal INTEGER PRIMARY KEY, digest BLOB NOT NULL)")
        self.conn.execute(
            "CREATE TABLE message_content (identity BLOB NOT NULL, content BLOB NOT NULL, "
            "multiplicity INTEGER NOT NULL, PRIMARY KEY(identity, content)) WITHOUT ROWID"
        )
        for table in ("mutable_message", "attachment_identity"):
            self.conn.execute(f"CREATE TABLE {table} (identity BLOB PRIMARY KEY) WITHOUT ROWID")
        for table in ("attachment_content", "event_content", "anchor_free_event"):
            second = "anchor_free" if table == "anchor_free_event" else "content"
            self.conn.execute(
                f"CREATE TABLE {table} (identity BLOB NOT NULL, {second} BLOB NOT NULL, "
                f"PRIMARY KEY(identity, {second})) WITHOUT ROWID"
            )

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if hasattr(self, "conn"):
            self.conn.close()
        if hasattr(self, "_scratch"):
            self._scratch.cleanup()

    def __del__(self) -> None:
        self.close()


class _DiskRevisionHashes(Sequence[bytes]):
    def __init__(self, store: _DiskRevisionStore, table: str, count: int) -> None:
        self._store = store
        self._table = table
        self._count = count

    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, index: int) -> bytes: ...

    @overload
    def __getitem__(self, index: slice) -> list[bytes]: ...

    def __getitem__(self, index: int | slice) -> bytes | list[bytes]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._count))]
        ordinal = index + self._count if index < 0 else index
        if ordinal < 0 or ordinal >= self._count:
            raise IndexError(index)
        row = self._store.conn.execute(f"SELECT digest FROM {self._table} WHERE ordinal = ?", (ordinal,)).fetchone()
        if row is None:
            raise ValueError("prepared revision hash row disappeared")
        return bytes(row[0])

    def __iter__(self) -> Iterator[bytes]:
        for (digest,) in self._store.conn.execute(f"SELECT digest FROM {self._table} ORDER BY ordinal"):
            yield bytes(digest)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence) or len(self) != len(other):
            return False
        return all(left == right for left, right in zip(self, other, strict=True))


class _DiskRevisionSet(Set[_T]):
    _TABLE_COLUMNS = {
        "message_content": ("identity", "content", "multiplicity"),
        "mutable_message": ("identity",),
        "attachment_identity": ("identity",),
        "attachment_content": ("identity", "content"),
        "event_content": ("identity", "content"),
        "anchor_free_event": ("identity", "anchor_free"),
    }

    def __init__(self, store: _DiskRevisionStore, table: str) -> None:
        self._store = store
        self._table = table
        self._columns = self._TABLE_COLUMNS[table]

    @property
    def scratch_parent(self) -> Path:
        return Path(self._store._scratch.name).parent

    def __len__(self) -> int:
        return int(self._store.conn.execute(f"SELECT COUNT(*) FROM {self._table}").fetchone()[0])

    def __contains__(self, value: object) -> bool:
        parts = value if isinstance(value, tuple) else (value,)
        if len(parts) != len(self._columns):
            return False
        where = " AND ".join(f"{column} = ?" for column in self._columns)
        return (
            self._store.conn.execute(f"SELECT 1 FROM {self._table} WHERE {where} LIMIT 1", parts).fetchone() is not None
        )

    def __iter__(self) -> Iterator[_T]:
        yield from self.iter_sorted()

    def iter_sorted(self) -> Iterator[_T]:
        columns = ", ".join(self._columns)
        order = ", ".join(self._columns)
        for row in self._store.conn.execute(f"SELECT {columns} FROM {self._table} ORDER BY {order}"):
            values = tuple(int(item) if isinstance(item, int) else bytes(item) for item in row)
            yield cast(_T, values[0] if len(values) == 1 else values)

    def lookup_second(self, identity: bytes) -> bytes | None:
        if self._table != "anchor_free_event":
            raise TypeError("second-value lookup is only defined for anchor-free events")
        row = self._store.conn.execute(
            "SELECT anchor_free FROM anchor_free_event WHERE identity = ? ORDER BY anchor_free LIMIT 1",
            (identity,),
        ).fetchone()
        return bytes(row[0]) if row is not None else None


class UnhashablePayloadValueError(TypeError):
    """A nested payload value outside the declared hash vocabulary.

    Raised by :func:`_normalize_nested_for_hash` instead of letting the JSON
    facade fail later with a backend-flavoured ``TypeError`` that names
    neither the hash vocabulary nor the field that carried the value. It
    subclasses :class:`TypeError` so the boundary's error contract is
    unchanged for anything that already caught the facade's own refusal.
    """


def _canonical_sort_key(value: object) -> str:
    """Total, type-independent ordering key for already-normalized values.

    Sorting a set's members directly is not total -- ``sorted({"z", 1})``
    raises, and the members of a ``dict[str, object]`` metadata payload carry
    no common order. Ordering by each member's canonical JSON text is total
    over everything the vocabulary below admits, and is the same idiom the
    publication encoder already uses (``sinex/material_adapter.py``).
    """
    return dumps(value, sort_keys=True)


def _normalize_nested_for_hash(value: object, *, path: str = "payload") -> object:
    """Canonicalize a nested payload for hashing, recursively.

    Two jobs, both required for the declared vocabulary to be *total* over
    what parsers actually emit into ``ParsedContentBlock.metadata`` /
    ``.tool_input`` and ``ParsedSessionEvent.payload`` -- all three typed
    ``object``-valued, so nothing at the parser boundary constrains them to
    JSON-native shapes.

    ``_normalize_for_hash`` covers scalar fields, but two nested payloads --
    ``ParsedContentBlock.tool_input`` and ``ParsedSessionEvent.payload`` --
    were passed straight to ``hash_payload``, which by its own docstring does
    NOT normalize the strings it serializes. So a tool_use block or session
    event whose nested content differed only in Unicode normalization form
    hashed as two distinct logical identities and could never dedupe, while
    the very same text in ``message.text`` hashed identically.

    Measured before fixing: 0 of 20,000 sampled ``tool_use.tool_input`` rows
    in the live archive carry non-NFC content, so no stored hash changes and
    nothing needs re-hashing -- this closes a latent trap rather than
    repairing active corruption. Plausible future sources are macOS-originated
    exports (HFS+ historically stored NFD) and browser-capture DOM extraction.

    Dict keys are normalized as well as values: a key is just as capable of
    carrying an NFD form, and an un-normalized key would split the hash the
    same way.

    Second job (polylogue-m706z): lower every non-JSON-native value the
    vocabulary admits to a declared canonical form. Nothing outside
    ``None``/``bool``/``int``/``float``/``str``/``list``/``tuple``/``dict``
    survives the digest encoder -- so a ``set`` in block metadata made
    :func:`message_content_identity` raise rather than hash, and the
    content-derived half of the message identity fallback was simply
    unavailable for that message. An unserializable member is a gap in this
    declaration, not a caller's mistake, so the cases below name the canonical
    form of each admitted type rather than leaving it to the JSON backend:

    - ``set``/``frozenset``: sorted by canonical JSON text. An unordered
      collection has no order to preserve, so ordering it is the only way to
      hash it deterministically at all (Python's set iteration order is
      hash-randomized per process). It lowers to a plain array, deliberately
      *not* a tagged wrapper: which container a parser chose for a member list
      is an implementation detail, and a parser normalizing ``{"a", "b"}`` to
      ``["a", "b"]`` must not move every affected ``content_identity``.
    - ``Decimal``: ``float``, the same lowering ``core/json.py`` declares for
      a JSON parser's ``Decimal`` (``_lower_decimals``, ``_default_encoder``).
      The ``QUERY`` digest profile ``hash_payload`` uses reaches stdlib
      ``json.dumps`` with no ``default`` hook at all, so a ``Decimal`` in
      block metadata raised here too.
    - ``datetime``/``date``/``time``: ISO-8601 text. ``bytes``-family: hex.
      ``Enum``: its value. Same canonical forms the publication encoder uses.

    Anything else raises :class:`UnhashablePayloadValueError` naming the type and
    the field path, so the next gap arrives as a named vocabulary refusal
    rather than a JSON backend message. Stringifying the unknown value instead
    would be worse than refusing: ``str(object())`` embeds a memory address,
    which would make an identity that is supposed to be content-derived vary
    per process.

    **No stored identity moves.** Every value type newly admitted here
    previously raised inside ``hash_payload``, and ``content_identity`` is
    computed on the write path (``archive_tiers/write.py``) before any row is
    inserted -- so a message carrying one of these shapes could never have been
    written to an archive in the first place. ``Decimal`` is the one type that
    already hashed successfully, and its lowering is unchanged.
    """
    if value is None:
        return _NULL_SENTINEL
    if isinstance(value, str):
        return _EMPTY_SENTINEL if value == "" else nfc(value)
    if isinstance(value, Mapping):
        return {
            nfc(key) if isinstance(key, str) else key: _normalize_nested_for_hash(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_nested_for_hash(item, path=f"{path}[]") for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_normalize_nested_for_hash(item, path=f"{path}{{}}") for item in value),
            key=_canonical_sort_key,
        )
    if isinstance(value, Enum):
        return _normalize_nested_for_hash(value.value, path=path)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (int, float)):
        return value
    raise UnhashablePayloadValueError(
        f"{type(value).__name__} at {path} is outside the declared hash vocabulary "
        f"(polylogue/pipeline/ids.py:_normalize_nested_for_hash); declare its canonical form there"
    )


def _normalize_for_hash(value: HashScalar) -> JSONValue:
    """Normalize a value for hashing, distinguishing None from empty.

    Args:
        value: Hash-compatible scalar value to normalize.

    Returns:
        Normalized JSON value with None → _NULL_SENTINEL and "" → _EMPTY_SENTINEL.
    """
    if value is None:
        return _NULL_SENTINEL
    if value == "":
        return _EMPTY_SENTINEL
    if isinstance(value, str):
        return nfc(value)
    return value


def session_id(source_name: Provider | Origin | str, provider_session_id: str) -> SessionId:
    """Generate the archive session ID from source/provider input.

    Args:
        provider_session_id: Provider's session identifier.

    Returns:
        Formatted session ID.

    Raises:
        ValueError: If source_name or provider_session_id is empty.
    """
    source_text = str(source_name).strip()
    if source_text == "":
        raise ValueError("source_name cannot be empty")
    if not provider_session_id or not provider_session_id.strip():
        raise ValueError("provider_session_id cannot be empty")
    origin = origin_from_provider(source_name)
    return SessionId(f"{origin.value}:{provider_session_id}")


#: Declared hash vocabulary for a conversation whose export carries no native
#: conversation id at all (Grok account-data exports: neither the
#: ``conversation`` object nor any response entry has an id field in any
#: confirmed wire shape). Without this, such a parser can only fall back to an
#: acquisition coordinate -- the file stem plus the conversation's array index
#: -- which is not identity: two files named ``grok.json`` in different
#: directories mint the SAME ``grok:grok`` session (the second ingest
#: full-replaces the first), and a re-export that reorders conversations
#: re-identifies every one of them (polylogue-31zag).
#:
#: The vocabulary is exactly the opening turn plus the conversation's declared
#: creation time: the earliest content the conversation has, and the one
#: timestamp a provider does not re-derive per export request. Both are
#: intrinsic to the conversation, so the identity survives reordering, a
#: renamed export file, and later turns being appended. Deliberately excluded:
#: the title (user- and provider-renameable after the fact), the response
#: count and any later turn (an ongoing conversation gains turns between
#: exports), and every acquisition coordinate (file stem, array index, member
#: path).
#:
#: Known, accepted limit stated rather than engineered around: two genuinely
#: distinct conversations that open with a byte-identical first turn at the
#: same declared creation time are indistinguishable by any signal this
#: vocabulary can offer.


def idless_session_identity(
    *,
    first_message_provider_id: JSONValue,
    first_message_text: JSONValue,
    created_at: JSONValue,
) -> str:
    """The sole constructor of a content-derived id for an id-less conversation.

    This is a fixed keyword-only signature, not a dict projected by a list of
    field names: passing a title, a message count, a file stem or an array
    index is a ``TypeError`` at the call boundary, not a value that has to be
    remembered and stripped. Extending what an id-less conversation's identity
    covers requires editing this signature and the vocabulary comment above
    it -- an explicit, reviewable decision.

    The result is a ``provider_session_id``, not a full session id; the caller
    still passes it through :func:`session_id` with its own source.
    """
    payload = {
        "created_at": _normalize_for_hash(created_at if isinstance(created_at, str) or created_at is None else None),
        "first_message_provider_id": _normalize_for_hash(
            first_message_provider_id if isinstance(first_message_provider_id, str) else None
        ),
        "first_message_text": _normalize_for_hash(first_message_text if isinstance(first_message_text, str) else None),
    }
    return f"conversation-{hash_payload(payload)[:24]}"


def message_id(session_id: SessionId, provider_message_id: str) -> MessageId:
    return MessageId(f"{session_id}:{provider_message_id}")


def _content_block_payload(block: ParsedContentBlock) -> dict[str, JSONValue]:
    """Build the declared semantic payload for a single content block."""
    return _model_hash_payload(block, _HASHED_FIELDS["ParsedContentBlock"])


def _is_redundant_text_only_block(message: ParsedMessage) -> bool:
    """True when ``message.blocks`` is a single TEXT block that just repeats ``message.text``.

    polylogue-0qfy: some claude-ai-export vintages parse an otherwise
    identical message with an empty ``blocks`` list, others with exactly one
    ``[{"type":"text","text": message.text}]`` block carrying no other
    field (tool_name/tool_id/tool_input/media_type all unset). This is a
    parser-shape artifact, not a second content axis -- both vintages carry
    the same real content -- so it must not make the hash-stable payload
    (and therefore the raw-authority membership comparison, which reuses
    this same payload) see the two vintages as a genuine conflict.
    """
    if len(message.blocks) != 1:
        return False
    block = message.blocks[0]
    return (
        block.type is BlockType.TEXT
        and block.text == message.text
        and not block.tool_name
        and not block.tool_id
        and block.tool_input is None
        and not block.media_type
        # Every remaining evidence-bearing field must also be absent. A block
        # carrying citations (``web_constructs``), parser metadata, or a tool
        # outcome is a second content axis, not the parser-shape artifact this
        # predicate exists to absorb, so collapsing it to the empty sentinel
        # would make a richer acquisition hash equal to a bare one.
        # ``signature`` stays excluded on purpose (see ``base_models.py``): it
        # is a provider attestation over content already hashed here.
        and not block.metadata
        and not block.web_constructs
        and block.is_error is None
        and block.exit_code is None
        and block.tool_outcome is None
        and not block.outcome_unknown_reason
        and block.file_edit is None
    )


def _message_hash_payload(message: ParsedMessage, message_id: str) -> dict[str, JSONValue]:
    """Build the complete semantic payload for a single message."""
    payload: dict[str, JSONValue] = {"id": message_id}
    payload.update(_message_semantic_payload(message))
    return payload


def _message_payload(message: ParsedMessage, fields: frozenset[str]) -> dict[str, JSONValue]:
    """Build a message payload with the requested semantic field boundary."""
    payload = _model_hash_payload(message, fields - {"blocks", "provider_message_id"})
    if message.blocks and not _is_redundant_text_only_block(message):
        payload["blocks"] = [_content_block_payload(b) for b in message.blocks]
    else:
        payload["blocks"] = _EMPTY_SENTINEL
    return payload


def _message_semantic_payload(message: ParsedMessage) -> dict[str, JSONValue]:
    """Build the complete semantic payload used by session content hashing."""
    return _message_payload(message, _HASHED_FIELDS["ParsedMessage"])


#: Hex characters retained from the message semantic digest when it stands in
#: for an absent provider id. The value only has to distinguish messages
#: *within one session*, so 128 bits is an enormous margin; keeping it short
#: bounds the length of ``message_id`` and of every ``block_id`` built on it.
MESSAGE_CONTENT_IDENTITY_HEX_CHARS = 32

#: One message's content-derived fallback identity: the semantic digest plus
#: the occurrence ordinal that separates two messages whose declared semantic
#: fields are byte-identical.
MessageContentIdentity: TypeAlias = tuple[str, int]


def message_content_identity(message: ParsedMessage) -> str:
    """Digest one message's declared semantic fields into a stable anchor.

    This is the identity used when a provider supplies no message id. It is
    computed from ``_HASHED_FIELDS["ParsedMessage"]`` -- the partition that
    already declares, exhaustively, which parser fields are content and which
    are parser-only coordinates or independently-owned measurements. Position,
    variant index, branch index, path markers, usage and timing are all on the
    excluded side, which is exactly the property this identity needs: a
    message that is inserted, removed or reordered *elsewhere* in the export
    must not change the identity of this one (polylogue-eqsri).
    """
    return hash_payload(_message_semantic_payload(message))[:MESSAGE_CONTENT_IDENTITY_HEX_CHARS]


def message_content_identities(
    messages: Sequence[ParsedMessage],
    *,
    occurrence_offsets: Mapping[str, int] | None = None,
) -> tuple[MessageContentIdentity, ...]:
    """Resolve the content-derived fallback identity for one message batch.

    Two messages in one session can carry byte-identical declared semantics.
    They are separated by their occurrence ordinal *among messages sharing
    that digest* -- not by their transcript position, so inserting or removing
    an unrelated message leaves every other message's ordinal untouched.

    ``occurrence_offsets`` carries the per-digest counts already stored for
    this session, so an append continues the session's numbering instead of
    restarting it and colliding with a row already written.
    """
    counts: Counter[str] = Counter()
    if occurrence_offsets:
        counts.update(dict(occurrence_offsets))
    identities: list[MessageContentIdentity] = []
    for message in messages:
        digest = message_content_identity(message)
        identities.append((digest, counts[digest]))
        counts[digest] += 1
    return tuple(identities)


class _DiskMessageContentIdentities(Sequence[MessageContentIdentity]):
    def __init__(self, conn: sqlite3.Connection, count: int) -> None:
        self._conn = conn
        self._count = count

    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, index: int) -> MessageContentIdentity: ...

    @overload
    def __getitem__(self, index: slice) -> list[MessageContentIdentity]: ...

    def __getitem__(self, index: int | slice) -> MessageContentIdentity | list[MessageContentIdentity]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._count))]
        ordinal = index + self._count if index < 0 else index
        if ordinal < 0 or ordinal >= self._count:
            raise IndexError(index)
        row = self._conn.execute("SELECT digest, occurrence FROM identity WHERE ordinal = ?", (ordinal,)).fetchone()
        if row is None:
            raise ValueError("prepared message identity row disappeared")
        return str(row[0]), int(row[1])

    def __iter__(self) -> Iterator[MessageContentIdentity]:
        for digest, occurrence in self._conn.execute("SELECT digest, occurrence FROM identity ORDER BY ordinal"):
            yield str(digest), int(occurrence)


@contextmanager
def disk_message_content_identities(
    messages: Sequence[ParsedMessage],
    *,
    occurrence_offsets: Mapping[str, int] | None = None,
) -> Iterator[Sequence[MessageContentIdentity]]:
    """Spool canonical fallback identities for bounded random-access lowering.

    The scratch database lives beside a prepared message sink when one is
    present. Its rows, including per-digest occurrence counts, are disposable;
    the caller must consume the sequence inside this context.
    """
    parent = getattr(messages, "path", None)
    directory = Path(parent).parent if parent is not None else None
    with (
        tempfile.TemporaryDirectory(prefix="polylogue-ids-", dir=directory) as scratch,
        closing(sqlite3.connect(Path(scratch) / "identities.db")) as conn,
    ):
        conn.execute("CREATE TABLE count (digest TEXT PRIMARY KEY, value INTEGER NOT NULL) WITHOUT ROWID")
        conn.execute(
            "CREATE TABLE identity (ordinal INTEGER PRIMARY KEY, digest TEXT NOT NULL, occurrence INTEGER NOT NULL)"
        )
        if occurrence_offsets:
            conn.executemany("INSERT INTO count VALUES (?, ?)", occurrence_offsets.items())
        count = 0
        for count, message in enumerate(messages, start=1):
            digest = message_content_identity(message)
            row = conn.execute("SELECT value FROM count WHERE digest = ?", (digest,)).fetchone()
            occurrence = int(row[0]) if row is not None else 0
            conn.execute(
                "INSERT INTO count (digest, value) VALUES (?, ?) "
                "ON CONFLICT(digest) DO UPDATE SET value = excluded.value",
                (digest, occurrence + 1),
            )
            conn.execute("INSERT INTO identity VALUES (?, ?, ?)", (count - 1, digest, occurrence))
        yield _DiskMessageContentIdentities(conn, count)


def _message_comparison_payload(message: ParsedMessage) -> dict[str, JSONValue]:
    """Build the owner-safe payload used to compare id-less messages.

    This deliberately excludes ``position``.  The owner resolver must retain
    the ability to report duplicate messages as ambiguous when position is
    their only distinguishing evidence.
    """
    return _message_payload(message, _OWNER_MATCH_FIELDS["ParsedMessage"])


#: Marker prefix for a content-derived message identity anchor, used only
#: when a message carries no native ``provider_message_id``. Namespaced so it
#: can never collide with a real provider id string (a provider id never
#: contains this literal token by construction).
_CONTENT_ANCHOR_PREFIX = "__polylogue_msg_content_anchor__"


def _message_revision_match_id(message: ParsedMessage) -> str:
    """Resolve the stable revision-match identity for one parsed message.

    Prefers the provider's own id -- stable across reordering even when the
    export's array ordering is not (polylogue-c429). When a parser could not
    populate one (``provider_message_id`` is empty), this used to fall back
    to ``f"msg-{index}"`` -- positionally-derived, exactly the same failure
    class the attachment identity fix (polylogue-hith/-d8al) removed for a
    different field: two parses of the same id-less message in a different
    array position would get two different fallback "ids" and compare as a
    conflict instead of the same message (polylogue-gysk3).

    Timestamped id-less messages use only role and timestamp here. Their text
    and blocks remain in ``_message_hash_payload`` as mutable content, so an
    edit shares one revision axis while still changing the session hash.
    Timestamp-less messages need their content to remain distinguishable.

    Parser normalization maintains the complementary invariant: a missing
    native id is never replaced with an array-position-derived value before
    reaching this function. Parser-local occurrence keys may use position,
    but they are not persisted as ``provider_message_id``.
    """
    native_id = message.provider_message_id.strip()
    if native_id:
        return native_id
    payload: dict[str, JSONValue] = {
        "role": str(message.role),
        "timestamp": _normalize_for_hash(message.timestamp),
    }
    if message.timestamp is None:
        payload["text"] = _normalize_for_hash(message.text)
        if message.blocks and not _is_redundant_text_only_block(message):
            payload["content_blocks"] = [_content_block_payload(b) for b in message.blocks]
    return f"{_CONTENT_ANCHOR_PREFIX}:{hash_payload(payload)}"


@dataclass(frozen=True, slots=True)
class MessageOwnerResolution:
    """One shared private owner-resolution contract for hash and write paths."""

    keys: Sequence[str]
    by_physical_coordinate: Mapping[tuple[int, int], str]
    ambiguous_physical_coordinates: Set[tuple[int, int]]
    by_stable_key: Mapping[str, str]
    ambiguous_stable_keys: Set[str]
    ambiguous_keys: Set[str]
    unique_provider_keys: Mapping[str, str]
    ambiguous_provider_ids: Set[str]


class _SqliteOwnerKeys(Sequence[str]):
    def __init__(self, conn: sqlite3.Connection, count: int) -> None:
        self._conn = conn
        self._count = count

    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> list[str]: ...

    def __getitem__(self, index: int | slice) -> str | list[str]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._count))]
        ordinal = index + self._count if index < 0 else index
        if ordinal < 0 or ordinal >= self._count:
            raise IndexError(index)
        row = self._conn.execute("SELECT owner_key FROM owner_message WHERE ordinal = ?", (ordinal,)).fetchone()
        if row is None or row[0] is None:
            raise ValueError("prepared message owner key row disappeared")
        return str(row[0])

    def __iter__(self) -> Iterator[str]:
        for (key,) in self._conn.execute("SELECT owner_key FROM owner_message ORDER BY ordinal"):
            if key is None:
                raise ValueError("prepared message owner key row is unresolved")
            yield str(key)


class _SqliteOwnerLookup(Mapping[_T, str]):
    def __init__(self, conn: sqlite3.Connection, kind: str) -> None:
        self._conn = conn
        self._kind = kind

    def __getitem__(self, key: _T) -> str:
        row = self._conn.execute(
            "SELECT value FROM owner_lookup WHERE kind = ? AND key = ?",
            (self._kind, json.dumps(key, separators=(",", ":"))),
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return str(row[0])

    def __iter__(self) -> Iterator[_T]:
        for (key,) in self._conn.execute("SELECT key FROM owner_lookup WHERE kind = ?", (self._kind,)):
            value = json.loads(key)
            yield cast(_T, tuple(value) if self._kind == "physical" else value)

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM owner_lookup WHERE kind = ?", (self._kind,)).fetchone()[0])


class _SqliteOwnerAmbiguities(Set[_T]):
    def __init__(self, conn: sqlite3.Connection, kind: str) -> None:
        self._conn = conn
        self._kind = kind

    def __contains__(self, key: object) -> bool:
        row = self._conn.execute(
            "SELECT count FROM owner_count WHERE kind = ? AND key = ?",
            (self._kind, json.dumps(key, separators=(",", ":"))),
        ).fetchone()
        return row is not None and int(row[0]) > 1

    def __iter__(self) -> Iterator[_T]:
        for (key,) in self._conn.execute("SELECT key FROM owner_count WHERE kind = ? AND count > 1", (self._kind,)):
            value = json.loads(key)
            yield cast(_T, tuple(value) if self._kind == "physical" else value)

    def __len__(self) -> int:
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM owner_count WHERE kind = ? AND count > 1", (self._kind,)
            ).fetchone()[0]
        )


@contextmanager
def disk_message_owner_resolution(messages: Sequence[ParsedMessage]) -> Iterator[MessageOwnerResolution]:
    """Resolve attachment anchors with disk-backed counts and lookup maps."""
    parent = getattr(messages, "path", None)
    directory = Path(parent).parent if parent is not None else None
    with (
        tempfile.TemporaryDirectory(prefix="polylogue-owners-", dir=directory) as scratch,
        closing(sqlite3.connect(Path(scratch) / "owners.db")) as conn,
    ):
        conn.execute(
            "CREATE TABLE owner_count (kind TEXT NOT NULL, key TEXT NOT NULL, count INTEGER NOT NULL, "
            "PRIMARY KEY (kind, key)) WITHOUT ROWID"
        )
        conn.execute(
            "CREATE TABLE owner_message (ordinal INTEGER PRIMARY KEY, revision TEXT NOT NULL, "
            "content TEXT NOT NULL, stable TEXT, physical TEXT, provider TEXT, owner_key TEXT)"
        )
        conn.execute(
            "CREATE TABLE owner_lookup (kind TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, "
            "PRIMARY KEY (kind, key)) WITHOUT ROWID"
        )

        def encoded(value: object) -> str:
            return json.dumps(value, separators=(",", ":"))

        def increment(kind: str, value: object) -> None:
            conn.execute(
                "INSERT INTO owner_count VALUES (?, ?, 1) ON CONFLICT(kind, key) DO UPDATE SET count = count + 1",
                (kind, encoded(value)),
            )

        def count(kind: str, value: object) -> int:
            row = conn.execute(
                "SELECT count FROM owner_count WHERE kind = ? AND key = ?", (kind, encoded(value))
            ).fetchone()
            return int(row[0]) if row is not None else 0

        total = 0
        for ordinal, message in enumerate(messages):
            total = ordinal + 1
            revision = _message_revision_match_id(message)
            content = f"{_CONTENT_ANCHOR_PREFIX}:{hash_payload(_message_comparison_payload(message))}"
            coordinate = _message_owner_coordinate(message, ordinal)
            stable = coordinate.stable_key
            physical = coordinate.physical_key
            provider = message.provider_message_id.strip() or None
            conn.execute(
                "INSERT INTO owner_message VALUES (?, ?, ?, ?, ?, ?, NULL)",
                (ordinal, revision, content, stable, encoded(physical) if physical is not None else None, provider),
            )
            increment("revision", revision)
            increment("content", content)
            if stable is not None:
                increment("stable", stable)
            if physical is not None:
                increment("physical", physical)
            if provider is not None:
                increment("provider", provider)
        for ordinal in range(total):
            revision, content, stable = conn.execute(
                "SELECT revision, content, stable FROM owner_message WHERE ordinal = ?", (ordinal,)
            ).fetchone()
            if stable is not None and count("stable", stable) == 1:
                key = stable
            elif count("revision", revision) == 1:
                key = revision
            elif count("content", content) == 1:
                key = content
            elif stable is not None:
                key = stable
            else:
                key = revision
            conn.execute("UPDATE owner_message SET owner_key = ? WHERE ordinal = ?", (key, ordinal))
            increment("key", key)
        for ordinal in range(total):
            stable, physical, provider, key = conn.execute(
                "SELECT stable, physical, provider, owner_key FROM owner_message WHERE ordinal = ?", (ordinal,)
            ).fetchone()
            if physical is not None and count("physical", json.loads(physical)) == 1:
                conn.execute("INSERT INTO owner_lookup VALUES (?, ?, ?)", ("physical", physical, key))
            if stable is not None and count("stable", stable) == 1 and count("key", key) == 1:
                conn.execute("INSERT INTO owner_lookup VALUES (?, ?, ?)", ("stable", encoded(stable), key))
            if provider is not None and count("provider", provider) == 1:
                conn.execute("INSERT INTO owner_lookup VALUES (?, ?, ?)", ("provider", encoded(provider), key))
        yield MessageOwnerResolution(
            keys=_SqliteOwnerKeys(conn, total),
            by_physical_coordinate=_SqliteOwnerLookup[tuple[int, int]](conn, "physical"),
            ambiguous_physical_coordinates=_SqliteOwnerAmbiguities[tuple[int, int]](conn, "physical"),
            by_stable_key=_SqliteOwnerLookup[str](conn, "stable"),
            ambiguous_stable_keys=_SqliteOwnerAmbiguities[str](conn, "stable"),
            ambiguous_keys=_SqliteOwnerAmbiguities[str](conn, "key"),
            unique_provider_keys=_SqliteOwnerLookup[str](conn, "provider"),
            ambiguous_provider_ids=_SqliteOwnerAmbiguities[str](conn, "provider"),
        )


def _message_owner_coordinate(message: ParsedMessage, fallback_position: int) -> MessageOwnerCoordinate:
    coordinate = message.owner_coordinate
    if coordinate is not None:
        return MessageOwnerCoordinate(
            stable_key=coordinate.stable_key,
            position=coordinate.position if coordinate.position is not None else message.position,
            variant_index=coordinate.variant_index,
        )
    return MessageOwnerCoordinate(
        position=message.position if message.position is not None else fallback_position,
        variant_index=message.variant_index or 0,
    )


def message_owner_resolution(messages: list[ParsedMessage]) -> MessageOwnerResolution:
    """Resolve stable private owner keys from one parsed message batch.

    Unique native ids and timestamped id-less role/timestamp anchors are
    stable across reordering and edits. Duplicate anchors use a unique
    content discriminator when the occurrences differ, then parser-provided
    reorder-stable evidence when their content is identical. If neither
    distinguishes the occurrences, the duplicate remains typed ambiguity
    instead of receiving a position-derived identity.

    The content discriminator covers a media block's ``metadata`` -- what an
    ``image``/``document`` turn cites (a Drive file id, an asset pointer, an
    inline-content digest). AI Studio exports carry runs of id-less,
    text-less turns that share one timestamp and differ only there, so
    ``metadata`` is the sole evidence keeping their attachments ownable
    (polylogue-prjai).
    """
    revision_ids = tuple(_message_revision_match_id(message) for message in messages)
    revision_counts = Counter(revision_ids)
    content_ids = tuple(
        f"{_CONTENT_ANCHOR_PREFIX}:{hash_payload(_message_comparison_payload(message))}" for message in messages
    )
    content_counts = Counter(content_ids)
    coordinates = tuple(_message_owner_coordinate(message, index) for index, message in enumerate(messages))
    stable_counts = Counter(coordinate.stable_key for coordinate in coordinates if coordinate.stable_key is not None)

    keys: list[str] = []
    for revision_id, content_id, coordinate in zip(revision_ids, content_ids, coordinates, strict=True):
        if coordinate.stable_key is not None and stable_counts[coordinate.stable_key] == 1:
            key = coordinate.stable_key
        elif revision_counts[revision_id] == 1:
            key = revision_id
        elif content_counts[content_id] == 1:
            key = content_id
        elif coordinate.stable_key is not None:
            key = coordinate.stable_key
        else:
            key = revision_id
        keys.append(key)

    key_counts = Counter(keys)
    ambiguous_keys = frozenset(key for key, count in key_counts.items() if count > 1)
    physical_counts = Counter(
        coordinate.physical_key for coordinate in coordinates if coordinate.physical_key is not None
    )
    by_physical_coordinate = {
        coordinate.physical_key: key
        for coordinate, key in zip(coordinates, keys, strict=True)
        if coordinate.physical_key is not None and physical_counts[coordinate.physical_key] == 1
    }
    by_stable_key = {
        coordinate.stable_key: key
        for coordinate, key in zip(coordinates, keys, strict=True)
        if (
            coordinate.stable_key is not None
            and stable_counts[coordinate.stable_key] == 1
            and key not in ambiguous_keys
        )
    }
    provider_keys: dict[str, str] = {}
    provider_counts = Counter(
        message.provider_message_id.strip() for message in messages if message.provider_message_id
    )
    for message, key in zip(messages, keys, strict=True):
        provider_id = message.provider_message_id.strip()
        if provider_id and provider_counts[provider_id] == 1:
            provider_keys[provider_id] = key
    return MessageOwnerResolution(
        keys=tuple(keys),
        by_physical_coordinate=by_physical_coordinate,
        ambiguous_physical_coordinates=frozenset(
            coordinate for coordinate, count in physical_counts.items() if count > 1
        ),
        by_stable_key=by_stable_key,
        ambiguous_stable_keys=frozenset(stable_key for stable_key, count in stable_counts.items() if count > 1),
        ambiguous_keys=ambiguous_keys,
        unique_provider_keys=provider_keys,
        ambiguous_provider_ids=frozenset(provider_id for provider_id, count in provider_counts.items() if count > 1),
    )


def _attachment_owner_coordinate(attachment: ParsedAttachment) -> MessageOwnerCoordinate:
    coordinate = attachment.owner_coordinate
    if coordinate is not None:
        return MessageOwnerCoordinate(
            stable_key=coordinate.stable_key,
            position=coordinate.position if coordinate.position is not None else attachment.message_position,
            variant_index=coordinate.variant_index,
        )
    return MessageOwnerCoordinate(
        position=attachment.message_position,
        variant_index=attachment.message_variant_index or 0,
    )


def attachment_message_owner_key(attachment: ParsedAttachment, resolution: MessageOwnerResolution) -> str | None:
    """Resolve one attachment to the same private owner key used by writes."""
    coordinate = _attachment_owner_coordinate(attachment)
    if (
        coordinate.stable_key is not None
        and coordinate.stable_key not in resolution.ambiguous_stable_keys
        and coordinate.stable_key not in resolution.ambiguous_keys
        and coordinate.stable_key in resolution.by_stable_key
    ):
        return resolution.by_stable_key[coordinate.stable_key]
    if coordinate.physical_key is not None:
        if coordinate.physical_key in resolution.ambiguous_physical_coordinates:
            raise MessageOwnerAmbiguityError(f"attachment owner coordinate is duplicated: {coordinate.physical_key!r}")
        key = resolution.by_physical_coordinate.get(coordinate.physical_key)
        if key is not None:
            if key in resolution.ambiguous_keys:
                raise MessageOwnerAmbiguityError(
                    "attachment owner coordinate is indistinguishable from another message: "
                    f"{coordinate.physical_key!r}"
                )
            return key
    if coordinate.stable_key in resolution.ambiguous_stable_keys:
        raise MessageOwnerAmbiguityError(f"attachment owner evidence is duplicated: {coordinate.stable_key!r}")
    if attachment.message_provider_id:
        provider_id = attachment.message_provider_id.strip()
        if provider_id in resolution.ambiguous_provider_ids:
            raise MessageOwnerAmbiguityError(
                f"attachment provider message id is duplicated without a private coordinate: {provider_id!r}"
            )
        return resolution.unique_provider_keys.get(provider_id, provider_id)
    return None


def message_identity_hash(*, id: str) -> bytes:
    """The sole constructor of a message's comparison identity (polylogue-aggz).

    A provider's own message id is stable across re-exports even when the
    export's array ordering is not (polylogue-c429) -- it is the only
    content field that answers *which message is this*, as opposed to *what
    does it currently say*.

    This is a fixed keyword-only signature, not a dict projected by a list
    of field names: passing ``role``/``text``/``timestamp``/anything else is
    a ``TypeError`` at the call boundary, not a value that has to be
    remembered and stripped. Extending what a message's comparison identity
    covers requires editing this signature -- an explicit, reviewable
    decision, never a side effect of a parser gaining a new field.
    """
    return bytes.fromhex(hash_payload({"id": id}))


#: Fields of an attachment hash payload that answer *which attachment is
#: this*, as opposed to *what have we managed to read about it* -- the
#: anchoring message plus content descriptors, content-derived and never the
#: provider's own attachment id. Claude.ai does not consistently emit
#: ``id``/``file_id``/``fileId``/``uuid``/``file_uuid`` for the same
#: attachment across separate export requests of the same conversation: one
#: vintage carries a real UUID-shaped id, the other has none (a positionally
#: -seeded synthetic id is not identity either -- polylogue-hith). No
#: id-minting scheme can make a real id and any synthetic value collide by
#: construction, so the id is excluded from identity altogether rather than
#: used when present (polylogue-d8al). ``size_bytes`` is excluded because for
#: lazily-fetched attachments (Drive/Gemini references, browser capture) the
#: provider states no size until the bytes are actually read, so treating it
#: as identity would make acquisition look like a different attachment
#: (polylogue-bu1i). ``inline_content_hash`` is excluded for the same reason
#: and is recovered separately as acquisition evidence in
#: ``attachment_contents``.
#:
#: Known, accepted limit stated explicitly rather than engineered around: two
#: genuinely distinct attachments that share one message/name/media-type and
#: carry no bytes on either side of a comparison are indistinguishable by any
#: signal this projection can offer.


def attachment_identity_hash(*, message_id: JSONValue, name: JSONValue, mime_type: JSONValue) -> bytes:
    """The sole constructor of an attachment's comparison identity (polylogue-aggz).

    Fixed to (anchoring message, name, media type) -- content-derived and
    never the provider's own attachment id (polylogue-d8al, polylogue-hith)
    or acquisition state such as ``size_bytes``/inline bytes
    (polylogue-bu1i). Those fields are not parameters here; passing them
    (e.g. spreading a full attachment payload dict as ``**kwargs``) is a
    ``TypeError``, not a value this function has to remember to strip.
    """
    return bytes.fromhex(hash_payload({"message_id": message_id, "name": name, "mime_type": mime_type}))


def _attachment_hash_payload(
    attachment: ParsedAttachment, *, message_owner_anchor: str | None = None
) -> dict[str, JSONValue]:
    """Build the full attachment payload using the shared owner coordinate."""
    owner_id = message_owner_anchor or attachment.message_provider_id
    payload: dict[str, JSONValue] = {
        "id": _normalize_for_hash(attachment.provider_attachment_id),
        "message_id": _normalize_for_hash(owner_id),
        "name": _normalize_for_hash(attachment.name),
        "mime_type": _normalize_for_hash(attachment.mime_type),
        "size_bytes": _normalize_for_hash(attachment.size_bytes),
    }
    if attachment.inline_bytes is not None:
        payload["inline_content_hash"] = hash_bytes(attachment.inline_bytes)
    elif attachment.precomputed_blob is not None:
        # polylogue-8ac0: bytes already streamed into the blob store during
        # sidecar discovery carry a known hash without needing to re-read
        # them here (mirrors the ``inline_bytes`` branch above, whose content
        # hash marks re-ingest content-changed once bytes newly arrive).
        payload["inline_content_hash"] = attachment.precomputed_blob[0]
    return payload


#: `generation_lifecycle` payload keys that are provider-reported measurement,
#: not identity, when the event's own payload declares them non-durable via
#: ``duration_semantics == "provider_reported_elapsed"``. ChatGPT re-derives
#: these from the raw export's own ``finished_duration_sec`` /
#: ``reasoning_start_time``/``reasoning_end_time`` metadata on every export
#: request, and the value is not stable across requests for the SAME
#: generation (observed varying non-monotonically, e.g. 13000 vs 21000ms;
#: 123000 vs 33000ms) even when the transcript is byte-identical
#: (polylogue-nuec).
_PROVIDER_REPORTED_ELAPSED_VOLATILE_PAYLOAD_KEYS = frozenset({"elapsed_duration_ms", "started_at_ms", "ended_at_ms"})
_PROVIDER_REPORTED_ELAPSED_MARKER_KEY = "duration_semantics"
_PROVIDER_REPORTED_ELAPSED_MARKER_VALUE = "provider_reported_elapsed"

#: Event payload ALLOWLIST by event type: only these fields, per type, ever
#: enter ``event_contents``. This is deliberately an allowlist, not a
#: denylist of fields discovered volatile after the fact -- three separate
#: volatility axes (attachment acquisition state, message array order,
#: provider-reported generation duration) were each found only after
#: shipping, one bead and one branch at a time, because a denylist design
#: means a NEW field silently enters identity the moment a parser starts
#: emitting it, until someone notices and adds it to the strip-list. Under
#: an allowlist, a field a parser adds later is excluded from comparison by
#: construction -- it takes an explicit decision to add it here before it
#: can affect identity, not an explicit decision to exclude it.
#:
#: Event types with no entry here compare their FULL payload (today's
#: behavior for every type except ``generation_lifecycle``, which is the
#: only one with proven volatility -- polylogue-nuec; the raw export's own
#: ``finished_duration_sec``/``reasoning_start_time``/``reasoning_end_time``
#: metadata is not stable across separate export requests for the SAME
#: generation, so ``elapsed_duration_ms``/``started_at_ms``/``ended_at_ms``
#: and the derived ``timestamp`` are excluded; ``duration_semantics`` merely
#: documents that fact and carries no content of its own).
_EVENT_CONTENT_PAYLOAD_ALLOWLIST: dict[str, frozenset[str]] = {
    "generation_lifecycle": frozenset({"state", "evidence_source", "fidelity"}),
}


def _anchor_is_remeasured(event: ParsedSessionEvent) -> bool:
    """Whether this event's anchoring message is provider measurement, not content.

    True only for an event type carrying a payload allowlist AND declaring
    ``duration_semantics == "provider_reported_elapsed"`` -- the ChatGPT
    generation_lifecycle shape. Same two-part test
    :func:`_event_content_payload` uses to strip remeasured payload fields, so
    the anchor tolerance can never be broader than the payload tolerance.
    """
    allowlist = _EVENT_CONTENT_PAYLOAD_ALLOWLIST.get(event.event_type)
    return (
        allowlist is not None
        and event.payload.get(_PROVIDER_REPORTED_ELAPSED_MARKER_KEY) == _PROVIDER_REPORTED_ELAPSED_MARKER_VALUE
    )


def _event_content_payload(event: ParsedSessionEvent) -> dict[str, JSONValue]:
    """Build the position- and measurement-independent CONTENT payload for one event.

    Array position is not identity for events any more than for messages:
    ChatGPT's ``generation_lifecycle`` events were independently observed to
    reorder alongside their duration values across separate export requests
    of the SAME conversation (same three durations, different array
    positions each time) -- the same volatility polylogue-c429 found for
    messages, on a different axis (polylogue-nuec). This never includes
    ``event_index``; ``session_revision_projection`` builds identity purely
    from this content plus the event's own type and anchoring message.

    The allowlist strip is narrowed to the specific provider-remeasured
    shape it targets: it applies only when the event's own payload declares
    itself non-durable via ``duration_semantics ==
    "provider_reported_elapsed"`` (set by the ChatGPT parser). The
    browser-capture parser emits the SAME ``event_type`` for its own DOM/UI
    generation observations, tagged with a different ``duration_semantics``
    (e.g. ``dom_observed_wall``, ``provider_ui_elapsed``) -- those are a real
    first-party measurement this projection has no other record of, not the
    re-derived-on-every-export ChatGPT value nuec exists for, so their
    observation id, timestamp, and duration/label/trigger fields remain
    content rather than being silently stripped.
    """
    allowlist = _EVENT_CONTENT_PAYLOAD_ALLOWLIST.get(event.event_type)
    provider_reported_elapsed = (
        allowlist is not None
        and event.payload.get(_PROVIDER_REPORTED_ELAPSED_MARKER_KEY) == _PROVIDER_REPORTED_ELAPSED_MARKER_VALUE
    )
    if allowlist is None or not provider_reported_elapsed:
        payload = event.payload
        timestamp = event.timestamp
    else:
        payload = {key: value for key, value in event.payload.items() if key in allowlist}
        # An event type with a registered allowlist also has its own
        # provider-remeasured timestamp excluded: for generation_lifecycle,
        # ChatGPT sets it from the same reasoning_end_time value the
        # duration is derived from, so it varies in tandem and is
        # measurement too, not content.
        timestamp = None
    return {
        "event_type": _normalize_for_hash(event.event_type),
        "timestamp": _normalize_for_hash(timestamp),
        "source_message_provider_id": _normalize_for_hash(event.source_message_provider_id),
        "payload": hash_payload(_normalize_nested_for_hash(payload)),
    }


#: The subset of an event content payload that answers *which event slot is
#: this*, as opposed to *what does it say*: anchoring message plus event
#: type, content-derived and never the array index. Can be shared by more
#: than one event within one revision (e.g. multiple
#: ``chatgpt_block_metadata`` events on the same message, one per block), so
#: ``session_revision_projection`` always folds the event's own content hash
#: into the FINAL identity on top of this base -- unconditionally, not only
#: when a sibling is present in that particular revision, so identity never
#: depends on what else happens to be in the set. Still content-derived
#: (each block's own content, including any content-intrinsic field such as
#: ``block_index``, already differs), never the array position.
def event_base_identity_hash(*, event_type: JSONValue, source_message_provider_id: JSONValue) -> bytes:
    """The sole constructor of an event's position-independent base identity.

    Anchoring message plus event type only -- content-derived, never the
    array index and never provider-reported measurement (polylogue-nuec),
    which is not a parameter here. Fixed keyword-only signature: passing a
    whole event content payload as ``**kwargs`` (which also carries
    ``timestamp``/``payload``) is a ``TypeError``.
    """
    return bytes.fromhex(
        hash_payload({"event_type": event_type, "source_message_provider_id": source_message_provider_id})
    )


def event_anchor_free_identity_hash(content_payload: Mapping[str, JSONValue]) -> bytes:
    """Identity for an event whose anchoring message is itself remeasured.

    Same content payload as the ordinary identity, minus
    ``source_message_provider_id``. ChatGPT anchors a ``generation_lifecycle``
    event to a different message id in different exports of one unchanged
    conversation (polylogue-uqwd), so the anchor is provider measurement on
    that shape, not content -- exactly the distinction
    ``_event_content_payload``'s allowlist already draws for the event's own
    duration and timestamp.

    Deliberately NOT the event's stored identity: dropping the anchor there
    would merge genuinely distinct events that differ only by which message
    they hang off, and would be a stored-identity change requiring a reparse.
    This value exists purely so the revision-membership comparison can pair a
    moved event with itself.
    """
    return bytes.fromhex(
        hash_payload({key: value for key, value in content_payload.items() if key != "source_message_provider_id"})
    )


def event_canonical_identity_hash(*, base_identity: bytes, content_hash: bytes) -> bytes:
    """Fold an event's base identity with its own content hash.

    Used only when a base identity (event type + anchoring message) may be
    shared by more than one event within one revision (e.g. multiple
    ``chatgpt_block_metadata`` events on the same message, one per block) --
    still content-derived, never the array index (polylogue-aggz).
    """
    return bytes.fromhex(hash_payload({"base_identity": base_identity.hex(), "content": content_hash.hex()}))


def _session_hash_payload(
    *,
    title: str | None,
    created_at: str | None,
    updated_at: str | None,
    messages: list[dict[str, JSONValue]],
    attachments: list[dict[str, JSONValue]],
    session_events: list[dict[str, JSONValue]],
) -> dict[str, object]:
    """Build the content-hash payload using the complete declared tree."""
    return {
        "title": _normalize_for_hash(title),
        "created_at": _normalize_for_hash(created_at),
        "updated_at": _normalize_for_hash(updated_at),
        "messages": messages,
        "session_events": session_events,
        "attachments": sorted(
            attachments,
            key=lambda item: (
                str(item.get("message_id") or ""),
                str(item.get("id") or ""),
                str(item.get("name") or ""),
            ),
        ),
    }


def _session_hash_components(
    convo: ParsedSession,
    *,
    tolerate_ambiguous_attachment_owners: bool = False,
) -> tuple[list[dict[str, JSONValue]], list[dict[str, JSONValue]], list[dict[str, JSONValue]]]:
    """Build the hash-stable message/attachment/event payloads once.

    ``session_content_hash`` and ``session_revision_projection`` both need
    these payloads (the former to hash the whole tree, the latter to also
    hash each item individually); building them once and sharing halves the
    payload-construction and nested tool_input hashing volume versus each
    caller re-deriving its own copy. Byte-identical to computing each
    payload independently -- pure sharing of an already-pure computation.
    """
    owner_resolution = message_owner_resolution(convo.messages)
    # Private owner keys may use duplicate-occurrence evidence. Revision
    # identity must remain the intrinsic role/timestamp axis for timestamped
    # id-less messages, independent of the sibling count in this acquisition.
    message_comparison_ids = [_message_revision_match_id(message) for message in convo.messages]
    messages_payload = [
        _message_hash_payload(message, comparison_id)
        for message, comparison_id in zip(convo.messages, message_comparison_ids, strict=True)
    ]
    attachments_payload: list[dict[str, JSONValue]] = []
    for attachment in convo.attachments:
        try:
            owner_anchor = attachment_message_owner_key(attachment, owner_resolution)
        except MessageOwnerAmbiguityError:
            if not tolerate_ambiguous_attachment_owners:
                raise
            owner_anchor = None
        attachments_payload.append(_attachment_hash_payload(attachment, message_owner_anchor=owner_anchor))
    session_events_payload = [
        {
            "event_index": event_index,
            "event_type": _normalize_for_hash(event.event_type),
            "timestamp": _normalize_for_hash(event.timestamp),
            "source_message_provider_id": _normalize_for_hash(event.source_message_provider_id),
            "payload": hash_payload(_normalize_nested_for_hash(event.payload)),
        }
        for event_index, event in enumerate(convo.session_events)
    ]
    return messages_payload, attachments_payload, session_events_payload


def _session_tree_hash(
    convo: ParsedSession,
    *,
    messages_payload: list[dict[str, JSONValue]],
    attachments_payload: list[dict[str, JSONValue]],
    session_events_payload: list[dict[str, JSONValue]],
) -> str:
    session_fields = _model_hash_payload(
        convo, _HASHED_FIELDS["ParsedSession"] - {"messages", "attachments", "session_events"}
    )
    return hash_payload(
        _session_hash_payload(
            title=convo.title,
            created_at=convo.created_at,
            updated_at=convo.updated_at,
            messages=messages_payload,
            attachments=attachments_payload,
            session_events=session_events_payload,
        )
        | {"semantic_session_fields": session_fields}
    )


def _stream_session_tree_hash(convo: ParsedSession) -> str:
    """Encode each prepared item through the QUERY codec without a tree list."""
    if QUERY.encoder != "stdlib" or QUERY.normalize_unicode or not QUERY.ensure_ascii:
        raise AssertionError("session hash streaming requires the declared QUERY codec")
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=QUERY.ensure_ascii)
    digest = hashlib.sha256()

    def write(value: object) -> None:
        for chunk in encoder.iterencode(value):
            digest.update(chunk.encode("utf-8"))

    def literal(value: str) -> None:
        digest.update(value.encode("ascii"))

    # The top-level keys are in the same sorted order as hash_payload's
    # JSONEncoder. Item encoding is delegated to that encoder as well.
    literal('{"attachments":[')
    if convo.attachments:
        attachments_payload: list[dict[str, JSONValue]] = []
        with disk_message_owner_resolution(convo.messages) as resolution:
            for attachment in convo.attachments:
                try:
                    owner_anchor = attachment_message_owner_key(attachment, resolution)
                except MessageOwnerAmbiguityError:
                    owner_anchor = None
                attachments_payload.append(_attachment_hash_payload(attachment, message_owner_anchor=owner_anchor))
        attachments_payload.sort(
            key=lambda item: (
                str(item.get("message_id") or ""),
                str(item.get("id") or ""),
                str(item.get("name") or ""),
            )
        )
        for index, payload in enumerate(attachments_payload):
            if index:
                literal(",")
            write(payload)
    literal('],"created_at":')
    write(_normalize_for_hash(convo.created_at))
    literal(',"messages":[')
    for index, message in enumerate(convo.messages):
        if index:
            literal(",")
        write(_message_hash_payload(message, _message_revision_match_id(message)))
    literal('],"semantic_session_fields":')
    write(_model_hash_payload(convo, _HASHED_FIELDS["ParsedSession"] - {"messages", "attachments", "session_events"}))
    literal(',"session_events":[')
    for event_index, event in enumerate(convo.session_events):
        if event_index:
            literal(",")
        write(
            {
                "event_index": event_index,
                "event_type": _normalize_for_hash(event.event_type),
                "timestamp": _normalize_for_hash(event.timestamp),
                "source_message_provider_id": _normalize_for_hash(event.source_message_provider_id),
                "payload": hash_payload(_normalize_nested_for_hash(event.payload)),
            }
        )
    literal('],"title":')
    write(_normalize_for_hash(convo.title))
    literal(',"updated_at":')
    write(_normalize_for_hash(convo.updated_at))
    literal("}")
    return digest.hexdigest()


def session_content_hash(convo: ParsedSession) -> ContentHash:
    """Generate semantic content identity from the declared field partition.

    Fields excluded from identity have an independent owner documented in
    ``_EXCLUDED_FIELDS``; parsed fields may not silently fall through.
    """
    validate_semantic_hash_partition()
    if hasattr(convo.messages, "path") or hasattr(convo.session_events, "path"):
        return ContentHash(_stream_session_tree_hash(convo))
    messages_payload, attachments_payload, session_events_payload = _session_hash_components(
        convo, tolerate_ambiguous_attachment_owners=True
    )
    return ContentHash(
        _session_tree_hash(
            convo,
            messages_payload=messages_payload,
            attachments_payload=attachments_payload,
            session_events_payload=session_events_payload,
        )
    )


def _disk_session_revision_projection(convo: ParsedSession) -> SessionRevisionProjection:
    """Project a prepared session without resident per-item collections."""
    bound_hash = bound_session_content_hash(convo)
    session_hash_hex = bound_hash if bound_hash is not None else session_content_hash(convo)
    parent = getattr(convo.messages, "path", None)
    store = _DiskRevisionStore(Path(parent).parent if parent is not None else None)
    conn = store.conn
    try:
        message_count = 0
        for message_count, message in enumerate(convo.messages, start=1):
            payload = _message_hash_payload(message, _message_revision_match_id(message))
            native_id = payload["id"]
            assert isinstance(native_id, str)
            identity = message_identity_hash(id=native_id)
            content = bytes.fromhex(hash_payload(payload))
            conn.execute("INSERT INTO message_hash VALUES (?, ?)", (message_count - 1, content))
            conn.execute(
                "INSERT INTO message_content VALUES (?, ?, 1) ON CONFLICT(identity, content) "
                "DO UPDATE SET multiplicity = multiplicity + 1",
                (identity, content),
            )
            if not message.provider_message_id.strip() and message.timestamp is not None:
                conn.execute("INSERT OR IGNORE INTO mutable_message VALUES (?)", (identity,))

        if convo.attachments:
            with disk_message_owner_resolution(convo.messages) as resolution:
                for attachment in convo.attachments:
                    owner_anchor = attachment_message_owner_key(attachment, resolution)
                    payload = _attachment_hash_payload(attachment, message_owner_anchor=owner_anchor)
                    identity = attachment_identity_hash(
                        message_id=payload["message_id"], name=payload["name"], mime_type=payload["mime_type"]
                    )
                    conn.execute("INSERT OR IGNORE INTO attachment_identity VALUES (?)", (identity,))
                    inline_hash = payload.get("inline_content_hash")
                    if isinstance(inline_hash, str):
                        conn.execute(
                            "INSERT OR IGNORE INTO attachment_content VALUES (?, ?)",
                            (identity, bytes.fromhex(inline_hash)),
                        )

        event_count = 0
        for event_count, event in enumerate(convo.session_events, start=1):
            payload = {
                "event_index": event_count - 1,
                "event_type": _normalize_for_hash(event.event_type),
                "timestamp": _normalize_for_hash(event.timestamp),
                "source_message_provider_id": _normalize_for_hash(event.source_message_provider_id),
                "payload": hash_payload(_normalize_nested_for_hash(event.payload)),
            }
            conn.execute(
                "INSERT INTO event_hash VALUES (?, ?)", (event_count - 1, bytes.fromhex(hash_payload(payload)))
            )
            content_payload = _event_content_payload(event)
            base_identity = event_base_identity_hash(
                event_type=content_payload["event_type"],
                source_message_provider_id=content_payload["source_message_provider_id"],
            )
            content = bytes.fromhex(hash_payload(content_payload))
            canonical_identity = event_canonical_identity_hash(base_identity=base_identity, content_hash=content)
            conn.execute("INSERT OR IGNORE INTO event_content VALUES (?, ?)", (canonical_identity, content))
            if _anchor_is_remeasured(event):
                anchor_free = event_anchor_free_identity_hash(content_payload)
                conn.execute("INSERT OR IGNORE INTO anchor_free_event VALUES (?, ?)", (canonical_identity, anchor_free))
        return SessionRevisionProjection(
            session_hash=bytes.fromhex(session_hash_hex),
            message_hashes=_DiskRevisionHashes(store, "message_hash", message_count),
            message_contents=_DiskRevisionSet[MessageContent](store, "message_content"),
            attachment_identities=_DiskRevisionSet[bytes](store, "attachment_identity"),
            attachment_contents=_DiskRevisionSet[tuple[bytes, bytes]](store, "attachment_content"),
            event_hashes=_DiskRevisionHashes(store, "event_hash", event_count),
            event_contents=_DiskRevisionSet[tuple[bytes, bytes]](store, "event_content"),
            anchor_free_event_identities=_DiskRevisionSet[tuple[bytes, bytes]](store, "anchor_free_event"),
            mutable_message_identities=_DiskRevisionSet[bytes](store, "mutable_message"),
        )
    except BaseException:
        store.close()
        raise


def session_revision_projection(convo: ParsedSession) -> SessionRevisionProjection:
    """Project canonical content hashes used to prove append-only session growth.

    Builds message/attachment/event payloads once (``_session_hash_components``)
    and reuses them for both the whole-tree ``session_hash`` and the per-item
    hashes below, instead of recomputing each payload from scratch a second
    time. Output is byte-identical to the previous double-computation --
    this is a pure elimination of redundant work, not an identity-hash
    change (polylogue-fqp0).

    ``session_hash`` still covers the full attachment payload, acquisition state
    included, so acquiring an attachment's bytes does change the session's
    content hash and does trigger a re-write. Only the *revision comparison*
    axes separate identity from acquisition (polylogue-bu1i).

    The same holds for message order (polylogue-c429), attachment identity
    presence (polylogue-d8al), and provider-reported generation-duration
    measurement (polylogue-nuec): ``session_hash`` still covers the full,
    order-sensitive message array, the full attachment payload including
    whatever id the provider did or didn't emit, and the full, unstripped
    event payload/timestamp, so a real reorder, a real id change, or a real
    duration change still triggers a re-write. Only ``message_contents`` /
    ``attachment_identities`` / ``attachment_contents`` / ``event_contents``
    -- the *revision comparison* axes -- are content-only (polylogue-aggz).
    """
    if hasattr(convo.messages, "path") or hasattr(convo.session_events, "path"):
        return _disk_session_revision_projection(convo)
    messages_payload, attachments_payload, session_events_payload = _session_hash_components(convo)
    bound_hash = bound_session_content_hash(convo)
    session_hash_hex = (
        bound_hash
        if bound_hash is not None
        else _session_tree_hash(
            convo,
            messages_payload=messages_payload,
            attachments_payload=attachments_payload,
            session_events_payload=session_events_payload,
        )
    )
    message_content_counts: Counter[tuple[bytes, bytes]] = Counter()
    message_hashes: list[bytes] = []
    mutable_message_identities: set[bytes] = set()
    for message, payload in zip(convo.messages, messages_payload, strict=True):
        message_native_id = payload["id"]
        assert isinstance(message_native_id, str)  # built as str above, never anything else
        identity = message_identity_hash(id=message_native_id)
        if not message.provider_message_id.strip() and message.timestamp is not None:
            mutable_message_identities.add(identity)
        content = bytes.fromhex(hash_payload(payload))
        message_content_counts[(identity, content)] += 1
        message_hashes.append(content)
    attachment_identities: set[bytes] = set()
    attachment_contents: set[tuple[bytes, bytes]] = set()
    for payload in attachments_payload:
        identity = attachment_identity_hash(
            message_id=payload["message_id"], name=payload["name"], mime_type=payload["mime_type"]
        )
        inline_content_hash = payload.get("inline_content_hash")
        attachment_identities.add(identity)
        if isinstance(inline_content_hash, str):
            attachment_contents.add((identity, bytes.fromhex(inline_content_hash)))
    event_hashes: list[bytes] = []
    event_base_identities: list[bytes] = []
    event_content_hashes: list[bytes] = []
    event_anchor_free_identities: list[bytes | None] = []
    for payload, event in zip(session_events_payload, convo.session_events, strict=True):
        event_hashes.append(bytes.fromhex(hash_payload(payload)))
        content_payload = _event_content_payload(event)
        event_base_identities.append(
            event_base_identity_hash(
                event_type=content_payload["event_type"],
                source_message_provider_id=content_payload["source_message_provider_id"],
            )
        )
        event_content_hashes.append(bytes.fromhex(hash_payload(content_payload)))
        # Narrowed to the same shape the payload allowlist targets: an event
        # type with a registered allowlist that declares itself
        # provider-remeasured. That is the ChatGPT generation_lifecycle shape
        # whose anchor was observed to move between export vintages
        # (polylogue-uqwd). Browser-capture emits the same event_type without
        # the marker, and its anchor is a real first-party observation, so it
        # is deliberately excluded and keeps strict anchor semantics.
        event_anchor_free_identities.append(
            event_anchor_free_identity_hash(content_payload) if _anchor_is_remeasured(event) else None
        )
    event_contents: set[tuple[bytes, bytes]] = set()
    anchor_free_pairs: set[tuple[bytes, bytes]] = set()
    for base_identity, content_hash, anchor_free in zip(
        event_base_identities, event_content_hashes, event_anchor_free_identities, strict=True
    ):
        # A base identity (event type + anchoring message) is ambiguous
        # whenever it is EVER possible for more than one event to share it
        # (e.g. one chatgpt_block_metadata event per block on a message), so
        # the event's own content is always folded into identity here --
        # unconditionally, not only when a sibling happens to be present in
        # THIS revision. An item's identity must not depend on what else is
        # in the set: computing it from this revision's own sibling count
        # made the same event's identity shift between `base_identity` (one
        # instance) and `hash(base_identity, content)` (two or more) purely
        # because a sibling appeared in a later revision, which made an
        # ordinary event-growth revision compare as a disjoint conflict
        # instead of containment. Folding content in always keeps identity
        # intrinsic to the event itself: still content-derived, never the
        # array index (distinct blocks already differ in content, e.g. a
        # content-intrinsic block_index), and true duplicates (same base
        # identity, same content, whether or not any sibling exists)
        # correctly collapse to one set entry either way.
        canonical_identity = event_canonical_identity_hash(base_identity=base_identity, content_hash=content_hash)
        event_contents.add((canonical_identity, content_hash))
        if anchor_free is not None:
            anchor_free_pairs.add((canonical_identity, anchor_free))
    return SessionRevisionProjection(
        session_hash=bytes.fromhex(session_hash_hex),
        message_hashes=tuple(message_hashes),
        message_contents=frozenset(
            (identity, content, multiplicity) for (identity, content), multiplicity in message_content_counts.items()
        ),
        attachment_identities=frozenset(attachment_identities),
        attachment_contents=frozenset(attachment_contents),
        event_hashes=tuple(event_hashes),
        event_contents=frozenset(event_contents),
        anchor_free_event_identities=frozenset(anchor_free_pairs),
        mutable_message_identities=frozenset(mutable_message_identities),
    )
