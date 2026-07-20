"""Canonical source, recipe, and output identities for archive embeddings."""

from __future__ import annotations

import hashlib
import sqlite3
import unicodedata
from dataclasses import dataclass

from polylogue.storage.derivation_identity import (
    DerivationIdentity,
    DerivationKey,
    DerivationSubject,
    compose_derivation_key_digest,
)
from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION

EMBEDDING_SUBJECT_GRAIN = "archive-message-vectors"
EMBEDDING_MESSAGE_GRAIN = "archive-message-vector"
# v2 (polylogue-q88p): the ordered aggregate now carries embedding_input_hash
# (identity-free: H(model, embedder input text)) instead of messages.content_hash
# (identity-contaminated: includes session_id/position/variant_index). Bumping
# this string changes recipe_hash, which deliberately busts every stale
# embedding_derivation_state row so the session-attempt ledger is
# re-evaluated -- while the vectors THEMSELVES stay valid and are reused for
# free wherever the underlying text is unchanged (hash presence is the only
# freshness signal at vector granularity).
EMBEDDING_SOURCE_CANONICALIZATION = "ordered-message-id-input-hash-v2"
EMBEDDING_TEXT_CANONICALIZATION = "ordered-text-block-prose-v1"
EMBEDDING_RECORD_SELECTOR = "authored-user-assistant-prose-v1"
EMBEDDING_CHUNKING_VERSION = "one-vector-per-message-v1"
EMBEDDING_PROVIDER = "voyage"
EMBEDDING_MODEL_REVISION = "provider-model-name"
EMBEDDING_TASK = "retrieval"
EMBEDDING_INPUT_TYPE = "document"
EMBEDDING_NORMALIZATION = "provider-default"
EMBEDDING_TOOL_IMPLEMENTATION = "polylogue.sqlite-vec-v1"
EMBEDDING_OUTPUT_SCHEMA_VERSION = 1
EMBEDDING_SOURCE_HASH_SQL_FUNCTION = "polylogue_embedding_source_hash"
EMBEDDING_DERIVATION_KEY_SQL_FUNCTION = "polylogue_embedding_derivation_key"
EMBEDDING_INPUT_HASH_SQL_FUNCTION = "polylogue_embedding_input_hash"

_SOURCE_HASH_DOMAIN = b"polylogue.embedding-source.v2\x00"
_INPUT_HASH_DOMAIN = b"polylogue.embedding-input.v1\x00"


@dataclass(frozen=True, slots=True)
class EmbeddingRecipe:
    """Every declared computational field that can change an embedding result."""

    canonicalization: str
    record_selector: str
    chunking_version: str
    provider: str
    model: str
    model_revision: str
    dimensions: int
    task: str
    input_type: str
    normalization: str
    tool_implementation: str
    input_schema_version: str

    @classmethod
    def current(cls, *, model: str, dimensions: int) -> EmbeddingRecipe:
        return cls(
            canonicalization=EMBEDDING_TEXT_CANONICALIZATION,
            record_selector=EMBEDDING_RECORD_SELECTOR,
            chunking_version=EMBEDDING_CHUNKING_VERSION,
            provider=EMBEDDING_PROVIDER,
            model=model,
            model_revision=EMBEDDING_MODEL_REVISION,
            dimensions=dimensions,
            task=EMBEDDING_TASK,
            input_type=EMBEDDING_INPUT_TYPE,
            normalization=EMBEDDING_NORMALIZATION,
            tool_implementation=EMBEDDING_TOOL_IMPLEMENTATION,
            input_schema_version=f"archive-index-v{INDEX_SCHEMA_VERSION}",
        )

    def identity(self) -> DerivationIdentity:
        return DerivationIdentity.from_mapping(
            "polylogue.embedding.recipe.v1",
            {
                "canonicalization": self.canonicalization,
                "chunking_version": self.chunking_version,
                "dimensions": self.dimensions,
                "input_schema_version": self.input_schema_version,
                "input_type": self.input_type,
                "model": self.model,
                "model_revision": self.model_revision,
                "normalization": self.normalization,
                "provider": self.provider,
                "record_selector": self.record_selector,
                "task": self.task,
                "tool_implementation": self.tool_implementation,
            },
        )

    def output_contract(self) -> DerivationIdentity:
        return DerivationIdentity.from_mapping(
            "polylogue.embedding.output.v1",
            {
                "dimensions": self.dimensions,
                "element_type": "float32",
                "kind": "dense-vector",
                "schema_version": EMBEDDING_OUTPUT_SCHEMA_VERSION,
            },
        )

    @property
    def recipe_hash(self) -> bytes:
        return self.identity().digest()

    @property
    def output_contract_hash(self) -> bytes:
        return self.output_contract().digest()


def embedding_input_hash(*, model: str, input_text: str) -> bytes:
    """Identity-free vector key: SHA-256 over exactly (model, embedder input text).

    This is a pure function of what is actually sent to the embedding
    provider -- nothing else. Session id, message id, position, variant
    index, role, material_origin, and every other identity-bearing field
    that ``messages.content_hash`` includes are excluded by construction, so
    a re-ingest, index rebuild, or lineage-normalization shift cannot
    invalidate a vector whose underlying text is unchanged (operator ruling
    2026-07-20, polylogue-q88p). The same philosophy as the svfj block
    evidence hash (``_block_content_hash``), applied to the embedding tier.

    ``input_text`` must be exactly the string passed to the embedder for
    this to hold -- callers must derive this hash from the same variable
    used to build the provider request payload, not a re-derivation, so
    hash validity equals vector validity by construction.

    NFC-normalizes the text first (matching the archive's existing
    content-hash normalization convention, ``core/hashing.py``) so
    Unicode-equivalent strings from different providers/encodings hash
    identically.
    """
    normalized = unicodedata.normalize("NFC", input_text)
    hasher = hashlib.sha256()
    hasher.update(_INPUT_HASH_DOMAIN)
    model_bytes = model.encode("utf-8", errors="surrogatepass")
    hasher.update(len(model_bytes).to_bytes(8, "big"))
    hasher.update(model_bytes)
    text_bytes = normalized.encode("utf-8", errors="surrogatepass")
    hasher.update(len(text_bytes).to_bytes(8, "big"))
    hasher.update(text_bytes)
    return hasher.digest()


def _embedding_input_hash_sql(model: object, input_text: object) -> bytes | None:
    if model is None or input_text is None:
        return None
    return embedding_input_hash(model=str(model), input_text=str(input_text))


def sql_string_literal(value: str) -> str:
    """Escape a Python string as a single-quoted SQL literal."""

    return "'" + value.replace("'", "''") + "'"


class EmbeddingSourceDigest:
    """Incremental canonical digest of an ordered embeddable-message set.

    v2 (polylogue-q88p): digests only ``embedding_input_hash`` VALUES, not
    ``(message_id, hash)`` pairs. Session-level "does this session's source
    set need a new embedding attempt" bookkeeping must stay identity-free
    too, or a message-set that is byte-identical in content but renumbered
    by a rebuild (message_id shifts, hash values unchanged) would still bust
    the session's derivation key and force a full re-embed pass through the
    provider -- defeating the point of content-addressing the vectors
    themselves. Message COUNT is still covered (duplicates are not
    collapsed; the multiset of hash values must match), which still detects
    a genuine add/remove of a message from the session.
    """

    __slots__ = ("_count", "_hasher")

    def __init__(self) -> None:
        self._hasher = hashlib.sha256()
        self._hasher.update(_SOURCE_HASH_DOMAIN)
        self._count = 0

    def update(self, input_hash: bytes) -> None:
        self._hasher.update(len(input_hash).to_bytes(8, "big"))
        self._hasher.update(input_hash)
        self._count += 1

    def digest(self) -> bytes:
        clone = self._hasher.copy()
        clone.update(self._count.to_bytes(8, "big"))
        return clone.digest()


class _EmbeddingSourceHashAggregate:
    """Order-independent SQLite adapter for the canonical source hash multiset."""

    def __init__(self) -> None:
        self._hashes: list[bytes] = []

    def step(self, input_hash: object) -> None:
        if input_hash is None:
            return
        self._hashes.append(_identity_bytes(input_hash))

    def finalize(self) -> bytes:
        digest = EmbeddingSourceDigest()
        for input_hash in sorted(self._hashes):
            digest.update(input_hash)
        return digest.digest()


def register_embedding_identity_sql(conn: sqlite3.Connection) -> None:
    """Install deterministic identity helpers used by the shared stale predicate."""

    # typeshed's _AggregateProtocol.step is narrowly typed for the single-int-arg
    # case; sqlite3 itself accepts any step arity matching n_arg (1 here).
    conn.create_aggregate(EMBEDDING_SOURCE_HASH_SQL_FUNCTION, 1, _EmbeddingSourceHashAggregate)  # type: ignore[arg-type]
    conn.create_function(
        EMBEDDING_DERIVATION_KEY_SQL_FUNCTION,
        4,
        _embedding_derivation_digest_sql,
        deterministic=True,
    )
    conn.create_function(
        EMBEDDING_INPUT_HASH_SQL_FUNCTION,
        2,
        _embedding_input_hash_sql,
        deterministic=True,
    )


def _embedding_derivation_digest_sql(
    session_id: object, source_hash: object, recipe_hash: object, output_contract_hash: object
) -> bytes | None:
    if session_id is None or source_hash is None or recipe_hash is None or output_contract_hash is None:
        return None
    return embedding_derivation_digest_from_hashes(
        session_id=str(session_id),
        source_hash=_identity_bytes(source_hash),
        recipe_hash=_identity_bytes(recipe_hash),
        output_contract_hash=_identity_bytes(output_contract_hash),
    )


def _identity_bytes(value: object) -> bytes:
    """Normalize SQLite BLOB/TEXT identity values without interpreting them."""

    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8", errors="surrogatepass")
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    raise TypeError(f"unsupported SQLite identity value type: {type(value)!r}")


def embedding_source_identity(source_hash: bytes) -> DerivationIdentity:
    return DerivationIdentity.from_mapping(
        "polylogue.embedding.source.v1",
        {
            "canonicalization": EMBEDDING_SOURCE_CANONICALIZATION,
            "message_set_sha256": source_hash,
        },
    )


def embedding_derivation_key(*, session_id: str, source_hash: bytes, recipe: EmbeddingRecipe) -> DerivationKey:
    return DerivationKey(
        subject=DerivationSubject(reference=session_id, grain=EMBEDDING_SUBJECT_GRAIN),
        source_identity=embedding_source_identity(source_hash),
        recipe_identity=recipe.identity(),
        output_contract=recipe.output_contract(),
    )


def embedding_derivation_digest_from_hashes(
    *,
    session_id: str,
    source_hash: bytes,
    recipe_hash: bytes,
    output_contract_hash: bytes,
) -> bytes:
    """Fast equivalent of :func:`embedding_derivation_key` for SQL comparisons."""

    return compose_derivation_key_digest(
        subject_digest=DerivationSubject(reference=session_id, grain=EMBEDDING_SUBJECT_GRAIN).digest(),
        source_identity_digest=embedding_source_identity(source_hash).digest(),
        recipe_identity_digest=recipe_hash,
        output_contract_digest=output_contract_hash,
    )


def message_embedding_derivation_digest_from_hashes(
    *,
    message_id: str,
    embedding_input_hash: bytes,
    recipe_hash: bytes,
    output_contract_hash: bytes,
) -> bytes:
    return compose_derivation_key_digest(
        subject_digest=DerivationSubject(reference=message_id, grain=EMBEDDING_MESSAGE_GRAIN).digest(),
        source_identity_digest=DerivationIdentity.from_mapping(
            "polylogue.embedding.message-source.v2",
            {
                "canonicalization": EMBEDDING_SOURCE_CANONICALIZATION,
                "embedding_input_sha256": embedding_input_hash,
            },
        ).digest(),
        recipe_identity_digest=recipe_hash,
        output_contract_digest=output_contract_hash,
    )


def message_embedding_derivation_key(
    *, message_id: str, embedding_input_hash: bytes, recipe: EmbeddingRecipe
) -> DerivationKey:
    return DerivationKey(
        subject=DerivationSubject(reference=message_id, grain=EMBEDDING_MESSAGE_GRAIN),
        source_identity=DerivationIdentity.from_mapping(
            "polylogue.embedding.message-source.v2",
            {
                "canonicalization": EMBEDDING_SOURCE_CANONICALIZATION,
                "embedding_input_sha256": embedding_input_hash,
            },
        ),
        recipe_identity=recipe.identity(),
        output_contract=recipe.output_contract(),
    )


__all__ = [
    "EMBEDDING_CHUNKING_VERSION",
    "EMBEDDING_DERIVATION_KEY_SQL_FUNCTION",
    "EMBEDDING_INPUT_HASH_SQL_FUNCTION",
    "EMBEDDING_INPUT_TYPE",
    "EMBEDDING_MODEL_REVISION",
    "EMBEDDING_NORMALIZATION",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_RECORD_SELECTOR",
    "EMBEDDING_SOURCE_CANONICALIZATION",
    "EMBEDDING_SOURCE_HASH_SQL_FUNCTION",
    "EMBEDDING_TASK",
    "EMBEDDING_TEXT_CANONICALIZATION",
    "EMBEDDING_TOOL_IMPLEMENTATION",
    "EmbeddingRecipe",
    "EmbeddingSourceDigest",
    "embedding_derivation_digest_from_hashes",
    "embedding_derivation_key",
    "embedding_input_hash",
    "embedding_source_identity",
    "message_embedding_derivation_digest_from_hashes",
    "message_embedding_derivation_key",
    "register_embedding_identity_sql",
    "sql_string_literal",
]
