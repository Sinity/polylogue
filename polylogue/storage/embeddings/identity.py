"""Canonical source, recipe, and output identities for archive embeddings."""

from __future__ import annotations

import hashlib
import sqlite3
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
EMBEDDING_SOURCE_CANONICALIZATION = "ordered-message-id-content-hash-v1"
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
EMBEDDING_MESSAGE_KEY_SQL_FUNCTION = "polylogue_embedding_message_key"

_SOURCE_HASH_DOMAIN = b"polylogue.embedding-source.v1\x00"


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


class EmbeddingSourceDigest:
    """Incremental canonical digest of an ordered embeddable-message set."""

    __slots__ = ("_count", "_hasher")

    def __init__(self) -> None:
        self._hasher = hashlib.sha256()
        self._hasher.update(_SOURCE_HASH_DOMAIN)
        self._count = 0

    def update(self, message_id: str, content_hash: bytes) -> None:
        encoded_id = message_id.encode("utf-8", errors="surrogatepass")
        self._hasher.update(len(encoded_id).to_bytes(8, "big"))
        self._hasher.update(encoded_id)
        self._hasher.update(len(content_hash).to_bytes(8, "big"))
        self._hasher.update(content_hash)
        self._count += 1

    def digest(self) -> bytes:
        clone = self._hasher.copy()
        clone.update(self._count.to_bytes(8, "big"))
        return clone.digest()


class _EmbeddingSourceHashAggregate:
    """Order-independent SQLite adapter for the canonical ordered source set."""

    def __init__(self) -> None:
        self._messages: list[tuple[str, bytes]] = []

    def step(self, message_id: object, content_hash: object) -> None:
        if message_id is None or content_hash is None:
            return
        self._messages.append((str(message_id), _identity_bytes(content_hash)))

    def finalize(self) -> bytes:
        digest = EmbeddingSourceDigest()
        for message_id, content_hash in sorted(self._messages):
            digest.update(message_id, content_hash)
        return digest.digest()


def register_embedding_identity_sql(conn: sqlite3.Connection) -> None:
    """Install deterministic identity helpers used by the shared stale predicate."""

    # typeshed's _AggregateProtocol.step is narrowly typed for the single-int-arg
    # case; sqlite3 itself accepts any step arity matching n_arg (2 here).
    conn.create_aggregate(EMBEDDING_SOURCE_HASH_SQL_FUNCTION, 2, _EmbeddingSourceHashAggregate)  # type: ignore[arg-type]
    conn.create_function(
        EMBEDDING_DERIVATION_KEY_SQL_FUNCTION,
        4,
        _embedding_derivation_digest_sql,
        deterministic=True,
    )
    conn.create_function(
        EMBEDDING_MESSAGE_KEY_SQL_FUNCTION,
        4,
        _message_embedding_derivation_digest_sql,
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


def _message_embedding_derivation_digest_sql(
    message_id: object, content_hash: object, recipe_hash: object, output_contract_hash: object
) -> bytes | None:
    if message_id is None or content_hash is None or recipe_hash is None or output_contract_hash is None:
        return None
    return message_embedding_derivation_digest_from_hashes(
        message_id=str(message_id),
        content_hash=_identity_bytes(content_hash),
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
    content_hash: bytes,
    recipe_hash: bytes,
    output_contract_hash: bytes,
) -> bytes:
    return compose_derivation_key_digest(
        subject_digest=DerivationSubject(reference=message_id, grain=EMBEDDING_MESSAGE_GRAIN).digest(),
        source_identity_digest=DerivationIdentity.from_mapping(
            "polylogue.embedding.message-source.v1",
            {
                "canonicalization": EMBEDDING_SOURCE_CANONICALIZATION,
                "content_sha256": content_hash,
            },
        ).digest(),
        recipe_identity_digest=recipe_hash,
        output_contract_digest=output_contract_hash,
    )


def message_embedding_derivation_key(*, message_id: str, content_hash: bytes, recipe: EmbeddingRecipe) -> DerivationKey:
    return DerivationKey(
        subject=DerivationSubject(reference=message_id, grain=EMBEDDING_MESSAGE_GRAIN),
        source_identity=DerivationIdentity.from_mapping(
            "polylogue.embedding.message-source.v1",
            {
                "canonicalization": EMBEDDING_SOURCE_CANONICALIZATION,
                "content_sha256": content_hash,
            },
        ),
        recipe_identity=recipe.identity(),
        output_contract=recipe.output_contract(),
    )


__all__ = [
    "EMBEDDING_CHUNKING_VERSION",
    "EMBEDDING_DERIVATION_KEY_SQL_FUNCTION",
    "EMBEDDING_INPUT_TYPE",
    "EMBEDDING_MESSAGE_KEY_SQL_FUNCTION",
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
    "embedding_source_identity",
    "message_embedding_derivation_digest_from_hashes",
    "message_embedding_derivation_key",
    "register_embedding_identity_sql",
]
