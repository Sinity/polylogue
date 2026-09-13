"""Authoritative message-vector derivation for the embeddings tier.

The independently replaceable embedding partition is one current message
reference.  ``message_embedding_refs`` records that membership and
``message_embeddings_meta`` proves the exact provider recipe and output
contract behind the referenced vector.  The content-addressed vector table is
shared storage, never a freshness ledger.

Session attempt/status rows deliberately do not participate in inspection.
They retain provider-attempt telemetry, but a stale attempt receipt cannot
make a vector current (or hide a missing vector).  A key is bound to its
message id, exact provider request hash, complete recipe, output contract, and
the active embeddings generation observed before computation.  Publication
checks all of those facts again under the writer lease before replacing just
that message's reference and, if necessary, inserting its vector.
"""

from __future__ import annotations

import contextlib
import hashlib
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.embeddings.generations import (
    EmbeddingGenerationBinding,
    EmbeddingGenerationError,
    EmbeddingGenerationStore,
)
from polylogue.storage.embeddings.identity import (
    EmbeddingRecipe,
    EmbeddingRequestSpec,
    message_embedding_derivation_key,
)
from polylogue.storage.embeddings.materialization import (
    EmbeddingWriteAdmission,
    _should_embed_archive_message,
    archive_embeddable_message_where,
    archive_embeddable_messages_relation,
    inline_embedding_admission,
    message_prose_sql,
)
from polylogue.storage.introspection import table_exists
from polylogue.storage.sqlite.archive_tiers.embedding_write import (
    ArchiveEmbeddingWrite,
    replace_message_embedding_derivation,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_isolated_write_connection, open_readonly_connection
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec
from polylogue.storage.sqlite.write_lease import require_write_lease


@runtime_checkable
class EmbeddingTextProvider(Protocol):
    """The provider capability the message derivation needs."""

    model: str
    dimension: int

    def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class EmbeddingMessageInput:
    """One immutable provider request reserved before lease-free computation."""

    message_id: str
    session_id: str
    origin: str
    message_content_hash: bytes
    text: str
    request: EmbeddingRequestSpec
    binding: EmbeddingGenerationBinding | None
    index_generation: str

    @property
    def input_binding(self) -> str:
        """Complete message semantic and recipe identity for this replacement."""

        message_key = message_embedding_derivation_key(
            message_id=self.message_id,
            vector_derivation_hash=self.request.vector_derivation_hash,
            recipe=self.request.recipe,
        ).digest()
        digest = hashlib.sha256()
        digest.update(b"polylogue.embedding.message-binding.v1\x00")
        digest.update(message_key)
        digest.update(self.message_content_hash)
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class EmbeddingMessageReplacement:
    """A computed message vector, or an empty replacement for a vanished key."""

    key: str
    input_binding: str
    payload: EmbeddingMessageInput | None
    vector: list[float] | None = None
    empty: bool = False


_REQUIRED_KEY_PREFIX = "message:"
_EXCESS_KEY_PREFIX = "orphan:"


def _message_id(key: str) -> tuple[str, bool]:
    """Decode a required or excess key without conflating their end states."""

    if key.startswith(_REQUIRED_KEY_PREFIX):
        return key.removeprefix(_REQUIRED_KEY_PREFIX), False
    if key.startswith(_EXCESS_KEY_PREFIX):
        return key.removeprefix(_EXCESS_KEY_PREFIX), True
    raise ValueError(f"invalid embedding derivation key: {key!r}")


def _initialize_embeddings(embeddings_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database

    if not embeddings_path.exists():
        initialize_archive_database(embeddings_path, ArchiveTier.EMBEDDINGS)


def _message_input(
    conn: sqlite3.Connection,
    message_id: str,
    recipe: EmbeddingRecipe,
    binding: EmbeddingGenerationBinding | None,
    index_generation: str,
) -> EmbeddingMessageInput | None:
    """Read precisely the text that is handed to the provider for one key."""

    if not table_exists(conn, "messages") or not table_exists(conn, "sessions"):
        return None
    message_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(messages)")}
    if "content_hash" not in message_columns:
        return None
    if table_exists(conn, "blocks"):
        prose = message_prose_sql("m", separator="char(10)||char(10)", block_types=("text",))
    else:
        prose = "m.text" if "text" in message_columns else "NULL"
    row = conn.execute(
        f"""
        SELECT m.message_id, m.session_id, s.origin, m.content_hash, m.role, m.material_origin,
               m.message_type, {prose} AS text
        FROM messages AS m
        JOIN sessions AS s ON s.session_id = m.session_id
        WHERE m.message_id = ? AND {archive_embeddable_message_where("m")}
        """,
        (message_id,),
    ).fetchone()
    if row is None:
        return None
    if row[3] is None or len(bytes(row[3])) != 32:
        return None
    text = None if row[7] is None else str(row[7])
    if not _should_embed_archive_message(str(row[5]), str(row[6]), str(row[4]), text):
        return None
    return EmbeddingMessageInput(
        message_id=str(row[0]),
        session_id=str(row[1]),
        origin=str(row[2]),
        message_content_hash=bytes(row[3]),
        text=str(text),
        request=EmbeddingRequestSpec(recipe=recipe, input_text=str(text)),
        binding=binding,
        index_generation=index_generation,
    )


def reserve_embedding_message(
    index_db_path: Path,
    embeddings_path: Path,
    message_id: str,
    recipe: EmbeddingRecipe,
    index_generation: str,
) -> EmbeddingMessageInput | None:
    """Capture an active generation and exact provider input under short admission.

    This is the reservation phase.  It may initialize the replaceable tier,
    but it never writes a vector, reference, or session completion projection.
    The lifecycle lock ends before the provider call.
    """

    require_write_lease("embedding message reservation", archive_root=embeddings_path.parent)
    _initialize_embeddings(embeddings_path)
    store = EmbeddingGenerationStore(embeddings_path.parent, active_path=embeddings_path)
    with (
        store.writer_lock() as binding,
        open_readonly_connection(
            index_db_path,
            timeout_class="background-read",
            validate_schema=False,
        ) as conn,
    ):
        return _message_input(conn, message_id, recipe, binding, index_generation)


def _current_input(
    index_db_path: Path,
    message_id: str,
    recipe: EmbeddingRecipe,
    binding: EmbeddingGenerationBinding | None,
    index_generation: str,
) -> EmbeddingMessageInput | None:
    with open_readonly_connection(index_db_path, timeout_class="background-read", validate_schema=False) as conn:
        return _message_input(conn, message_id, recipe, binding, index_generation)


class EmbeddingDerivationAdapter:
    """Paged common-kernel adapter for authoritative message-vector membership.

    The key is ``message_id``.  Required keys are exactly embeddable messages;
    an archive with no embeddable prose is therefore valid-empty without a
    synthetic completion row.  Excess keys are orphaned refs and publication
    deletes only that ref.  Vectors remain reusable by address.
    """

    domain = "embeddings"
    prerequisites: tuple[str, ...] = ()

    def __init__(
        self,
        index_db_path: Path,
        provider: EmbeddingTextProvider,
        *,
        embeddings_path: Path | None = None,
        archive_root: Path | None = None,
        reserve: EmbeddingWriteAdmission | None = None,
        quiet: Callable[[object, str], bool] | None = None,
    ) -> None:
        self._index_db_path = index_db_path
        self._embeddings_path = embeddings_path or index_db_path.with_name("embeddings.db")
        self._archive_root = archive_root or index_db_path.parent
        self._provider = provider
        self._recipe = EmbeddingRecipe.current(model=str(provider.model), dimensions=int(provider.dimension))
        self._reserve = reserve or inline_embedding_admission
        self._quiet = quiet

    @property
    def recipe_version(self) -> str:
        return f"{self._recipe.recipe_hash.hex()}:{self._recipe.output_contract_hash.hex()}"

    def _active_index_path(self) -> Path:
        return resolve_active_index_path(self._archive_root).resolve()

    def _assert_frame(self, frame: object) -> Path:
        index_path = self._active_index_path()
        if getattr(frame, "source_revision", None) != f"index-generation:{index_path}":
            raise RuntimeError("embedding frame names a retired index generation")
        version = getattr(frame, "recipe_version", None)
        if not callable(version) or version(self.domain) != self.recipe_version:
            raise RuntimeError("embedding frame recipe identity changed")
        return index_path

    @staticmethod
    def _scope(frame: object) -> tuple[str, ...] | None:
        value = getattr(frame, "scope", None)
        if value is None:
            return None
        if not isinstance(value, tuple):
            raise TypeError("embedding frame scope must be a tuple of session ids or None")
        return tuple(dict.fromkeys(str(item) for item in value))

    def required_page(self, frame: object, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], str | None]:
        """Keyset-page every message whose current provider input is non-empty."""

        index_path = self._assert_frame(frame)
        scope = self._scope(frame)
        if scope == ():
            return (), None
        with open_readonly_connection(index_path, timeout_class="background-read", validate_schema=False) as conn:
            if not table_exists(conn, "messages"):
                return (), None
            relation = archive_embeddable_messages_relation(conn, alias="desired", model=self._recipe.model)
            params: list[object] = []
            predicates: list[str] = []
            if cursor is not None:
                predicates.append("desired.message_id > ?")
                params.append(cursor)
            if scope:
                predicates.append(f"desired.session_id IN ({', '.join('?' for _ in scope)})")
                params.extend(scope)
            predicate = f"WHERE {' AND '.join(predicates)}" if predicates else ""
            rows = conn.execute(
                f"""
                SELECT desired.message_id FROM {relation}
                {predicate}
                ORDER BY desired.message_id
                LIMIT ?
                """,
                (*params, limit + 1),
            ).fetchall()
        keys = tuple(f"{_REQUIRED_KEY_PREFIX}{row[0]}" for row in rows[:limit])
        return keys, (str(rows[limit - 1][0]) if len(rows) > limit and keys else None)

    def excess_page(self, frame: object, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], str | None]:
        """Page ref keys no longer named by the required message relation."""

        index_path = self._assert_frame(frame)
        scope = self._scope(frame)
        if scope == () or not self._embeddings_path.exists():
            return (), None
        with open_readonly_connection(index_path, timeout_class="background-read", validate_schema=False) as conn:
            if not table_exists(conn, "messages"):
                return (), None
            conn.execute("ATTACH DATABASE ? AS embeddings", (str(self._embeddings_path),))
            if not table_exists(conn, "message_embedding_refs", schema="embeddings"):
                return (), None
            relation = archive_embeddable_messages_relation(conn, alias="desired", model=self._recipe.model)
            params: list[object] = []
            predicates: list[str] = []
            if cursor is not None:
                predicates.append("refs.message_id > ?")
                params.append(cursor)
            if scope:
                predicates.append(f"refs.session_id IN ({', '.join('?' for _ in scope)})")
                params.extend(scope)
            predicate = f"AND {' AND '.join(predicates)}" if predicates else ""
            rows = conn.execute(
                f"""
                SELECT refs.message_id
                FROM embeddings.message_embedding_refs AS refs
                LEFT JOIN {relation} ON desired.message_id = refs.message_id
                WHERE desired.message_id IS NULL {predicate}
                ORDER BY refs.message_id
                LIMIT ?
                """,
                (*params, limit + 1),
            ).fetchall()
        keys = tuple(f"{_EXCESS_KEY_PREFIX}{row[0]}" for row in rows[:limit])
        return keys, (str(rows[limit - 1][0]) if len(rows) > limit and keys else None)

    def inspect(self, frame: object, keys: Sequence[str]) -> Mapping[str, str]:
        """Classify current refs/meta from their authoritative membership relation."""

        index_path = self._assert_frame(frame)
        wanted = tuple(dict.fromkeys(keys))
        if not wanted:
            return {}
        statuses: dict[str, str] = {}
        with open_readonly_connection(index_path, timeout_class="background-read", validate_schema=False) as index:
            embeddings_attached = self._embeddings_path.exists()
            if embeddings_attached:
                index.execute("ATTACH DATABASE ? AS embeddings", (str(self._embeddings_path),))
            has_refs = embeddings_attached and table_exists(index, "message_embedding_refs", schema="embeddings")
            has_meta = embeddings_attached and table_exists(index, "message_embeddings_meta", schema="embeddings")
            has_vectors = embeddings_attached and table_exists(index, "message_embeddings", schema="embeddings")
            if has_vectors:
                loaded, error = try_load_sqlite_vec(index)
                if not loaded:
                    raise RuntimeError(f"embedding vector inspection unavailable: {error}")
            for key in wanted:
                message_id, excess = _message_id(key)
                # Inspection needs no generation token: the current source and
                # recipe define the desired relation; publish captures a token.
                current = _message_input(
                    index,
                    message_id,
                    self._recipe,
                    None,
                    f"index-generation:{index_path}",
                )
                if current is None:
                    if (
                        has_refs
                        and index.execute(
                            "SELECT 1 FROM embeddings.message_embedding_refs WHERE message_id = ?", (message_id,)
                        ).fetchone()
                    ):
                        statuses[key] = "excess"
                    else:
                        statuses[key] = "missing" if excess else "valid"
                    continue
                if not has_refs or not has_meta or not has_vectors:
                    statuses[key] = "missing"
                    continue
                row = index.execute(
                    """
                    SELECT refs.session_id, refs.origin, refs.vector_derivation_hash,
                           meta.recipe_hash, meta.output_contract_hash, meta.model, meta.dimension,
                           refs.message_content_hash,
                           EXISTS(
                             SELECT 1 FROM embeddings.message_embeddings AS vectors
                             WHERE vectors.vector_derivation_hash = lower(hex(refs.vector_derivation_hash))
                           ) AS vector_present
                    FROM embeddings.message_embedding_refs AS refs
                    LEFT JOIN embeddings.message_embeddings_meta AS meta
                      ON meta.vector_derivation_hash = refs.vector_derivation_hash
                    WHERE refs.message_id = ?
                    """,
                    (message_id,),
                ).fetchone()
                if row is None:
                    statuses[key] = "missing"
                    continue
                if row[3] is None:
                    statuses[key] = "stale"
                    continue
                request = current.request
                statuses[key] = (
                    "valid"
                    if (
                        str(row[0]) == current.session_id
                        and str(row[1]) == current.origin
                        and bytes(row[2]) == request.vector_derivation_hash
                        and bytes(row[3]) == request.recipe.recipe_hash
                        and bytes(row[4]) == request.recipe.output_contract_hash
                        and str(row[5]) == request.recipe.model
                        and int(row[6]) == request.recipe.dimensions
                        and int(row[8]) == 1
                        and row[7] is not None
                        and bytes(row[7]) == current.message_content_hash
                    )
                    else "stale"
                )
        return statuses

    def prerequisite_keys(self, frame: object, key: str) -> tuple[()]:
        del frame, key
        return ()

    def quiet(self, frame: object, key: str) -> bool:
        return bool(self._quiet and self._quiet(frame, key))

    def compute(self, frame: object, key: str) -> EmbeddingMessageReplacement:
        """Reserve the input briefly, then call the provider outside every lease."""

        index_path = self._assert_frame(frame)
        message_id, excess = _message_id(key)
        if excess:
            # Discovery is advisory; publication rechecks the source before
            # deleting this ref.  Retirement has no provider input to reserve.
            return EmbeddingMessageReplacement(key=key, input_binding="", payload=None, empty=True)
        reserved = self._reserve(
            "embedding.reserve",
            lambda: reserve_embedding_message(
                index_path,
                self._embeddings_path,
                message_id,
                self._recipe,
                f"index-generation:{index_path}",
            ),
        )
        if reserved is None:
            return EmbeddingMessageReplacement(key=key, input_binding="", payload=None, empty=True)
        vectors = self._provider._get_embeddings([reserved.text], input_type=reserved.request.recipe.input_type)
        if len(vectors) != 1:
            raise RuntimeError("embedding provider returned a mismatched vector count")
        return EmbeddingMessageReplacement(
            key=key,
            input_binding=reserved.input_binding,
            payload=reserved,
            vector=vectors[0],
        )

    def publish(self, frame: object, replacement: EmbeddingMessageReplacement) -> bool:
        """Revalidate source/recipe/generation and atomically replace one ref."""

        try:
            index_path = self._assert_frame(frame)
        except RuntimeError:
            return False
        require_write_lease("embedding message publication", archive_root=self._embeddings_path.parent)
        if replacement.payload is None:
            return self._retire_vanished_ref(replacement)
        if replacement.vector is None:
            raise ValueError("embedding replacement has no vector")
        expected = replacement.payload
        if expected.binding is None:
            raise ValueError("computed embedding replacement lacks a generation binding")
        store = EmbeddingGenerationStore(self._embeddings_path.parent, active_path=self._embeddings_path)
        try:
            with store.writer_lock() as binding:
                store.assert_binding(expected.binding)
                if expected.index_generation != f"index-generation:{index_path}":
                    return False
                current = _current_input(
                    index_path,
                    expected.message_id,
                    self._recipe,
                    binding,
                    f"index-generation:{index_path}",
                )
                if current is None or current.input_binding != replacement.input_binding:
                    return False
                conn = open_isolated_write_connection(
                    Path(binding.database_path),
                    purpose="embedding message publication",
                    timeout=30.0,
                    archive_root=binding.archive_root,
                )
                try:
                    now_ms = int(datetime.now(UTC).timestamp() * 1000)
                    replace_message_embedding_derivation(
                        conn,
                        ArchiveEmbeddingWrite(
                            message_id=current.message_id,
                            session_id=current.session_id,
                            origin=current.origin,
                            message_content_hash=current.message_content_hash,
                            embedding=replacement.vector,
                            model=current.request.recipe.model,
                            embedded_at_ms=now_ms,
                            vector_derivation_hash=current.request.vector_derivation_hash,
                            recipe_hash=current.request.recipe.recipe_hash,
                            output_contract_hash=current.request.recipe.output_contract_hash,
                            derivation_key=bytes.fromhex(current.input_binding),
                        ),
                    )
                finally:
                    with contextlib.suppress(sqlite3.Error):
                        conn.close()
                store.refresh_binding_contract(binding)
                return True
        except EmbeddingGenerationError:
            return False

    def _retire_vanished_ref(self, replacement: EmbeddingMessageReplacement) -> bool:
        """Delete one orphan ref only after source inspection proves it excess."""

        message_id, excess = _message_id(replacement.key)
        if not excess:
            # The source vanished after required discovery.  No output is
            # published; the next required page omits it.
            return False
        store = EmbeddingGenerationStore(self._embeddings_path.parent, active_path=self._embeddings_path)
        try:
            with store.writer_lock() as binding:
                index_path = self._active_index_path()
                if (
                    _current_input(
                        index_path,
                        message_id,
                        self._recipe,
                        binding,
                        f"index-generation:{index_path}",
                    )
                    is not None
                ):
                    return False
                conn = open_isolated_write_connection(
                    Path(binding.database_path),
                    purpose="embedding orphan-ref retirement",
                    timeout=30.0,
                    archive_root=binding.archive_root,
                )
                try:
                    with conn:
                        conn.execute("DELETE FROM message_embedding_refs WHERE message_id = ?", (message_id,))
                finally:
                    with contextlib.suppress(sqlite3.Error):
                        conn.close()
                store.refresh_binding_contract(binding)
                return True
        except EmbeddingGenerationError:
            return False


__all__ = [
    "EmbeddingDerivationAdapter",
    "EmbeddingMessageInput",
    "EmbeddingMessageReplacement",
    "EmbeddingTextProvider",
    "reserve_embedding_message",
]
