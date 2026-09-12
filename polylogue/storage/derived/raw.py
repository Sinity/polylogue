"""Raw-observation derivation over retained bytes and logical membership.

The adapter owns discovery and output inspection. Publication uses the existing
revision-governance replay seam, which still owns durable arbitration and its
per-logical-key transactions. This is not an observation-wide atomic publisher.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from polylogue.archive.revision_authority import (
    BYTE_AUTHORITY_CENSUS_DETAIL,
    RAW_AUTHORITY_PARSER_FINGERPRINT,
    decided_unresolved_membership_sql,
    durable_authority_logical_keys,
    parser_census_is_complete,
)
from polylogue.core.enums import Origin
from polylogue.core.raw_failure_evidence import (
    RAW_FAILURE_DEFERRED_SUPPORT_STATUS,
    RAW_FAILURE_REPLAY_AUTHORITY_EVIDENCE_KINDS,
)
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.raw_authority import parser_census_logical_keys

if TYPE_CHECKING:
    from polylogue.sources.revision_backfill import RawParsePrefetchCache

RAW_OBSERVATION_DOMAIN = "raw_observation"


class RawFrame(Protocol):
    @property
    def archive_root(self) -> str: ...

    @property
    def source_revision(self) -> str: ...

    @property
    def scope(self) -> object | None: ...

    def recipe_version(self, domain: str) -> str: ...


@dataclass(frozen=True, slots=True)
class RawObservationScope:
    source_roots: tuple[Path, ...] = ()
    raw_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RawObservationReplacement:
    key: str
    input_binding: str
    payload: RawParsePrefetchCache
    raw_ids: tuple[str, ...]
    empty: bool = False


class RawObservationDerivation:
    """A paged raw adapter; no ops hint or backlog census certifies validity.

    Preparation can run without a writer lease. Existing synchronous recovery
    callers still hold their enclosing lease; their composition must move
    before that production route can claim lease-free computation.
    """

    domain = RAW_OBSERVATION_DOMAIN
    prerequisites: tuple[str, ...] = ()
    recipe_version = RAW_AUTHORITY_PARSER_FINGERPRINT

    def __init__(self, archive_root: Path, *, max_payload_bytes: int = 64 * 1024 * 1024) -> None:
        self.archive_root = archive_root
        self.max_payload_bytes = max_payload_bytes

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        source = self.archive_root / "source.db"
        conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True, timeout=5.0)
        try:
            conn.row_factory = sqlite3.Row
            index = ArchiveLocation.resolve(self.archive_root).active_index_path
            conn.execute("ATTACH DATABASE ? AS index_tier", (f"file:{index}?mode=ro",))
            conn.execute("BEGIN")
            yield conn
        finally:
            conn.close()

    def _current(self, frame: RawFrame) -> bool:
        return (
            Path(frame.archive_root).resolve() == self.archive_root.resolve()
            and frame.source_revision == str(ArchiveLocation.resolve(self.archive_root).active_index_path.resolve())
            and frame.recipe_version(self.domain) == self.recipe_version
        )

    def required_page(self, frame: RawFrame, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], str | None]:
        if limit < 1:
            return (), None
        if not (self.archive_root / "source.db").exists():
            if (self.archive_root / "index.db").exists() or (self.archive_root / ".index-active-pointer").exists():
                raise FileNotFoundError(f"durable source tier is missing: {self.archive_root / 'source.db'}")
            return (), None
        scope = frame.scope if isinstance(frame.scope, RawObservationScope) else RawObservationScope()
        predicates = ["r.raw_id > ?"]
        parameters: list[object] = [cursor or ""]
        if scope.raw_ids:
            predicates.append(f"r.raw_id IN ({','.join('?' for _ in scope.raw_ids)})")
            parameters.extend(scope.raw_ids)
        if scope.source_roots:
            bounds = []
            for path in scope.source_roots:
                root = str(path).rstrip("/")
                bounds.append("(r.source_path = ? OR (r.source_path >= ? AND r.source_path < ?))")
                parameters.extend((root, root + "/", root + "0"))
            predicates.append("(" + " OR ".join(bounds) + ")")
        with self._read() as conn:
            rows = conn.execute(
                f"SELECT r.raw_id FROM raw_sessions r WHERE {' AND '.join(predicates)} ORDER BY r.raw_id LIMIT ?",
                (*parameters, limit),
            ).fetchall()
        keys = tuple(str(row[0]) for row in rows)
        return keys, keys[-1] if len(keys) == limit else None

    def excess_page(self, frame: RawFrame, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], None]:
        # Durable raws are retained. Excess logical identities are inspected
        # within their observation, never deleted by guessing an absent owner.
        return (), None

    def prerequisite_keys(self, frame: RawFrame, key: str) -> tuple[()]:
        return ()

    def quiet(self, frame: RawFrame, key: str) -> bool:
        return False

    def inspect(self, frame: RawFrame, keys: Sequence[str]) -> Mapping[str, str]:
        if not self._current(frame):
            return dict.fromkeys(keys, "stale")
        with self._read() as conn:
            return {key: self._inspect(conn, key) for key in keys}

    def source_paths(self, keys: Sequence[str]) -> Mapping[str, str]:
        if not keys:
            return {}
        with self._read() as conn:
            return dict(
                conn.execute(
                    f"SELECT raw_id, source_path FROM raw_sessions WHERE raw_id IN ({','.join('?' for _ in keys)})",
                    tuple(keys),
                )
            )

    def _inspect(self, conn: sqlite3.Connection, key: str) -> str:
        from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin

        raw = conn.execute(
            f"SELECT r.*, ({decided_unresolved_membership_sql('r')}) AS decided FROM raw_sessions r WHERE raw_id = ?",
            (key,),
        ).fetchone()
        if raw is None:
            return "missing"
        # These are durable refusals, not missing parser work. Preserve their
        # existing authority contract without re-arbitrating rejected bytes.
        if raw["decided"] or (
            raw["validation_status"] == "failed"
            and (
                raw["parsed_at_ms"] is None
                or raw["validated_at_ms"] is None
                or raw["validated_at_ms"] >= raw["parsed_at_ms"]
            )
        ):
            return "valid"
        error = raw["parse_error"]
        if error and not (
            error == "OperationalError: database is locked"
            or str(error).startswith("membership_replay_conflict:")
            or (str(error).startswith("decode:") and "No such file or directory:" in error)
        ):
            retry = conn.execute(
                f"""SELECT 1 FROM raw_artifacts WHERE raw_id = ? AND origin IS ?
                AND source_path IS ? AND source_index IS ? AND support_status = ?
                AND artifact_kind IN ({",".join("?" for _ in RAW_FAILURE_REPLAY_AUTHORITY_EVIDENCE_KINDS)}) LIMIT 1""",
                (
                    key,
                    raw["origin"],
                    raw["source_path"],
                    raw["source_index"],
                    RAW_FAILURE_DEFERRED_SUPPORT_STATUS,
                    *sorted(RAW_FAILURE_REPLAY_AUTHORITY_EVIDENCE_KINDS),
                ),
            ).fetchone()
            if retry is None:
                return "valid"
        census = conn.execute("SELECT * FROM raw_authority_parser_census WHERE raw_id = ?", (key,)).fetchone()
        membership = conn.execute("SELECT * FROM raw_membership_census WHERE raw_id = ?", (key,)).fetchone()
        members = conn.execute(
            "SELECT logical_source_key, decision FROM raw_session_memberships WHERE raw_id = ? ORDER BY logical_source_key",
            (key,),
        ).fetchall()
        if census is None:
            return "missing"
        if census["parser_fingerprint"] != self.recipe_version or census["status"] != "complete":
            return "stale"
        expected = durable_authority_logical_keys(
            raw_logical_key=raw["logical_source_key"],
            revision_kind=raw["revision_kind"],
            membership_logical_keys=(row[0] for row in members),
        )
        non_session = (
            conn.execute(
                "SELECT 1 FROM raw_artifacts WHERE raw_id = ? AND parse_as_session = 0 LIMIT 1",
                (key,),
            ).fetchone()
            is not None
        )
        if not parser_census_is_complete(
            recorded_keys=parser_census_logical_keys(census["logical_keys_json"]),
            durable_keys=expected,
            typed_non_session=non_session,
            parser_confirmed_non_session=membership is not None
            and membership["status"] == "non_session"
            and membership["parser_fingerprint"] == self.recipe_version,
            byte_governed_fragment=raw["source_index"] < 0
            and membership is not None
            and membership["detail"] == BYTE_AUTHORITY_CENSUS_DETAIL,
        ):
            return "stale"
        if membership is not None and membership["status"] == "complete" and membership["member_count"] != len(members):
            return "stale"
        owned = {
            str(row[0]) for row in conn.execute("SELECT session_id FROM index_tier.sessions WHERE raw_id = ?", (key,))
        }
        if owned - set(expected or ()):
            return "excess"
        if not expected:
            return "valid"
        decisions = {str(row[0]): row[1] for row in members}
        for logical_key in expected:
            if logical_key in decisions and decisions[logical_key] in {"ambiguous", "deferred"}:
                continue
            if logical_key in decisions and decisions[logical_key] is None:
                return "missing"
            application = conn.execute(
                """SELECT 1 FROM index_tier.raw_revision_applications
                WHERE raw_id = ? AND logical_source_key = ?
                  AND decision IN ('selected_baseline', 'applied_append', 'superseded')
                LIMIT 1""",
                (key, logical_key),
            ).fetchone()
            if application is None:
                return "missing"
            output = conn.execute(
                """SELECT s.origin, s.parser_fingerprint, s.lowering_fingerprint
                FROM index_tier.raw_revision_heads h JOIN index_tier.sessions s
                  ON s.session_id = h.session_id AND s.raw_id = h.accepted_raw_id
                 AND s.content_hash = h.accepted_content_hash
                WHERE h.logical_source_key = ?""",
                (logical_key,),
            ).fetchone()
            if output is None:
                return "missing"
            if (
                output["parser_fingerprint"] != parser_fingerprint_for_origin(Origin(output["origin"]))
                or output["lowering_fingerprint"] != lowering_fingerprint()
            ):
                return "stale"
        return "valid"

    def _binding(self, raw_ids: tuple[str, ...]) -> str:
        with self._read() as conn:
            parts = []
            for raw_id in raw_ids:
                for table, order in (
                    ("raw_sessions", "raw_id"),
                    ("raw_session_memberships", "logical_source_key"),
                    ("raw_membership_census", "raw_id"),
                    ("raw_authority_parser_census", "raw_id"),
                    ("raw_artifacts", "artifact_id"),
                ):
                    parts.append(
                        [
                            tuple(row)
                            for row in conn.execute(
                                f"SELECT * FROM {table} WHERE raw_id = ? ORDER BY {order}", (raw_id,)
                            )
                        ]
                    )
            return hashlib.sha256(repr(parts).encode()).hexdigest()

    def compute(self, frame: RawFrame, key: str) -> RawObservationReplacement:
        from polylogue.sources.revision_backfill import RawParsePrefetchCache, parse_retained_raw_sessions
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(self.archive_root, read_only=True) as archive:
            raw_ids, _keys = archive.expand_raw_membership_selection([key])
            binding = self._binding(raw_ids)
            sizes = archive.raw_payload_sizes(raw_ids)
            if sum(sizes.values()) > self.max_payload_bytes:
                raise ValueError("raw observation component exceeds its payload budget")
            cache = RawParsePrefetchCache(max_inflight_bytes=self.max_payload_bytes)
            empty = True
            for raw_id in raw_ids:
                _provider, blob_hash, _path, kind, size = archive.raw_revision_descriptor(raw_id)
                if not BlobStore(self.archive_root / "blob").verify(blob_hash):
                    raise ValueError(f"retained raw blob does not match its identity: {raw_id}")
                sessions = parse_retained_raw_sessions(archive, raw_id)
                empty = empty and not sessions
                if not cache.try_admit(raw_id, sessions, payload_bytes=size, revision_kind=kind):
                    raise ValueError("raw observation parse preparation exceeds its payload budget")
        return RawObservationReplacement(key, binding, cache, raw_ids, empty)

    def publish(self, frame: RawFrame, replacement: RawObservationReplacement) -> bool:
        from polylogue.sources.revision_backfill import backfill_historical_revision_evidence
        from polylogue.storage.index_generation import ActiveWriterLease
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        lease = ActiveWriterLease(self.archive_root)
        lease.acquire()
        try:
            if not self._current(frame) or self._binding(replacement.raw_ids) != replacement.input_binding:
                return False
            with ArchiveStore.open_existing(self.archive_root, read_only=True) as archive:
                raw_ids, _keys = archive.expand_raw_membership_selection([replacement.key])
                if raw_ids != replacement.raw_ids:
                    return False
                for raw_id in raw_ids:
                    _provider, blob_hash, _path, _kind, _size = archive.raw_revision_descriptor(raw_id)
                    if not BlobStore(self.archive_root / "blob").verify(blob_hash):
                        return False
            backfill_historical_revision_evidence(
                self.archive_root,
                active_index_path=Path(frame.source_revision),
                selected_raw_ids=list(replacement.raw_ids),
                max_payload_bytes=self.max_payload_bytes,
                prefetch_cache=replacement.payload,
                pipeline_decode=False,
            )
            return True
        finally:
            lease.close()
