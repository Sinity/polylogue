"""Raw-observation derivation over retained bytes and logical membership.

The adapter owns discovery and output inspection. Publication uses the existing
revision-governance replay seam, which still owns durable arbitration and its
per-logical-key transactions. This is not an observation-wide atomic publisher.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from polylogue.archive.revision_authority import (
    RAW_AUTHORITY_PARSER_FINGERPRINT,
    RawRevisionAuthority,
    durable_authority_logical_keys,
    parser_census_is_complete,
)
from polylogue.core.enums import Origin
from polylogue.core.raw_failure_evidence import (
    RAW_FAILURE_DEFERRED_SUPPORT_STATUS,
    RAW_FAILURE_REPLAY_AUTHORITY_EVIDENCE_KINDS,
    RAW_FAILURE_TERMINAL_EVIDENCE_SUPPORT_STATUS_PAIRS,
)
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.raw_authority import (
    SUPERSEDED_MEMBERSHIP_FINGERPRINTS,
    build_raw_replay_plan,
    parser_census_logical_keys,
    raw_replay_application_receipt_from_connection,
    validate_raw_replay_application_receipt,
)
from polylogue.storage.sqlite.queries.raw_state import raw_provider_origin_sql

if TYPE_CHECKING:
    from polylogue.sources.revision_backfill import RawParsePrefetchCache

RAW_OBSERVATION_DOMAIN = "raw_observation"


def raw_replay_error_is_retryable(error: object, durable_retryable: bool = False) -> bool:
    """Recognize durable retry evidence and exact historical replay refusals."""
    if durable_retryable:
        return True
    if not isinstance(error, str):
        return False
    return (
        error == "OperationalError: database is locked"
        or error.startswith(("MembershipReplayConflictError:", "membership_replay_conflict:"))
        or (error.startswith("decode:") and "No such file or directory" in error)
        or error
        in {
            "RuntimeError: raw revision CAS rejected an older accepted frontier",
            "RuntimeError: membership replay cannot replace an unconvertible byte head",
        }
    )


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
    already_valid: bool = False


class RawObservationDerivation:
    """A paged raw adapter; no ops hint or backlog census certifies validity.

    Preparation can run without a writer lease. Existing synchronous recovery
    callers still hold their enclosing lease; their composition must move
    before that production route can claim lease-free computation.
    """

    domain = RAW_OBSERVATION_DOMAIN
    prerequisites: tuple[str, ...] = ()
    recipe_version = RAW_AUTHORITY_PARSER_FINGERPRINT

    def __init__(
        self, archive_root: Path, *, max_payload_bytes: int = 64 * 1024 * 1024, stream_safe_only: bool = False
    ) -> None:
        self.archive_root = archive_root
        self.max_payload_bytes = max_payload_bytes
        self.stream_safe_only = stream_safe_only

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

    def _require_bootstrapped_source_tier(self, conn: sqlite3.Connection) -> None:
        """Refuse a present-but-unbootstrapped source tier with a typed reason.

        ``source.db`` existing as a file is not proof that the durable source
        tier was ever bootstrapped: a bare ``sqlite3.connect(...)`` creates the
        file with no schema at all. Every caller already handles
        ``FileNotFoundError`` as "this backlog is unavailable, and here is why",
        so an absent ``raw_sessions`` table reports through that same channel
        rather than escaping as a bare ``OperationalError``.
        """
        if (
            conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'raw_sessions'").fetchone()
            is None
        ):
            raise FileNotFoundError(
                f"durable source tier is not bootstrapped: {self.archive_root / 'source.db'} has no raw_sessions table"
            )

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
        if scope.source_roots and not scope.raw_ids:
            return self._source_scope_page(scope.source_roots, cursor=cursor, limit=limit)
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
            self._require_bootstrapped_source_tier(conn)
            rows = conn.execute(
                f"SELECT r.raw_id FROM raw_sessions r WHERE {' AND '.join(predicates)} ORDER BY r.raw_id LIMIT ?",
                (*parameters, limit),
            ).fetchall()
        keys = tuple(str(row[0]) for row in rows)
        return keys, keys[-1] if len(keys) == limit else None

    def _source_scope_page(
        self, roots: tuple[Path, ...], *, cursor: str | None, limit: int
    ) -> tuple[tuple[str, ...], str | None]:
        """Seek the existing source-path index; never sort an excluded archive.

        Each root has an exact-path interval and a descendant interval. The
        disposable cursor tracks that interval and the index's natural key,
        including rowid for observations sharing a path/index coordinate.
        Both returned rows and empty interval probes have one call-wide bound.
        """
        ordered = tuple(sorted({str(root).rstrip("/") for root in roots}))
        position = 0
        after: tuple[str, int, int] | None = None
        if cursor is not None:
            position, serialized_after = json.loads(cursor)
            if serialized_after is not None:
                after = (str(serialized_after[0]), int(serialized_after[1]), int(serialized_after[2]))
        keys: list[str] = []
        probes = 0
        with self._read() as conn:
            self._require_bootstrapped_source_tier(conn)
            while position < 2 * len(ordered) and len(keys) < limit and probes < max(2, limit):
                root = ordered[position // 2]
                if position % 2:
                    predicate = "source_path >= ? AND source_path < ?"
                    parameters: list[object] = [root + "/", root + "0"]
                else:
                    predicate = "source_path = ?"
                    parameters = [root]
                if after is not None:
                    predicate += " AND (source_path, source_index, rowid) > (?, ?, ?)"
                    parameters.extend(after)
                remaining = limit - len(keys)
                rows = conn.execute(
                    f"SELECT raw_id, source_path, source_index, rowid FROM raw_sessions "
                    f"INDEXED BY idx_raw_sessions_source_path WHERE {predicate} "
                    "ORDER BY source_path, source_index, rowid LIMIT ?",
                    (*parameters, remaining),
                ).fetchall()
                probes += 1
                keys.extend(str(row[0]) for row in rows)
                if len(rows) == remaining:
                    last = rows[-1]
                    after = (str(last[1]), int(last[2]), int(last[3]))
                    break
                position += 1
                after = None
        continuation = json.dumps((position, after)) if position < 2 * len(ordered) else None
        return tuple(keys), continuation

    def excess_page(self, frame: RawFrame, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], None]:
        # Durable raws are retained. Excess logical identities are inspected
        # within their observation, never deleted by guessing an absent owner.
        return (), None

    def prerequisite_keys(self, frame: RawFrame, key: str) -> tuple[()]:
        return ()

    def quiet(self, frame: RawFrame, key: str) -> bool:
        return False

    def inspect(self, frame: RawFrame, keys: Sequence[str]) -> Mapping[str, str]:
        if not keys:
            # Nothing to inspect: opening the read connection here would demand
            # an existing source.db purely to answer the empty question, which
            # is how a probe of an archive with no source tier used to die on
            # "unable to open database file". ``source_paths`` already guards
            # the same way.
            return {}
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

    @staticmethod
    def _terminal_revision_refusal(conn: sqlite3.Connection, key: str, parser_fingerprint: str | None) -> bool:
        classifier_superseded = parser_fingerprint in SUPERSEDED_MEMBERSHIP_FINGERPRINTS
        unresolved = conn.execute(
            """SELECT decision FROM raw_session_memberships WHERE raw_id = ? AND decision IN ('ambiguous', 'deferred')
            UNION ALL SELECT decision FROM index_tier.raw_revision_applications
            WHERE raw_id = ? AND decision IN ('ambiguous', 'deferred')""",
            (key, key),
        ).fetchall()
        return any(row[0] == "deferred" or not classifier_superseded for row in unresolved)

    def _inspect(self, conn: sqlite3.Connection, key: str) -> str:
        from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin

        raw = conn.execute(
            f"SELECT r.*, {raw_provider_origin_sql(table_alias='r')} AS effective_origin FROM raw_sessions r WHERE raw_id = ?",
            (key,),
        ).fetchone()
        if raw is None:
            return "missing"
        # These are durable refusals, not missing parser work. Preserve their
        # existing authority contract without re-arbitrating rejected bytes.
        census = conn.execute("SELECT * FROM raw_authority_parser_census WHERE raw_id = ?", (key,)).fetchone()
        if self._terminal_revision_refusal(conn, key, census["parser_fingerprint"] if census else None) or (
            raw["validation_status"] == "failed"
            and (
                raw["parsed_at_ms"] is None
                or raw["validated_at_ms"] is None
                or raw["validated_at_ms"] >= raw["parsed_at_ms"]
            )
        ):
            return "valid"
        error = raw["parse_error"]
        coordinates = (key, raw["origin"], raw["effective_origin"], raw["source_path"], raw["source_index"])
        exact_coordinate = "raw_id = ? AND (origin IS ? OR origin IS ?) AND source_path IS ? AND source_index IS ?"
        if (
            error
            and conn.execute(
                f"SELECT 1 FROM raw_artifacts WHERE {exact_coordinate} "
                f"AND (artifact_kind, support_status) IN ({','.join('(?, ?)' for _ in RAW_FAILURE_TERMINAL_EVIDENCE_SUPPORT_STATUS_PAIRS)}) LIMIT 1",
                (
                    *coordinates,
                    *(value for pair in RAW_FAILURE_TERMINAL_EVIDENCE_SUPPORT_STATUS_PAIRS for value in pair),
                ),
            ).fetchone()
            is not None
        ):
            return "valid"
        if error and not raw_replay_error_is_retryable(error):
            retry = conn.execute(
                f"""SELECT 1 FROM raw_artifacts WHERE {exact_coordinate} AND support_status = ?
                AND artifact_kind IN ({",".join("?" for _ in RAW_FAILURE_REPLAY_AUTHORITY_EVIDENCE_KINDS)}) LIMIT 1""",
                (
                    *coordinates,
                    RAW_FAILURE_DEFERRED_SUPPORT_STATUS,
                    *sorted(RAW_FAILURE_REPLAY_AUTHORITY_EVIDENCE_KINDS),
                ),
            ).fetchone()
            if retry is None:
                return "valid"
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
            and membership["revision_authority"] == RawRevisionAuthority.BYTE_PROVEN.value,
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
                  AND decision IN ('selected_baseline', 'applied_append', 'superseded', 'reparse_reaffirmation')
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
        from polylogue.storage.sqlite.archive_tiers.revision_governance import expand_raw_membership_selection_sync

        component, _logical_keys = expand_raw_membership_selection_sync(conn, [key])
        # Refused siblings have their own terminal authority; they did not
        # execute and cannot supply an execution receipt for accepted siblings.
        # A superseded classifier must still leave that sibling in repair debt.
        execution_component = tuple(
            str(row["raw_id"])
            for row in conn.execute(
                f"""SELECT r.raw_id, c.parser_fingerprint FROM raw_sessions r
                LEFT JOIN raw_authority_parser_census c ON c.raw_id = r.raw_id
                WHERE r.raw_id IN ({",".join("?" for _ in component)}) ORDER BY r.raw_id""",
                component,
            ).fetchall()
            if not self._terminal_revision_refusal(conn, str(row["raw_id"]), row["parser_fingerprint"])
        )
        plan = build_raw_replay_plan(conn, execution_component)
        receipt = raw_replay_application_receipt_from_connection(
            conn, plan, index_db_path=ArchiveLocation.resolve(self.archive_root).active_index_path
        )
        exact, _problems = validate_raw_replay_application_receipt(plan, receipt)
        return "valid" if exact else "stale"

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
        from polylogue.operations.operation_context import open_operation_read
        from polylogue.sources.dispatch import is_stream_record_provider
        from polylogue.sources.revision_backfill import RawParsePrefetchCache, parse_retained_raw_sessions

        # One component replay settles every member. The kernel classified the
        # page before it began publishing, so a sibling can still arrive here
        # with that old ``stale`` verdict after an earlier member has made the
        # shared authoritative output current. Re-inspect at the compute
        # boundary before parsing retained bytes again. The publication method
        # repeats this check, so a later race remains pending rather than being
        # certified from this observation alone.
        if self.inspect(frame, (key,)).get(key) == "valid":
            return RawObservationReplacement(
                key,
                "",
                RawParsePrefetchCache(max_inflight_bytes=self.max_payload_bytes),
                (),
                already_valid=True,
            )

        with open_operation_read(self.archive_root) as pinned:
            archive = pinned.archive
            raw_ids, _keys = archive.expand_raw_membership_selection([key])
            binding = self._binding(raw_ids)
            sizes = archive.raw_payload_sizes(raw_ids)
            if sum(sizes.values()) > self.max_payload_bytes:
                raise ValueError("raw observation component exceeds its payload budget")
            if self.stream_safe_only:
                for raw_id in raw_ids:
                    provider, _blob_hash, source_path, _kind, _size = archive.raw_revision_descriptor(raw_id)
                    if not is_stream_record_provider(source_path, str(provider)):
                        raise ValueError("oversized raw observation component is not entirely stream-safe")
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
        from polylogue.storage.raw_retention import raw_frontier_blocked_raw_ids

        if replacement.already_valid:
            return self._current(frame) and self.inspect(frame, (replacement.key,)).get(replacement.key) == "valid"

        lease = ActiveWriterLease(self.archive_root)
        lease.acquire()
        try:
            if not self._current(frame) or self._binding(replacement.raw_ids) != replacement.input_binding:
                return False
            refusal = raw_frontier_blocked_raw_ids(self.archive_root, replacement.raw_ids)
            selected_paths = set(self.source_paths(replacement.raw_ids).values())
            if refusal.unattributed_reason is not None or selected_paths.intersection(refusal.source_paths):
                return False
            from polylogue.operations.operation_context import open_operation_read

            with open_operation_read(self.archive_root) as pinned:
                archive = pinned.archive
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
