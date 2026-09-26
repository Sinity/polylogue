"""Raw-observation derivation over retained bytes and logical membership.

The adapter owns discovery and output inspection. Publication uses the existing
revision-governance replay seam, which still owns durable arbitration and its
per-logical-key transactions. This is not an observation-wide atomic publisher.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import closing, contextmanager
from dataclasses import dataclass
from multiprocessing import get_context
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
from polylogue.pipeline.services.process_pool import terminate_process_pool
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
    from polylogue.sources.parsers.base import ParsedSession
    from polylogue.sources.prepared_jsonl import PreparedJsonl
    from polylogue.sources.revision_backfill import (
        PreparedRetainedAggregate,
        PreparedRetainedInput,
        RawParsePrefetchCache,
    )
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.revision_governance import PreparedRawRevisionClassification
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionWrite

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
    payload: RawParsePrefetchCache | None
    raw_ids: tuple[str, ...]
    prepared_inputs: Mapping[str, PreparedRetainedInput] | None = None
    prepared_aggregates: Mapping[str, PreparedRetainedAggregate] | None = None
    prepared_writes: Mapping[str, PreparedSessionWrite] | None = None
    classification_proofs: Mapping[str, PreparedRawRevisionClassification] | None = None
    verified_blob_stats: Mapping[str, tuple[int, int, int, int, int]] | None = None
    planned_accepted_raw_ids: Mapping[str, tuple[str, ...]] | None = None
    needs_source_census: bool = False
    needs_source_classification: bool = False
    scratch_directory: Path | None = None
    scratch_owner: tempfile.TemporaryDirectory[str] | None = None
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

    def __init__(self, archive_root: Path) -> None:
        self.archive_root = archive_root

    @staticmethod
    def _blob_stat_identity(path: Path) -> tuple[int, int, int, int, int]:
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

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

    @staticmethod
    def _source_replay_plans(archive: ArchiveStore, logical_keys: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
        """Capture only source-backed byte plans; census may establish others later."""
        plans: dict[str, tuple[str, ...]] = {}
        for logical_key in logical_keys:
            candidate = archive.source_connection.execute(
                "SELECT 1 FROM raw_sessions WHERE logical_source_key = ? AND source_revision IS NOT NULL LIMIT 1",
                (logical_key,),
            ).fetchone()
            if candidate is not None:
                plans[logical_key] = archive.raw_revision_replay_plan(logical_key).accepted_raw_ids
        return plans

    def compute(self, frame: RawFrame, key: str) -> RawObservationReplacement:
        from polylogue.operations.operation_context import open_operation_read
        from polylogue.sources.dispatch import is_jsonl_source_path
        from polylogue.sources.revision_backfill import (
            PreparedRetainedInput,
            RetainedPreparationRetryableError,
            prepare_retained_jsonl_artifact,
        )
        from polylogue.sources.sqlite_export import looks_like_logical_source_path

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
                None,
                (),
                already_valid=True,
            )

        with open_operation_read(self.archive_root) as pinned:
            archive = pinned.archive
            raw_ids, logical_keys = archive.expand_raw_membership_selection([key])
            binding = self._binding(raw_ids)
            descriptors = {raw_id: archive.raw_revision_descriptor(raw_id) for raw_id in raw_ids}
            process_prepared = bool(descriptors)
            if process_prepared:
                from polylogue.core.sources import origin_from_provider
                from polylogue.sources.prepared_merge import (
                    prepare_retained_cohort_artifact,
                    prepared_cohort_source_hash,
                )
                from polylogue.sources.revision_backfill import (
                    PreparedRetainedAggregate,
                    prepare_retained_non_json_artifact,
                    selected_prepared_membership_head,
                )
                from polylogue.storage.sqlite.archive_tiers.revision_governance import (
                    prepare_raw_revision_rebuild_classification,
                    prepared_raw_revision_classification_current,
                )
                from polylogue.storage.sqlite.archive_tiers.write import (
                    prepare_session_write,
                    prepared_lineage_bindings,
                    prepared_session_rows_from_shard,
                )

                census_rows = archive.source_connection.execute(
                    f"SELECT raw_id, parser_fingerprint, status FROM raw_authority_parser_census "
                    f"WHERE raw_id IN ({','.join('?' for _ in raw_ids)})",
                    raw_ids,
                ).fetchall()
                complete_census = {
                    str(row[0]) for row in census_rows if row[1] == self.recipe_version and row[2] == "complete"
                }
                needs_source_census = len(raw_ids) > 1 and len(complete_census) != len(raw_ids)
                classification_proofs: dict[str, PreparedRawRevisionClassification] = {}
                if not needs_source_census:
                    for logical_key in logical_keys:
                        has_byte_candidate = archive.source_connection.execute(
                            "SELECT 1 FROM raw_sessions WHERE logical_source_key = ? "
                            "AND source_revision IS NOT NULL LIMIT 1",
                            (logical_key,),
                        ).fetchone()
                        if has_byte_candidate is None:
                            continue
                        classification_proofs[logical_key] = prepare_raw_revision_rebuild_classification(
                            archive, logical_key
                        )
                    if any(
                        not prepared_raw_revision_classification_current(archive, proof)
                        for proof in classification_proofs.values()
                    ):
                        # The source must own the byte decision before the
                        # accepted tip can select its sealed writer carrier.
                        return RawObservationReplacement(
                            key,
                            binding,
                            None,
                            raw_ids,
                            classification_proofs=classification_proofs,
                            needs_source_classification=True,
                        )
                planned_accepted_raw_ids = self._source_replay_plans(archive, logical_keys)
                scratch_owner = tempfile.TemporaryDirectory(
                    prefix=".raw-prepared-", dir=Path(frame.source_revision).resolve().parent
                )
                scratch = Path(scratch_owner.name)
                prepared: dict[str, PreparedRetainedInput] = {}
                aggregates: dict[str, PreparedRetainedAggregate] = {}
                prepared_writes: dict[str, PreparedSessionWrite] = {}
                verified_blob_stats: dict[str, tuple[int, int, int, int, int]] = {}
                prepared_artifacts: dict[tuple[object, ...], PreparedJsonl] = {}
                try:
                    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
                        for raw_id in raw_ids:
                            provider, blob_hash, path, kind, size = descriptors[raw_id]
                            blob_store = BlobStore(self.archive_root / "blob")
                            blob_path = blob_store.blob_path(blob_hash)
                            try:
                                before = self._blob_stat_identity(blob_path)
                            except OSError as exc:
                                raise RetainedPreparationRetryableError(
                                    f"retained raw blob disappeared: {raw_id}"
                                ) from exc
                            if not blob_store.verify(blob_hash):
                                raise RetainedPreparationRetryableError(f"retained raw blob changed: {raw_id}")
                            native_id = archive.raw_native_id(raw_id) if kind.value == "append" else None
                            fallback_timestamp = archive.raw_revision_file_mtime(raw_id)
                            artifact_key = (provider, blob_hash, path, kind, native_id, fallback_timestamp)
                            artifact = prepared_artifacts.get(artifact_key)
                            if artifact is None:
                                try:
                                    is_json = (
                                        is_jsonl_source_path(path) or Path(path).suffix.lower() == ".json"
                                    ) and not looks_like_logical_source_path(blob_path)
                                    artifact = pool.submit(
                                        prepare_retained_jsonl_artifact
                                        if is_json
                                        else prepare_retained_non_json_artifact,
                                        raw_id,
                                        provider.value,
                                        blob_hash,
                                        path,
                                        kind.value,
                                        native_id,
                                        str(self.archive_root / "blob"),
                                        str(self.archive_root / "source.db"),
                                        frame.source_revision,
                                        str(scratch),
                                        fallback_timestamp,
                                    ).result(timeout=600)
                                except TimeoutError as exc:
                                    terminate_process_pool(pool)
                                    raise RetainedPreparationRetryableError(
                                        f"retained preparation timed out for raw {raw_id}"
                                    ) from exc
                                except BrokenProcessPool as exc:
                                    raise RetainedPreparationRetryableError(
                                        f"retained worker exited before preparing raw {raw_id}"
                                    ) from exc
                            if not blob_store.verify(blob_hash):
                                raise RetainedPreparationRetryableError(f"retained raw blob changed: {raw_id}")
                            try:
                                after = self._blob_stat_identity(blob_path)
                            except OSError as exc:
                                raise RetainedPreparationRetryableError(
                                    f"retained raw blob disappeared: {raw_id}"
                                ) from exc
                            if before != after:
                                raise RetainedPreparationRetryableError(f"retained raw blob changed: {raw_id}")
                            verified_blob_stats[raw_id] = after
                            if artifact.deferred:
                                raise RetainedPreparationRetryableError(
                                    f"retained preparation deferred for raw {raw_id}: {artifact.error}"
                                )
                            if artifact.error is None and artifact.blob_hash != blob_hash:
                                raise RetainedPreparationRetryableError(
                                    f"retained preparation hash changed for raw {raw_id}"
                                )
                            if artifact.error is None:
                                try:
                                    artifact.verify_files(full=artifact_key not in prepared_artifacts)
                                except (OSError, ValueError) as exc:
                                    raise RetainedPreparationRetryableError(
                                        f"retained preparation seal changed for raw {raw_id}"
                                    ) from exc
                            prepared_artifacts[artifact_key] = artifact
                            prepared[raw_id] = PreparedRetainedInput(
                                raw_id,
                                provider,
                                blob_hash,
                                path,
                                kind,
                                size,
                                native_id,
                                self.recipe_version,
                                fallback_timestamp,
                                None,
                                artifact.error,
                                prepared_artifact=artifact if artifact.error is None else None,
                            )
                        if needs_source_census and not planned_accepted_raw_ids:

                            def empty_artifact(artifact: PreparedJsonl) -> bool:
                                if artifact.error is not None:
                                    return False
                                with closing(artifact.iter_sessions()) as sessions:
                                    return next(sessions, None) is None

                            if all(empty_artifact(artifact) for artifact in prepared_artifacts.values()):
                                needs_source_census = False
                        if not needs_source_census and len(complete_census) != len(raw_ids):
                            for raw_id, retained in prepared.items():
                                retained_artifact = retained.prepared_artifact
                                if retained_artifact is None or archive.index_connection is None:
                                    continue
                                for parsed_session in retained_artifact.iter_sessions():
                                    session_id = (
                                        f"{origin_from_provider(parsed_session.source_name).value}:"
                                        f"{parsed_session.provider_session_id}"
                                    )
                                    existing = archive.index_connection.execute(
                                        "SELECT raw_id FROM sessions WHERE session_id = ?", (session_id,)
                                    ).fetchone()
                                    if existing is not None and str(existing[0]) != raw_id:
                                        needs_source_census = True
                                if needs_source_census:
                                    break
                        for logical_key, accepted_raw_ids in (
                            planned_accepted_raw_ids.items() if not needs_source_census else ()
                        ):
                            if len(accepted_raw_ids) < 2:
                                continue
                            ordered: list[tuple[str, PreparedJsonl]] = []
                            for raw_id in accepted_raw_ids:
                                ordered_input = prepared.get(raw_id)
                                retained_artifact = (
                                    ordered_input.prepared_artifact if ordered_input is not None else None
                                )
                                if retained_artifact is not None:
                                    ordered.append((raw_id, retained_artifact))
                            if len(ordered) != len(accepted_raw_ids):
                                continue
                            try:
                                aggregate = pool.submit(prepare_retained_cohort_artifact, ordered, scratch).result(
                                    timeout=600
                                )
                            except TimeoutError as exc:
                                terminate_process_pool(pool)
                                raise RetainedPreparationRetryableError(
                                    f"retained cohort preparation timed out for {logical_key}"
                                ) from exc
                            except BrokenProcessPool as exc:
                                raise RetainedPreparationRetryableError(
                                    f"retained cohort worker exited before preparing {logical_key}"
                                ) from exc
                            if aggregate.deferred or aggregate.error is not None:
                                raise RetainedPreparationRetryableError(
                                    f"retained cohort preparation deferred for {logical_key}: {aggregate.error}"
                                )
                            if aggregate.blob_hash != prepared_cohort_source_hash(ordered):
                                raise RetainedPreparationRetryableError(
                                    f"retained cohort source dependency changed for {logical_key}"
                                )
                            try:
                                aggregate.verify_files(full=True)
                            except (OSError, ValueError) as exc:
                                raise RetainedPreparationRetryableError(
                                    f"retained cohort preparation seal changed for {logical_key}"
                                ) from exc
                            aggregates[logical_key] = PreparedRetainedAggregate(accepted_raw_ids, aggregate)
                    if not needs_source_census and archive.index_connection is not None:
                        selected_writes: dict[str, tuple[ParsedSession, PreparedJsonl]] = {}
                        for logical_key, accepted_raw_ids in planned_accepted_raw_ids.items():
                            if not accepted_raw_ids:
                                continue
                            tip_raw_id = accepted_raw_ids[-1]
                            selected_artifact = (
                                aggregates[logical_key].artifact
                                if len(accepted_raw_ids) > 1 and logical_key in aggregates
                                else prepared[tip_raw_id].prepared_artifact
                            )
                            if selected_artifact is None:
                                continue
                            selected_session: ParsedSession | None = None
                            for candidate_session in selected_artifact.iter_sessions():
                                if selected_session is not None:
                                    raise RetainedPreparationRetryableError(
                                        f"retained replay plan has no single prepared session for {logical_key}"
                                    )
                                selected_session = candidate_session
                            if selected_session is None:
                                raise RetainedPreparationRetryableError(
                                    f"retained replay plan has no single prepared session for {logical_key}"
                                )
                            selected_writes[tip_raw_id] = (selected_session, selected_artifact)
                        for logical_key in logical_keys:
                            if planned_accepted_raw_ids.get(logical_key):
                                continue
                            selected = selected_prepared_membership_head(archive, logical_key, prepared)
                            if selected is None:
                                continue
                            accepted_raw_id, accepted_session = selected
                            membership_artifact = prepared[accepted_raw_id].prepared_artifact
                            if membership_artifact is not None:
                                selected_writes[accepted_raw_id] = (accepted_session, membership_artifact)
                        for tip_raw_id, (session, artifact) in selected_writes.items():
                            session_id = (
                                f"{origin_from_provider(session.source_name).value}:{session.provider_session_id}"
                            )
                            existing = archive.index_connection.execute(
                                "SELECT raw_id FROM sessions WHERE session_id = ?", (session_id,)
                            ).fetchone()
                            hook_parent_id, resolved_parent_id = prepared_lineage_bindings(
                                archive.index_connection,
                                session,
                                source_conn=archive.source_connection,
                            )
                            has_parent_claim = (
                                session.parent_session_provider_id is not None
                                or hook_parent_id is not None
                                or resolved_parent_id is not None
                            )
                            if (existing is None or str(existing[0]) == tip_raw_id) and not has_parent_claim:
                                continue
                            if artifact.shard_path is None:
                                raise RetainedPreparationRetryableError(
                                    f"retained replay shard is absent for raw {tip_raw_id}"
                                )
                            prepared_writes[tip_raw_id] = prepare_session_write(
                                archive.index_connection,
                                session,
                                merge_append=False,
                                fallback_timestamp=archive.raw_revision_file_mtime(tip_raw_id),
                                source_conn=archive.source_connection,
                                raw_id=tip_raw_id,
                                force_replace=True,
                                prepared_rows=prepared_session_rows_from_shard(artifact.shard_path, session_id),
                            )
                except BaseException:
                    for prepared_write in prepared_writes.values():
                        prepared_write.close()
                    scratch_owner.cleanup()
                    raise
                return RawObservationReplacement(
                    key,
                    binding,
                    None,
                    raw_ids,
                    prepared_inputs=prepared,
                    prepared_aggregates=aggregates,
                    prepared_writes=prepared_writes,
                    classification_proofs=classification_proofs,
                    verified_blob_stats=verified_blob_stats,
                    planned_accepted_raw_ids=planned_accepted_raw_ids,
                    needs_source_census=needs_source_census,
                    scratch_directory=scratch,
                    scratch_owner=scratch_owner,
                )
        # An empty raw selection has no parse payload; the canonical replay
        # owner remains responsible for its final authority verdict.
        return RawObservationReplacement(key, binding, None, raw_ids)

    def publish(self, frame: RawFrame, replacement: RawObservationReplacement) -> bool:
        from polylogue.sources.revision_backfill import (
            RetainedPreparationRetryableError,
            backfill_historical_revision_evidence,
            census_historical_revision_evidence,
        )
        from polylogue.storage.index_generation import ActiveWriterLease
        from polylogue.storage.raw_retention import raw_frontier_blocked_raw_ids
        from polylogue.storage.sqlite.archive_tiers.revision_governance import PreparedRawClassificationStaleError

        if replacement.already_valid:
            return self._current(frame) and self.inspect(frame, (replacement.key,)).get(replacement.key) == "valid"

        lease = ActiveWriterLease(self.archive_root)
        try:
            lease.acquire()
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
                    blob_store = BlobStore(self.archive_root / "blob")
                    if replacement.prepared_inputs is not None:
                        expected_stat = (replacement.verified_blob_stats or {}).get(raw_id)
                        try:
                            current_stat = self._blob_stat_identity(blob_store.blob_path(blob_hash))
                        except OSError:
                            return False
                        if expected_stat is None or current_stat != expected_stat:
                            return False
                    elif not replacement.needs_source_classification and not blob_store.verify(blob_hash):
                        return False
                if (
                    replacement.planned_accepted_raw_ids is not None
                    and self._source_replay_plans(archive, _keys) != replacement.planned_accepted_raw_ids
                ):
                    return False
            for prepared in (replacement.prepared_inputs or {}).values():
                if prepared.prepared_artifact is not None:
                    try:
                        prepared.prepared_artifact.verify_files(full=False)
                    except (OSError, ValueError):
                        return False
            for aggregate in (replacement.prepared_aggregates or {}).values():
                try:
                    aggregate.artifact.verify_files(full=False)
                except (OSError, ValueError):
                    return False
            if replacement.needs_source_census:
                if replacement.prepared_inputs is None:
                    return False
                census_historical_revision_evidence(
                    self.archive_root,
                    active_index_path=Path(frame.source_revision),
                    selected_raw_ids=list(replacement.raw_ids),
                    max_payload_bytes=None,
                    prepared_inputs=replacement.prepared_inputs,
                    ingest_workers=1,
                )
                return False
            if replacement.needs_source_classification:
                if replacement.classification_proofs is None:
                    return False
                try:
                    census_historical_revision_evidence(
                        self.archive_root,
                        active_index_path=Path(frame.source_revision),
                        selected_raw_ids=list(replacement.raw_ids),
                        max_payload_bytes=None,
                        classification_proofs=replacement.classification_proofs,
                        ingest_workers=1,
                    )
                except (PreparedRawClassificationStaleError, RetainedPreparationRetryableError):
                    return False
                return False
            try:
                backfill_historical_revision_evidence(
                    self.archive_root,
                    active_index_path=Path(frame.source_revision),
                    selected_raw_ids=list(replacement.raw_ids),
                    max_payload_bytes=None,
                    prefetch_cache=replacement.payload,
                    prepared_inputs=replacement.prepared_inputs,
                    prepared_aggregates=replacement.prepared_aggregates,
                    prepared_writes=replacement.prepared_writes,
                    prepared_replay_plans=replacement.planned_accepted_raw_ids,
                    pipeline_decode=False,
                    use_session_shards=replacement.prepared_inputs is not None,
                )
            except RetainedPreparationRetryableError:
                return False
            return True
        finally:
            try:
                lease.close()
            finally:
                try:
                    for prepared_write in (replacement.prepared_writes or {}).values():
                        prepared_write.close()
                finally:
                    if replacement.scratch_owner is not None:
                        replacement.scratch_owner.cleanup()
