"""Raw-observation derivation over retained bytes and logical membership.

The adapter owns discovery and output inspection. Publication uses the existing
revision-governance replay seam, which still owns durable arbitration and its
per-logical-key transactions. This is not an observation-wide atomic publisher.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import tempfile
import weakref
import zipfile
from builtins import BaseExceptionGroup
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

from polylogue.archive.revision_authority import (
    RawRevisionAuthority,
    durable_authority_logical_keys,
    parser_census_is_complete,
    raw_authority_parser_fingerprint,
)
from polylogue.core.compute import (
    DaemonBackpressureError,
    DaemonOperationCancelled,
    SubmittedOperation,
    compute_adapter,
)
from polylogue.core.compute_cancel import compute_cancel_requested
from polylogue.core.content_identity import ContentIdentityRefusal
from polylogue.core.enums import Origin, Provider
from polylogue.core.raw_failure_evidence import (
    RAW_FAILURE_DEFERRED_SUPPORT_STATUS,
    RAW_FAILURE_REPLAY_AUTHORITY_EVIDENCE_KINDS,
    RAW_FAILURE_TERMINAL_EVIDENCE_SUPPORT_STATUS_PAIRS,
    RAW_FAILURE_VALIDATION_FAILURE_KINDS,
    RetainedRawDecodeRefusalError,
    raw_failure_outcome_code,
    validated_raw_failure_evidence_kind,
)
from polylogue.core.sql_settlement import retain_native_sql_lifetimes
from polylogue.logging import WARNING, emit
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.blob_store import BlobStore, BlobVerificationCancelledError, PreparedBlob
from polylogue.storage.raw_authority import (
    build_raw_replay_plan,
    parser_census_logical_keys,
    raw_replay_application_receipt_from_connection,
    validate_raw_replay_application_receipt,
)
from polylogue.storage.source_blob_restoration import (
    read_prior_full_source_receipts,
    read_raw_source_evidence,
    retained_blob_source_candidates,
    retained_source_location,
    stage_exact_blob,
    stage_exact_source_window_blob,
)
from polylogue.storage.sqlite.archive_tiers.source_write import PENDING_RAW_LOGICAL_SOURCE_PREFIX
from polylogue.storage.sqlite.connection_profile import attach_readonly_database, open_readonly_connection
from polylogue.storage.sqlite.queries.raw_state import raw_provider_origin_sql

if TYPE_CHECKING:
    from polylogue.sources.parsers.base import ParsedSession
    from polylogue.sources.prepared_jsonl import PreparedJsonl
    from polylogue.sources.revision_backfill import (
        PreparedMembershipReplay,
        PreparedRetainedAggregate,
        PreparedRetainedInput,
    )
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.revision_governance import PreparedRawRevisionClassification
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionWrite

RAW_OBSERVATION_DOMAIN = "raw_observation"

#: Origins whose enrichment reads session-scoped retained evidence. Both
#: origins map to exactly one provider, so the reverse lookup is not a guess.
_EVIDENCE_PROVIDER_BY_ORIGIN: Mapping[str, Provider] = {
    Origin.CLAUDE_CODE_SESSION.value: Provider.CLAUDE_CODE,
    Origin.CODEX_SESSION.value: Provider.CODEX,
}


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


def _discard_staged_blobs(store: BlobStore, prepared: tuple[PreparedBlob, ...]) -> None:
    for item in prepared:
        store.discard_prepared(item)


class StagedBlobRestorations:
    """Exact source bytes staged for absent retained blobs, awaiting the writer.

    Compute stages them without a lease; ``publish`` reserves and publishes
    them under the archive writer. A replacement the kernel never hands to
    ``publish`` still has its staged files discarded when this owner is
    collected.
    """

    def __init__(self, store: BlobStore, staged: Sequence[tuple[str, PreparedBlob]]) -> None:
        self.store = store
        self.staged = tuple(staged)
        self._finalizer = weakref.finalize(
            self, _discard_staged_blobs, store, tuple(prepared for _raw_id, prepared in self.staged)
        )

    def published(self) -> None:
        """The writer moved the staged files into place; nothing is left to discard."""
        self._finalizer.detach()

    def discard(self) -> None:
        self._finalizer()


@dataclass(frozen=True, slots=True)
class RawObservationReplacement:
    key: str
    input_binding: str
    # ``ReplacementLike`` requires a payload; this adapter carries its inputs
    # in the typed fields below instead.
    payload: None
    raw_ids: tuple[str, ...]
    prepared_inputs: Mapping[str, PreparedRetainedInput] | None = None
    prepared_aggregates: Mapping[str, PreparedRetainedAggregate] | None = None
    prepared_writes: Mapping[tuple[str, str], PreparedSessionWrite] | None = None
    classification_proofs: Mapping[str, PreparedRawRevisionClassification] | None = None
    verified_blob_stats: Mapping[str, tuple[int, int, int, int, int]] | None = None
    planned_accepted_raw_ids: Mapping[str, tuple[str, ...]] | None = None
    prepared_membership_plans: Mapping[str, PreparedMembershipReplay] | None = None
    needs_source_census: bool = False
    needs_source_classification: bool = False
    scratch_directory: Path | None = None
    scratch_owner: tempfile.TemporaryDirectory[str] | None = None
    empty: bool = False
    already_valid: bool = False
    blob_restorations: StagedBlobRestorations | None = None

    def close(self) -> None:
        """Drain this prepared carrier when publication is cancelled or ends."""
        with retain_native_sql_lifetimes(*(() if self.scratch_owner is None else (self.scratch_owner,))):
            _close_prepared_carriers(self.prepared_writes or {}, self.prepared_membership_plans or {})
            if self.scratch_owner is not None:
                _cleanup_scratch(self.scratch_owner)
            if self.blob_restorations is not None:
                self.blob_restorations.discard()


def _session_id(session: ParsedSession) -> str:
    from polylogue.core.sources import origin_from_provider

    return f"{origin_from_provider(session.source_name).value}:{session.provider_session_id}"


def _holds_several_sessions(artifact: PreparedJsonl | None) -> bool:
    if artifact is None:
        return False
    with closing(artifact.iter_sessions()) as sessions:
        return next(sessions, None) is not None and next(sessions, None) is not None


class RawObservationDerivation:
    """A paged raw adapter; no ops hint or backlog census certifies validity.

    Preparation can run without a writer lease. Existing synchronous recovery
    callers still hold their enclosing lease; their composition must move
    before that production route can claim lease-free computation.
    """

    domain = RAW_OBSERVATION_DOMAIN
    prerequisites: tuple[str, ...] = ()

    @property
    def recipe_version(self) -> str:
        return raw_authority_parser_fingerprint()

    def __init__(
        self,
        archive_root: Path,
        *,
        prepare_non_json_artifact: Callable[..., PreparedJsonl] | None = None,
        index_db_path: Path | None = None,
    ) -> None:
        self.archive_root = archive_root
        self._prepare_non_json_artifact = prepare_non_json_artifact
        self._index_db_path = index_db_path

    @staticmethod
    def _blob_stat_identity(path: Path) -> tuple[int, int, int, int, int]:
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        source = self.archive_root / "source.db"
        conn = open_readonly_connection(source, timeout_class="background-read", validate_schema=False)
        try:
            conn.row_factory = sqlite3.Row
            index = self._index_db_path or ArchiveLocation.resolve(self.archive_root).active_index_path
            attach_readonly_database(conn, index, alias="index_tier")
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
            and frame.source_revision
            == str((self._index_db_path or ArchiveLocation.resolve(self.archive_root).active_index_path).resolve())
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
        classifier_superseded = (
            parser_fingerprint is not None and parser_fingerprint != raw_authority_parser_fingerprint()
        )
        unresolved = conn.execute(
            """SELECT decision FROM raw_session_memberships WHERE raw_id = ? AND decision IN ('ambiguous', 'deferred')
            UNION ALL SELECT decision FROM index_tier.raw_revision_applications
            WHERE raw_id = ? AND decision IN ('ambiguous', 'deferred')""",
            (key, key),
        ).fetchall()
        return bool(unresolved) and not classifier_superseded

    def _decode_refusal(self, conn: sqlite3.Connection, key: str) -> RetainedRawDecodeRefusalError | None:
        row = conn.execute(
            f"""SELECT a.artifact_kind, r.parse_error, a.support_status,
              r.validation_status, a.classification_reason FROM raw_sessions r
            JOIN raw_authority_parser_census c ON c.raw_id = r.raw_id
            JOIN raw_artifacts a ON a.raw_id = r.raw_id
              AND (a.origin IS r.origin OR a.origin IS {raw_provider_origin_sql(table_alias="r")})
              AND a.source_path IS r.source_path AND a.source_index IS r.source_index
            WHERE r.raw_id = ? AND c.parser_fingerprint = ? AND c.status = 'complete'
              AND r.parse_error IS NOT NULL
              AND a.support_status = 'decode_failed'
              AND a.artifact_kind IN ({",".join("?" for _ in RAW_FAILURE_VALIDATION_FAILURE_KINDS)})
            ORDER BY a.artifact_kind LIMIT 1""",
            (key, self.recipe_version, *sorted(RAW_FAILURE_VALIDATION_FAILURE_KINDS)),
        ).fetchone()
        if row is None:
            return None
        kind = validated_raw_failure_evidence_kind(
            row[0],
            row[2],
            validation_failed=row[3] == "failed",
            classification_reason=row[4],
            outcome_code=raw_failure_outcome_code(row[4]),
        )
        if kind is None:
            return None
        return RetainedRawDecodeRefusalError(key, kind, str(row[1]))

    def terminal_decode_refusals(self, keys: Sequence[str]) -> Mapping[str, RetainedRawDecodeRefusalError]:
        """Read the same exact current receipt used by inspection and compute."""
        if not keys:
            return {}
        with self._read() as conn:
            return {key: refusal for key in keys if (refusal := self._decode_refusal(conn, key)) is not None}

    def _inspect(self, conn: sqlite3.Connection, key: str) -> str:
        from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin

        raw = conn.execute(
            f"SELECT r.*, {raw_provider_origin_sql(table_alias='r')} AS effective_origin FROM raw_sessions r WHERE raw_id = ?",
            (key,),
        ).fetchone()
        if raw is None:
            return "missing"
        # Semantic refusals belong to the classifier that produced them. A
        # changed classifier must recensus retained bytes before accepting
        # either its ambiguity or its deferral as current evidence.
        census = conn.execute("SELECT * FROM raw_authority_parser_census WHERE raw_id = ?", (key,)).fetchone()
        if census is not None and census["parser_fingerprint"] != self.recipe_version:
            return "stale"
        if self._decode_refusal(conn, key) is not None:
            # Settled failure evidence is not a successfully derived output.
            # Compute reports the permanent refusal without parsing it again.
            return "stale"
        if (
            raw["parse_error"]
            and conn.execute(
                f"""SELECT 1 FROM raw_artifacts WHERE raw_id = ?
            AND (origin IS ? OR origin IS ?) AND source_path IS ? AND source_index IS ?
            AND artifact_kind IN ({",".join("?" for _ in RAW_FAILURE_VALIDATION_FAILURE_KINDS)}) LIMIT 1""",
                (
                    key,
                    raw["origin"],
                    raw["effective_origin"],
                    raw["source_path"],
                    raw["source_index"],
                    *sorted(RAW_FAILURE_VALIDATION_FAILURE_KINDS),
                ),
            ).fetchone()
            is not None
        ):
            # A malformed terminal carrier cannot acquire authority through a
            # generic validation marker or the older parse-error fast paths.
            return "stale"
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
                """SELECT s.origin, s.parser_fingerprint, s.lowering_fingerprint, s.session_id, s.native_id,
                       b.evidence_key, accepted.source_path AS accepted_source_path
                FROM index_tier.raw_revision_heads h JOIN index_tier.sessions s
                  ON s.session_id = h.session_id AND s.raw_id = h.accepted_raw_id
                 AND s.content_hash = h.accepted_content_hash
                LEFT JOIN index_tier.session_enrichment_bindings b ON b.session_id = s.session_id
                LEFT JOIN raw_sessions accepted ON accepted.raw_id = h.accepted_raw_id
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
            # The binding was written from the accepted head's raw; a
            # superseded sibling from another directory reads different
            # evidence and could never match it, so compare the head's own.
            evidence_path = output["accepted_source_path"] or raw["source_path"]
            if self._enrichment_evidence_moved(conn, evidence_path, output):
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

    def _enrichment_evidence_moved(self, conn: sqlite3.Connection, source_path: object, output: sqlite3.Row) -> bool:
        """Whether the evidence this output was enriched from is no longer current.

        A session index, prompt history or thread-state export can be admitted
        after the session it describes, in any order. The writer binds each
        output to the evidence it read; here the evidence the archive holds now
        is compared with that binding. An absent binding cannot certify the
        output, so it is derived again on the retained route -- ordinary
        derivation from durable evidence, whatever order the bytes arrived in.
        """
        from polylogue.sources.revision_backfill import session_enrichment_evidence_key

        provider = _EVIDENCE_PROVIDER_BY_ORIGIN.get(str(output["origin"]))
        if provider is None or not isinstance(source_path, str):
            return False
        # ``session_enrichment_evidence_key`` (and ``read_thread_titles``
        # beneath it) queries ``work_evidence_*`` unqualified, expecting
        # ``index_conn``'s own ``main`` schema to be index.db. ``conn`` here
        # has source.db as ``main`` and the index attached only as
        # ``index_tier``, so passing it as ``index_conn`` resolved those
        # tables against the wrong schema (silently empty, not an error the
        # caller sees), and Codex evidence never registered as moved. Open a
        # real index connection for this specific read.
        index_path = ArchiveLocation.resolve(self.archive_root).active_index_path
        with closing(
            open_readonly_connection(index_path, timeout_class="background-read", validate_schema=False)
        ) as index_conn:
            current = session_enrichment_evidence_key(
                provider=provider,
                source_path=source_path,
                native_id=str(output["native_id"]),
                index_conn=index_conn,
                source_conn=conn,
                blob_root=self.archive_root / "blob",
            )
        return current is not None and output["evidence_key"] != current

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
        """Capture only source-backed byte plans; census may establish others later.

        A multi-session raw's pending envelope is governed per session through
        its memberships, so it has no one-session byte plan to capture.
        """
        from polylogue.storage.sqlite.archive_tiers.revision_governance import (
            pending_raw_envelope_has_membership_authority,
        )

        plans: dict[str, tuple[str, ...]] = {}
        for logical_key in logical_keys:
            if pending_raw_envelope_has_membership_authority(archive.source_connection, logical_key):
                continue
            candidate = archive.source_connection.execute(
                "SELECT 1 FROM raw_sessions WHERE logical_source_key = ? AND source_revision IS NOT NULL LIMIT 1",
                (logical_key,),
            ).fetchone()
            if candidate is not None:
                plans[logical_key] = archive.raw_revision_replay_plan(logical_key).accepted_raw_ids
        return plans

    def compute(self, frame: RawFrame, key: str, *, replay_current: bool = False) -> RawObservationReplacement:
        from polylogue.core.prepared_file import VerificationCancelledError
        from polylogue.operations.operation_context import open_operation_read
        from polylogue.sources.dispatch import is_jsonl_source_path
        from polylogue.sources.revision_backfill import (
            PreparedRetainedInput,
            RetainedPreparationRetryableError,
            prepare_retained_jsonl_artifact,
        )
        from polylogue.sources.sqlite_export import looks_like_logical_source_path

        with self._read() as conn:
            refusal = self._decode_refusal(conn, key)
        if refusal is not None:
            raise refusal

        # One component replay settles every member. The kernel classified the
        # page before it began publishing, so a sibling can still arrive here
        # with that old ``stale`` verdict after an earlier member has made the
        # shared authoritative output current. Re-inspect at the compute
        # boundary before parsing retained bytes again. The publication method
        # repeats this check, so a later race remains pending rather than being
        # certified from this observation alone.
        if not replay_current and self.inspect(frame, (key,)).get(key) == "valid":
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
            # Classification and preparation both read retained bytes, so an
            # absent blob is restored before either runs.
            restorations = self._stage_absent_blob_restorations(archive, raw_ids, descriptors)
            if restorations is not None:
                return RawObservationReplacement(key, binding, None, raw_ids, blob_restorations=restorations)
            process_prepared = bool(descriptors)
            if process_prepared:
                from polylogue.core.sources import origin_from_provider
                from polylogue.sources.prepared_merge import (
                    prepare_retained_cohort_artifact,
                    prepared_cohort_source_hash,
                )
                from polylogue.sources.revision_backfill import (
                    PreparedRetainedAggregate,
                    prepare_membership_replay,
                )
                from polylogue.storage.sqlite.archive_tiers.revision_governance import (
                    membership_key_has_pending_envelope_member,
                    pending_raw_envelope_has_membership_authority,
                    prepare_raw_revision_rebuild_classification,
                    prepared_raw_revision_classification_current,
                )
                from polylogue.storage.sqlite.archive_tiers.write import (
                    prepare_session_write,
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
                        if has_byte_candidate is None or pending_raw_envelope_has_membership_authority(
                            archive.source_connection, logical_key
                        ):
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
                material_store = BlobStore(self.archive_root / "blob")
                material_staging = material_store._ensure_private_staging_root()
                scratch_owner = tempfile.TemporaryDirectory(prefix=".raw-prepared-", dir=material_staging)
                scratch = Path(scratch_owner.name)
                with retain_native_sql_lifetimes(scratch_owner):
                    # Whatever path removes the scratch tree (publication, an
                    # exception, or the directory's own finalizer when publication
                    # is bypassed), the decodes cached from it go with it.
                    from polylogue.sources.prepared_message_sink import discard_decoded_sessions_under

                    weakref.finalize(scratch_owner, discard_decoded_sessions_under, scratch)
                    prepared: dict[str, PreparedRetainedInput] = {}
                    aggregates: dict[str, PreparedRetainedAggregate] = {}
                    prepared_writes: dict[tuple[str, str], PreparedSessionWrite] = {}
                    membership_plans: dict[str, PreparedMembershipReplay] = {}
                    verified_blob_stats: dict[str, tuple[int, int, int, int, int]] = {}
                    prepared_artifacts: dict[tuple[object, ...], PreparedJsonl] = {}
                    try:
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
                            if not blob_store.verify(blob_hash, stop=compute_cancel_requested):
                                raise RetainedPreparationRetryableError(f"retained raw blob changed: {raw_id}")
                            native_id = archive.raw_native_id(raw_id) if kind.value == "append" else None
                            fallback_timestamp = archive.raw_revision_file_mtime(raw_id)
                            profile_identity = archive.raw_profile_identity(raw_id)
                            artifact_key = (
                                provider,
                                blob_hash,
                                path,
                                kind,
                                native_id,
                                fallback_timestamp,
                                profile_identity,
                            )
                            artifact = prepared_artifacts.get(artifact_key)
                            if artifact is None:
                                try:
                                    is_json = (
                                        is_jsonl_source_path(path) or Path(path).suffix.lower() == ".json"
                                    ) and not looks_like_logical_source_path(blob_path)
                                    worker = (
                                        prepare_retained_jsonl_artifact if is_json else self._prepare_non_json_artifact
                                    )
                                    if worker is None:
                                        raise RetainedPreparationRetryableError(
                                            "non-JSON retained preparation requires an operations worker"
                                        )
                                    submitted = compute_adapter().submit(
                                        partial(
                                            worker,
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
                                        ),
                                        admission_class="incremental-background",
                                        estimated_bytes=size,
                                    )
                                    artifact = _await_reporting_stalls(submitted, subject=f"raw {raw_id}")
                                except (DaemonBackpressureError, DaemonOperationCancelled) as exc:
                                    raise RetainedPreparationRetryableError(
                                        f"retained compute unavailable while preparing raw {raw_id}"
                                    ) from exc
                            if not blob_store.verify(blob_hash, stop=compute_cancel_requested):
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
                                    artifact.verify_files(
                                        full=artifact_key not in prepared_artifacts, stop=compute_cancel_requested
                                    )
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
                                verified_blob_stat=after,
                                parser_error=artifact.error,
                                parser_decode_failure=artifact.decode_failure,
                                missing_profile_identity=artifact.missing_profile_identity,
                                retained_zip_membership_unproved=artifact.retained_zip_membership_unproved,
                                captured_profile_key=profile_identity,
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
                                    _raise_if_compute_cancelled(f"raw {raw_id}")
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
                                aggregate = _await_reporting_stalls(
                                    compute_adapter().submit(
                                        partial(prepare_retained_cohort_artifact, ordered, scratch),
                                        admission_class="incremental-background",
                                        estimated_bytes=sum(descriptors[raw_id][4] for raw_id, _artifact in ordered),
                                    ),
                                    subject=f"cohort {logical_key}",
                                )
                            except (DaemonBackpressureError, DaemonOperationCancelled) as exc:
                                raise RetainedPreparationRetryableError(
                                    f"retained cohort compute unavailable while preparing {logical_key}"
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
                                aggregate.verify_files(full=True, stop=compute_cancel_requested)
                            except (OSError, ValueError) as exc:
                                raise RetainedPreparationRetryableError(
                                    f"retained cohort preparation seal changed for {logical_key}"
                                ) from exc
                            aggregates[logical_key] = PreparedRetainedAggregate(accepted_raw_ids, aggregate)
                        if not needs_source_census:
                            # A pending-raw envelope whose bytes hold several
                            # sessions has not yet been censused into per-session
                            # memberships; the census, not a one-session write,
                            # settles it.
                            needs_source_census = any(
                                logical_key.startswith(PENDING_RAW_LOGICAL_SOURCE_PREFIX)
                                and _holds_several_sessions(prepared[accepted_raw_ids[-1]].prepared_artifact)
                                for logical_key, accepted_raw_ids in planned_accepted_raw_ids.items()
                                if accepted_raw_ids
                            )
                        if not needs_source_census and archive.index_connection is not None:
                            # Keyed by (raw, session): one raw may carry several sessions.
                            selected_writes: dict[tuple[str, str], tuple[ParsedSession, PreparedJsonl]] = {}
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
                                    _raise_if_compute_cancelled(f"cohort {logical_key}")
                                    if selected_session is not None:
                                        raise RetainedPreparationRetryableError(
                                            f"retained replay plan has no single prepared session for {logical_key}"
                                        )
                                    selected_session = candidate_session
                                if selected_session is None:
                                    raise RetainedPreparationRetryableError(
                                        f"retained replay plan has no single prepared session for {logical_key}"
                                    )
                                selected_writes[(tip_raw_id, _session_id(selected_session))] = (
                                    selected_session,
                                    selected_artifact,
                                )
                            for logical_key in logical_keys:
                                if planned_accepted_raw_ids.get(
                                    logical_key
                                ) and not membership_key_has_pending_envelope_member(
                                    archive.source_connection, logical_key
                                ):
                                    continue
                                membership_plan = prepare_membership_replay(
                                    archive, logical_key, prepared, stop=compute_cancel_requested
                                )
                                membership_plans[logical_key] = membership_plan
                                if not membership_plan.classification.accepted_raw_ids:
                                    continue
                                accepted_raw_id = membership_plan.classification.accepted_raw_ids[-1]
                                accepted_session = membership_plan.sessions[accepted_raw_id]
                                membership_artifact = prepared[accepted_raw_id].prepared_artifact
                                if membership_artifact is not None:
                                    selected_writes[(accepted_raw_id, _session_id(accepted_session))] = (
                                        accepted_session,
                                        membership_artifact,
                                    )
                            for (tip_raw_id, session_id), (session, artifact) in selected_writes.items():
                                _raise_if_compute_cancelled(f"raw {tip_raw_id}")
                                if artifact.shard_path is None:
                                    raise RetainedPreparationRetryableError(
                                        f"retained replay shard is absent for raw {tip_raw_id}"
                                    )
                                prepared_writes[(tip_raw_id, session_id)] = prepare_session_write(
                                    archive.index_connection,
                                    session,
                                    merge_append=False,
                                    fallback_timestamp=archive.raw_revision_file_mtime(tip_raw_id),
                                    source_conn=archive.source_connection,
                                    raw_id=tip_raw_id,
                                    force_replace=True,
                                    prepared_rows=prepared_session_rows_from_shard(artifact.shard_path, session_id),
                                )
                    except (BlobVerificationCancelledError, VerificationCancelledError) as exc:
                        _close_prepared_carriers(prepared_writes, membership_plans)
                        _cleanup_scratch(scratch_owner)
                        raise RetainedPreparationRetryableError(
                            "retained preparation cancelled during verification"
                        ) from exc
                    except BaseException:
                        _close_prepared_carriers(prepared_writes, membership_plans)
                        _cleanup_scratch(scratch_owner)
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
                        prepared_membership_plans=membership_plans,
                        needs_source_census=needs_source_census,
                        scratch_directory=scratch,
                        scratch_owner=scratch_owner,
                    )
            # An empty raw selection has no parse payload; the canonical replay
            # owner remains responsible for its final authority verdict.
            return RawObservationReplacement(key, binding, None, raw_ids)

    def _stage_absent_blob_restorations(
        self,
        archive: ArchiveStore,
        raw_ids: tuple[str, ...],
        descriptors: Mapping[str, tuple[Provider, str, str, object, int]],
    ) -> StagedBlobRestorations | None:
        """Stage exact source bytes for every retained blob the component lacks.

        Preparation cannot read an absent blob, and retrying it cannot make
        the bytes reappear. When a recorded source window of the raw -- a
        direct file window or its ZIP member -- still holds bytes whose
        SHA-256 and size equal the raw's, those bytes are staged here and
        published by the writer in ``publish``; the next pass then prepares
        over present bytes. Otherwise the retryable refusal names why
        (``source_missing``, ``hash_mismatch``, ``container_member_rejected``,
        ``inexact_payload`` ...) instead of reporting a vanished file; it
        stays retryable because the source or a restored backup can still
        bring the bytes back.
        """
        from polylogue.sources.revision_backfill import RetainedPreparationRetryableError

        blob_store = BlobStore(self.archive_root / "blob")
        staged: list[tuple[str, PreparedBlob]] = []
        staged_hashes: set[str] = set()
        try:
            for raw_id in raw_ids:
                _provider, blob_hash, path, _kind, _size = descriptors[raw_id]
                if blob_hash in staged_hashes:
                    continue
                try:
                    self._blob_stat_identity(blob_store.blob_path(blob_hash))
                    continue
                except FileNotFoundError:
                    pass
                except OSError:
                    # Present but unreadable: preparation reports it as a
                    # retryable disappearance, not as lost bytes.
                    continue
                prepared, reason = self._stage_blob_from_recorded_source(
                    archive, blob_store, raw_id, blob_hash=blob_hash, source_path=path
                )
                if prepared is None:
                    raise RetainedPreparationRetryableError(
                        f"retained raw blob absent and not restorable from its source ({reason}): {raw_id}"
                    )
                staged.append((raw_id, prepared))
                staged_hashes.add(blob_hash)
        except BlobVerificationCancelledError as exc:
            _discard_staged_blobs(blob_store, tuple(prepared for _raw_id, prepared in staged))
            raise RetainedPreparationRetryableError("retained blob restoration cancelled") from exc
        except BaseException:
            _discard_staged_blobs(blob_store, tuple(prepared for _raw_id, prepared in staged))
            raise
        return StagedBlobRestorations(blob_store, staged) if staged else None

    def _stage_blob_from_recorded_source(
        self,
        archive: ArchiveStore,
        blob_store: BlobStore,
        raw_id: str,
        *,
        blob_hash: str,
        source_path: str,
    ) -> tuple[PreparedBlob | None, str | None]:
        """Stage one absent blob from the first recorded source window holding its exact bytes.

        The candidate windows come from ``retained_blob_source_candidates``,
        the owner backup recoverability reads too. A ZIP member is replayed
        through acquisition's ZIP admission (``zip_reacquired_unit``)
        and staged only when the replayed value is byte-identical to the
        blob; a structural-only match is ``inexact_payload``. Returns the
        staged blob, or ``None`` with the last candidate's refusal reason.
        """
        conn = archive.source_connection
        row = read_raw_source_evidence(conn, raw_id)
        if row is None:
            raise KeyError(raw_id)
        prior_full_observations = read_prior_full_source_receipts(conn, row)
        source_path, container_member = retained_source_location(row, self.archive_root)
        candidates = retained_blob_source_candidates(
            row,
            container_member=container_member,
            prior_full_observations=prior_full_observations,
        )
        if not candidates:
            return None, "no_source_window"
        reason: str | None = None
        for candidate in candidates:
            if candidate.window is not None:
                prepared, reason = stage_exact_source_window_blob(
                    blob_store,
                    source_path=Path(source_path),
                    window=candidate.window,
                    blob_hash=blob_hash,
                    stop=compute_cancel_requested,
                )
            else:
                from polylogue.storage.source_zip_replay import zip_reacquired_unit

                # The resolved unit streams from its member again: a preserved
                # member can be gigabytes, so its bytes are never held whole.
                unit, reason = zip_reacquired_unit(row, source_path=source_path, zip_payload_cache={})
                prepared = None
                if unit is not None and unit.open_payload is not None:
                    try:
                        with unit.open_payload() as unit_stream:
                            prepared = stage_exact_blob(
                                blob_store,
                                unit_stream,
                                blob_hash=blob_hash,
                                size_bytes=unit.size_bytes,
                                stop=compute_cancel_requested,
                            )
                    except (OSError, zipfile.BadZipFile, LookupError, ContentIdentityRefusal) as exc:
                        reason = f"error:{type(exc).__name__}"
                    else:
                        reason = None if prepared is not None else "inexact_payload"
            if prepared is not None:
                return prepared, None
        return None, reason

    def _publish_blob_restorations(self, restorations: StagedBlobRestorations) -> None:
        """Reserve and publish staged restorations, then consume their receipts."""
        from polylogue.storage.blob_publication import ArchiveBlobPublisher, consume_restored_raw_blob_receipts

        source_db = self.archive_root / "source.db"
        publisher = ArchiveBlobPublisher(source_db, restorations.store.root, store=restorations.store)
        raw_by_hash = {prepared.hash_hex: raw_id for raw_id, prepared in restorations.staged}
        for _raw_id, prepared in restorations.staged:
            publisher.queue_prepared(prepared)
        receipts = publisher.flush()
        restorations.published()
        consume_restored_raw_blob_receipts(
            source_db, tuple((receipt, raw_by_hash[receipt.blob_hash]) for receipt in receipts)
        )

    def publish(self, frame: RawFrame, replacement: RawObservationReplacement) -> bool:
        from polylogue.sources.revision_backfill import (
            RetainedPreparationRetryableError,
            apply_prepared_revision_census,
            apply_prepared_revision_replay,
        )
        from polylogue.storage.index_generation import ActiveWriterLease
        from polylogue.storage.raw_retention import raw_frontier_blocked_raw_ids
        from polylogue.storage.sqlite.archive_tiers.revision_governance import PreparedRawClassificationStaleError

        if replacement.already_valid:
            return self._current(frame) and self.inspect(frame, (replacement.key,)).get(replacement.key) == "valid"

        with retain_native_sql_lifetimes(*(() if replacement.scratch_owner is None else (replacement.scratch_owner,))):
            lease = ActiveWriterLease(self.archive_root)
            try:
                lease.acquire()
                if not self._current(frame) or self._binding(replacement.raw_ids) != replacement.input_binding:
                    return False
                refusal = raw_frontier_blocked_raw_ids(self.archive_root, replacement.raw_ids)
                selected_paths = set(self.source_paths(replacement.raw_ids).values())
                if refusal.unattributed_reason is not None or selected_paths.intersection(refusal.source_paths):
                    return False
                if replacement.blob_restorations is not None:
                    self._publish_blob_restorations(replacement.blob_restorations)
                    # The restored bytes are prepared on the next pass, which now
                    # finds them present; this publication certifies no output.
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
                        elif not replacement.needs_source_classification:
                            return False
                    if (
                        replacement.planned_accepted_raw_ids is not None
                        and self._source_replay_plans(archive, _keys) != replacement.planned_accepted_raw_ids
                    ):
                        return False
                for prepared_input in (replacement.prepared_inputs or {}).values():
                    if prepared_input.prepared_artifact is not None:
                        prepared_input.prepared_artifact.publish_blobs()
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
                    apply_prepared_revision_census(
                        self.archive_root,
                        active_index_path=Path(frame.source_revision),
                        selected_raw_ids=list(replacement.raw_ids),
                        prepared_inputs=replacement.prepared_inputs,
                    )
                    refusal = self.terminal_decode_refusals((replacement.key,)).get(replacement.key)
                    if refusal is not None:
                        raise refusal
                    return False
                if replacement.needs_source_classification:
                    if replacement.classification_proofs is None:
                        return False
                    try:
                        apply_prepared_revision_census(
                            self.archive_root,
                            active_index_path=Path(frame.source_revision),
                            selected_raw_ids=list(replacement.raw_ids),
                            classification_proofs=replacement.classification_proofs,
                        )
                    except (PreparedRawClassificationStaleError, RetainedPreparationRetryableError):
                        return False
                    return False
                if (
                    replacement.prepared_inputs is None
                    or replacement.prepared_aggregates is None
                    or replacement.prepared_writes is None
                    or replacement.planned_accepted_raw_ids is None
                    or replacement.prepared_membership_plans is None
                ):
                    return False
                try:
                    apply_prepared_revision_replay(
                        self.archive_root,
                        active_index_path=Path(frame.source_revision),
                        selected_raw_ids=list(replacement.raw_ids),
                        prepared_inputs=replacement.prepared_inputs,
                        prepared_aggregates=replacement.prepared_aggregates,
                        prepared_writes=replacement.prepared_writes,
                        prepared_replay_plans=replacement.planned_accepted_raw_ids,
                        prepared_membership_plans=replacement.prepared_membership_plans,
                        bulk_fts=True,
                        # One publication is one component of a pass: each
                        # replayed session proves its own FTS rows, and the
                        # archive-wide audit belongs to the daemon's
                        # fts_readiness_binding stage after the burst.
                        exact_fts_audit=False,
                    )
                except RetainedPreparationRetryableError:
                    return False
                refusal = self.terminal_decode_refusals((replacement.key,)).get(replacement.key)
                if refusal is not None:
                    raise refusal
                return True
            finally:
                try:
                    lease.close()
                finally:
                    replacement.close()


def _close_prepared_carriers(
    writes: Mapping[tuple[str, str], PreparedSessionWrite],
    plans: Mapping[str, PreparedMembershipReplay],
) -> None:
    """Attempt each owner close before deleting their containing scratch tree."""
    failures: list[BaseException] = []
    for carrier in (*writes.values(), *plans.values()):
        try:
            carrier.close()
        except BaseException as exc:
            failures.append(exc)
    if failures:
        raise BaseExceptionGroup("retained replay carrier cleanup failed", failures)


def _cleanup_scratch(scratch_owner: tempfile.TemporaryDirectory[str]) -> None:
    """Remove a replay scratch tree and the decodes cached from its carriers."""
    from polylogue.sources.prepared_message_sink import discard_decoded_sessions_under
    from polylogue.storage.sqlite.connection_profile import (
        NativeConnectionSettlementError,
        retained_native_sql_owners_for_lifetime,
    )

    pending = retained_native_sql_owners_for_lifetime(scratch_owner)
    if pending:
        raise NativeConnectionSettlementError(pending[0], RuntimeError("scratch cleanup requires native drain"))
    discard_decoded_sessions_under(Path(scratch_owner.name))
    scratch_owner.cleanup()


T = TypeVar("T")

#: Seconds without a result before a retained preparation is reported
#: stalled. A report, not a deadline (polylogue-slc55): terminating the pool
#: at a fixed time discarded a large raw's preparation and retried it from
#: scratch, so it never finished.
_RETAINED_PREPARATION_STALL_REPORT_SECONDS = 600.0


_STILL_RUNNING = object()


def _raise_if_compute_cancelled(subject: str) -> None:
    """A checkpoint between retained-preparation phases (hashing, reconciliation)."""
    if compute_cancel_requested():
        from polylogue.sources.revision_backfill import RetainedPreparationRetryableError

        raise RetainedPreparationRetryableError(f"retained preparation cancelled for {subject}")


#: How often a wait checks its caller's cancellation.
_RETAINED_PREPARATION_CANCEL_POLL_SECONDS = 1.0


def _await_reporting_stalls(operation: SubmittedOperation[T], *, subject: str) -> T:
    """Wait for ``future``; report every window it runs without finishing.

    A retained preparation has no deadline, so the compute owner's
    cancellation is cooperative. A cancelled caller drains its own submitted
    work before scratch cleanup and raises a retryable refusal, retaining the
    raw for a later pass. It never closes the shared adapter.
    """
    from polylogue.sources.revision_backfill import RetainedPreparationRetryableError

    future = operation.future
    window = _RETAINED_PREPARATION_STALL_REPORT_SECONDS
    step = min(window, _RETAINED_PREPARATION_CANCEL_POLL_SECONDS)

    def refuse_if_cancelled() -> None:
        if compute_cancel_requested() or operation.cancellation.cancelled:
            operation.cancellation.cancel()
            # Running threads retain scratch ownership until they settle.
            # Cooperative checkpoints observe the compute owner's event.
            with contextlib.suppress(BaseException):
                future.result()
            raise RetainedPreparationRetryableError(f"retained preparation cancelled for {subject}")

    waited = 0.0
    unreported = 0.0
    while True:
        # Checked before each wait and before a finished result is accepted,
        # so a run of fast preparations cannot carry a cancelled pass on.
        refuse_if_cancelled()
        result: object
        try:
            result = future.result(timeout=step)
        except TimeoutError:
            # A future that finished between the wait expiring and this
            # check is re-read for its value, or for the worker's exception.
            result = future.result() if future.done() else _STILL_RUNNING
        if result is not _STILL_RUNNING:
            refuse_if_cancelled()
            return cast(T, result)
        waited += step
        unreported += step
        if unreported >= window:
            unreported = 0.0
            emit(
                "storage.raw_observation.preparation_stalled",
                level=WARNING,
                outcome="degraded",
                reason="no_result_in_window",
                subject_kind=subject.split(" ", 1)[0],
                wait_ms=round(waited * 1000),
            )
