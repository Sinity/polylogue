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
import sys
import tempfile
import time
import weakref
from builtins import BaseExceptionGroup
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from functools import partial
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, cast

from polylogue.archive.revision_authority import (
    RawRevisionAuthority,
    is_work_event_raw_id,
    parser_census_identity_measurement,
    raw_authority_parser_fingerprint,
)
from polylogue.archive.revision_replay import RevisionReplayPlan
from polylogue.archive.session_revision_membership import MembershipClassification
from polylogue.core.compute import (
    BoundedComputeAdapter,
    DaemonBackpressureError,
    DaemonOperationCancelled,
    SubmittedOperation,
)
from polylogue.core.compute_cancel import check_compute_cancelled, compute_cancel_requested
from polylogue.core.enums import Origin, Provider
from polylogue.core.raw_failure_evidence import (
    RAW_FAILURE_DEFERRED_SUPPORT_STATUS,
    RAW_FAILURE_REPLAY_AUTHORITY_EVIDENCE_KINDS,
    RAW_FAILURE_TERMINAL_EVIDENCE_SUPPORT_STATUS_PAIRS,
    RAW_FAILURE_VALIDATION_FAILURE_KINDS,
    CohortMembershipRefusalError,
    RetainedRawDecodeRefusalError,
)
from polylogue.core.sql_settlement import retain_native_sql_lifetimes
from polylogue.logging import WARNING, emit
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.blob_store import (
    BlobStore,
    BlobVerificationCancelledError,
    PreparedBlob,
    blob_store_for_connection,
)
from polylogue.storage.raw_authority import (
    build_raw_replay_plan,
    iter_parser_census_logical_keys,
    raw_replay_application_receipt_from_connection,
    validate_raw_replay_application_receipt,
)
from polylogue.storage.source_blob_restoration import stage_blob_from_recorded_source
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
        PreparedRevisionReplayResult,
        RevisionCensusResult,
    )
    from polylogue.storage.index_generation import IndexGeneration
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionWrite
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

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
    # Kernel diagnostic for selected coordinates/recipe, not a Source-value
    # fingerprint. The original witness supplies actual publication authority.
    input_binding: str
    # ``ReplacementLike`` requires a payload; this adapter carries its inputs
    # in the typed fields below instead.
    payload: None
    raw_ids: tuple[str, ...]
    prepared_inputs: Mapping[str, PreparedRetainedInput] | None = None
    prepared_aggregates: Mapping[str, PreparedRetainedAggregate] | None = None
    prepared_writes: Mapping[tuple[str, str], PreparedSessionWrite] | None = None
    prepared_replay_adoption: Mapping[tuple[str, tuple[str, ...]], PreparedRevisionAdoption] | None = None
    prepared_source_classification: PreparedRevisionSourceClassification | None = None
    verified_blob_stats: Mapping[str, tuple[int, int, int, int, int]] | None = None
    planned_accepted_raw_ids: Mapping[str, tuple[str, ...]] | None = None
    prepared_revision_plans: Mapping[str, RevisionReplayPlan] | None = None
    prepared_byte_outcomes: Mapping[str, PreparedRevisionReplayOutcome] | None = None
    prepared_membership_plans: Mapping[str, PreparedMembershipReplay] | None = None
    prepared_replay_source: PreparedRetainedReplaySource | None = None
    needs_source_census: bool = False
    prepared_source_census: PreparedRevisionSourceCensus | None = None
    prepared_replay_schedule: ReplaySchedule | None = None
    prepared_logical_keys: tuple[str, ...] = ()
    prepared_membership_keys: tuple[str, ...] = ()
    prepared_byte_logical_keys: tuple[str, ...] = ()
    prepared_key_refusals: tuple[CohortMembershipRefusalError, ...] = ()
    #: Logical keys whose parent publishes earlier in this same unit; they
    #: publish nothing here and are re-prepared against that parent.
    prepared_lineage_deferrals: tuple[str, ...] = ()
    #: The retained raws of those deferred keys; the rest of the unit published.
    lineage_deferred_raw_ids: tuple[str, ...] = ()
    #: Retained raws outside this unit whose sessions a write claims as its
    #: absent parent; compute widens the unit to them and prepares again.
    lineage_parent_raw_ids: tuple[str, ...] = ()
    needs_source_classification: bool = False
    scratch_directory: Path | None = None
    scratch_owner: tempfile.TemporaryDirectory[str] | None = None
    empty: bool = False
    already_valid: bool = False
    blob_restorations: StagedBlobRestorations | None = None
    reference_seal: PreparedIndexMutation | None = None
    # Wall time this compute spent in the provider parsers that sealed its
    # carriers. Publication reports it beside its own writer timings.
    provider_parse_seconds: float = 0.0

    def close(self) -> None:
        """Drain this prepared carrier when publication is cancelled or ends."""
        if self.reference_seal is not None and self.reference_seal.publication_lifetime_bound:
            self.reference_seal.close()
            return
        failures: list[BaseException] = []
        for close in (
            self._close_prepared_payload,
            *(() if self.reference_seal is None else (self.reference_seal.close,)),
        ):
            try:
                close()
            except BaseException as failure:
                failures.append(failure)
        if failures:
            raise BaseExceptionGroup("retained preparation cleanup failed", failures)

    def _close_prepared_payload(self) -> None:
        """Settle the carrier payload without recursing into its retained seal."""
        with retain_native_sql_lifetimes(*(() if self.scratch_owner is None else (self.scratch_owner,))):
            failures: list[BaseException] = []
            for close in (
                partial(
                    _close_prepared_carriers,
                    self.prepared_writes or {},
                    self.prepared_membership_plans or {},
                    chain(
                        (
                            item.prepared_artifact
                            for item in (self.prepared_inputs or {}).values()
                            if item.prepared_artifact is not None
                        ),
                        (item.artifact for item in (self.prepared_aggregates or {}).values()),
                    ),
                ),
                *(() if self.blob_restorations is None else (self.blob_restorations.discard,)),
            ):
                try:
                    close()
                except BaseException as failure:
                    failures.append(failure)
            # Scratch files depend on every native preparation owner. Keep
            # them on any failed close for the original creator's retry.
            if not failures and self.scratch_owner is not None:
                try:
                    _cleanup_scratch(self.scratch_owner)
                except BaseException as failure:
                    failures.append(failure)
            if failures:
                raise BaseExceptionGroup("retained preparation cleanup failed", failures)


def _session_id(session: ParsedSession) -> str:
    from polylogue.core.sources import origin_from_provider

    return f"{origin_from_provider(session.source_name).value}:{session.provider_session_id}"


def _holds_several_sessions(artifact: PreparedJsonl | None) -> bool:
    if artifact is None:
        return False
    with closing(artifact.iter_sessions()) as sessions:
        return next(sessions, None) is not None and next(sessions, None) is not None


class RawObservationInspection:
    """Read retained Raw state without admitting computation or publication."""

    domain = RAW_OBSERVATION_DOMAIN
    prerequisites: tuple[str, ...] = ()

    def __init__(self, archive_root: Path, *, index_db_path: Path | None = None) -> None:
        self.archive_root = archive_root
        self._index_db_path = index_db_path

    @property
    def recipe_version(self) -> str:
        return raw_authority_parser_fingerprint()

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
        from polylogue.core.raw_failure_evidence import retained_raw_decode_refusal_from_row
        from polylogue.storage.sqlite.queries.raw_state import retained_raw_decode_refusal_sql

        row = conn.execute(
            retained_raw_decode_refusal_sql(),
            (key, self.recipe_version, *sorted(RAW_FAILURE_VALIDATION_FAILURE_KINDS)),
        ).fetchone()
        return retained_raw_decode_refusal_from_row(key, row)

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
        if census is None:
            return "missing"
        if census["parser_fingerprint"] != self.recipe_version or census["status"] != "complete":
            return "stale"
        non_session = (
            conn.execute(
                "SELECT 1 FROM raw_artifacts WHERE raw_id = ? AND parse_as_session = 0 LIMIT 1",
                (key,),
            ).fetchone()
            is not None
        )
        member_count = 0

        def member_keys(rows: sqlite3.Cursor) -> Iterator[object]:
            nonlocal member_count
            for row in rows:
                member_count += 1
                yield row[0]

        with (
            closing(
                conn.execute(
                    "SELECT logical_source_key FROM raw_session_memberships WHERE raw_id=? ORDER BY logical_source_key",
                    (key,),
                )
            ) as members,
            parser_census_identity_measurement(
                raw_logical_key=raw["logical_source_key"],
                revision_kind=raw["revision_kind"],
                membership_logical_keys=member_keys(members),
                observed_logical_keys=iter_parser_census_logical_keys(census["logical_keys_json"]),
                observed_are_receipt=True,
            ) as measured,
        ):
            if not measured.complete(
                typed_non_session=non_session,
                parser_confirmed_non_session=membership is not None
                and membership["status"] == "non_session"
                and membership["parser_fingerprint"] == self.recipe_version,
                byte_governed_fragment=raw["source_index"] < 0
                and membership is not None
                and membership["revision_authority"] == RawRevisionAuthority.BYTE_PROVEN.value,
            ):
                return "stale"
            if (
                membership is not None
                and membership["status"] == "complete"
                and membership["member_count"] != member_count
            ):
                return "stale"
            with closing(conn.execute("SELECT session_id FROM index_tier.sessions WHERE raw_id=?", (key,))) as owned:
                for row in owned:
                    with closing(
                        measured.connection.execute(
                            "SELECT 1 FROM census_identity WHERE kind=1 AND logical_key=?", (str(row[0]),)
                        )
                    ) as identity:
                        if identity.fetchone() is None:
                            return "excess"
            if not measured.observed_count:
                return "valid"
            with closing(measured.iter_durable_bindings()) as bindings:
                for logical_key, source_key in bindings:
                    decision = None
                    if source_key is not None:
                        with closing(
                            conn.execute(
                                "SELECT decision FROM raw_session_memberships WHERE raw_id=? AND logical_source_key=?",
                                (key, source_key),
                            )
                        ) as rows:
                            decision = rows.fetchone()
                    if decision is not None and decision[0] in {"ambiguous", "deferred"}:
                        continue
                    if decision is not None and decision[0] is None:
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
            conn,
            plan,
            index_db_path=self._index_db_path or ArchiveLocation.resolve(self.archive_root).active_index_path,
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


def _publish_acquired_attachment_refs(
    seal: PreparedIndexMutation,
    prepared_inputs: Mapping[str, PreparedRetainedInput],
    *,
    blob_store: BlobStore,
) -> None:
    """Settle exact acquired claims before derived replay can fail or skip."""
    from polylogue.core.iterator_lifetime import settled_iterator
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.storage.sqlite.archive_tiers.revision_governance import (
        _PreparedSourceProducer,
        publish_prepared_revision_source,
    )
    from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceBlobRef, _write_source_blob_refs
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead

    artifacts: dict[int, PreparedJsonl] = {}
    for retained in prepared_inputs.values():
        artifact = retained.prepared_artifact
        if artifact is not None and (artifact.error is not None or retained.parser_error is not None):
            raise ValueError("errored retained input cannot publish attachment claims")
        if artifact is not None and id(artifact) not in artifacts:
            with closing(artifact.iter_attachment_claims()) as claims:
                if next(claims, None) is not None:
                    artifacts[id(artifact)] = artifact
    if not artifacts:
        return
    has_refs = False
    with seal.original_read_snapshot(), seal.source_producer():
        reader = PreparedSessionSourceRead(seal, blob_store=blob_store)
        producer = _PreparedSourceProducer(seal)
        for artifact in artifacts.values():
            original_raw_id, original_input = next(
                (raw_id, item) for raw_id, item in prepared_inputs.items() if item.prepared_artifact is artifact
            )
            source_path = original_input.source_path
            with settled_iterator(
                artifact.iter_attachment_refs(
                    source_path=source_path,
                    acquired_at_ms=reader.raw_revision_observation_order(original_raw_id)[0],
                    source_read=reader,
                    before_input=seal.retain_prepared_blob_input,
                )
            ) as refs:
                for ref in refs:
                    # One exact published claim can prove several identical Raw
                    # carriers; read its reservation once before receipt consumption.
                    consumed = False
                    for raw_id, retained in prepared_inputs.items():
                        if retained.prepared_artifact is artifact:
                            if retained.blob_hash != artifact.blob_hash:
                                raise ValueError("acquired attachment carrier differs from its original Raw bytes")
                            current_ref = replace(
                                ref,
                                acquired_at_ms=reader.raw_revision_observation_order(raw_id)[0],
                                publication_receipt_id=None if consumed else ref.publication_receipt_id,
                            )

                            def current_refs(
                                current_ref: ArchiveSourceBlobRef = current_ref,
                            ) -> Iterator[ArchiveSourceBlobRef]:
                                yield current_ref

                            _write_source_blob_refs(producer, raw_id, current_refs)
                            consumed = True
                            has_refs = True
    if has_refs:
        permit = seal.prepare_source_mutation()
        admit_stage_write("retained-acquired-attachment-refs", partial(publish_prepared_revision_source, seal, permit))


class RawObservationDerivation(RawObservationInspection):
    """A paged raw adapter; no ops hint or backlog census certifies validity.

    Preparation can run without a writer lease. Existing synchronous recovery
    callers still hold their enclosing lease; their composition must move
    before that production route can claim lease-free computation.
    """

    domain = RAW_OBSERVATION_DOMAIN
    prerequisites: tuple[str, ...] = ()
    # Restored bytes, Source census, classification and in-unit lineage
    # deferral each commit before the work that depends on them can be
    # prepared off the writer; the kernel continues those phases within one
    # pass while :meth:`publication_advanced` reports committed progress.
    # Every advance is one that cannot repeat for the same state, so the
    # continuation is bounded by progress rather than a phase count.

    def __init__(
        self,
        archive_root: Path,
        *,
        prepare_non_json_artifact: RetainedArtifactPreparer | None = None,
        compute_adapter: BoundedComputeAdapter,
        prepaid_blob_inputs: tuple[tuple[str, bytes, int], ...] = (),
        index_db_path: Path | None = None,
        owned_generation: IndexGeneration | None = None,
    ) -> None:
        super().__init__(archive_root, index_db_path=index_db_path)
        self._prepare_non_json_artifact = prepare_non_json_artifact
        self._compute_adapter = compute_adapter
        self._prepaid_blob_inputs = prepaid_blob_inputs
        self._index_db_path = index_db_path
        self._owned_generation = owned_generation
        #: Replacements whose publication committed a prerequisite phase,
        #: consumed by :meth:`publication_advanced` on the same key.
        self._phase_committed: dict[int, str] = {}
        #: Logical keys already deferred once for in-unit lineage. A key is
        #: deferred at most once per adapter, so re-preparation always makes
        #: progress even when its parent's publication is itself refused.
        self._lineage_deferred: set[str] = set()
        if owned_generation is not None:
            from polylogue.storage.sqlite.reference_seal import IndexMutationDestination

            destination = IndexMutationDestination.owned_inactive(owned_generation)
            if Path(owned_generation.archive_root).resolve(strict=True) != archive_root.resolve(strict=True):
                raise ValueError("retained replay generation belongs to another archive")
            if index_db_path is not None and index_db_path.resolve(strict=True) != destination.index_path:
                raise ValueError("retained replay Index differs from its owned generation")
            self._index_db_path = destination.index_path

    def _lineage_deferrals(
        self,
        prepared_writes: Mapping[tuple[str, str], PreparedSessionWrite],
        selected_writes: Mapping[tuple[str, str], tuple[ParsedSession, PreparedJsonl]],
        *,
        write_keys: Mapping[str, tuple[str, str]],
    ) -> tuple[str, ...]:
        """Logical keys whose claimed parent session publishes earlier in this unit.

        Each write was prepared against the Index as it stood before the unit,
        so a child whose parent is absent there expects no parent. Only a
        child whose parent another write of this unit produces is deferred.
        """
        produced = {session_id for _raw_id, session_id in write_keys.values()}
        deferred: list[str] = []
        for logical_key, write_key in write_keys.items():
            write = prepared_writes.get(write_key)
            selected = selected_writes.get(write_key)
            if write is None or selected is None or write.context.parent_session_id is not None:
                continue
            session = selected[0]
            claimed = write.context.hook_parent_native_id or session.parent_session_provider_id
            if not claimed:
                continue
            from polylogue.core.sources import origin_from_provider

            parent_session_id = f"{origin_from_provider(session.source_name).value}:{claimed.strip()}"
            if parent_session_id == write_key[1] or parent_session_id not in produced:
                continue
            if logical_key in self._lineage_deferred:
                continue
            deferred.append(logical_key)
        self._lineage_deferred.update(deferred)
        return tuple(sorted(deferred))

    @staticmethod
    def _lineage_parent_raw_ids(
        reference_seal: PreparedIndexMutation,
        read: PreparedSessionSourceRead,
        selected_writes: Mapping[tuple[str, str], tuple[ParsedSession, PreparedJsonl]],
        *,
        unit_raw_ids: Sequence[str],
    ) -> tuple[str, ...]:
        """Retained raws of claimed parents that are absent from the Index and this unit.

        A parent shares its child's origin. It is found by its logical key once
        censused, and by its acquired native id before that. A parent whose
        retained bytes refuse to decode is left out: widening must not make a
        child fail on it.
        """
        from polylogue.core.sources import origin_from_provider

        produced = {session_id for _raw_id, session_id in selected_writes}
        claims: dict[str, tuple[str, str]] = {}
        for (_raw_id, session_id), (session, _artifact) in selected_writes.items():
            claimed = (session.parent_session_provider_id or "").strip()
            if not claimed:
                continue
            origin = origin_from_provider(session.source_name).value
            parent_session_id = f"{origin}:{claimed}"
            if parent_session_id == session_id or parent_session_id in produced:
                continue
            reference_seal.before_index_input(
                "sessions", ("session_id",), "SELECT rowid FROM sessions WHERE session_id=?", (parent_session_id,)
            )
            with closing(
                reference_seal.observer("index").execute(
                    "SELECT 1 FROM sessions WHERE session_id=?", (parent_session_id,)
                )
            ) as present:
                if present.fetchone() is not None:
                    continue
            claims.setdefault(parent_session_id, (origin, claimed))
        if not claims:
            return ()
        candidates: set[str] = set()
        with read.replay_representative_rows(sorted(claims)) as rows:
            for _logical_source_key, raw_id in rows:
                candidates.add(str(raw_id))
        for origin, native_id in sorted(set(claims.values())):
            candidates.update(read.raw_ids_for_native_session(origin, native_id))
        unit = set(unit_raw_ids)
        return tuple(
            raw_id
            for raw_id in sorted(candidates)
            if raw_id not in unit and read.raw_terminal_decode_refusal(raw_id) is None
        )

    def publication_advanced(self, replacement: RawObservationReplacement) -> bool:
        """Whether this replacement's publication committed a prerequisite phase.

        False from :meth:`publish` otherwise means a refusal or a moved input;
        only a committed restoration, census or classification is the key's
        own progress that a fresh preparation can continue.
        """
        return self._phase_committed.pop(id(replacement), None) == replacement.key

    #: The durable Source rows a census or classification phase may change.
    _CENSUS_STATE_TABLES = (
        "raw_sessions",
        "raw_artifacts",
        "raw_session_memberships",
        "raw_membership_census",
        "raw_authority_parser_census",
    )

    def _census_state(self, raw_ids: Sequence[str]) -> tuple[tuple[object, ...], ...]:
        """Committed census state of ``raw_ids``; equal before and after means no progress."""
        from polylogue.storage.sqlite.connection_profile import readonly_connection_context

        selected = tuple(sorted(raw_ids))
        marks = ",".join("?" for _ in selected)
        state: list[tuple[object, ...]] = []
        with readonly_connection_context(self.archive_root / "source.db") as source:
            for table in self._CENSUS_STATE_TABLES:
                with closing(
                    source.execute(f"SELECT * FROM {table} WHERE raw_id IN ({marks}) ORDER BY rowid", selected)
                ) as rows:
                    state.extend((table, *row) for row in rows)
        return tuple(state)

    @staticmethod
    def _blob_stat_identity(path: Path) -> tuple[int, int, int, int, int]:
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    @contextmanager
    def _preparation_archive(self) -> Iterator[ArchiveStore]:
        if self._owned_generation is None:
            from polylogue.operations.operation_context import open_operation_read

            with open_operation_read(self.archive_root) as pinned:
                yield pinned.archive
            return
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.storage.sqlite.reference_seal import IndexMutationDestination

        destination = IndexMutationDestination.owned_inactive(self._owned_generation)
        with ArchiveStore.open_owned_inactive_read(self._owned_generation) as archive:
            yield archive
            destination.validate()

    def _selection_diagnostic(self, raw_ids: tuple[str, ...]) -> str:
        digest = hashlib.sha256(self.recipe_version.encode())
        for raw_id in raw_ids:
            encoded = raw_id.encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    @staticmethod
    def _source_replay_plans(
        reader: PreparedSessionSourceRead,
        logical_keys: tuple[str, ...],
    ) -> dict[str, RevisionReplayPlan]:
        """Capture byte plans on the parent's same original/selected Source view."""
        plans: dict[str, RevisionReplayPlan] = {}
        for logical_key in logical_keys:
            if reader.pending_raw_envelope_has_membership_authority(
                logical_key
            ) or reader.raw_membership_logical_raw_ids(logical_key):
                continue
            plan = reader.raw_revision_replay_plan(logical_key)
            if plan.applications:
                plans[logical_key] = plan
        return plans

    def compute(
        self,
        frame: RawFrame,
        key: str,
        *,
        replay_current: bool = False,
        select_retained_raw_ids: Callable[[PreparedSessionSourceRead], Sequence[str]] | None = None,
    ) -> RawObservationReplacement:
        self._compute_adapter.require_current_creator()
        # Even an initially empty discovery holds exclusive byte admission
        # before its witness can hydrate durable reference proof inputs.
        self._compute_adapter.amend_current_input_demand(0)
        # A child replayed before its parent stores the shared prefix whole and
        # is normalized again once the parent arrives. When a write claims a
        # parent retained in another component and absent from the Index, the
        # unit widens to that parent and prepares again; in-unit deferral then
        # publishes the parent first. The selection only grows, so this ends.
        widened: tuple[str, ...] = ()
        while True:
            selection = select_retained_raw_ids
            if widened:

                def selection(read: PreparedSessionSourceRead, extra: tuple[str, ...] = widened) -> Sequence[str]:
                    base = (key,) if select_retained_raw_ids is None else tuple(select_retained_raw_ids(read))
                    return (*base, *(raw_id for raw_id in extra if raw_id not in base))

            replacement = self._compute_once(frame, key, replay_current=replay_current, selection=selection)
            additional = tuple(raw_id for raw_id in replacement.lineage_parent_raw_ids if raw_id not in widened)
            if not additional:
                return replacement
            replacement.close()
            widened = (*widened, *additional)

    def _compute_once(
        self,
        frame: RawFrame,
        key: str,
        *,
        replay_current: bool,
        selection: Callable[[PreparedSessionSourceRead], Sequence[str]] | None,
    ) -> RawObservationReplacement:
        from polylogue.storage.sqlite.reference_seal import IndexMutationDestination, PreparedIndexMutation

        index_path = self._index_db_path or ArchiveLocation.resolve(self.archive_root).active_index_path
        destination = (
            None if self._owned_generation is None else IndexMutationDestination.owned_inactive(self._owned_generation)
        )
        seal = PreparedIndexMutation(
            index_path,
            archive_root=self.archive_root,
            destination=destination,
            input_demand=self._compute_adapter.amend_current_input_demand,
        )
        replacement: RawObservationReplacement | None = None
        try:
            replacement = replace(
                self._compute_prepared(
                    frame,
                    key,
                    replay_current=replay_current,
                    reference_seal=seal,
                    select_retained_raw_ids=selection,
                ),
                reference_seal=seal,
            )
            seal.validate_observers_current()
            return replacement
        except BaseException as primary:
            try:
                if replacement is None:
                    seal.close()
                else:
                    replacement.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup("retained preparation and cleanup failed", [primary, cleanup]) from primary
            raise

    def _compute_prepared(
        self,
        frame: RawFrame,
        key: str,
        *,
        replay_current: bool,
        reference_seal: PreparedIndexMutation,
        select_retained_raw_ids: Callable[[PreparedSessionSourceRead], Sequence[str]] | None = None,
    ) -> RawObservationReplacement:
        from polylogue.core.prepared_file import VerificationCancelledError
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

        with self._preparation_archive() as archive:
            from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead

            with (
                reference_seal.original_read_snapshot(
                    input_demand=self._compute_adapter.amend_current_input_demand,
                    prepaid_blob_inputs=self._prepaid_blob_inputs,
                ),
                reference_seal.source_producer(),
            ):
                selection_read = PreparedSessionSourceRead(
                    reference_seal,
                    blob_store=BlobStore(self.archive_root / "blob"),
                )
                selected = (key,) if select_retained_raw_ids is None else tuple(select_retained_raw_ids(selection_read))
                raw_ids, logical_keys = selection_read.expand_raw_membership_selection(selected)
                for selected_raw_id in raw_ids:
                    refusal = selection_read.raw_terminal_decode_refusal(selected_raw_id)
                    if refusal is not None:
                        raise refusal
                descriptors = {raw_id: selection_read.raw_revision_descriptor(raw_id) for raw_id in raw_ids}
            if (
                select_retained_raw_ids is None
                and not replay_current
                and self.inspect(frame, (key,)).get(key) == "valid"
            ):
                return RawObservationReplacement(
                    key,
                    "",
                    None,
                    (),
                    already_valid=True,
                )
            binding = self._selection_diagnostic(raw_ids)
            # Classification and preparation both read retained bytes, so an
            # absent blob is restored before either runs.
            restorations = self._stage_absent_blob_restorations(archive, raw_ids, descriptors)
            if restorations is not None:
                return RawObservationReplacement(key, binding, None, raw_ids, blob_restorations=restorations)
            process_prepared = bool(descriptors)
            if process_prepared:
                from polylogue.sources.prepared_merge import (
                    prepare_retained_cohort_artifact,
                    prepared_cohort_source_hash,
                )
                from polylogue.sources.revision_backfill import (
                    PreparedRetainedAggregate,
                    prepare_membership_replay,
                    prepare_retained_replay_source,
                )
                from polylogue.storage.sqlite.archive_tiers.revision_governance import (
                    membership_head_input_from_revision_head,
                    prepare_membership_head_plan,
                    prepare_membership_head_plan_from_inputs,
                    prepared_parser_census_is_current,
                )
                from polylogue.storage.sqlite.archive_tiers.write import (
                    prepare_session_write,
                    prepared_session_rows_from_shard,
                )

                with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                    complete_census = {
                        raw_id for raw_id in raw_ids if prepared_parser_census_is_current(reference_seal, raw_id)
                    }
                # Every retained raw needs its actual parser authority before
                # replay can select a session. A singleton can still refine an
                # opaque acquisition identity or prove a non-session artifact.
                needs_source_census = len(complete_census) != len(raw_ids)
                prepared_source_classification = None
                if not needs_source_census:
                    from polylogue.sources.revision_backfill import prepare_revision_source_classification

                    prepared_source_classification = prepare_revision_source_classification(
                        reference_seal,
                        selection_read,
                        selected_raw_ids=raw_ids,
                        logical_keys=logical_keys,
                        payload_store=BlobStore(self.archive_root / "blob"),
                    )
                    if prepared_source_classification is not None:
                        return RawObservationReplacement(
                            key,
                            binding,
                            None,
                            raw_ids,
                            prepared_source_classification=prepared_source_classification,
                            needs_source_classification=True,
                        )
                with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                    prepared_revision_plans = self._source_replay_plans(
                        selection_read, selection_read.raw_revision_rebuild_logical_keys(raw_ids)
                    )
                    planned_accepted_raw_ids = {
                        key: plan.accepted_raw_ids for key, plan in prepared_revision_plans.items()
                    }
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
                    prepared_replay_adoption: dict[tuple[str, tuple[str, ...]], PreparedRevisionAdoption] = {}
                    membership_plans: dict[str, PreparedMembershipReplay] = {}
                    prepared_byte_outcomes: dict[str, PreparedRevisionReplayOutcome] = {}
                    prepared_lineage_deferrals: tuple[str, ...] = ()
                    lineage_deferred_raw_ids: set[str] = set()
                    prepared_replay_source: PreparedRetainedReplaySource | None = None
                    verified_blob_stats: dict[str, tuple[int, int, int, int, int]] = {}
                    prepared_source_census: PreparedRevisionSourceCensus | None = None
                    prepared_replay_schedule: ReplaySchedule | None = None
                    prepared_logical_keys: tuple[str, ...] = ()
                    prepared_membership_keys: tuple[str, ...] = ()
                    prepared_byte_logical_keys: tuple[str, ...] = ()
                    prepared_key_refusals: dict[str, CohortMembershipRefusalError] = {}
                    prepared_artifacts: dict[tuple[object, ...], PreparedJsonl] = {}
                    provider_parse_seconds = 0.0
                    try:
                        for raw_id in raw_ids:
                            provider, blob_hash, path, kind, size = descriptors[raw_id]
                            blob_store = BlobStore(self.archive_root / "blob")
                            # Verification consumes the same canonical CAS
                            # input as parsing, so enroll it before the first
                            # byte read rather than charging after verification.
                            with reference_seal.original_read_snapshot():
                                input_hash, input_size = reference_seal.retain_original_blob_input(raw_id)
                                if input_hash != bytes.fromhex(blob_hash) or input_size != size:
                                    raise RetainedPreparationRetryableError(
                                        f"retained raw input identity changed before verification: {raw_id}"
                                    )
                            blob_path = blob_store.blob_path(blob_hash)
                            try:
                                before = self._blob_stat_identity(blob_path)
                            except OSError as exc:
                                raise RetainedPreparationRetryableError(
                                    f"retained raw blob disappeared: {raw_id}"
                                ) from exc
                            if not blob_store.verify(blob_hash, stop=compute_cancel_requested):
                                raise RetainedPreparationRetryableError(f"retained raw blob changed: {raw_id}")

                            with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                                retained_read = PreparedSessionSourceRead(reference_seal, blob_store=blob_store)
                                native_id = retained_read.raw_native_id(raw_id) if kind.value == "append" else None
                                fallback_timestamp = retained_read.raw_revision_file_mtime(raw_id)
                                profile_identity = retained_read.raw_profile_identity(raw_id)
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
                                    # This carrier stays under the already admitted Raw
                                    # reservation through publication and physical close.
                                    check_compute_cancelled()

                                    with (
                                        reference_seal.original_read_snapshot(),
                                        reference_seal.source_producer(),
                                    ):
                                        retained_read = PreparedSessionSourceRead(
                                            reference_seal,
                                            blob_store=blob_store,
                                        )
                                        parse_started = time.perf_counter()
                                        artifact = worker(retained_read, raw_id, directory=scratch)
                                        provider_parse_seconds += time.perf_counter() - parse_started
                                    check_compute_cancelled()
                                except DaemonOperationCancelled:
                                    raise
                                except DaemonBackpressureError as exc:
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
                                unsupported_shape=artifact.unsupported_shape,
                                captured_profile_key=profile_identity,
                                prepared_artifact=artifact if artifact.error is None else None,
                            )
                        # Each publisher commits/exposes its first Source unit on
                        # this creator before the later census/session tape exists.
                        for artifact in prepared_artifacts.values():
                            if artifact.error is None:
                                artifact.publish_blobs(reference_seal=reference_seal)
                        _publish_acquired_attachment_refs(reference_seal, prepared, blob_store=material_store)
                        from polylogue.sources.prepared_jsonl import complete_thread_projection_cohort

                        with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                            state_read = PreparedSessionSourceRead(reference_seal, blob_store=material_store)
                            for retained_raw_id, retained_input in prepared.items():
                                state_artifact = retained_input.prepared_artifact
                                if state_artifact is None or state_artifact.codex_state_kind != "thread_state":
                                    continue
                                state_observed_at, state_order = state_read.raw_revision_observation_order(
                                    retained_raw_id
                                )
                                state_artifact.prepare_thread_projection(
                                    reference_seal,
                                    source_read=state_read,
                                    raw_id=retained_raw_id,
                                    blob_hash=retained_input.blob_hash,
                                    observed_at_ms=state_observed_at,
                                    observation_order=state_order,
                                    source_path=retained_input.source_path,
                                )
                            complete_thread_projection_cohort(
                                (
                                    item.prepared_artifact
                                    for item in prepared.values()
                                    if item.prepared_artifact is not None
                                ),
                                reference_seal,
                                source_read=state_read,
                            )
                        if not needs_source_census:
                            from polylogue.sources.revision_backfill import (
                                _prepared_retained_outcome,
                                prepared_session_for_revision_key,
                            )

                            with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                                member_read = PreparedSessionSourceRead(reference_seal, blob_store=material_store)
                                for logical_key, accepted_raw_ids in planned_accepted_raw_ids.items():
                                    if logical_key.startswith(PENDING_RAW_LOGICAL_SOURCE_PREFIX):
                                        continue
                                    for member_raw_id in accepted_raw_ids:
                                        try:
                                            original_output = _prepared_retained_outcome(
                                                member_read, member_raw_id, prepared, stop=compute_cancel_requested
                                            )
                                            if isinstance(original_output, Exception):
                                                raise CohortMembershipRefusalError(
                                                    logical_key,
                                                    member_raw_id,
                                                    f"selector member did not parse: {original_output}",
                                                ) from original_output
                                            prepared_session_for_revision_key(
                                                original_output[0], raw_id=member_raw_id, logical_source_key=logical_key
                                            )
                                        except CohortMembershipRefusalError as refusal:
                                            prepared_key_refusals[logical_key] = refusal
                                            break
                        for logical_key, accepted_raw_ids in (
                            planned_accepted_raw_ids.items() if not needs_source_census else ()
                        ):
                            if logical_key in prepared_key_refusals or len(accepted_raw_ids) < 2:
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
                                check_compute_cancelled()
                                aggregate = prepare_retained_cohort_artifact(ordered, scratch)
                                check_compute_cancelled()
                            except DaemonOperationCancelled:
                                raise
                            except DaemonBackpressureError as exc:
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
                        if needs_source_census:
                            from polylogue.sources.revision_backfill import (
                                PreparedRevisionSourceCensus,
                                RevisionCensusResult,
                                prepare_revision_source_census,
                            )

                            with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                                census_read = PreparedSessionSourceRead(reference_seal, blob_store=material_store)
                                census_state = prepare_revision_source_census(
                                    reference_seal,
                                    census_read,
                                    selected_raw_ids=raw_ids,
                                    prepared_inputs=prepared,
                                )
                                census_raw_ids, census_keys = census_read.expand_raw_membership_selection(raw_ids)
                            prepared_source_census = PreparedRevisionSourceCensus(
                                reference_seal.prepare_source_mutation(),
                                RevisionCensusResult(
                                    census_state.scanned,
                                    census_state.classified,
                                    census_state.quarantined,
                                    census_raw_ids,
                                    census_keys,
                                ),
                            )
                        if not needs_source_census:
                            from polylogue.sources.revision_backfill import (
                                PreparedRevisionSourceCensus,
                                RevisionCensusResult,
                                prepare_revision_source_membership_conversion,
                            )

                            with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                                conversion_read = PreparedSessionSourceRead(reference_seal, blob_store=material_store)
                                converted = prepare_revision_source_membership_conversion(
                                    reference_seal,
                                    conversion_read,
                                    logical_keys=tuple(key for key in logical_keys if key not in prepared_key_refusals),
                                    prepared_inputs=prepared,
                                    prepared_replay_plans=planned_accepted_raw_ids,
                                )
                                conversion_raw_ids: tuple[str, ...] = ()
                                conversion_keys: tuple[str, ...] = ()
                                if converted:
                                    conversion_raw_ids, conversion_keys = (
                                        conversion_read.expand_raw_membership_selection(raw_ids)
                                    )
                            if converted:
                                prepared_source_census = PreparedRevisionSourceCensus(
                                    reference_seal.prepare_source_mutation(),
                                    RevisionCensusResult(0, converted, 0, conversion_raw_ids, conversion_keys),
                                )
                                needs_source_census = True
                        if not needs_source_census and archive.index_connection is not None:
                            # Keyed by (raw, session): one raw may carry several sessions.
                            from polylogue.sources.revision_backfill import prepared_session_for_revision_key

                            selected_writes: dict[tuple[str, str], tuple[ParsedSession, PreparedJsonl]] = {}
                            for logical_key, accepted_raw_ids in planned_accepted_raw_ids.items():
                                if logical_key in prepared_key_refusals or not accepted_raw_ids:
                                    continue
                                tip_raw_id = accepted_raw_ids[-1]
                                selected_artifact = (
                                    aggregates[logical_key].artifact
                                    if len(accepted_raw_ids) > 1 and logical_key in aggregates
                                    else prepared[tip_raw_id].prepared_artifact
                                )
                                if selected_artifact is None:
                                    continue
                                try:
                                    selected_session = prepared_session_for_revision_key(
                                        selected_artifact.session_sequence(),
                                        raw_id=tip_raw_id,
                                        logical_source_key=logical_key,
                                    )
                                except CohortMembershipRefusalError as refusal:
                                    prepared_key_refusals[logical_key] = refusal
                                    continue
                                selected_writes[(tip_raw_id, _session_id(selected_session))] = (
                                    selected_session,
                                    selected_artifact,
                                )
                                if is_work_event_raw_id(logical_key):
                                    continue
                                with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                                    adoption_read = PreparedSessionSourceRead(reference_seal, blob_store=material_store)
                                    prepared_replay_adoption[(logical_key, accepted_raw_ids)] = (
                                        adoption_read.prepare_raw_revision_replay_adoption(
                                            [selected_session],
                                            logical_source_key=logical_key,
                                            raw_ids=accepted_raw_ids,
                                        )
                                    )
                            with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                                lineage_parent_raw_ids = self._lineage_parent_raw_ids(
                                    reference_seal,
                                    PreparedSessionSourceRead(reference_seal, blob_store=material_store),
                                    selected_writes,
                                    unit_raw_ids=raw_ids,
                                )
                            if lineage_parent_raw_ids:
                                # The widened unit prepares everything again; this
                                # carrier only drains what was prepared so far.
                                return RawObservationReplacement(
                                    key,
                                    binding,
                                    None,
                                    raw_ids,
                                    prepared_inputs=prepared,
                                    prepared_aggregates=aggregates,
                                    lineage_parent_raw_ids=lineage_parent_raw_ids,
                                    scratch_directory=scratch,
                                    scratch_owner=scratch_owner,
                                )
                            from polylogue.storage.sqlite.archive_tiers.revision_governance import (
                                prepared_raw_revision_file_mtime,
                            )

                            with (
                                reference_seal.original_read_snapshot(
                                    input_demand=self._compute_adapter.amend_current_input_demand,
                                ),
                                reference_seal.source_producer(),
                            ):
                                source_read = PreparedSessionSourceRead(
                                    reference_seal,
                                    blob_store=blob_store_for_connection(reference_seal.observer("source")),
                                )
                                from polylogue.sources.revision_backfill import (
                                    _lineage_aware_replay_schedule,
                                    _PreparedReplayInputs,
                                )

                                prepared_logical_keys = source_read.raw_revision_rebuild_logical_keys(raw_ids)
                                _expanded_raw_ids, expanded_membership_keys = (
                                    source_read.expand_raw_membership_selection(raw_ids)
                                )
                                # Event envelopes have singleton byte authority and
                                # append to an existing target; they never replace
                                # transcript membership or its accepted head.
                                prepared_membership_keys = tuple(
                                    key for key in expanded_membership_keys if not is_work_event_raw_id(key)
                                )
                                # Membership-keyed sessions are ordered with the
                                # byte-typed ones, so a parent replays before its
                                # children whichever authority keys either.
                                prepared_replay_schedule = _lineage_aware_replay_schedule(
                                    {
                                        logical_key
                                        for logical_key in (*prepared_logical_keys, *prepared_membership_keys)
                                        if not is_work_event_raw_id(logical_key)
                                    },
                                    source_read,
                                    _PreparedReplayInputs(prepared),
                                )
                                prepared_byte_logical_keys = tuple(
                                    logical_key
                                    for logical_key in prepared_replay_schedule.order
                                    if not (
                                        source_read.pending_raw_envelope_has_membership_authority(logical_key)
                                        or source_read.raw_membership_logical_raw_ids(logical_key)
                                    )
                                )
                                for (tip_raw_id, session_id), (session, artifact) in selected_writes.items():
                                    _raise_if_compute_cancelled(f"raw {tip_raw_id}")
                                    if artifact.shard_path is None:
                                        raise RetainedPreparationRetryableError(
                                            f"retained replay shard is absent for raw {tip_raw_id}"
                                        )
                                    event_only = is_work_event_raw_id(tip_raw_id)
                                    prepared_writes[(tip_raw_id, session_id)] = prepare_session_write(
                                        reference_seal.observer("index"),
                                        session,
                                        merge_append=event_only,
                                        fallback_timestamp=prepared_raw_revision_file_mtime(reference_seal, tip_raw_id),
                                        source_read=source_read,
                                        raw_id=tip_raw_id,
                                        force_replace=not event_only,
                                        prepared_rows=(
                                            None
                                            if event_only
                                            else prepared_session_rows_from_shard(artifact.shard_path, session_id)
                                        ),
                                        before_input=reference_seal.before_index_input,
                                    )
                                from polylogue.storage.sqlite.archive_tiers.revision_governance import (
                                    prepare_revision_replay_outcome,
                                )

                                for logical_key in prepared_byte_logical_keys:
                                    if logical_key in prepared_key_refusals:
                                        continue
                                    byte_plan = prepared_revision_plans[logical_key]
                                    if not byte_plan.accepted_raw_ids:
                                        continue
                                    byte_adoption = prepared_replay_adoption[(logical_key, byte_plan.accepted_raw_ids)]
                                    if byte_adoption.session_id is None:
                                        raise RetainedPreparationRetryableError(
                                            "byte replay has no prepared adoption identity"
                                        )
                                    byte_tip = byte_plan.accepted_raw_ids[-1]
                                    byte_session, _byte_artifact = selected_writes[(byte_tip, byte_adoption.session_id)]
                                    byte_write = prepared_writes[(byte_tip, byte_adoption.session_id)]
                                    prepared_byte_outcomes[logical_key] = prepare_revision_replay_outcome(
                                        reference_seal,
                                        source_read,
                                        byte_plan,
                                        byte_adoption,
                                        aggregate_session=byte_session,
                                        aggregate_content_hash=byte_write.rows.session_content_hash,
                                        prepared_write=byte_write,
                                    )
                            for logical_key in logical_keys:
                                if logical_key in prepared_key_refusals or is_work_event_raw_id(logical_key):
                                    continue
                                with (
                                    reference_seal.original_read_snapshot(),
                                    reference_seal.source_producer(),
                                ):
                                    membership_read = PreparedSessionSourceRead(
                                        reference_seal,
                                        blob_store=material_store,
                                    )
                                    if planned_accepted_raw_ids.get(logical_key) and not (
                                        membership_read.membership_key_has_pending_envelope_member(logical_key)
                                        or membership_read.raw_membership_logical_raw_ids(logical_key)
                                    ):
                                        continue
                                    preceding_byte = prepared_byte_outcomes.get(logical_key)
                                    effective_head_raw_id = (
                                        (
                                            None
                                            if preceding_byte.effective_head is None
                                            else str(preceding_byte.effective_head[1])
                                        )
                                        if preceding_byte is not None
                                        else membership_read.raw_revision_head_raw_id(logical_key)
                                    )
                                    try:
                                        membership_plan = prepare_membership_replay(
                                            membership_read,
                                            logical_key,
                                            prepared,
                                            head_raw_id=effective_head_raw_id,
                                            stop=compute_cancel_requested,
                                        )
                                    except CohortMembershipRefusalError as refusal:
                                        prepared_key_refusals[logical_key] = refusal
                                        continue
                                    if preceding_byte is None:
                                        head_plan = prepare_membership_head_plan(
                                            reference_seal.observer("index"),
                                            membership_read,
                                            logical_key,
                                            membership_plan.classification,
                                            before_input=reference_seal.before_index_input,
                                        )
                                    else:
                                        head_plan = prepare_membership_head_plan_from_inputs(
                                            membership_read,
                                            logical_key,
                                            membership_plan.classification,
                                            existing_head=membership_head_input_from_revision_head(
                                                preceding_byte.effective_head
                                            ),
                                            persisted_session=preceding_byte.effective_session_revision,
                                        )
                                    membership_plan = replace(membership_plan, head_plan=head_plan)
                                    if head_plan.conflict is not None:
                                        # A refused cohort publishes nothing this pass;
                                        # its members carry the retryable evidence.
                                        membership_plan = replace(
                                            membership_plan, classification=MembershipClassification((), (), ())
                                        )
                                    accepted_members = membership_plan.classification.accepted_raw_ids
                                    if accepted_members:
                                        prepared_replay_adoption[(logical_key, accepted_members)] = (
                                            membership_read.prepare_raw_revision_replay_adoption(
                                                [membership_plan.sessions[raw_id] for raw_id in accepted_members],
                                                logical_source_key=logical_key,
                                                raw_ids=accepted_members,
                                            )
                                        )
                                membership_plans[logical_key] = membership_plan
                                if (
                                    not membership_plan.classification.accepted_raw_ids
                                    or head_plan.yield_to_head_raw_id is not None
                                ):
                                    continue
                                accepted_raw_id = membership_plan.classification.accepted_raw_ids[-1]
                                accepted_session = membership_plan.sessions[accepted_raw_id]
                                membership_artifact = prepared[accepted_raw_id].prepared_artifact
                                if membership_artifact is not None:
                                    selected_writes[(accepted_raw_id, _session_id(accepted_session))] = (
                                        accepted_session,
                                        membership_artifact,
                                    )
                                    with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                                        membership_source = PreparedSessionSourceRead(
                                            reference_seal, blob_store=material_store
                                        )
                                        if membership_artifact.shard_path is None:
                                            raise RetainedPreparationRetryableError("membership replay shard is absent")
                                        member_session_id = _session_id(accepted_session)
                                        prepared_writes[(accepted_raw_id, member_session_id)] = prepare_session_write(
                                            reference_seal.observer("index"),
                                            accepted_session,
                                            merge_append=False,
                                            fallback_timestamp=prepared_raw_revision_file_mtime(
                                                reference_seal, accepted_raw_id
                                            ),
                                            source_read=membership_source,
                                            raw_id=accepted_raw_id,
                                            force_replace=True,
                                            prepared_rows=prepared_session_rows_from_shard(
                                                membership_artifact.shard_path, member_session_id
                                            ),
                                            before_input=reference_seal.before_index_input,
                                        )
                            # A child prepared against an Index without its parent
                            # expects no parent; if that parent publishes earlier in
                            # this same unit, the child's write would find it and
                            # refuse as moved lineage. Defer the child: the rest
                            # publishes, and the child is re-prepared in this same
                            # pass against its published parent.
                            write_keys = {
                                **{
                                    logical_key: (
                                        prepared_revision_plans[logical_key].accepted_raw_ids[-1],
                                        str(
                                            prepared_replay_adoption[
                                                (logical_key, prepared_revision_plans[logical_key].accepted_raw_ids)
                                            ].session_id
                                        ),
                                    )
                                    for logical_key in prepared_byte_outcomes
                                },
                                **{
                                    logical_key: (
                                        plan.classification.accepted_raw_ids[-1],
                                        _session_id(plan.sessions[plan.classification.accepted_raw_ids[-1]]),
                                    )
                                    for logical_key, plan in membership_plans.items()
                                    if plan.classification.accepted_raw_ids
                                },
                            }
                            prepared_lineage_deferrals = self._lineage_deferrals(
                                prepared_writes, selected_writes, write_keys=write_keys
                            )
                            for deferred_key in prepared_lineage_deferrals:
                                prepared_byte_outcomes.pop(deferred_key, None)
                                deferred_plan = membership_plans.pop(deferred_key, None)
                                lineage_deferred_raw_ids.update(
                                    deferred_plan.candidate_raw_ids
                                    if deferred_plan is not None
                                    else prepared_revision_plans[deferred_key].accepted_raw_ids
                                )
                            # A later membership refusal supersedes any earlier
                            # prepared byte outcome for that same original key.
                            # The Source acknowledgement and writer consume the
                            # same remaining outcomes, including shared Raw IDs.
                            for refused_key in prepared_key_refusals:
                                prepared_byte_outcomes.pop(refused_key, None)
                            prepared_replay_source = prepare_retained_replay_source(
                                reference_seal,
                                prepared_inputs=prepared,
                                byte_outcomes=prepared_byte_outcomes,
                                membership_plans=membership_plans,
                                adoptions=prepared_replay_adoption,
                                prepared_writes=prepared_writes,
                            )
                            membership_plans = dict(prepared_replay_source.membership_plans)
                    except BaseException as primary:
                        failure = (
                            DaemonOperationCancelled("retained preparation cancelled during verification")
                            if isinstance(primary, (BlobVerificationCancelledError, VerificationCancelledError))
                            else primary
                        )

                        def close_failed_payload() -> None:
                            _close_prepared_carriers(
                                prepared_writes,
                                membership_plans,
                                chain(prepared_artifacts.values(), (item.artifact for item in aggregates.values())),
                            )
                            _cleanup_scratch(scratch_owner)

                        # The original seal releases settled SQL dependencies
                        # before invoking this payload, and retains it on any
                        # physical close failure for the original owner's retry.
                        reference_seal.retain_preparation_payload(close_failed_payload)
                        if failure is not primary:
                            raise failure from primary
                        raise
                    return RawObservationReplacement(
                        key,
                        binding,
                        None,
                        raw_ids,
                        prepared_inputs=prepared,
                        prepared_aggregates=aggregates,
                        prepared_writes=prepared_writes,
                        prepared_replay_adoption=prepared_replay_adoption,
                        prepared_source_classification=prepared_source_classification,
                        verified_blob_stats=verified_blob_stats,
                        planned_accepted_raw_ids=planned_accepted_raw_ids,
                        prepared_revision_plans=prepared_revision_plans,
                        prepared_byte_outcomes=prepared_byte_outcomes,
                        prepared_membership_plans=membership_plans,
                        prepared_replay_source=prepared_replay_source,
                        needs_source_census=needs_source_census,
                        prepared_source_census=prepared_source_census,
                        prepared_replay_schedule=prepared_replay_schedule,
                        prepared_logical_keys=prepared_logical_keys,
                        prepared_membership_keys=prepared_membership_keys,
                        prepared_byte_logical_keys=prepared_byte_logical_keys,
                        prepared_key_refusals=tuple(prepared_key_refusals.values()),
                        prepared_lineage_deferrals=prepared_lineage_deferrals,
                        lineage_deferred_raw_ids=tuple(sorted(lineage_deferred_raw_ids)),
                        scratch_directory=scratch,
                        scratch_owner=scratch_owner,
                        provider_parse_seconds=provider_parse_seconds,
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
                prepared, reason = stage_blob_from_recorded_source(
                    archive.source_connection,
                    self.archive_root,
                    blob_store,
                    raw_id,
                    blob_hash=blob_hash,
                    source_path=path,
                    stop=compute_cancel_requested,
                )
                if prepared is None:
                    raise RetainedPreparationRetryableError(
                        f"retained raw blob absent and not restorable from its source ({reason}): {raw_id}"
                    )
                staged.append((raw_id, prepared))
                staged_hashes.add(blob_hash)
        except BlobVerificationCancelledError as exc:
            _discard_staged_blobs(blob_store, tuple(prepared for _raw_id, prepared in staged))
            raise DaemonOperationCancelled("retained blob restoration cancelled") from exc
        except BaseException:
            _discard_staged_blobs(blob_store, tuple(prepared for _raw_id, prepared in staged))
            raise
        return StagedBlobRestorations(blob_store, staged) if staged else None

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

    def publish(
        self,
        frame: RawFrame,
        replacement: RawObservationReplacement,
        *,
        phase_receipt: Callable[
            [Literal["census", "classification", "replay"], RevisionCensusResult | PreparedRevisionReplayResult], None
        ]
        | None = None,
        publication_failure: Callable[[BaseException], None] | None = None,
    ) -> bool:
        from polylogue.sources.revision_backfill import (
            RetainedPreparationNoProgressError,
            RetainedPreparationRetryableError,
            apply_prepared_revision_census,
            apply_prepared_revision_classification,
            apply_prepared_revision_replay,
        )
        from polylogue.storage.index_generation import ActiveWriterLease
        from polylogue.storage.raw_retention import raw_frontier_blocked_raw_ids
        from polylogue.storage.sqlite.archive_tiers.revision_governance import PreparedRawClassificationStaleError

        if replacement.already_valid:
            try:
                return self._current(frame) and self.inspect(frame, (replacement.key,)).get(replacement.key) == "valid"
            finally:
                replacement.close()

        with retain_native_sql_lifetimes(*(() if replacement.scratch_owner is None else (replacement.scratch_owner,))):
            lease = ActiveWriterLease(self.archive_root)
            lifetime_bound = False
            try:
                lease.acquire()
                if replacement.reference_seal is None:
                    raise RetainedPreparationRetryableError("retained publication lacks its original reference seal")
                replacement.reference_seal.retain_publication_lifetime(lease, replacement._close_prepared_payload)
                lifetime_bound = True
                if not self._current(frame):
                    return False
                # This gate also covers early Blob restoration, whose branch
                # does not reach a Source or Index permit. It must precede any
                # exposure, and checks the same original observers after actual
                # writer admission instead of copying a second value digest.
                replacement.reference_seal.validate_observers_current()
                frontier_refusal = raw_frontier_blocked_raw_ids(self.archive_root, replacement.raw_ids)
                selected_paths = set(self.source_paths(replacement.raw_ids).values())
                if frontier_refusal.unattributed_reason is not None or selected_paths.intersection(
                    frontier_refusal.source_paths
                ):
                    return False
                if replacement.blob_restorations is not None:
                    self._publish_blob_restorations(replacement.blob_restorations)
                    # The restored bytes are prepared by the next phase, which
                    # now finds them present; this publication certifies no output.
                    self._phase_committed[id(replacement)] = replacement.key
                    return False
                with self._preparation_archive() as archive:
                    raw_ids, _keys = archive.expand_raw_membership_selection(list(replacement.raw_ids))
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
                    if replacement.prepared_source_census is None:
                        raise RetainedPreparationRetryableError(
                            "retained census lacks its original prepared Source tape"
                        )
                    before = self._census_state(replacement.raw_ids)
                    receipt = apply_prepared_revision_census(
                        replacement.reference_seal,
                        replacement.prepared_source_census,
                    )
                    if self._census_state(replacement.raw_ids) == before:
                        raise RetainedPreparationNoProgressError(
                            f"retained census left its durable inputs unchanged: {replacement.key}"
                        )
                    if phase_receipt is not None:
                        phase_receipt("census", receipt)
                    refusal = next(iter(self.terminal_decode_refusals(replacement.raw_ids).values()), None)
                    if refusal is not None:
                        raise refusal
                    self._phase_committed[id(replacement)] = replacement.key
                    return False
                if replacement.needs_source_classification:
                    if replacement.prepared_source_classification is None:
                        raise RetainedPreparationRetryableError(
                            "retained classification lacks its original Source tape"
                        )
                    before = self._census_state(replacement.raw_ids)
                    try:
                        receipt = apply_prepared_revision_classification(
                            replacement.reference_seal,
                            replacement.prepared_source_classification,
                            payload_store=BlobStore(self.archive_root / "blob"),
                        )
                    except (PreparedRawClassificationStaleError, RetainedPreparationRetryableError) as failure:
                        if publication_failure is not None:
                            publication_failure(failure)
                        return False
                    if self._census_state(replacement.raw_ids) == before:
                        raise RetainedPreparationNoProgressError(
                            f"retained classification left its durable inputs unchanged: {replacement.key}"
                        )
                    if phase_receipt is not None:
                        phase_receipt("classification", receipt)
                    refusal = next(iter(self.terminal_decode_refusals(replacement.raw_ids).values()), None)
                    if refusal is not None:
                        raise refusal
                    self._phase_committed[id(replacement)] = replacement.key
                    return False
                if (
                    replacement.prepared_inputs is None
                    or replacement.prepared_aggregates is None
                    or replacement.prepared_writes is None
                    or replacement.planned_accepted_raw_ids is None
                    or replacement.prepared_revision_plans is None
                    or replacement.prepared_byte_outcomes is None
                    or replacement.prepared_membership_plans is None
                    or replacement.prepared_replay_source is None
                    or replacement.prepared_replay_adoption is None
                    or replacement.prepared_replay_schedule is None
                ):
                    return False
                try:
                    replay_receipt = apply_prepared_revision_replay(
                        self.archive_root,
                        reference_seal=replacement.reference_seal,
                        active_index_path=Path(frame.source_revision),
                        selected_raw_ids=list(replacement.raw_ids),
                        prepared_inputs=replacement.prepared_inputs,
                        prepared_aggregates=replacement.prepared_aggregates,
                        prepared_writes=replacement.prepared_writes,
                        prepared_replay_adoption=replacement.prepared_replay_adoption,
                        prepared_replay_plans=replacement.prepared_revision_plans,
                        prepared_byte_outcomes=replacement.prepared_byte_outcomes,
                        prepared_membership_plans=replacement.prepared_membership_plans,
                        prepared_replay_source=replacement.prepared_replay_source,
                        prepared_replay_schedule=replacement.prepared_replay_schedule,
                        prepared_logical_keys=replacement.prepared_logical_keys,
                        prepared_membership_keys=replacement.prepared_membership_keys,
                        prepared_byte_logical_keys=replacement.prepared_byte_logical_keys,
                        prepared_key_refusals=replacement.prepared_key_refusals,
                        prepared_lineage_deferrals=replacement.prepared_lineage_deferrals,
                        bulk_fts=True,
                        # One publication is one component of a pass: each
                        # replayed session proves its own FTS rows, and the
                        # archive-wide audit belongs to the daemon's
                        # fts_readiness_binding stage after the burst.
                        exact_fts_audit=False,
                    )
                except RetainedPreparationRetryableError as failure:
                    if publication_failure is not None:
                        publication_failure(failure)
                    return False
                replay_receipt.stage_timings_s["provider_parse"] = (
                    replay_receipt.stage_timings_s.get("provider_parse", 0.0) + replacement.provider_parse_seconds
                )
                if phase_receipt is not None:
                    phase_receipt("replay", replay_receipt)
                if replay_receipt.changed_session_ids:
                    # Searches answered before this publication are cached by
                    # epoch; a changed session must never be served stale.
                    from polylogue.storage.search.cache import invalidate_search_cache

                    invalidate_search_cache()
                refusal = next(iter(self.terminal_decode_refusals(replacement.raw_ids).values()), None)
                if refusal is not None:
                    raise refusal
                if replacement.prepared_lineage_deferrals:
                    # The parents published; their deferred children are
                    # prepared against them by the next phase of this pass.
                    self._phase_committed[id(replacement)] = replacement.key
                    return False
                return True
            finally:
                if lifetime_bound:
                    primary = sys.exception()
                    try:
                        replacement.close()
                    except BaseException as cleanup:
                        if primary is not None:
                            raise BaseExceptionGroup(
                                "retained publication and cleanup failed", [primary, cleanup]
                            ) from primary
                        raise
                else:
                    primary = sys.exception()
                    failures: list[BaseException] = []
                    for close in (replacement.close, lease.close):
                        try:
                            close()
                        except BaseException as failure:
                            failures.append(failure)
                    if failures:
                        if primary is not None:
                            failures.insert(0, primary)
                        raise BaseExceptionGroup("retained publication admission cleanup failed", failures)


def _close_prepared_carriers(
    writes: Mapping[tuple[str, str], PreparedSessionWrite],
    plans: Mapping[str, PreparedMembershipReplay],
    artifacts: Iterable[PreparedJsonl] = (),
) -> None:
    """Attempt each owner close before deleting their containing scratch tree."""
    failures: list[BaseException] = []
    carriers: Iterator[SQLCustodyOwner] = chain(writes.values(), plans.values())
    for carrier in carriers:
        try:
            carrier.close()
        except BaseException as exc:
            failures.append(exc)
    seen: set[int] = set()
    for artifact in artifacts:
        if id(artifact) in seen:
            continue
        seen.add(id(artifact))
        try:
            artifact.discard()
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
    """Stop an owned retained-preparation phase without surrendering cleanup."""
    if compute_cancel_requested():
        raise DaemonOperationCancelled(f"retained preparation cancelled for {subject}")


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


if TYPE_CHECKING:
    from polylogue.core.sql_settlement import SQLCustodyOwner
    from polylogue.sources.parsers.base import ParsedSession
    from polylogue.sources.prepared_jsonl import PreparedJsonl
    from polylogue.sources.revision_backfill import (
        PreparedMembershipReplay,
        PreparedRetainedAggregate,
        PreparedRetainedInput,
        PreparedRetainedReplaySource,
        PreparedRevisionSourceCensus,
        PreparedRevisionSourceClassification,
        ReplaySchedule,
        RetainedArtifactPreparer,
    )
    from polylogue.storage.index_generation import IndexGeneration
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.revision_governance import (
        PreparedRevisionAdoption,
        PreparedRevisionReplayOutcome,
    )
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead, PreparedSessionWrite
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
