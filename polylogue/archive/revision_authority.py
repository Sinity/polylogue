"""Durable source-revision evidence and conservative legacy classification."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import TYPE_CHECKING, BinaryIO, Literal

if TYPE_CHECKING:
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner

from polylogue.core.enums import Origin, PolylogueStrEnum, Provider
from polylogue.core.sources import origin_from_provider


class RawRevisionKind(StrEnum):
    FULL = "full"
    APPEND = "append"
    UNKNOWN = "unknown"


class RawRevisionAuthority(PolylogueStrEnum):
    """Typed source-revision authority, validated at durable write boundaries."""

    ASSERTED = "asserted"
    BYTE_PROVEN = "byte_proven"
    QUARANTINED = "quarantined"


BYTE_AUTHORITY_CENSUS_DETAIL = "append fragments are governed by byte revision authority"


def raw_authority_parser_fingerprint() -> str:
    """Return current parser/replay semantics without paying at module import.

    This is derived from executable OriginSpec routes, their recursive parser
    closures, and shared lowering. Keeping the computation lazy avoids parsing
    the whole provider tree for imports that only need durable revision types.
    """
    from polylogue.sources.origin_specs import parser_semantic_authority_fingerprint

    return parser_semantic_authority_fingerprint()


def raw_receipt_order_sql(table_alias: str = "r") -> str:
    """SQL expression ranking a raw by its latest ``raw_payload`` receipt.

    Currency among raws is decided by durable receipt order: ``blob_refs``
    re-records a raw's payload receipt on every observation (``INSERT OR
    REPLACE``), so the receipt's ``rowid`` is a monotonic observation
    sequence. ``raw_sessions.acquired_at_ms`` is the first time the bytes were
    seen, and a live source that goes A -> B -> A re-mints A's content-derived
    raw id, so ordering by it keeps B current; ordering by a receipt's wall
    clock breaks on a clock rollback. The source tier is never ``VACUUM``-ed,
    which is what keeps this implicit ``rowid`` stable; a migration that
    rebuilds ``blob_refs`` must copy ``rowid`` explicitly. A raw with no
    receipt yields NULL and ranks oldest under ``DESC``.
    """
    return f"""(
        SELECT MAX(receipt.rowid) FROM blob_refs AS receipt
        WHERE receipt.ref_id = {table_alias}.raw_id AND receipt.ref_type = 'raw_payload'
    )"""


def decided_unresolved_membership_sql(table_alias: str = "r") -> str:
    """SQL predicate for a raw whose membership arbitration concluded unresolved.

    Census complete, every membership row arbitrated, the verdict
    ``ambiguous``/``deferred``, and the raw durably quarantined: replay is
    fail-closed and only new bytes or new evidence can change it, so the
    passage of time and re-reading the same source bytes cannot. Consumers
    that ask "is there still pending work here?" must read this as a decided
    outcome rather than as an unfinished parse (which ``parsed_at_ms IS NULL``
    alone looks like).
    """
    return f"""
        {table_alias}.revision_authority = 'quarantined'
        AND EXISTS (
            SELECT 1 FROM raw_membership_census AS decided_census
            WHERE decided_census.raw_id = {table_alias}.raw_id
              AND decided_census.status = 'complete'
        )
        AND EXISTS (
            SELECT 1 FROM raw_session_memberships AS decided_membership
            WHERE decided_membership.raw_id = {table_alias}.raw_id
              AND decided_membership.decision IN ('ambiguous', 'deferred')
        )
        AND NOT EXISTS (
            SELECT 1 FROM raw_session_memberships AS pending_membership
            WHERE pending_membership.raw_id = {table_alias}.raw_id
              AND pending_membership.decision IS NULL
        )
    """


#: ``raw_id`` prefix of a retained agent work event (``ArchiveStore.admit_work_event``).
#: A work event is its own logical source: its raw id is its logical key, so it
#: never joins the byte-revision cohort of the transcript it annotates.
WORK_EVENT_RAW_ID_PREFIX = "agent-work-event:"


def is_work_event_raw_id(raw_id: str | None) -> bool:
    return raw_id is not None and raw_id.startswith(WORK_EVENT_RAW_ID_PREFIX)


def canonical_authority_logical_key(logical_key: str) -> str:
    """Normalize a provider or public-origin authority key to public origin form.

    A work-event key names one retained event rather than a session, so it has
    no origin form and is already canonical.
    """
    if is_work_event_raw_id(logical_key) and len(logical_key) > len(WORK_EVENT_RAW_ID_PREFIX):
        return logical_key
    prefix, separator, native_id = logical_key.partition(":")
    if not separator or not native_id:
        raise ValueError(f"invalid logical source key: {logical_key!r}")
    try:
        origin = Origin(prefix)
    except ValueError:
        try:
            origin = origin_from_provider(Provider(prefix))
        except ValueError as exc:
            raise ValueError(f"unknown logical source key prefix: {prefix!r}") from exc
    return f"{origin.value}:{native_id}"


def logical_head_cohort_sql(conn: sqlite3.Connection, *, raw_alias: str, has_memberships: bool) -> str:
    """Return the canonical, membership-aware SQL partition key for raw heads."""

    def _canonical_or_original(value: object) -> str | None:
        if value is None:
            return None
        text = str(value)
        try:
            return canonical_authority_logical_key(text)
        except ValueError:
            # A malformed legacy key must remain observable as its own cohort,
            # never make a read-only verifier fail while grouping heads.
            return text

    conn.create_function("canonical_authority_logical_key", 1, _canonical_or_original, deterministic=True)
    membership_key = "NULL"
    if has_memberships:
        membership_key = f"""
            canonical_authority_logical_key(
                (
                    SELECT CASE
                        WHEN COUNT(DISTINCT canonical_authority_logical_key(m.logical_source_key)) = 1
                        THEN MIN(canonical_authority_logical_key(m.logical_source_key))
                    END
                    FROM raw_session_memberships AS m
                    WHERE m.raw_id = {raw_alias}.raw_id
                )
            )
        """
    return (
        f"COALESCE(canonical_authority_logical_key({raw_alias}.logical_source_key), "
        f"{membership_key}, {raw_alias}.native_id, {raw_alias}.source_path)"
    )


def parser_census_identity_is_complete(
    *,
    durable_valid: bool,
    observed_valid: bool,
    identities_match: bool,
    observed_count: int,
    typed_non_session: bool,
    parser_confirmed_non_session: bool,
    byte_governed_fragment: bool,
) -> bool:
    """One completion law for exact measured producer and retained identity sets."""
    return (
        durable_valid
        and observed_valid
        and identities_match
        and (observed_count > 0 or typed_non_session or parser_confirmed_non_session or byte_governed_fragment)
    )


class InvalidParserCensusKeysError(ValueError):
    """A retained identity receipt cannot establish its declared key set."""


@dataclass(frozen=True, slots=True)
class ParserCensusIdentityMeasurement:
    """Exact identity comparison scoped to its existing Native disk owner."""

    connection: sqlite3.Connection
    durable_valid: bool
    observed_valid: bool
    identities_match: bool
    observed_count: int
    owner: NativeSQLCustodyOwner

    def iter_keys(self, *, observed: bool) -> Generator[str, None, None]:
        with closing(
            self.connection.execute(
                "SELECT logical_key FROM census_identity WHERE kind=? ORDER BY logical_key", (int(not observed),)
            )
        ) as rows:
            for row in rows:
                yield str(row[0])

    def iter_durable_bindings(self) -> Generator[tuple[str, str | None], None, None]:
        """Yield canonical identity and its exact captured membership spelling."""
        with closing(
            self.connection.execute(
                "SELECT logical_key, source_key FROM census_identity WHERE kind=1 ORDER BY logical_key"
            )
        ) as rows:
            for key, source_key in rows:
                yield str(key), None if source_key is None else str(source_key)

    def complete(
        self,
        *,
        typed_non_session: bool,
        parser_confirmed_non_session: bool,
        byte_governed_fragment: bool,
    ) -> bool:
        return parser_census_identity_is_complete(
            durable_valid=self.durable_valid,
            observed_valid=self.observed_valid,
            identities_match=self.identities_match,
            observed_count=self.observed_count,
            typed_non_session=typed_non_session,
            parser_confirmed_non_session=parser_confirmed_non_session,
            byte_governed_fragment=byte_governed_fragment,
        )

    def _stdlib_keys_json_chunks(self, check_stop: Callable[[], None]) -> Generator[bytes, None, None]:
        """Encode each exact key through its original readonly TEXT handle."""
        import codecs
        import json

        check_stop()
        yield b"["
        first = True
        with closing(
            self.connection.execute("SELECT rowid, occurrences FROM census_identity WHERE kind=0 ORDER BY logical_key")
        ) as rows:
            for rowid, occurrences in rows:
                check_stop()
                for _ in range(occurrences):
                    check_stop()
                    if not first:
                        yield b", "
                    yield b'"'
                    with self.owner.readonly_blob("census_identity", "logical_key", rowid) as blob:
                        decoder = codecs.getincrementaldecoder("utf-8")()
                        remaining = len(blob)
                        while remaining:
                            check_stop()
                            chunk = blob.read(min(4096, remaining))
                            if not chunk:
                                raise ValueError("parser census key ended before its retained byte length")
                            remaining -= len(chunk)
                            text = decoder.decode(chunk, final=remaining == 0)
                            # The stdlib string encoder preserves its exact
                            # ASCII/control/non-BMP spelling. Incremental UTF8
                            # decoding never splits a Unicode code point.
                            yield json.dumps(text)[1:-1].encode("ascii")
                    yield b'"'
                    first = False
        check_stop()
        yield b"]"

    @contextmanager
    def keys_json_stream(
        self, *, sqlite_encoding: bool, check_stop: Callable[[], None] | None = None
    ) -> Iterator[tuple[int, Generator[bytes, None, None]]]:
        """Lend exact receipt bytes while their original measurement owner is live.

        SQLite's aggregate remains native and may allocate its complete value;
        it is never fetched into Python. The stdlib route repeats its existing
        occurrence spelling, measuring and delivering on this same snapshot.
        """
        from polylogue.core.compute_cancel import check_compute_cancelled

        def check() -> None:
            check_compute_cancelled()
            if check_stop is not None:
                check_stop()

        if self.owner.require_connection() is not self.connection:
            raise RuntimeError("parser census stream differs from its original native owner")
        check()
        if not sqlite_encoding:
            with closing(self._stdlib_keys_json_chunks(check)) as measured:
                byte_length = sum(len(chunk) for chunk in measured)
            with closing(self._stdlib_keys_json_chunks(check)) as chunks:
                yield byte_length, chunks
            return

        # This is the original SQLite JSON aggregate, including ordered
        # distinct keys and its exact Unicode/control-character spelling.
        # Keep the scalar on the same private measurement connection.
        stopped: BaseException | None = None

        def progress() -> int:
            nonlocal stopped
            try:
                check()
            except BaseException as error:
                stopped = error
                return 1
            return 0

        self.connection.set_progress_handler(progress, 1000)
        try:
            with closing(self.connection.execute("DELETE FROM census_encoded_keys")):
                pass
            with closing(
                self.connection.execute(
                    "INSERT INTO census_encoded_keys(id,value) SELECT 1,json_group_array(logical_key) FROM ("
                    "SELECT logical_key FROM census_identity WHERE kind=0 ORDER BY logical_key)"
                )
            ):
                pass
        except sqlite3.OperationalError as interrupted:
            if stopped is not None:
                raise stopped from interrupted
            raise
        finally:
            self.connection.set_progress_handler(None, 0)
        check()
        with self.owner.readonly_blob("census_encoded_keys", "value", 1) as blob:
            byte_length = len(blob)

            def read_chunks() -> Generator[bytes, None, None]:
                remaining = byte_length
                while remaining:
                    check()
                    chunk = blob.read(min(65536, remaining))
                    if not chunk:
                        raise ValueError("parser census JSON ended before its measured byte length")
                    remaining -= len(chunk)
                    yield chunk
                check()

            with closing(read_chunks()) as chunks:
                yield byte_length, chunks


@contextmanager
def parser_census_identity_measurement(
    *,
    raw_logical_key: object,
    revision_kind: object,
    membership_logical_keys: Iterable[object],
    observed_logical_keys: Iterable[str] | None,
    observed_are_receipt: bool = False,
    inherit_durable_keys: bool = False,
    check_stop: Callable[[], None] | None = None,
) -> Iterator[ParserCensusIdentityMeasurement]:
    """Measure producer or retained receipt keys under the same identity law.

    Inputs come from the caller's admitted Source snapshot. The scratch owner
    never reopens that snapshot; its regular indexed table lives on disk.
    """
    from polylogue.storage.sqlite.connection_profile import (
        retained_native_sql_owners_on_current_thread,
        scratch_connection_context,
    )

    if observed_logical_keys is not None and inherit_durable_keys:
        raise ValueError("parser census cannot combine observed and inherited durable keys")
    with scratch_connection_context(prefix="polylogue-parser-census-", filename="identities.sqlite") as scratch:
        owners = tuple(owner for owner in retained_native_sql_owners_on_current_thread() if owner.connection is scratch)
        if len(owners) != 1:
            raise RuntimeError("parser census measurement requires its single original native owner")
        owner = owners[0]
        scratch.execute("PRAGMA cache_size=-2048").close()
        scratch.execute("PRAGMA journal_mode=DELETE").close()
        scratch.execute("PRAGMA temp_store=FILE").close()
        scratch.execute("BEGIN").close()
        scratch.execute(
            "CREATE TABLE census_identity(kind INTEGER NOT NULL, logical_key TEXT NOT NULL, source_key TEXT, "
            "occurrences INTEGER NOT NULL DEFAULT 1, "
            "PRIMARY KEY(kind, logical_key))"
        ).close()
        scratch.execute("CREATE TABLE census_encoded_keys(id INTEGER PRIMARY KEY, value TEXT NOT NULL)").close()
        durable_valid = True
        for value in membership_logical_keys:
            if check_stop is not None:
                check_stop()
            if value is None:
                continue
            try:
                key = canonical_authority_logical_key(str(value))
            except ValueError:
                durable_valid = False
                continue
            scratch.execute(
                "INSERT OR IGNORE INTO census_identity(kind, logical_key, source_key) VALUES (1, ?, ?)",
                (key, str(value)),
            ).close()
        if raw_logical_key is not None and str(revision_kind) != RawRevisionKind.UNKNOWN.value:
            typed_key = str(raw_logical_key)
            if not typed_key.startswith("pending-raw:"):
                try:
                    key = canonical_authority_logical_key(typed_key)
                except ValueError:
                    durable_valid = False
                else:
                    scratch.execute(
                        "INSERT OR IGNORE INTO census_identity(kind, logical_key) VALUES (1, ?)", (key,)
                    ).close()
        observed_valid = observed_logical_keys is not None or inherit_durable_keys
        if inherit_durable_keys and durable_valid:
            scratch.execute(
                "INSERT INTO census_identity(kind, logical_key) SELECT 0, logical_key FROM census_identity WHERE kind=1"
            ).close()
        elif observed_logical_keys is not None:
            observed = iter(observed_logical_keys)
            try:
                while True:
                    if check_stop is not None:
                        check_stop()
                    try:
                        value = next(observed)
                    except StopIteration:
                        break
                    except InvalidParserCensusKeysError:
                        observed_valid = False
                        break
                    key = canonical_authority_logical_key(value)
                    with closing(
                        scratch.execute(
                            "INSERT OR IGNORE INTO census_identity(kind, logical_key) VALUES (0, ?)", (key,)
                        )
                    ) as inserted:
                        if observed_are_receipt and not inserted.rowcount:
                            observed_valid = False
                            scratch.execute(
                                "UPDATE census_identity SET occurrences=occurrences+1 WHERE kind=0 AND logical_key=?",
                                (key,),
                            ).close()
            finally:
                close_observed = getattr(observed, "close", None)
                if callable(close_observed):
                    close_observed()
        with closing(scratch.execute("SELECT COUNT(*) FROM census_identity WHERE kind=0")) as counted:
            observed_count = int(counted.fetchone()[0])
        with closing(
            scratch.execute(
                "SELECT 1 FROM census_identity AS observed WHERE observed.kind=0 AND NOT EXISTS ("
                "SELECT 1 FROM census_identity AS durable WHERE durable.kind=1 AND durable.logical_key=observed.logical_key) "
                "UNION ALL SELECT 1 FROM census_identity AS durable WHERE durable.kind=1 AND NOT EXISTS ("
                "SELECT 1 FROM census_identity AS observed WHERE observed.kind=0 AND observed.logical_key=durable.logical_key) LIMIT 1"
            )
        ) as compared:
            differs = compared.fetchone()
        yield ParserCensusIdentityMeasurement(
            scratch, durable_valid, observed_valid, differs is None, observed_count, owner
        )


#: ``raw_membership_census.detail`` marker written when a full-only,
#: non-prefix-chain cohort is retired from byte-revision governance to
#: membership governance (``ArchiveStore.replace_raw_membership_census``,
#: ``retire_full_revision_governance=True``). The write boundary translates it
#: to the typed ``quarantined`` census authority that the polylogue-52l2 guard
#: reads to detect that a logical identity already has retired,
#: previously-ambiguous sibling evidence before ever accepting a
#: later-discovered raw as an unconditional singleton byte-proven baseline.
HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL = "historical non-prefix full revision governance"
#: A typed full revision whose stored identity the current parser no longer
#: derives retires to membership governance under the parsed identity.
SUPERSEDED_IDENTITY_GOVERNANCE_DETAIL = "superseded full revision identity"


@dataclass(frozen=True)
class RawRevisionEnvelope:
    """Evidence captured with raw bytes, before any derived write occurs."""

    logical_source_key: str
    kind: RawRevisionKind
    source_revision: str
    acquisition_generation: int
    predecessor_source_revision: str | None = None
    predecessor_raw_id: str | None = None
    baseline_raw_id: str | None = None
    append_start_offset: int | None = None
    append_end_offset: int | None = None
    authority: RawRevisionAuthority = RawRevisionAuthority.ASSERTED

    def __post_init__(self) -> None:
        if not self.logical_source_key or not self.source_revision:
            raise ValueError("revision envelope identity must be non-empty")
        if self.acquisition_generation < 0:
            raise ValueError("acquisition_generation must be non-negative")
        offsets = (self.append_start_offset, self.append_end_offset)
        if self.kind is RawRevisionKind.APPEND:
            if self.predecessor_source_revision is None or None in offsets:
                raise ValueError("append evidence requires predecessor revision and offsets")
            assert self.append_start_offset is not None and self.append_end_offset is not None
            if self.append_start_offset < 0 or self.append_end_offset <= self.append_start_offset:
                raise ValueError("append offsets must describe a non-empty forward range")
            raw_predecessors = (self.predecessor_raw_id, self.baseline_raw_id)
            if self.authority is RawRevisionAuthority.QUARANTINED:
                if any(value is not None for value in raw_predecessors):
                    raise ValueError("quarantined append evidence may not claim raw predecessors")
            elif any(value is None for value in raw_predecessors):
                raise ValueError("replay-eligible append evidence requires baseline and raw predecessor")
        elif self.predecessor_source_revision is not None or any(value is not None for value in offsets):
            raise ValueError("only append evidence may carry predecessor revision or byte offsets")


@dataclass(frozen=True)
class HistoricalRawRevision:
    raw_id: str
    payload: bytes


@dataclass(frozen=True)
class HistoricalRawRevisionStream:
    """A retained full revision whose bytes can be compared without loading it."""

    raw_id: str
    payload_size: int
    open_payload: Callable[[], BinaryIO]


@dataclass(frozen=True)
class HistoricalRevisionDecision:
    raw_id: str
    authority: RawRevisionAuthority
    relation: Literal["baseline", "predecessor", "ambiguous", "duplicate"]
    predecessor_raw_id: str | None = None
    #: For ``relation="duplicate"`` only: the representative raw_id whose
    #: verdict (and, for callers that derive chain position such as
    #: ``revision_governance.classify_raw_revision_cohort``, generation
    #: number) this duplicate mirrors. Deliberately separate from
    #: ``predecessor_raw_id`` -- a duplicate is NOT a chain-continuing child
    #: of anything; it is a second copy of its representative's own bytes.
    #: Mirroring the representative's ``predecessor_raw_id`` onto the
    #: duplicate instead (the naive fix CodeRabbit flagged on #3574) would
    #: make the duplicate and its representative collide on the same
    #: predecessor-keyed dict entry in a plain-dict generation walk, risking
    #: silently dropping the real chain-continuing representative. This
    #: field lets a caller copy the representative's already-computed
    #: generation in a separate post-pass instead of participating in the
    #: predecessor-keyed walk at all (polylogue-5unky).
    duplicate_of_raw_id: str | None = None


def append_source_revision(predecessor_revision: str, payload_hash: str) -> str:
    """Return the exact content fingerprint committed by the append cursor."""
    return sha256(f"{predecessor_revision}\0{payload_hash}".encode()).hexdigest()


def _classify_deduped_nodes(
    node_ids: list[str],
    sizes: dict[str, int],
    is_prefix: Callable[[str, str], bool],
) -> dict[str, HistoricalRevisionDecision]:
    """Prove a byte-prefix DAG over already-deduplicated (distinct-content) nodes.

    Implements I4/I5 (polylogue-lb39z): a divergence quarantines only the
    genuinely-divergent suffix, never a proven prefix chain or an unrelated
    sibling that happens to share a cohort with it.

    ``is_prefix(parent, child)`` must only be evaluated for pairs where
    ``sizes[parent] < sizes[child]`` (the caller guarantees this so streamed
    implementations can skip same-size/reverse-size comparisons entirely).

    A node is only ever classified ``BYTE_PROVEN`` if it sits on a single,
    unbranched path back to the cohort's one true root: every ancestor,
    including the node itself, has exactly one maximal parent candidate and
    is the sole child of that parent. The moment a node's parent has more
    than one child (a fork) or a node has more than one incomparable maximal
    parent (an ambiguous parent set), that node -- and everything built on
    top of it -- is quarantined, while everything already proven up to that
    point (including the fork point itself) is left alone. If the cohort has
    zero or more than one root (no shared ancestor at all, e.g. two
    unrelated same-size captures), there is no anchor to localize from and
    the whole cohort is quarantined, matching prior behavior for that case.
    """
    ordered = sorted(node_ids, key=lambda raw_id: (sizes[raw_id], raw_id))
    parents: dict[str, list[str]] = {}
    children: dict[str, list[str]] = {raw_id: [] for raw_id in ordered}
    # Prefix is transitive: if parent1 and parent2 are both prefixes of child
    # and sizes[parent1] < sizes[parent2], then parent1 is necessarily a
    # prefix of parent2 (parent1's bytes equal child's first N bytes, which
    # equal parent2's first N bytes since parent2 is itself a prefix of child
    # of length >= N). So a node's prefix ancestors form a totally ordered
    # chain and its maximal parent is the deepest prefix ancestor -- the
    # all-pairs scan that used to find it is never needed.
    #
    # Nodes are placed in ascending size order into a growing prefix forest,
    # so every possible parent of the node being placed is already in it.
    # Leaves of that forest are pairwise incomparable (if leaf X were a prefix
    # of leaf Y then Y's maximal parent would be at or below X, giving X a
    # child), so at most one leaf can match and a hit there is immediately the
    # deepest ancestor. That is the whole cost for an incrementally-growing
    # cohort: one comparison per node. Only when no leaf matches does the
    # search walk down from the roots, paying the depth of the branch it
    # descends -- and at most one child per level can match, for the same
    # incomparability reason.
    roots: list[str] = []
    leaves: list[str] = []
    for child in ordered:
        child_size = sizes[child]
        verdicts: dict[str, bool] = {}

        def matches(
            candidate: str, *, child: str = child, child_size: int = child_size, verdicts: dict[str, bool] = verdicts
        ) -> bool:
            if sizes[candidate] >= child_size:
                return False
            cached = verdicts.get(candidate)
            if cached is None:
                cached = is_prefix(candidate, child)
                verdicts[candidate] = cached
            return cached

        deepest: str | None = None
        # Largest leaf first: the tail of a growing chain is the likely parent.
        for leaf in sorted(leaves, key=lambda raw_id: (-sizes[raw_id], raw_id)):
            if matches(leaf):
                deepest = leaf
                break
        if deepest is None:
            frontier = roots
            while True:
                step = next((node for node in frontier if matches(node)), None)
                if step is None:
                    break
                deepest = step
                frontier = children[step]

        parents[child] = [] if deepest is None else [deepest]
        if deepest is None:
            roots.append(child)
        else:
            if not children[deepest]:
                leaves.remove(deepest)
            children[deepest].append(child)
        leaves.append(child)

    roots = [raw_id for raw_id in ordered if not parents[raw_id]]
    if len(roots) != 1:
        return {
            raw_id: HistoricalRevisionDecision(
                raw_id=raw_id, authority=RawRevisionAuthority.QUARANTINED, relation="ambiguous"
            )
            for raw_id in ordered
        }
    root = roots[0]
    clean: dict[str, bool] = {root: True}
    for raw_id in ordered:
        if raw_id == root:
            continue
        parent_set = parents[raw_id]
        if len(parent_set) != 1:
            clean[raw_id] = False
            continue
        (parent,) = parent_set
        clean[raw_id] = clean.get(parent, False) and len(children[parent]) == 1

    decisions: dict[str, HistoricalRevisionDecision] = {}
    for raw_id in ordered:
        if clean.get(raw_id, False):
            parent_set = parents[raw_id]
            predecessor = parent_set[0] if parent_set else None
            relation: Literal["baseline", "predecessor"] = "baseline" if predecessor is None else "predecessor"
            decisions[raw_id] = HistoricalRevisionDecision(
                raw_id=raw_id,
                authority=RawRevisionAuthority.BYTE_PROVEN,
                relation=relation,
                predecessor_raw_id=predecessor,
            )
        else:
            decisions[raw_id] = HistoricalRevisionDecision(
                raw_id=raw_id, authority=RawRevisionAuthority.QUARANTINED, relation="ambiguous"
            )
    return decisions


def classify_historical_full_revisions(
    revisions: list[HistoricalRawRevision],
) -> list[HistoricalRevisionDecision]:
    """Prove a unique byte-prefix chain; quarantine only the divergent suffix.

    Acquisition time, source path, provider timestamps, and raw-id ordering are
    intentionally absent. Equal or divergent payloads do not establish which
    capture is newer.

    I4 (polylogue-lb39z): byte-identical captures of one source are one
    evidence node observed twice, not competing revisions -- they are
    collapsed onto a single representative (the lexicographically smallest
    ``raw_id``) before the chain proof runs, and every non-representative
    duplicate mirrors its representative's verdict with ``relation=
    "duplicate"``. Output order matches input size order (ascending, ties
    broken by ``raw_id``): the first is oldest and the last is the head.
    """
    if not revisions:
        return []
    groups: dict[bytes, list[HistoricalRawRevision]] = {}
    for revision in revisions:
        groups.setdefault(revision.payload, []).append(revision)
    representative_payload: dict[str, bytes] = {}
    members_of: dict[str, list[str]] = {}
    for payload, members in groups.items():
        representative = min(members, key=lambda revision: revision.raw_id)
        representative_payload[representative.raw_id] = payload
        members_of[representative.raw_id] = sorted(member.raw_id for member in members)

    def is_prefix(parent: str, child: str) -> bool:
        return representative_payload[child].startswith(representative_payload[parent])

    sizes = {raw_id: len(payload) for raw_id, payload in representative_payload.items()}
    rep_decisions = _classify_deduped_nodes(list(representative_payload), sizes, is_prefix)
    return _expand_duplicate_decisions(rep_decisions, members_of, sizes)


def _expand_duplicate_decisions(
    rep_decisions: dict[str, HistoricalRevisionDecision],
    members_of: dict[str, list[str]],
    rep_sizes: dict[str, int],
) -> list[HistoricalRevisionDecision]:
    size_by_raw_id: dict[str, int] = {}
    decision_by_raw_id: dict[str, HistoricalRevisionDecision] = {}
    for representative_id, member_ids in members_of.items():
        rep_decision = rep_decisions[representative_id]
        decision_by_raw_id[representative_id] = rep_decision
        size_by_raw_id[representative_id] = rep_sizes[representative_id]
        for member_id in member_ids:
            if member_id == representative_id:
                continue
            decision_by_raw_id[member_id] = HistoricalRevisionDecision(
                raw_id=member_id,
                authority=rep_decision.authority,
                relation="duplicate",
                predecessor_raw_id=None,
                duplicate_of_raw_id=representative_id,
            )
            size_by_raw_id[member_id] = rep_sizes[representative_id]
    ordered_ids = sorted(decision_by_raw_id, key=lambda raw_id: (size_by_raw_id[raw_id], raw_id))
    return [decision_by_raw_id[raw_id] for raw_id in ordered_ids]


def _never_compared(parent: str, child: str) -> bool:
    raise AssertionError(f"a single-revision cohort compared {parent} with {child}")


def _stream_size_and_hash(revision: HistoricalRawRevisionStream) -> tuple[int, str]:
    size = 0
    digest = sha256()
    with revision.open_payload() as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _stream_is_prefix(
    parent: HistoricalRawRevisionStream,
    child: HistoricalRawRevisionStream,
    *,
    parent_size: int,
    child_size: int,
) -> bool:
    """Return whether *parent* is an exact proper byte prefix of *child*."""
    if parent_size >= child_size:
        return False
    remaining = parent_size
    with parent.open_payload() as parent_handle, child.open_payload() as child_handle:
        while remaining:
            chunk = parent_handle.read(min(1024 * 1024, remaining))
            if not chunk or child_handle.read(len(chunk)) != chunk:
                return False
            remaining -= len(chunk)
        return parent_handle.read(1) == b""


def classify_historical_full_revision_streams(
    revisions: list[HistoricalRawRevisionStream],
) -> list[HistoricalRevisionDecision]:
    """Stream the same dedup-first, localized-ambiguity proof, without eager payloads.

    In a cohort of two or more, each stream is read exactly once to learn its
    true size and content hash (never trusting the caller-supplied
    ``payload_size`` alone); streams whose
    hash matches are byte-identical duplicates and are collapsed per I4 before
    any prefix comparison runs. Prefix comparisons between distinct-content
    representatives are themselves streamed (``_stream_is_prefix``), so no
    full payload is ever held in memory at once.
    """
    if not revisions:
        return []
    if len(revisions) == 1:
        # A lone revision has no duplicate to collapse and no prefix to
        # prove: it is its cohort's baseline whatever its bytes are, so they
        # are not read. Hashing them anyway re-read every retained file of a
        # fresh build (a 440 MB rollout in full) under the writer hold.
        (only,) = revisions
        sizes = {only.raw_id: only.payload_size}
        return _expand_duplicate_decisions(
            _classify_deduped_nodes([only.raw_id], sizes, _never_compared),
            {only.raw_id: [only.raw_id]},
            sizes,
        )
    size_and_hash = {revision.raw_id: _stream_size_and_hash(revision) for revision in revisions}
    by_hash: dict[str, list[HistoricalRawRevisionStream]] = {}
    for revision in revisions:
        _, digest = size_and_hash[revision.raw_id]
        by_hash.setdefault(digest, []).append(revision)

    representative_stream: dict[str, HistoricalRawRevisionStream] = {}
    representative_size: dict[str, int] = {}
    members_of: dict[str, list[str]] = {}
    for members in by_hash.values():
        representative = min(members, key=lambda revision: revision.raw_id)
        representative_stream[representative.raw_id] = representative
        representative_size[representative.raw_id] = size_and_hash[representative.raw_id][0]
        members_of[representative.raw_id] = sorted(member.raw_id for member in members)

    def is_prefix(parent: str, child: str) -> bool:
        return _stream_is_prefix(
            representative_stream[parent],
            representative_stream[child],
            parent_size=representative_size[parent],
            child_size=representative_size[child],
        )

    rep_decisions = _classify_deduped_nodes(list(representative_stream), representative_size, is_prefix)
    return _expand_duplicate_decisions(rep_decisions, members_of, representative_size)


__all__ = [
    "HistoricalRawRevision",
    "HistoricalRawRevisionStream",
    "HistoricalRevisionDecision",
    "RawRevisionAuthority",
    "RawRevisionEnvelope",
    "RawRevisionKind",
    "append_source_revision",
    "classify_historical_full_revisions",
    "classify_historical_full_revision_streams",
    "raw_authority_parser_fingerprint",
]
