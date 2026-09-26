"""Read-only derived-model snapshots for real archive rebuild differentials.

The current index DDL is the census authority.  A new ordinary index table
must either be compared here or receive an explicit non-comparison reason;
otherwise the differential fails before it can quietly lose coverage.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import re
import sqlite3
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from polylogue.storage.fts.sql import FTS_INDEXABLE_MESSAGE_COUNT_SQL
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from tests.infra.workload_artifacts import FinishedBuildResourceMeasurement, FinishedBuildResourceProbe

SqlValue = str | int | float | bytes | None
FactRow = tuple[SqlValue, ...]

_CREATE_TABLE = re.compile(r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_CREATE_VIRTUAL_TABLE = re.compile(
    r"CREATE\s+VIRTUAL\s+TABLE\s+IF\s+NOT\s+EXISTS\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE
)

# These tables record maintenance attempts rather than the logical index
# model. Their stable semantic consequences are compared through the current
# revision heads, FTS membership, materialization markers, and debt state.
_NON_COMPARABLE_TABLES: dict[str, str] = {
    "ingest_index_incarnation": "physical index-file identity; each isolated candidate has its own inode/incarnation",
    "messages_fts_identity": "FTS support relation compared through public search and exact membership counts",
    "ingest_marker_witnesses": "live-ingest idempotency receipts are route history, not finished-build output",
    "query_unit_frame_state": "cursor invalidation epoch depends on write-route history",
    "raw_revision_applications": "attempt receipts contain generated decision ids and wall-clock timestamps",
    "schema_identity": "stores a hash of the DDL identity itself, not derived model data",
}

# One explicit entry per comparable DDL table. Empty sets are declarations:
# they make an added table fail this test until its volatility is considered.
_VOLATILE_COLUMNS: dict[str, frozenset[str]] = {
    "action_pairs": frozenset(),
    "attachment_native_ids": frozenset(),
    "attachment_refs": frozenset(),
    "attachments": frozenset(),
    "blocks": frozenset(),
    "delegation_facts": frozenset(),
    "delegation_refresh_scope": frozenset(),
    "derived_refresh_guard": frozenset(),
    "file_edits": frozenset(),
    "messages": frozenset(),
    "messages_fts_readiness_binding": frozenset(),
    "paste_spans": frozenset(),
    "raw_revision_heads": frozenset({"decided_at_ms"}),
    "repo_checkouts": frozenset(),
    "repos": frozenset(),
    "session_agent_policies": frozenset(),
    "session_commits": frozenset(),
    "session_events": frozenset(),
    "session_identity_claims": frozenset(),
    "session_latency_profiles": frozenset({"materialized_at"}),
    "session_links": frozenset({"observed_at_ms", "resolved_at_ms"}),
    "session_model_usage": frozenset(),
    # Completed arms must agree on outstanding profile work and the recipe
    # seed that produced it. A pending demand is not a finished build.
    "session_profile_demand": frozenset(),
    "session_profile_demand_state": frozenset(),
    "session_profiles": frozenset({"materialized_at"}),
    "session_provider_usage_events": frozenset(),
    "session_refs": frozenset(),
    "session_repos": frozenset(),
    "session_summary_bindings": frozenset(),
    "session_tags": frozenset(),
    # (session_id, input_binding, recipe_version): the inputs the canonical
    # usage rollup was computed from and the recipe that computed them. Every
    # value is derived from the session's own usage evidence -- no clock, no
    # attempt id, no generation-local coordinate -- so two builds of the same
    # sealed input must reproduce it exactly.
    "session_usage_rollup_bindings": frozenset(),
    "session_working_dirs": frozenset(),
    "sessions": frozenset(),
    "web_content_constructs": frozenset(),
    # Retained runtime state exports reach these relations as source-derived
    # projections. The replay route must reproduce their complete row sets,
    # including the source receipt that prevents an older export overwriting a
    # newer one. Those timestamps/order are durable source evidence, not
    # run-local clock observations, so they have no volatile exclusions.
    "work_evidence_edges": frozenset(),
    "work_evidence_graphs": frozenset(),
    "work_evidence_nodes": frozenset(),
}

_PUBLIC_VOLATILE_FIELDS = frozenset({"materialized_at", "checked_at", "priced_at"})


@dataclass(frozen=True, slots=True)
class TableProjection:
    columns: tuple[str, ...]
    rows: tuple[FactRow, ...]


@dataclass(frozen=True, slots=True)
class FtsReadiness:
    source_rows: int
    indexed_rows: int
    public_index_count: int
    public_searches: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class DerivedModelSnapshot:
    tables: tuple[tuple[str, TableProjection], ...]
    public_reads: tuple[tuple[str, object], ...]
    fts: FtsReadiness
    open_debt: tuple[FactRow, ...]


@dataclass(frozen=True, slots=True)
class FinishedBuildWorkIdentity:
    """The selected logical work, independent from its production arm."""

    source_identity: str
    code_identity: str
    profile_identity: str

    def __post_init__(self) -> None:
        if not all((self.source_identity, self.code_identity, self.profile_identity)):
            raise ValueError("finished-build work identity requires source, code, and profile identities")


@dataclass(frozen=True, slots=True)
class FinishedBuildRoute:
    """A declared arm bound to the production callable that executed it."""

    variant: str
    callable_identity: str

    def __post_init__(self) -> None:
        if not self.variant or not self.callable_identity:
            raise ValueError("finished-build route requires an arm and production callable identity")
        if not self.callable_identity.startswith("polylogue."):
            raise ValueError("finished-build route must name a Polylogue production callable")

    @classmethod
    def from_production_callable(cls, variant: str, route: Callable[..., object]) -> FinishedBuildRoute:
        module = getattr(route, "__module__", None)
        qualname = getattr(route, "__qualname__", None)
        if not isinstance(module, str) or not isinstance(qualname, str):
            raise ValueError("finished-build route callable has no stable Python identity")
        return cls(variant=variant, callable_identity=f"{module}.{qualname}")


@dataclass(frozen=True, slots=True)
class SealedRawInput:
    """The raw population one finished-build comparison is measured over.

    The identity is read back from the durable source tier rather than
    declared by the builder: an arm that lost, gained, or rewrote a retained
    raw cannot present itself as having run the same work.
    """

    digest: str
    byte_count: int
    raw_count: int

    def __post_init__(self) -> None:
        if not self.digest or self.raw_count < 1 or self.byte_count < 1:
            raise ValueError("a sealed raw input requires a digest and a non-empty population")


def seal_raw_input(archive_root: Path) -> SealedRawInput:
    """Read the durable raw identity/byte manifest without changing it."""
    with closing(
        open_readonly_connection(
            archive_root / "source.db",
            tier=ArchiveTier.SOURCE,
            timeout_class="background-read",
        )
    ) as conn:
        rows = conn.execute("SELECT raw_id, blob_hash FROM raw_sessions ORDER BY raw_id").fetchall()
    digest = hashlib.sha256()
    byte_count = 0
    for raw_id, blob_hash in rows:
        blob_hex = bytes(blob_hash).hex() if isinstance(blob_hash, bytes) else str(blob_hash)
        size = (archive_root / "blob" / blob_hex[:2] / blob_hex[2:]).stat().st_size
        byte_count += size
        digest.update(f"{raw_id}:{blob_hex}:{size}\n".encode())
    return SealedRawInput(digest=digest.hexdigest(), byte_count=byte_count, raw_count=len(rows))


def clone_sealed_arm(template: Path, destination: Path, sealed: SealedRawInput) -> Path:
    """Clone one sealed source tree into an isolated arm carrying that input."""
    from tests.infra.archive_templates import clone_archive_template

    clone_archive_template(template, destination)
    cloned = seal_raw_input(destination)
    if cloned != sealed:
        raise AssertionError(f"cloned finished-build arm does not carry the sealed input: {cloned} != {sealed}")
    return destination


def finished_build_work_identity(
    sealed: SealedRawInput,
    *,
    profile: str,
    routes: Sequence[Callable[..., object] | type],
) -> FinishedBuildWorkIdentity:
    """Bind one sealed input, the exact route code, and one selected profile.

    Every arm of a comparison takes the SAME identity: the work is what is
    being held fixed, and each arm's own production callable is recorded
    separately in its :class:`FinishedBuildRoute`.
    """
    if not routes:
        raise ValueError("finished-build code identity requires at least one production route")
    code_digest = hashlib.sha256("".join(inspect.getsource(route) for route in routes).encode()).hexdigest()
    return FinishedBuildWorkIdentity(
        source_identity=f"sha256:{sealed.digest}",
        code_identity=f"sha256:{code_digest}",
        profile_identity=profile,
    )


@dataclass(frozen=True, slots=True)
class FinishedBuildOutput:
    """A completed, read-through output snapshot for one production arm."""

    work: FinishedBuildWorkIdentity
    route: FinishedBuildRoute
    canonical_logical_digest: str
    schema_object_census: tuple[tuple[str, str], ...]
    schema_identity: str
    output_session_count: int
    output_message_count: int
    output_block_count: int
    resources: FinishedBuildResourceMeasurement
    snapshot: DerivedModelSnapshot

    def __post_init__(self) -> None:
        if not self.canonical_logical_digest or not self.schema_identity:
            raise ValueError("finished-build output requires canonical and schema identities")
        if min(self.output_session_count, self.output_message_count, self.output_block_count) < 0:
            raise ValueError("finished-build output counts cannot be negative")


@dataclass(frozen=True, slots=True)
class StreamedFinishedBuildFingerprint:
    """Bounded-memory canonical identity for a completed archive generation."""

    canonical_logical_digest: str
    schema_object_census: tuple[tuple[str, str], ...]
    schema_identity: str
    output_session_count: int
    output_message_count: int
    output_block_count: int
    fts_source_rows: int
    fts_indexed_rows: int
    public_index_count: int


def compared_table_census() -> tuple[str, ...]:
    """Return all ordinary current-DDL index tables with a declared policy."""
    tables = frozenset(_CREATE_TABLE.findall(INDEX_DDL))
    virtual_tables = frozenset(_CREATE_VIRTUAL_TABLE.findall(INDEX_DDL))
    if virtual_tables != {"messages_fts"}:
        raise AssertionError(f"unclassified virtual index tables: {sorted(virtual_tables)}")
    classified = set(_VOLATILE_COLUMNS) | set(_NON_COMPARABLE_TABLES)
    if missing := tables - classified:
        raise AssertionError(f"current index DDL tables lack differential classification: {sorted(missing)}")
    if stale := classified - tables:
        raise AssertionError(f"differential table declarations no longer exist in index DDL: {sorted(stale)}")
    return tuple(sorted(tables - set(_NON_COMPARABLE_TABLES)))


def snapshot_derived_model(
    archive_root: Path,
    index_path: Path,
    *,
    session_ids: tuple[str, ...],
    search_queries: tuple[str, ...],
    include_threads: bool = True,
) -> DerivedModelSnapshot:
    """Read one archive generation without mutating any archive tier."""
    census = compared_table_census()
    with _connect(index_path) as conn:
        tables = tuple((table, _project_table(conn, table)) for table in census)
        source_rows = int(conn.execute(FTS_INDEXABLE_MESSAGE_COUNT_SQL).fetchone()[0])
        indexed_rows = int(conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0])
    with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
        public_reads = _public_reads(archive, session_ids, include_threads=include_threads)
        searches = tuple((query, tuple(archive.search_blocks(query))) for query in search_queries)
        public_index_count = int(archive.index_status()["count"])
    return DerivedModelSnapshot(
        tables=tables,
        public_reads=public_reads,
        fts=FtsReadiness(
            source_rows=source_rows,
            indexed_rows=indexed_rows,
            public_index_count=public_index_count,
            public_searches=searches,
        ),
        open_debt=_open_debt_rows(archive_root / "ops.db"),
    )


def capture_streamed_finished_build_fingerprint(
    archive_root: Path,
    index_path: Path,
    *,
    scratch_root: Path,
    session_ids: tuple[str, ...],
    search_queries: tuple[str, ...] = (),
    include_threads: bool = False,
) -> StreamedFinishedBuildFingerprint:
    """Hash a finished index without retaining its rows in memory.

    Row ordering, volatile-column exclusions, schema coverage and canonical
    JSON match :func:`_canonical_logical_digest`. A disk-backed SQLite sorter
    keeps the cost proportional to output bytes on scratch storage rather than
    retaining every index row in the supervisor. ``include_threads`` is for
    small differential fixtures; production qualification leaves it false
    because thread listings are archive-wide.
    """
    census = compared_table_census()
    read_conn = _connect(index_path)
    spool_dir = tempfile.TemporaryDirectory(prefix="finished-build-fingerprint-", dir=scratch_root)
    try:
        spool_path = Path(spool_dir.name) / "rows.sqlite"
        with sqlite3.connect(spool_path) as spool:
            spool.execute("PRAGMA journal_mode=OFF")
            spool.execute("PRAGMA temp_store=FILE")
            spool.execute("PRAGMA cache_size=-8192")
            spool.execute("PRAGMA synchronous=OFF")
            spool.execute(
                "CREATE TABLE row_facts (table_name TEXT NOT NULL, sort_key TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            for table in census:
                volatile = _VOLATILE_COLUMNS[table]
                table_info = tuple(read_conn.execute(f'PRAGMA table_xinfo("{table}")'))
                actual_columns = {str(row["name"]) for row in table_info}
                if unknown := volatile - actual_columns:
                    raise AssertionError(f"volatile declaration for {table} names missing columns: {sorted(unknown)}")
                columns = tuple(str(row["name"]) for row in table_info if str(row["name"]) not in volatile)
                selected_columns = ", ".join(f'"{column}"' for column in columns)
                cursor = read_conn.execute(f'SELECT {selected_columns} FROM "{table}"')
                for row in cursor:
                    fact = _fact_row(row)
                    spool.execute(
                        "INSERT INTO row_facts(table_name, sort_key, payload) VALUES (?, ?, ?)",
                        (
                            table,
                            repr(fact),
                            json.dumps(fact, ensure_ascii=True, separators=(",", ":")),
                        ),
                    )
            spool.execute("CREATE INDEX row_facts_order ON row_facts(table_name, sort_key, payload)")

            fts_source_rows = int(read_conn.execute(FTS_INDEXABLE_MESSAGE_COUNT_SQL).fetchone()[0])
            fts_indexed_rows = int(read_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0])
            output_session_count, output_message_count, output_block_count = (
                int(read_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in ("sessions", "messages", "blocks")
            )
            if fts_source_rows != fts_indexed_rows:
                raise AssertionError(
                    f"finished fingerprint requires exact FTS readiness: source={fts_source_rows}, "
                    f"indexed={fts_indexed_rows}"
                )
            schema_object_census, schema_identity = _finished_schema_census(index_path)
            with closing(
                open_readonly_connection(
                    archive_root / "ops.db",
                    tier=ArchiveTier.OPS,
                    timeout_class="background-read",
                )
            ) as ops:
                debt_table = ops.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='convergence_debt'"
                ).fetchone()
                open_debt = (
                    int(ops.execute("SELECT COUNT(*) FROM convergence_debt WHERE status != 'resolved'").fetchone()[0])
                    if debt_table is not None
                    else 0
                )
            if open_debt:
                raise AssertionError(f"finished fingerprint requires zero convergence debt; found {open_debt}")

            with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
                public_reads = _public_reads(archive, session_ids, include_threads=include_threads)
                public_searches = tuple((query, tuple(archive.search_blocks(query))) for query in search_queries)
                public_index_count = int(archive.index_status()["count"])
            fts_payload = {
                "indexed_rows": fts_indexed_rows,
                "public_index_count": public_index_count,
                "public_searches": public_searches,
                "source_rows": fts_source_rows,
            }
            digest = hashlib.sha256()
            digest.update(
                b'{"fts":'
                + _canonical_json(fts_payload)
                + b',"open_debt":[],"public_reads":'
                + _canonical_json(public_reads)
                + b',"tables":['
            )
            for table_index, table in enumerate(census):
                if table_index:
                    digest.update(b",")
                volatile = _VOLATILE_COLUMNS[table]
                columns = tuple(
                    str(row["name"])
                    for row in read_conn.execute(f'PRAGMA table_xinfo("{table}")')
                    if str(row["name"]) not in volatile
                )
                digest.update(b"[" + _canonical_json(table) + b',{"columns":' + _canonical_json(columns) + b',"rows":[')
                first = True
                for (payload,) in spool.execute(
                    "SELECT payload FROM row_facts WHERE table_name = ? ORDER BY sort_key, payload", (table,)
                ):
                    if not first:
                        digest.update(b",")
                    digest.update(str(payload).encode("ascii"))
                    first = False
                digest.update(b"]}]")
            digest.update(b"]}")
            return StreamedFinishedBuildFingerprint(
                canonical_logical_digest=digest.hexdigest(),
                schema_object_census=schema_object_census,
                schema_identity=schema_identity,
                output_session_count=output_session_count,
                output_message_count=output_message_count,
                output_block_count=output_block_count,
                fts_source_rows=fts_source_rows,
                fts_indexed_rows=fts_indexed_rows,
                public_index_count=public_index_count,
            )
    finally:
        read_conn.close()
        spool_dir.cleanup()


def capture_finished_build_output(
    archive_root: Path,
    index_path: Path,
    *,
    work: FinishedBuildWorkIdentity,
    route: FinishedBuildRoute,
    resource_probe: FinishedBuildResourceProbe,
    session_ids: tuple[str, ...],
    search_queries: tuple[str, ...],
) -> FinishedBuildOutput:
    """Capture only an output that has passed every terminal read condition.

    The snapshot uses the regular archive readers and FTS search route. Its
    readiness assertion is intentionally before all receipt values: a caller
    cannot time/release a replay, then hide final FTS work behind a later
    operation while still claiming a completed comparison.
    """
    snapshot = snapshot_derived_model(
        archive_root,
        index_path,
        session_ids=session_ids,
        search_queries=search_queries,
    )
    assert_derived_model_ready(snapshot)
    schema_object_census, schema_identity = _finished_schema_census(index_path)
    with _connect(index_path) as conn:
        output_session_count, output_message_count, output_block_count = (
            int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("sessions", "messages", "blocks")
        )
    resources = resource_probe.finish(archive_root)
    return FinishedBuildOutput(
        work=work,
        route=route,
        canonical_logical_digest=_canonical_logical_digest(snapshot),
        schema_object_census=schema_object_census,
        schema_identity=schema_identity,
        output_session_count=output_session_count,
        output_message_count=output_message_count,
        output_block_count=output_block_count,
        resources=resources,
        snapshot=snapshot,
    )


def assert_derived_models_equivalent(expected: DerivedModelSnapshot, actual: DerivedModelSnapshot) -> None:
    """Fail with the first durable/public differential, named for diagnosis."""
    expected_tables = dict(expected.tables)
    actual_tables = dict(actual.tables)
    if expected_tables.keys() != actual_tables.keys():
        raise AssertionError(
            f"derived-table census differs: expected={sorted(expected_tables)}, actual={sorted(actual_tables)}"
        )
    for table in expected_tables:
        if expected_tables[table] != actual_tables[table]:
            raise AssertionError(
                f"derived table {table} differs: {_table_difference(expected_tables[table], actual_tables[table])}"
            )
    if expected.public_reads != actual.public_reads:
        raise AssertionError(
            f"public insight reads differ: {_value_difference(expected.public_reads, actual.public_reads)}"
        )
    if expected.fts != actual.fts:
        raise AssertionError("FTS membership or public FTS reads differ")
    if expected.open_debt != actual.open_debt:
        raise AssertionError(f"open convergence debt differs: expected={expected.open_debt}, actual={actual.open_debt}")


def assert_finished_builds_equivalent(expected: FinishedBuildOutput, actual: FinishedBuildOutput) -> None:
    """Require same selected work and identical completed logical output.

    Routes are deliberately allowed to differ. That is the comparison being
    made, while each receipt retains its exact production callable and arm
    variant for an audit of what actually ran.
    """
    if expected.work != actual.work:
        raise AssertionError(f"finished-build work identity differs: expected={expected.work}, actual={actual.work}")
    assert_derived_model_ready(expected.snapshot)
    assert_derived_model_ready(actual.snapshot)
    assert_derived_models_equivalent(expected.snapshot, actual.snapshot)
    for field_name in (
        "canonical_logical_digest",
        "schema_object_census",
        "schema_identity",
        "output_session_count",
        "output_message_count",
        "output_block_count",
    ):
        if getattr(expected, field_name) != getattr(actual, field_name):
            raise AssertionError(
                f"finished-build {field_name} differs: "
                f"expected={getattr(expected, field_name)!r}, actual={getattr(actual, field_name)!r}"
            )


def assert_derived_model_ready(snapshot: DerivedModelSnapshot) -> None:
    """Keep a matching but jointly stale generation from passing the lane."""
    if snapshot.fts.source_rows != snapshot.fts.indexed_rows:
        raise AssertionError(
            f"FTS is not ready: source_rows={snapshot.fts.source_rows}, indexed_rows={snapshot.fts.indexed_rows}"
        )
    if snapshot.open_debt:
        raise AssertionError(f"convergence debt remains: {snapshot.open_debt}")


def _finished_schema_census(index_path: Path) -> tuple[tuple[tuple[str, str], ...], str]:
    """Read the declared schema shape after the route's terminal read checks."""
    with _connect(index_path) as conn:
        objects = tuple(
            (str(kind), str(name))
            for kind, name in conn.execute(
                """
                SELECT type, name
                FROM sqlite_master
                WHERE name NOT LIKE 'sqlite_%'
                ORDER BY type, name
                """
            )
        )
        row = conn.execute("SELECT identity FROM schema_identity WHERE tier = 'index'").fetchone()
    if row is None or not str(row[0]):
        raise AssertionError("finished build has no index schema identity")
    return objects, str(row[0])


def _canonical_logical_digest(snapshot: DerivedModelSnapshot) -> str:
    """Hash the sorted logical differential, not a database page image."""

    def default(value: object) -> str:
        if isinstance(value, bytes):
            return value.hex()
        raise TypeError(f"cannot canonically encode {type(value).__name__}")

    payload = json.dumps(asdict(snapshot), default=default, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _canonical_json(value: object) -> bytes:
    def default(item: object) -> str:
        if isinstance(item, bytes):
            return item.hex()
        raise TypeError(f"cannot canonically encode {type(item).__name__}")

    return json.dumps(value, default=default, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def _connect(path: Path) -> sqlite3.Connection:
    conn = open_readonly_connection(path, tier=ArchiveTier.INDEX, timeout_class="background-read")
    conn.row_factory = sqlite3.Row
    return conn


def _project_table(conn: sqlite3.Connection, table: str) -> TableProjection:
    volatile = _VOLATILE_COLUMNS[table]
    actual_columns = {str(row["name"]) for row in conn.execute(f'PRAGMA table_xinfo("{table}")')}
    if unknown := volatile - actual_columns:
        raise AssertionError(f"volatile declaration for {table} names missing columns: {sorted(unknown)}")
    columns = tuple(
        str(row["name"]) for row in conn.execute(f'PRAGMA table_xinfo("{table}")') if str(row["name"]) not in volatile
    )
    quoted = ", ".join(f'"{column}"' for column in columns)
    rows = tuple(sorted((_fact_row(row) for row in conn.execute(f'SELECT {quoted} FROM "{table}"')), key=repr))
    return TableProjection(columns=columns, rows=rows)


def _open_debt_rows(ops_path: Path) -> tuple[FactRow, ...]:
    if not ops_path.exists():
        return ()
    with closing(open_readonly_connection(ops_path, tier=ArchiveTier.OPS, timeout_class="background-read")) as conn:
        row = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'convergence_debt'").fetchone()
        if row is None:
            return ()
        rows = conn.execute(
            """
            SELECT stage, target_type, target_id, status, attempts,
                   last_error, next_retry_at, materializer_version
            FROM convergence_debt
            WHERE status != 'resolved'
            ORDER BY stage, target_type, target_id
            """
        )
        return tuple(tuple(_normalize(value) for value in row) for row in rows)


def _public_reads(
    archive: ArchiveStore,
    session_ids: tuple[str, ...],
    *,
    include_threads: bool = True,
) -> tuple[tuple[str, object], ...]:
    values: list[tuple[str, object]] = []
    for session_id in session_ids:
        values.extend(((f"profile:{session_id}", _freeze_public(archive.get_session_profile_insight(session_id))),))
    if include_threads:
        values.append(("threads", _freeze_public(archive.list_thread_insights(limit=None))))
    return tuple(values)


def _freeze_public(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _freeze_public(dataclasses.asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _freeze_public(model_dump())
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_public(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _PUBLIC_VOLATILE_FIELDS
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(_freeze_public(item) for item in value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _fact_row(row: sqlite3.Row) -> FactRow:
    return tuple(_normalize(value) for value in row)


def _normalize(value: Any) -> SqlValue:
    if isinstance(value, bytes):
        return value.hex()
    if value is None or isinstance(value, (str, int, float)):
        return value
    raise TypeError(f"unsupported SQLite fact value: {type(value)!r}")


def _table_difference(expected: TableProjection, actual: TableProjection) -> str:
    if expected.columns != actual.columns:
        return f"columns expected={expected.columns}, actual={actual.columns}"
    expected_rows = set(expected.rows)
    actual_rows = set(actual.rows)
    return f"only_expected={sorted(expected_rows - actual_rows, key=repr)[:2]!r}, only_actual={sorted(actual_rows - expected_rows, key=repr)[:2]!r}"


def _value_difference(expected: object, actual: object) -> str:
    expected_items = dict(expected) if isinstance(expected, tuple) else {"value": expected}
    actual_items = dict(actual) if isinstance(actual, tuple) else {"value": actual}
    keys = sorted(set(expected_items) | set(actual_items))
    for key in keys:
        if expected_items.get(key) != actual_items.get(key):
            return f"{key}: expected={expected_items.get(key)!r}, actual={actual_items.get(key)!r}"
    return f"expected={expected!r}, actual={actual!r}"
