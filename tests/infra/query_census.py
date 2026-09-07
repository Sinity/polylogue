"""Serialized workload census for the declared query families.

The census answers three questions about one query family, with receipts:
what plan SQLite chose (``EXPLAIN QUERY PLAN``: scans and temp B-trees), what
the route physically cost (VM steps, rows, wall/CPU, RSS, swap, temp and
read/write I/O, response bytes), and whether the cheapest correct primitive
returns the same identities for less work.

Safety is structural, not advisory. The census runs against a reflink copy
made by :func:`reflink_archive_snapshot`, never a live archive; one module
lock makes concurrent censuses impossible, so there is never a second dbstat
or EQP walk in flight; and the plan walk uses one read-only connection with a
declared VM-step ceiling.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import sqlite3
import subprocess
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from polylogue.scenarios.workload import (
    MeasurementScope,
    WorkloadEnvelopeSpec,
    WorkloadInputRef,
    WorkloadPhaseObservation,
    WorkloadReceipt,
    WorkloadRunStatus,
)
from tests.infra.query_contract import CENSUS_FAMILIES, CensusFamily

#: Archive tier files a census copy must contain to be usable evidence.
REQUIRED_TIERS: tuple[str, ...] = ("index.db", "source.db", "user.db")

#: Durable tiers the census must leave byte-identical. Derived and disposable
#: tiers may legitimately move: a read updates cursors and telemetry.
IMMUTABLE_TIERS: tuple[str, ...] = ("source.db", "user.db")

#: Ceiling for the plan walk's own reader. Far above any census statement;
#: it exists so a pathological plan aborts instead of walking a live-sized
#: archive without a bound.
CENSUS_READER_VM_STEP_CEILING = 200_000_000

#: Declared phases of one family's census, in order.
CENSUS_PHASES: tuple[str, ...] = ("route", "plan", "cheapest")

_CENSUS_LOCK = threading.Lock()


class CensusConcurrencyError(RuntimeError):
    """Raised when a second census tries to walk an archive in parallel."""


class CensusSnapshotError(RuntimeError):
    """Raised when a census copy is stale, partial, or the live archive."""


class CensusBudgetExceededError(RuntimeError):
    """Raised when the bounded census reader exhausts its VM-step ceiling."""


@dataclass(frozen=True, slots=True)
class ArchiveSnapshot:
    """One reflink copy of an archive, verified usable before it is read."""

    source: Path
    root: Path
    reflinked: bool
    digests: Mapping[str, str]

    @property
    def index_path(self) -> Path:
        return self.root / "index.db"


def _file_digest(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reflink_archive_snapshot(source: Path, destination: Path) -> ArchiveSnapshot:
    """Copy ``source`` to ``destination`` by reflink and verify it is readable.

    A reflink is the point: it costs no space on a copy-on-write filesystem,
    so censusing a live-sized archive never needs a second copy of it. On a
    filesystem without reflinks the copy still happens, and the receipt says
    so rather than silently changing the safety story.
    """

    source = source.resolve()
    destination = destination.resolve()
    if destination == source:
        raise CensusSnapshotError("a census snapshot cannot be the archive it copies")
    if destination.exists():
        raise CensusSnapshotError(f"census destination already exists: {destination}")
    missing = tuple(name for name in REQUIRED_TIERS if not (source / name).exists())
    if missing:
        raise CensusSnapshotError(f"archive {source} is missing tiers {missing}; refusing to census partial state")

    destination.parent.mkdir(parents=True, exist_ok=True)
    reflinked = True
    completed = subprocess.run(
        ["cp", "--reflink=always", "-a", str(source), str(destination)],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        reflinked = False
        shutil.copytree(source, destination)

    for name in REQUIRED_TIERS:
        copied = destination / name
        if not copied.exists():
            raise CensusSnapshotError(f"census copy lost tier {name}")
        with closing(sqlite3.connect(f"file:{copied}?mode=ro", uri=True)) as conn:
            verdict = conn.execute("PRAGMA quick_check(1)").fetchone()
            if verdict is None or str(verdict[0]).lower() != "ok":
                raise CensusSnapshotError(f"census copy of {name} is not consistent: {verdict}")

    digests = {name: _file_digest(destination / name) for name in IMMUTABLE_TIERS}
    return ArchiveSnapshot(source=source, root=destination, reflinked=reflinked, digests=digests)


def assert_snapshot_unmutated(snapshot: ArchiveSnapshot) -> None:
    """Fail if the census wrote to a durable tier of its own copy."""

    changed = tuple(name for name, digest in snapshot.digests.items() if _file_digest(snapshot.root / name) != digest)
    if changed:
        raise CensusSnapshotError(f"the census mutated durable tiers of its copy: {changed}")


# ---------------------------------------------------------------------------
# Process measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProcessSample:
    """One point-in-time read of this process's kernel accounting."""

    wall_s: float
    cpu_s: float
    peak_rss_bytes: int
    anon_bytes: int | None
    file_cache_bytes: int | None
    swap_bytes: int | None
    read_io_bytes: int | None
    write_io_bytes: int | None
    unavailable: tuple[str, ...]


def _proc_status_kib(field_name: str) -> int | None:
    try:
        text = Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith(f"{field_name}:"):
            return int(line.split()[1]) * 1024
    return None


def _proc_io(field_name: str) -> int | None:
    try:
        text = Path("/proc/self/io").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith(f"{field_name}:"):
            return int(line.split()[1])
    return None


def sample_process() -> ProcessSample:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    anon = _proc_status_kib("RssAnon")
    cache = _proc_status_kib("RssFile")
    swap = _proc_status_kib("VmSwap")
    read_io = _proc_io("read_bytes")
    write_io = _proc_io("write_bytes")
    unavailable: list[str] = []
    for name, value in (
        ("anon_bytes", anon),
        ("file_cache_bytes", cache),
        ("swap_bytes", swap),
        ("read_io_bytes", read_io),
        ("write_io_bytes", write_io),
    ):
        if value is None:
            unavailable.append(name)
    return ProcessSample(
        wall_s=time.perf_counter(),
        cpu_s=usage.ru_utime + usage.ru_stime + children.ru_utime + children.ru_stime,
        # ru_maxrss is in kilobytes on Linux and is a high-water mark for the
        # whole process, so it is reported as peak, never as current.
        peak_rss_bytes=usage.ru_maxrss * 1024,
        anon_bytes=anon,
        file_cache_bytes=cache,
        swap_bytes=swap,
        read_io_bytes=read_io,
        write_io_bytes=write_io,
        unavailable=tuple(unavailable),
    )


def _delta(after: int | None, before: int | None) -> int | None:
    if after is None or before is None:
        return None
    return max(0, after - before)


# ---------------------------------------------------------------------------
# SQL capture and the bounded plan reader
# ---------------------------------------------------------------------------


@dataclass
class StatementTrace:
    """SELECT statements the routed read actually issued."""

    statements: list[str] = field(default_factory=list)

    def record(self, statement: str) -> None:
        text = statement.strip()
        head = text.split(None, 1)[0].upper() if text else ""
        if head in {"SELECT", "WITH"} and text not in self.statements:
            self.statements.append(text)


@contextmanager
def _trace_sqlite(trace: StatementTrace) -> Iterator[None]:
    """Record every SELECT any connection issues while the block runs."""

    real_connect = sqlite3.connect
    lock = threading.Lock()

    def connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = cast(sqlite3.Connection, real_connect(*args, **kwargs))

        def _callback(statement: str) -> None:
            with lock:
                trace.record(statement)

        conn.set_trace_callback(_callback)
        return conn

    sqlite3.connect = connect  # type: ignore[assignment]
    try:
        yield
    finally:
        sqlite3.connect = real_connect


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One ``EXPLAIN QUERY PLAN`` row, reduced to what a budget cares about."""

    detail: str

    @property
    def is_scan(self) -> bool:
        return self.detail.startswith("SCAN")

    @property
    def is_temp_btree(self) -> bool:
        return "USE TEMP B-TREE" in self.detail

    @property
    def target(self) -> str:
        """The table or index the step names, for allowance matching."""

        words = self.detail.replace("SCAN ", "").replace("SEARCH ", "").split()
        return words[0] if words else self.detail


class BoundedCensusReader:
    """One read-only connection with a declared VM-step ceiling.

    Every plan walk and every cheapest-primitive execution in a census runs
    through this one reader. There is no second connection and no parallel
    walk: the module lock the census holds makes that unrepresentable.
    """

    def __init__(self, snapshot: ArchiveSnapshot, *, vm_step_ceiling: int = CENSUS_READER_VM_STEP_CEILING) -> None:
        self._snapshot = snapshot
        self._ceiling = vm_step_ceiling
        self._steps = 0
        self._conn: sqlite3.Connection | None = None

    @property
    def vm_steps(self) -> int:
        """Lower bound on VM steps executed, at progress-handler granularity."""

        return self._steps

    def reset_steps(self) -> None:
        self._steps = 0

    def __enter__(self) -> BoundedCensusReader:
        conn = sqlite3.connect(f"file:{self._snapshot.index_path}?mode=ro", uri=True, timeout=30.0)
        for schema, filename in (("source_tier", "source.db"), ("user_tier", "user.db"), ("ops_tier", "ops.db")):
            sibling = self._snapshot.root / filename
            if sibling.exists():
                conn.execute(f"ATTACH DATABASE ? AS {schema}", (f"file:{sibling}?mode=ro",))
        conn.set_progress_handler(self._guard, 1000)
        self._conn = conn
        return self

    def __exit__(self, *_exc: object) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.set_progress_handler(None, 0)
            conn.close()

    def _guard(self) -> int:
        self._steps += 1000
        return 1 if self._steps > self._ceiling else 0

    def _require(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("the census reader is used outside its context")
        return self._conn

    def explain(self, statement: str) -> tuple[PlanStep, ...]:
        try:
            rows = self._require().execute(f"EXPLAIN QUERY PLAN {statement}").fetchall()
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                raise CensusBudgetExceededError(
                    f"census reader exhausted its VM-step ceiling: {statement[:120]}"
                ) from exc
            # A statement the plan walk cannot prepare (a temp table the route
            # created and dropped) has no plan to classify; say so instead of
            # inventing one.
            return (PlanStep(f"UNPLANNABLE {type(exc).__name__}: {exc}"),)
        return tuple(PlanStep(str(row[3])) for row in rows)

    def identities(self, statement: str) -> tuple[str, ...]:
        try:
            rows = self._require().execute(statement).fetchall()
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                raise CensusBudgetExceededError("census reader exhausted its VM-step ceiling") from exc
            raise
        return tuple(str(row[0]) for row in rows)


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScanFinding:
    """One full scan or materialization the plan chose."""

    statement: str
    detail: str
    target: str
    classified: bool
    owner: str = ""
    reason: str = ""
    allowance_detail: str = ""


@dataclass(frozen=True, slots=True)
class CensusObservation:
    """Everything one family's census measured, plus its receipt."""

    family: CensusFamily
    receipt: WorkloadReceipt
    rows_emitted: int
    rows_visited_cheapest: int
    response_bytes: int
    routed_vm_steps: int
    cheapest_vm_steps: int
    statements: tuple[str, ...]
    scans: tuple[ScanFinding, ...]
    temp_btrees: tuple[ScanFinding, ...]
    identity_match: bool
    identity_detail: str
    #: Steps in the cheapest correct primitive's own plan, for comparison.
    cheapest_plan: tuple[PlanStep, ...]
    #: Declared allowances that matched nothing this run: a stale declaration
    #: is as much a defect in the census as an unclassified scan.
    stale_allowances: tuple[str, ...]
    pushdown_detail: str

    @property
    def unclassified_scans(self) -> tuple[ScanFinding, ...]:
        return tuple(finding for finding in (*self.scans, *self.temp_btrees) if not finding.classified)

    @property
    def pushdown_held(self) -> bool:
        return not self.pushdown_detail

    @property
    def routed_plan_weight(self) -> int:
        """Full scans plus materializations in the routed plan."""

        return len({(finding.statement, finding.detail) for finding in (*self.scans, *self.temp_btrees)})

    @property
    def cheapest_plan_weight(self) -> int:
        return sum(1 for step in self.cheapest_plan if step.is_scan or step.is_temp_btree)

    @property
    def cheaper_primitive(self) -> bool:
        """Whether the cheapest correct primitive really has the smaller plan."""

        return self.cheapest_plan_weight < self.routed_plan_weight


class _ReceiptCollector:
    """Capture production's own execution receipts off its debug logger."""

    def __init__(self) -> None:
        self.receipts: list[Mapping[str, object]] = []

    def handle(self, record: Any) -> None:
        # ``logging`` unwraps a lone Mapping argument into ``record.args``
        # itself rather than a one-tuple, so both shapes have to be read.
        args = record.args
        candidate = args if isinstance(args, Mapping) else (args[0] if isinstance(args, tuple) and args else None)
        if isinstance(candidate, Mapping) and "sqlite_vm_steps_lower_bound" in candidate:
            self.receipts.append(cast(Mapping[str, object], candidate))

    @property
    def vm_steps(self) -> int:
        return sum(int(cast(int, receipt.get("sqlite_vm_steps_lower_bound", 0))) for receipt in self.receipts)

    @property
    def rows_emitted(self) -> int:
        return sum(int(cast(int, receipt.get("rows_emitted", 0))) for receipt in self.receipts)

    @property
    def cleanup_complete(self) -> bool:
        return bool(self.receipts) and all(bool(receipt.get("cleanup_complete")) for receipt in self.receipts)


@contextmanager
def _collect_receipts() -> Iterator[_ReceiptCollector]:
    import logging

    collector = _ReceiptCollector()

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            collector.handle(record)

    logger = logging.getLogger("polylogue.archive.query.execution_control")
    handler = _Handler(level=logging.DEBUG)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield collector
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def _pushdown_detail(family: CensusFamily, statements: Sequence[str]) -> str:
    """Return why the family's selective predicate is not pushed down, if so.

    The anchor is the statement's first ``LIMIT``: that is where the first
    ranked window closes. A selective restriction applied after it filters an
    already-materialized archive-wide ranking, which is the plan shape behind
    the 2026-07-15 incident -- so the census states it as a defect rather than
    as a slow query. An ``ORDER BY`` alone is not the anchor: a correlated
    subquery may legitimately order rows it has already restricted.
    """

    marker = family.pushdown_marker
    for statement in statements:
        collapsed = " ".join(statement.split())
        position = collapsed.find(marker)
        if position < 0:
            continue
        window = collapsed.find("LIMIT ")
        if window < 0 or position < window:
            return ""
        return f"{marker!r} is applied after the first ranked window closes in: {collapsed[:160]}"
    return f"no routed statement carries the declared restriction {marker!r}"


def _classify(statement: str, step: PlanStep, family: CensusFamily) -> ScanFinding:
    for allowance in family.scan_allowances:
        if allowance.detail in step.detail:
            return ScanFinding(
                statement=statement,
                detail=step.detail,
                target=step.target,
                classified=True,
                owner=allowance.owner,
                reason=allowance.reason,
                allowance_detail=allowance.detail,
            )
    return ScanFinding(statement=statement, detail=step.detail, target=step.target, classified=False)


def _spec(family: CensusFamily, snapshot: ArchiveSnapshot) -> WorkloadEnvelopeSpec:
    return WorkloadEnvelopeSpec(
        workload_id=f"query-census:{family.family_id}",
        family_id=family.family_id,
        version=1,
        inputs=(
            WorkloadInputRef(
                input_id=f"archive-snapshot:{snapshot.root.name}",
                corpus_id="tests.infra.query_corpus",
                selectivity_tier=family.selectivity,
                distribution_refs=("index.session_shapes.message_count", "source.blob_size"),
            ),
        ),
        # The census budgets one phase, the routed read; the plan and cheapest
        # phases are the census's own measurement work, not the workload.
        phases=("query",),
        measurement_scope=MeasurementScope.PROCESS_TREE,
        budgets=family.budgets,
    )


async def census_family(
    family: CensusFamily,
    snapshot: ArchiveSnapshot,
    reader: BoundedCensusReader,
) -> CensusObservation:
    """Measure one declared query family against the snapshot."""

    from polylogue.api import Polylogue

    trace = StatementTrace()
    archive = Polylogue(archive_root=snapshot.root)
    before = sample_process()
    try:
        with _collect_receipts() as receipts, _trace_sqlite(trace):
            envelope = await archive.query_units(family.expression, limit=10_000)
        payload = envelope.model_dump(mode="json")
    finally:
        await archive.close()
    after = sample_process()
    response_bytes = len(json.dumps(payload).encode("utf-8"))
    rows = cast(Sequence[Mapping[str, Any]], payload.get("items", ()))
    routed_identities = tuple(str(row.get(family.cheapest_primitive_identity)) for row in rows)

    scans: list[ScanFinding] = []
    temp_btrees: list[ScanFinding] = []
    matched_allowances: set[str] = set()
    for statement in trace.statements:
        for step in reader.explain(statement):
            if not (step.is_scan or step.is_temp_btree):
                continue
            finding = _classify(statement, step, family)
            if finding.classified:
                matched_allowances.add(finding.allowance_detail)
            (scans if step.is_scan else temp_btrees).append(finding)
    stale = tuple(
        sorted(allowance.detail for allowance in family.scan_allowances if allowance.detail not in matched_allowances)
    )

    reader.reset_steps()
    cheapest_identities = reader.identities(family.cheapest_primitive_sql)
    cheapest_vm_steps = reader.vm_steps
    cheapest_plan = reader.explain(family.cheapest_primitive_sql)
    pushdown_detail = _pushdown_detail(family, trace.statements)

    missing = sorted(set(cheapest_identities) - set(routed_identities))
    extra = sorted(set(routed_identities) - set(cheapest_identities))
    identity_match = not missing and not extra
    identity_detail = "" if identity_match else f"missing={missing[:5]} extra={extra[:5]}"

    observation = WorkloadPhaseObservation(
        name="query",
        measurement_scope=MeasurementScope.PROCESS_TREE,
        wall_ms=(after.wall_s - before.wall_s) * 1000.0,
        cpu_ms=(after.cpu_s - before.cpu_s) * 1000.0,
        peak_rss_bytes=after.peak_rss_bytes,
        anon_bytes=after.anon_bytes,
        file_cache_bytes=after.file_cache_bytes,
        swap_bytes=after.swap_bytes,
        read_io_bytes=_delta(after.read_io_bytes, before.read_io_bytes),
        write_io_bytes=_delta(after.write_io_bytes, before.write_io_bytes),
        response_bytes=response_bytes,
        sqlite_vm_steps=receipts.vm_steps,
        progress_completed=len(rows),
        progress_total=len(cheapest_identities),
        cleanup_complete=receipts.cleanup_complete,
        quiescent=True,
        # The census measures one in-process read; there is no temp spool and
        # no child process to charge, and saying so beats reporting a zero.
        unavailable=("temp_storage_bytes", "peak_pss_bytes"),
    )
    receipt = WorkloadReceipt.from_observations(
        spec=_spec(family, snapshot),
        status=WorkloadRunStatus.SUCCEEDED,
        build_id=None,
        runtime_id=f"pid:{os.getpid()}",
        archive_id=snapshot.digests.get("source.db"),
        generation_id=None,
        frame_id=None,
        phases=(observation,),
        cleanup_complete=receipts.cleanup_complete,
        notes=(f"reflinked={snapshot.reflinked}",),
    )
    return CensusObservation(
        family=family,
        receipt=receipt,
        rows_emitted=len(rows),
        rows_visited_cheapest=len(cheapest_identities),
        response_bytes=response_bytes,
        routed_vm_steps=receipts.vm_steps,
        cheapest_vm_steps=cheapest_vm_steps,
        statements=tuple(trace.statements),
        scans=tuple(scans),
        temp_btrees=tuple(temp_btrees),
        identity_match=identity_match,
        identity_detail=identity_detail,
        cheapest_plan=cheapest_plan,
        stale_allowances=stale,
        pushdown_detail=pushdown_detail,
    )


async def run_workload_census(
    snapshot: ArchiveSnapshot,
    *,
    families: Sequence[CensusFamily] = CENSUS_FAMILIES,
) -> tuple[CensusObservation, ...]:
    """Census every declared family serially, through one bounded reader."""

    if not _CENSUS_LOCK.acquire(blocking=False):
        raise CensusConcurrencyError("a workload census is already walking an archive in this process")
    try:
        observations: list[CensusObservation] = []
        with BoundedCensusReader(snapshot) as reader:
            for family in families:
                observations.append(await census_family(family, snapshot, reader))
        assert_snapshot_unmutated(snapshot)
        return tuple(observations)
    finally:
        _CENSUS_LOCK.release()


__all__ = [
    "CENSUS_PHASES",
    "ArchiveSnapshot",
    "BoundedCensusReader",
    "CensusBudgetExceededError",
    "CensusConcurrencyError",
    "CensusObservation",
    "CensusSnapshotError",
    "PlanStep",
    "ScanFinding",
    "StatementTrace",
    "assert_snapshot_unmutated",
    "census_family",
    "reflink_archive_snapshot",
    "run_workload_census",
    "sample_process",
]
