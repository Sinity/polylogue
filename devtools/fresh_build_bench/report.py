"""Fold one fresh-build run into a comparable receipt, and compare two.

Every number in a receipt comes from one of four sources, named per section:
the daemon's own structured event log (``events.jsonl``), the ops-tier batch
ledger (``daemon_events`` rows of kind ``ingestion_batch``), the driver's
process-tree samples, and a read-only pass over the finished archive. Stage
names are the daemon's own timer names; the rollup into acquire / parse /
materialize / index / fts / derived is declared once in :data:`STAGE_ROLLUP`.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

RECEIPT_FORMAT: Final = "polylogue.fresh-build-receipt.v1"

#: Ordered (pattern, bucket). First match wins. Keys are the daemon's
#: ``stage_timings_s`` names from the ``ingestion_batch`` ledger rows.
STAGE_ROLLUP: Final[tuple[tuple[str, str], ...]] = (
    (r"fts", "fts"),
    (r"^full\.source_|^raw_|blob|^source\.", "acquire"),
    (r"provider_parse|parse_stage|^census", "parse"),
    (r"^full\.index\.prepare", "materialize"),
    # Derived work timed inside the index phase (``full.index.graph_resolve``,
    # ``full.index.delegation_facts``) is its own bucket, so it must match
    # before the generic index prefix.
    (r"graph_resolve|lineage|delegation|hook_paste|paste", "derived_inline"),
    (r"^full\.index\.", "index"),
    (r"^full\.index_parsed_write$", "index_total"),
)

_TS_FORMAT: Final = "%Y-%m-%dT%H:%M:%S.%fZ"


def _ts(value: str) -> float:
    # The trailing ``Z`` is UTC; a naive parse would read it as local time.
    return datetime.strptime(value, _TS_FORMAT).replace(tzinfo=UTC).timestamp()


def _bucket(stage: str) -> str:
    for pattern, bucket in STAGE_ROLLUP:
        if re.search(pattern, stage):
            return bucket
    return "other"


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1) + 0.5))]


# ---------------------------------------------------------------------------
# event log


def _iter_events(path: Path) -> Iterator[dict[str, Any]]:
    # A daemon that dies before configuring logging leaves no event file.
    if not path.exists():
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield event


def analyse_events(path: Path, *, origin_unix: float | None = None) -> dict[str, Any]:
    """Reduce the daemon's event log in one streaming pass.

    Times are seconds from ``origin_unix`` -- the driver's launch, so event
    milestones share a clock with the driver's own observations and process
    samples -- or from ``daemon.run.start`` when no origin is given. Only
    aggregates and per-chunk timings are retained, never the events: a long
    build logs millions of writer events.
    """
    start = origin_unix
    if start is None:
        first: float | None = None
        for event in _iter_events(path):
            if first is None:
                first = _ts(event["ts"])
            if event.get("event") == "daemon.run.start":
                start = _ts(event["ts"])
                break
        start = start if start is not None else first
    if start is None:
        return {"event_count": 0}
    origin = start

    def rel(event: dict[str, Any]) -> float:
        return round(_ts(event["ts"]) - origin, 3)

    writer: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "hold_s": 0.0, "wait_s": 0.0, "max_hold_s": 0.0}
    )
    busy_total = 0.0
    busy_build = 0.0
    queued_max = 0
    queued_sum = 0
    queued_count = 0
    chunks: list[tuple[float, int, int, float]] = []
    pages: dict[str, dict[str, float]] = defaultdict(lambda: {"pages": 0, "files": 0, "bytes": 0, "seconds": 0.0})
    by_source: dict[str, dict[str, float]] = defaultdict(lambda: {"groups": 0, "files": 0, "seconds": 0.0})
    problems: Counter[str] = Counter()
    milestones: dict[str, float] = {}
    preparation_done: float | None = None
    event_count = 0
    last = 0.0
    for event in _iter_events(path):
        event_count += 1
        last = rel(event)
        promoted = milestones.get("promoted_s")
        name = event.get("event", "")
        if name == "daemon.writer.released":
            actor = str(event.get("actor"))
            hold = float(event.get("hold_ms") or 0) / 1000
            entry = writer[actor]
            entry["count"] += 1
            entry["hold_s"] += hold
            entry["wait_s"] += float(event.get("wait_ms") or 0) / 1000
            entry["max_hold_s"] = max(entry["max_hold_s"], hold)
            released = last
            begin = released - hold
            busy_total += hold
            # Busy time to promotion: holds are logged in release order, so
            # every hold before the promotion event ends by it, and one after
            # it counts only its part before promotion.
            end = released if promoted is None else min(released, promoted)
            busy_build += max(0.0, end - max(begin, 0.0))
            queued = int(event.get("queued") or 0)
            queued_max = max(queued_max, queued)
            queued_sum += queued
            queued_count += 1
        elif name == "live.ingest.chunk":
            chunks.append(
                (
                    last,
                    int(event.get("files") or 0),
                    int(event.get("bytes") or 0),
                    float(event.get("duration_ms") or 0) / 1000,
                )
            )
            milestones.setdefault("first_chunk_done_s", last)
        elif name == "daemon.intake.page":
            entry = pages[str(event.get("component"))]
            entry["pages"] += 1
            entry["files"] += int(event.get("files") or 0)
            entry["bytes"] += int(event.get("bytes") or 0)
            entry["seconds"] += float(event.get("duration_ms") or 0) / 1000
        elif name == "live.ingest.source_group":
            # Per-origin intake rates cover the same window as the intake
            # wall they are scaled against: groups re-offered after promotion
            # are derived-phase passes.
            if promoted is None or last <= promoted:
                entry = by_source[str(event.get("source_name"))]
                entry["groups"] += 1
                entry["files"] += int(event.get("files") or 0)
                entry["seconds"] += float(event.get("duration_ms") or 0) / 1000
        elif name == "daemon.cold_build.preparation":
            preparation_done = last if preparation_done is None else max(preparation_done, last)
        elif name == "daemon.cold_build.generation_created":
            milestones["generation_created_s"] = last
        elif name == "daemon.cold_build.generation_promoted":
            milestones["promoted_s"] = last
        if event.get("level") in {"warning", "error"}:
            problems[f"{name}:{event.get('reason')}"] += 1
    if event_count == 0:
        return {"event_count": 0}
    if preparation_done is not None:
        milestones["preparation_done_s"] = preparation_done
    # Intake ends at the last chunk before promotion. Chunks after it are the
    # promoted generation's own live passes (raw retention, re-offered
    # files), which belong to the derived phase, not to intake throughput.
    intake_chunks = [
        chunk[0] for chunk in chunks if "promoted_s" not in milestones or chunk[0] <= milestones["promoted_s"]
    ]
    if intake_chunks:
        milestones["last_chunk_done_s"] = max(intake_chunks)
    milestones["post_promotion_chunks"] = len(chunks) - len(intake_chunks)
    build_end = milestones.get("promoted_s", last)
    chunk_seconds = [chunk[3] for chunk in chunks]
    return {
        "event_count": event_count,
        "milestones_s": milestones,
        "writer": {
            "busy_s_to_promotion": round(busy_build, 3),
            "busy_share_to_promotion": round(busy_build / build_end, 4) if build_end > 0 else None,
            "busy_s_total": round(busy_total, 3),
            "queue_depth_max": queued_max,
            "queue_depth_mean": round(queued_sum / queued_count, 3) if queued_count else 0.0,
            # Lists, not objects: receipts are written with sorted keys, and
            # the order here is the ranking.
            "by_actor": [
                [actor, {key: round(value, 4) for key, value in stats.items()}]
                for actor, stats in sorted(writer.items(), key=lambda item: -item[1]["hold_s"])
            ],
        },
        "chunks": {
            "count": len(chunks),
            "files": sum(chunk[1] for chunk in chunks),
            "bytes": sum(chunk[2] for chunk in chunks),
            "seconds": round(sum(chunk_seconds), 3),
            "seconds_p50": round(_percentile(chunk_seconds, 0.5), 3),
            "seconds_max": round(max(chunk_seconds, default=0.0), 3),
        },
        "intake_pages_by_class": {key: {k: round(v, 3) for k, v in value.items()} for key, value in pages.items()},
        "by_source": {key: {k: round(v, 3) for k, v in value.items()} for key, value in sorted(by_source.items())},
        "warnings_and_errors": dict(problems.most_common()),
    }


# ---------------------------------------------------------------------------
# ops-tier batch ledger


def analyse_batches(ops_path: Path) -> dict[str, Any]:
    if not ops_path.exists():
        return {"batches": 0}
    stages: defaultdict[str, float] = defaultdict(float)
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    with closing(sqlite3.connect(f"file:{ops_path}?mode=ro", uri=True)) as conn:
        # A daemon that exited during bootstrap may leave ops.db without its
        # ledger; that is zero batches, and the failure receipt still follows.
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'daemon_events'").fetchone():
            return {"batches": 0}
        for (payload,) in conn.execute("SELECT payload_json FROM daemon_events WHERE kind = 'ingestion_batch'"):
            data = json.loads(payload)
            count += 1
            for key in (
                "input_bytes",
                "ingested_bytes",
                "source_payload_read_bytes",
                "cursor_fingerprint_read_bytes",
                "succeeded_file_count",
                "failed_file_count",
                "ingested_session_count",
                "ingested_message_count",
                "parse_time_s",
                "convergence_time_s",
                "total_time_s",
            ):
                totals[key] += data.get(key) or 0
            for stage, seconds in (data.get("stage_timings_s") or {}).items():
                stages[stage] += float(seconds)
    buckets: defaultdict[str, float] = defaultdict(float)
    for stage, seconds in stages.items():
        # Parent timers include their children; roll up only leaf names so a
        # bucket never double-counts a nested timer.
        if any(other.startswith(stage + ".") for other in stages):
            continue
        buckets[_bucket(stage)] += seconds
    return {
        "batches": count,
        "totals": {key: round(value, 3) for key, value in totals.items()},
        "stage_seconds": [[key, round(value, 3)] for key, value in sorted(stages.items(), key=lambda item: -item[1])],
        "stage_buckets_seconds": [
            [key, round(value, 3)] for key, value in sorted(buckets.items(), key=lambda item: -item[1])
        ],
    }


# ---------------------------------------------------------------------------
# finished archive


def archive_census(archive: Path, promoted_index: str | None) -> dict[str, Any]:
    tiers = {}
    for name in ("source.db", "ops.db", "user.db", "audit.db", "embeddings.db"):
        path = archive / name
        if path.exists():
            tiers[name] = path.stat().st_size
    blob_bytes = sum(path.stat().st_size for path in (archive / "blob").rglob("*") if path.is_file())
    result: dict[str, Any] = {"tier_bytes": tiers, "blob_bytes": blob_bytes}
    if promoted_index is not None:
        index = Path(promoted_index)
        result["index_bytes"] = index.stat().st_size
        with closing(sqlite3.connect(f"file:{index}?mode=ro", uri=True)) as conn:
            result["rows"] = {
                table: int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
                for table in ("sessions", "messages", "blocks", "file_edits", "action_pairs", "session_links")
            }
            result["messages_fts_rows"] = int(conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0])
            from polylogue.storage.fts.sql import FTS_INDEXABLE_MESSAGE_COUNT_SQL

            result["fts_indexable_rows"] = int(conn.execute(FTS_INDEXABLE_MESSAGE_COUNT_SQL).fetchone()[0])
    return result


def _fts_postings(read: sqlite3.Connection) -> dict[str, Any]:
    """Digest every posting ``messages_fts`` holds, keyed by block identity.

    ``messages_fts`` is contentless, so its text cannot be read back and the
    table census skips it. Its index can: ``fts5vocab`` in ``instance`` mode
    lists every (term, document, column, offset) posting. Each posting is
    named by its block's ``block_id`` (FTS rowids are not stable across
    builds; a posting whose rowid names no block digests as ``null``), and
    the postings are folded into an order-independent sum of SHA-256
    digests, so the pass streams in constant memory whatever the archive
    size and still sees a moved, dropped or added posting for any term.
    """
    read.execute("DROP TABLE IF EXISTS temp.bench_fts_postings")
    read.execute("CREATE VIRTUAL TABLE temp.bench_fts_postings USING fts5vocab(main, messages_fts, instance)")
    total = 0
    postings = 0
    terms = 0
    previous: str | None = None
    try:
        for term, block_id, column, offset in read.execute(
            "SELECT v.term, b.block_id, v.col, v.offset FROM temp.bench_fts_postings v "
            "LEFT JOIN blocks b ON b.rowid = v.doc"
        ):
            payload = json.dumps([term, block_id, column, offset], ensure_ascii=True, separators=(",", ":"))
            total = (total + int.from_bytes(hashlib.sha256(payload.encode()).digest(), "big")) % (1 << 256)
            postings += 1
            if term != previous:
                terms += 1
                previous = term
    finally:
        read.execute("DROP TABLE IF EXISTS temp.bench_fts_postings")
    return {"rows": postings, "terms": terms, "sha256": f"{total:064x}"}


class FingerprintCancelledError(RuntimeError):
    """A cancellation arrived while the output fingerprint was being taken."""


def output_fingerprint(
    archive: Path, promoted_index: str, scratch: Path, *, cancelled: Callable[[], bool] = lambda: False
) -> dict[str, Any]:
    """Per-table digests over the comparable index relations.

    Uses the differential harness's declared census, volatile-column policy
    and row normalisation, so an equivalence claim means what the finished-
    build differential tests mean by it. Per-table digests localise a
    mismatch without retaining rows.
    """
    from tests.infra.reindex_differential import _VOLATILE_COLUMNS, _fact_row, compared_table_census

    scratch.mkdir(parents=True, exist_ok=True)
    spool_path = scratch / "fingerprint.sqlite"
    spool_path.unlink(missing_ok=True)
    tables: dict[str, dict[str, Any]] = {}
    with (
        closing(sqlite3.connect(f"file:{promoted_index}?mode=ro", uri=True)) as read,
        closing(sqlite3.connect(spool_path)) as spool,
    ):
        read.row_factory = sqlite3.Row
        spool.execute("PRAGMA journal_mode=OFF")
        spool.execute("PRAGMA synchronous=OFF")
        for table in compared_table_census():
            volatile = _VOLATILE_COLUMNS[table]
            columns = [
                str(row["name"])
                for row in read.execute(f'PRAGMA table_xinfo("{table}")')
                if row["name"] not in volatile
            ]
            spool.execute("DROP TABLE IF EXISTS facts")
            spool.execute("CREATE TABLE facts (payload TEXT NOT NULL)")
            selected = ", ".join(f'"{column}"' for column in columns)
            batch: list[tuple[str]] = []
            batch_bytes = 0
            rows = 0
            for row in read.execute(f'SELECT {selected} FROM "{table}"'):
                if rows % 1000 == 0 and cancelled():
                    # The one step that scales with the archive stops on a
                    # cancellation, so the receipt is written in time.
                    raise FingerprintCancelledError("cancelled during the output fingerprint")
                payload = json.dumps(_fact_row(row), ensure_ascii=True, separators=(",", ":"))
                batch.append((payload,))
                batch_bytes += len(payload)
                rows += 1
                # Flushed by bytes as well as rows: a few thousand
                # megabyte-sized rows must not all be resident at once.
                if len(batch) >= 5000 or batch_bytes >= _FACT_BATCH_BYTES:
                    spool.executemany("INSERT INTO facts VALUES (?)", batch)
                    batch.clear()
                    batch_bytes = 0
            if batch:
                spool.executemany("INSERT INTO facts VALUES (?)", batch)
            digest = hashlib.sha256()
            digest.update(json.dumps(columns).encode())
            for (payload,) in spool.execute("SELECT payload FROM facts ORDER BY payload"):
                digest.update(payload.encode())
                digest.update(b"\n")
            tables[table] = {"rows": rows, "sha256": digest.hexdigest()}
        tables["messages_fts:postings"] = _fts_postings(read)
    spool_path.unlink(missing_ok=True)
    overall = hashlib.sha256(json.dumps(tables, sort_keys=True).encode()).hexdigest()
    return {"digest": overall, "tables": tables}


# ---------------------------------------------------------------------------
# receipt


def _log_delivery(stacks_path: Path) -> dict[str, int] | None:
    """The daemon's event-sink delivery counters, as the sampler saw them at exit."""
    if not stacks_path.exists():
        return None
    delivery = json.loads(stacks_path.read_text(encoding="utf-8")).get("log_delivery")
    return delivery if isinstance(delivery, dict) else None


def _tree_summary(samples: list[tuple[float, int, float, int, int, int]], build_end_s: float | None) -> dict[str, Any]:
    if not samples:
        return {}
    rss = [sample[1] for sample in samples]
    last = samples[-1]
    # A run that never promoted has no promotion-scoped CPU: the whole run's
    # CPU is ``cpu_seconds_total``, never a milestone measurement.
    to_end = [sample for sample in samples if build_end_s is not None and sample[0] <= build_end_s]
    cpu_to_end = (to_end[-1][2] if to_end else 0.0) if build_end_s is not None else None
    return {
        "rss_peak_bytes": max(rss),
        "rss_p95_bytes": int(_percentile([float(value) for value in rss], 0.95)),
        "rss_final_bytes": rss[-1],
        "cpu_seconds_total": round(last[2], 2),
        "cpu_seconds_to_promotion": round(cpu_to_end, 2) if cpu_to_end is not None else None,
        "mean_cores_to_promotion": round(cpu_to_end / to_end[-1][0], 3)
        if cpu_to_end is not None and to_end and to_end[-1][0] > 0
        else None,
        "threads_max": max(sample[3] for sample in samples),
        "io_read_bytes": last[4],
        "io_write_bytes": last[5],
        # Compact timeline for plotting: (t, rss MiB, cpu s, write MiB).
        "timeline": [
            (sample[0], round(sample[1] / 2**20, 1), round(sample[2], 1), round(sample[5] / 2**20, 1))
            for sample in samples[:: max(1, len(samples) // 400)]
        ],
    }


#: Watch-source names whose batches carry each corpus origin.
_SOURCE_ORIGIN = {
    "claude-code": "claude-code",
    "codex": "codex",
    "gemini-cli": "gemini-cli",
    "chatgpt": "chatgpt",
    "claude-ai": "claude-ai",
}


def projection(manifest: dict[str, Any], by_source: dict[str, Any], intake_wall_s: float | None) -> dict[str, Any]:
    """Scale measured per-origin intake cost to the population the sample came from.

    Per origin: the batch time the daemon spent on that origin's files per
    sampled MiB, times the population's MiB. The per-origin times are writer
    publication plus the parse wait of each source group, so their sum is the
    serial-equivalent intake time; ``wall_scale`` rescales it to the measured
    intake wall so overlap between groups is not double counted. Whales are
    excluded from a default sample and priced at the sample's per-MiB rate,
    which the separate whale qualifications must confirm. This is intake
    only: promotion and derived convergence are reported beside it, measured,
    not scaled.
    """
    population = (manifest.get("parameters") or {}).get("population") or {}
    sampled = manifest.get("by_origin") or {}
    rows: dict[str, Any] = {}
    serial_sample = 0.0
    serial_projected = 0.0
    for source, stats in (by_source or {}).items():
        origin = _SOURCE_ORIGIN.get(source)
        if origin is None or origin not in sampled or not sampled[origin]["bytes"]:
            continue
        sample_mib = sampled[origin]["bytes"] / 2**20
        seconds = float(stats["seconds"])
        rate = seconds / sample_mib
        entry: dict[str, Any] = {
            "sample_mib": round(sample_mib, 1),
            "seconds": round(seconds, 1),
            "s_per_mib": round(rate, 4),
        }
        pop = population.get(origin)
        if pop:
            pop_mib = pop["bytes"] / 2**20
            entry["population_mib"] = round(pop_mib, 1)
            entry["projected_s"] = round(rate * pop_mib, 1)
            serial_projected += rate * pop_mib
        serial_sample += seconds
        rows[origin] = entry
    scale = (intake_wall_s / serial_sample) if intake_wall_s and serial_sample else None
    # A populated origin without a measured rate is not priced at zero: the
    # projection then covers only part of the census and says so.
    unmeasured = sorted(origin for origin, pop in population.items() if pop.get("bytes") and origin not in rows)
    return {
        "complete": not unmeasured,
        "unmeasured_origins": unmeasured,
        "by_origin": rows,
        "serial_sample_s": round(serial_sample, 1),
        "intake_wall_s": intake_wall_s,
        "wall_scale": round(scale, 4) if scale else None,
        "projected_intake_hours": round(serial_projected * scale / 3600, 2)
        if scale and serial_projected and not unmeasured
        else None,
        "projected_measured_origins_hours": round(serial_projected * scale / 3600, 2)
        if scale and serial_projected
        else None,
    }


#: Budgets a run may assert, and where each observed value comes from.
BUDGETS: Final[dict[str, str]] = {
    "rss_peak_mib": "peak RSS of the daemon's whole process tree",
    "promotion_s": "seconds from daemon start to candidate promotion",
    "terminal_s": "seconds from daemon start to terminal convergence",
}


def evaluate_budgets(limits: dict[str, float], observed: dict[str, float | None]) -> dict[str, Any]:
    """One row per asserted budget; an unobserved value fails its budget."""
    rows: dict[str, Any] = {}
    for name, limit in sorted(limits.items()):
        if name not in BUDGETS:
            raise ValueError(f"unknown budget {name!r}; known: {sorted(BUDGETS)}")
        value = observed.get(name)
        rows[name] = {"limit": limit, "observed": value, "pass": value is not None and value <= limit}
    return rows


def _progress_timeline(observations: list[Any]) -> list[tuple[Any, ...]]:
    """One row per observable change: (t, cursors complete, raw rows, open
    debt, promoted, ready domains, open debt by stage, debt in backoff by
    stage)."""
    rows: list[tuple[Any, ...]] = []
    for observation in observations:
        row = (
            observation.t,
            observation.cursor_complete,
            observation.raw_rows,
            observation.open_debt,
            observation.promoted_index is not None,
            sorted(domain for domain, ready in observation.readiness.items() if ready),
            dict(sorted(observation.debt_by_stage.items())),
            dict(sorted(observation.debt_waiting_by_stage.items())),
        )
        if not rows or rows[-1][1:] != row[1:]:
            rows.append(row)
    return rows


def thread_cpu_summary(document: dict[str, Any]) -> dict[str, Any]:
    """Daemon CPU by thread: every writer actor, the writer total, the rest.

    The writer total is the single-writer ceiling: intake cannot go faster
    than the writer's own CPU allows, whatever the host's load does to wall
    time. Sampling sees live threads only, so the per-thread figures are
    lower bounds: a thread that started and ended between samples, or the
    last interval of one that ended, lands in ``unattributed`` (the process
    total minus every sampled thread).
    """
    ticks_per_s = float(document["clock_ticks_per_s"])
    by_thread = {name: ticks / ticks_per_s for name, ticks in document.get("thread_cpu_ticks", {}).items()}
    writer = {name: seconds for name, seconds in by_thread.items() if name.startswith("polylogue-writer:")}
    process_ticks = document.get("process_cpu_ticks")
    return {
        "process_total": round(process_ticks / ticks_per_s, 2) if process_ticks is not None else None,
        "unattributed": round(process_ticks / ticks_per_s - sum(by_thread.values()), 2)
        if process_ticks is not None
        else None,
        "writer_total_lower_bound": round(sum(writer.values()), 2),
        "writer_by_actor": [
            [name.removeprefix("polylogue-writer:"), round(seconds, 2)]
            for name, seconds in sorted(writer.items(), key=lambda item: -item[1])
        ],
        "other_threads": [
            [name, round(seconds, 2)]
            for name, seconds in sorted(by_thread.items(), key=lambda item: -item[1])
            if name not in writer
        ][:15],
    }


#: Largest wall-versus-monotonic drift over one run still read as a steady
#: clock (slewing NTP moves far less; a step moves far more).
_MAX_CLOCK_STEP_S: Final = 1.0
#: Serialized fact bytes buffered before one spool insert.
_FACT_BATCH_BYTES: Final = 64 << 20


def benchmark_implementation_sha256() -> str:
    """Identity of the driver and report code that produced a receipt.

    ``--candidate`` lets the measured checkout differ from the one supplying
    this instrumentation, so the sampler, reducer, observer and fingerprint
    implementation are an input of their own.
    """
    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    for path in (*sorted(package.glob("*.py")), *_FINGERPRINT_DEPENDENCIES):
        digest.update(path.name.encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


#: Modules outside the package whose code decides the output fingerprint
#: (table census, volatile columns, row normalization).
_FINGERPRINT_DEPENDENCIES: Final = (
    Path(__file__).resolve().parents[2] / "tests" / "infra" / "reindex_differential.py",
)


def config_digest(config: Any) -> str:
    """What must match, besides the corpus, for two receipts to compare."""
    payload = {
        "profile": config.profile,
        "profile_interval_s": config.profile_interval_s if config.profile else None,
        # The observation loop decides when an unsettled run stops, so it is
        # part of what two receipts must share.
        "stall_timeout_s": config.stall_timeout_s,
        "poll_s": config.poll_s,
        "stable_polls": config.stable_polls,
        "extra_env": sorted(config.extra_env),
        "daemon_argv": ["run", "--no-browser-capture", "--no-api", "--cold-build-index"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _derive_dependents(receipt: dict[str, Any]) -> None:
    """Recompute every field derived from the timing and the process tree.

    One place, so a refreshed receipt cannot keep a throughput, budget or
    qualification computed from a milestone it has since replaced.
    """
    timing = receipt["timing_s"]
    promoted, terminal_at = timing.get("promotion"), timing.get("terminal")
    # The receipt's timings are reductions of the event log (see
    # build_receipt); a refresh that re-reads a moved milestone must move
    # this check with it, or a formerly missing milestone that is now
    # recovered stays falsely unqualified, and one no longer recovered can
    # leave a terminal receipt qualified with "promotion": null.
    if "checks" in receipt:
        receipt["checks"]["milestones_recorded"] = promoted is not None
    total_mib = receipt["corpus"]["total_bytes"] / 2**20
    timing["derived_after_promotion"] = (
        round(terminal_at - promoted, 3) if terminal_at is not None and promoted is not None else None
    )
    receipt["throughput"] = {
        "mib_per_s_to_promotion": round(total_mib / promoted, 3) if promoted else None,
        "mib_per_s_to_terminal": round(total_mib / terminal_at, 3) if terminal_at else None,
        "files_per_s_to_promotion": round(receipt["corpus"]["file_count"] / promoted, 3) if promoted else None,
    }
    tree = receipt.get("process_tree") or {}
    rss = tree.get("rss_budget_peak_bytes", tree.get("rss_peak_bytes"))
    budgets = evaluate_budgets(
        {name: row["limit"] for name, row in (receipt.get("budgets") or {}).items()},
        {"rss_peak_mib": rss / 2**20 if rss else None, "promotion_s": promoted, "terminal_s": terminal_at},
    )
    receipt["budgets"] = budgets
    # A qualified build finished, satisfied every terminal check and stayed
    # inside every asserted budget.
    receipt["qualified"] = (
        receipt["outcome"] == "terminal"
        and all(receipt["checks"].values())
        and all(row["pass"] for row in budgets.values())
    )


def build_receipt(
    *,
    config: Any,
    manifest: dict[str, Any],
    paths: dict[str, Path],
    identity: dict[str, Any],
    environment: dict[str, Any],
    command: list[str],
    started_wall: float,
    wall_s: float,
    outcome: str,
    terminal_at: float | None,
    exit_code: int | None,
    shutdown_s: float,
    observations: list[Any],
    final: Any,
    tree_samples: list[tuple[float, int, float, int, int, int]],
    daemon_rss_hwm_bytes: int = 0,
    corpus_unchanged: bool = True,
    clock_step_s: float = 0.0,
    cancelled: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    events = analyse_events(paths["events"], origin_unix=started_wall)
    batches = analyse_batches(paths["archive"] / "ops.db")
    milestones = events.get("milestones_s", {})
    promoted = milestones.get("promoted_s")
    census = archive_census(paths["archive"], final.promoted_index)
    fingerprint: dict[str, Any] | None = None
    # An interrupted run has seconds before its supervisor escalates; the
    # fingerprint is the one step that scales with the archive.
    if config.fingerprint and final.promoted_index is not None and outcome != "interrupted":
        try:
            fingerprint = output_fingerprint(
                paths["archive"], final.promoted_index, paths["work"] / "tmp", cancelled=cancelled
            )
        except FingerprintCancelledError:
            fingerprint = None
    total_bytes = int(manifest["total_bytes"])
    tree = _tree_summary(tree_samples, promoted)
    if tree:
        tree["daemon_rss_hwm_bytes"] = daemon_rss_hwm_bytes
        # The budgeted peak: the sampled tree peak (4 Hz) or the daemon's own
        # exact high-water mark, whichever is larger. A worker child's spike
        # shorter than the interval is still unobserved.
        tree["rss_budget_peak_bytes"] = max(tree["rss_peak_bytes"], daemon_rss_hwm_bytes)
    delivery = _log_delivery(paths["stacks"])
    checks = {
        "promoted": final.promoted_index is not None,
        "intake_complete": final.intake_complete,
        "raw_parse_failures_zero": final.raw_failed == 0,
        "memberships_settled": final.memberships_pending == 0,
        "debt_zero": final.open_debt == 0,
        "readiness_complete": final.readiness_complete,
        "fts_exact": census.get("messages_fts_rows") is not None
        and census.get("messages_fts_rows") == census.get("fts_indexable_rows"),
        "candidate_unchanged": identity["unchanged_during_run"],
        # A daemon that reached terminal but had to be killed on shutdown is
        # not a finished build.
        "clean_shutdown": exit_code == 0,
        # The receipt's timings are reductions of the event log; a promoted
        # pointer without the promotion event leaves them unmeasured.
        "milestones_recorded": promoted is not None,
        "corpus_unchanged": corpus_unchanged,
        # Event milestones are wall-clock (the daemon's ``ts``), samples and
        # terminal time monotonic. A wall clock stepped during the run (a VM
        # or NTP step) moves the milestones against the samples, so the run
        # does not qualify rather than report misaligned timings.
        "wall_clock_steady": abs(clock_step_s) <= _MAX_CLOCK_STEP_S,
        # Receipt sections are reductions of the event log; a dropped or
        # undelivered event makes them understate.
        "events_lossless": delivery is not None
        and not (delivery.get("dropped") or delivery.get("failures") or delivery.get("undrained")),
        # A nonempty corpus (total_bytes > 0) that qualifies with zero
        # sessions/messages materialized did no conversational work: an
        # accepted-but-empty export (e.g. a ChatGPT "[]" file) can settle its
        # cursor and promote an empty index, and "0 == 0" alone already
        # passes the fts_exact check above. Require positive output whenever
        # there was anything to materialize.
        "positive_output": total_bytes == 0
        or bool(census.get("rows", {}).get("sessions"))
        and bool(census.get("rows", {}).get("messages")),
    }
    receipt: dict[str, Any] = {
        "format": RECEIPT_FORMAT,
        "label": config.label,
        "outcome": outcome,
        "candidate": identity,
        "environment": environment,
        "benchmark_implementation_sha256": benchmark_implementation_sha256(),
        "config": {
            "digest": config_digest(config),
            "argv": command[3:],
            "profile": config.profile,
            "overrides": sorted(key for key, _value in config.extra_env),
        },
        "corpus": {
            "path": str(config.corpus),
            "kind": manifest["kind"],
            "digest": manifest["digest"],
            "file_count": manifest["file_count"],
            "total_bytes": total_bytes,
            "by_origin": manifest["by_origin"],
            "parameters": {k: v for k, v in manifest.get("parameters", {}).items() if k != "population"},
            "population": manifest.get("parameters", {}).get("population"),
        },
        "timing_s": {
            "wall": round(wall_s, 3),
            "terminal": terminal_at,
            "promotion": promoted,
            "first_chunk_done": milestones.get("first_chunk_done_s"),
            "last_chunk_done": milestones.get("last_chunk_done_s"),
            "preparation_done": milestones.get("preparation_done_s"),
            "shutdown": round(shutdown_s, 3),
        },
        "stages": batches,
        "writer": events.get("writer"),
        "chunks": events.get("chunks"),
        "intake_pages_by_class": events.get("intake_pages_by_class"),
        "by_source": events.get("by_source"),
        "projection": projection(
            manifest,
            events.get("by_source") or {},
            (milestones.get("last_chunk_done_s") or 0) - (milestones.get("preparation_done_s") or 0) or None,
        ),
        "process_tree": tree,
        "checks": checks,
        "budgets": {name: {"limit": limit} for name, limit in sorted(dict(config.budgets).items())},
        "final_observation": {
            key: getattr(final, key)
            for key in (
                "cursor_rows",
                "cursor_complete",
                "cursor_excluded",
                "cursor_failing",
                "cursor_deferred",
                "raw_rows",
                "raw_pending",
                "raw_failed",
                "memberships_pending",
                "open_debt",
                "debt_by_stage",
                "readiness",
            )
        },
        "archive": census,
        "output_fingerprint": fingerprint,
        "warnings_and_errors": events.get("warnings_and_errors"),
        "daemon_exit_code": exit_code,
        "observation_count": len(observations),
        "progress": _progress_timeline(observations),
        # Unrounded: ``refresh`` re-reduces the event log from this origin.
        "started_at_unix": started_wall,
        "clock_step_s": round(clock_step_s, 3),
    }
    _derive_dependents(receipt)
    if paths["stacks"].exists():
        document = json.loads(paths["stacks"].read_text(encoding="utf-8"))
        receipt["thread_cpu_s"] = thread_cpu_summary(document)
    if config.profile and paths["stacks"].exists():
        from devtools.fresh_build_bench.profile_report import summarise

        document = json.loads(paths["stacks"].read_text(encoding="utf-8"))
        summary = summarise(document, top=25, thread_filter=None)
        writer_summary = summarise(document, top=25, thread_filter="polylogue-writer:watcher.live_ingest.full")
        receipt["profile"] = {
            "sampler_overhead_s": summary["sampler_overhead_s"],
            "threads_by_cpu_s": summary["threads_by_cpu_s"],
            "leaf_kind_cpu_s": summary["leaf_kind_cpu_s"],
            "ingest_writer_leaf_kind_wall_s": writer_summary["leaf_kind_wall_s"],
            "ingest_writer_leaf_kind_cpu_s": writer_summary["leaf_kind_cpu_s"],
        }
    return receipt


# ---------------------------------------------------------------------------
# rendering and comparison


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render(receipt: dict[str, Any]) -> str:
    lines = [
        f"fresh-build receipt '{receipt['label']}' outcome={receipt['outcome']}"
        f" qualified={receipt.get('qualified')}"
        f" candidate={receipt['candidate']['git_sha'][:12]}{' (dirty)' if receipt['candidate']['dirty'] else ''}",
        f"corpus {receipt['corpus']['kind']} {receipt['corpus']['digest'][:12]}"
        f" files={_fmt(receipt['corpus']['file_count'])} bytes={receipt['corpus']['total_bytes'] / 2**20:,.1f} MiB",
    ]
    timing = receipt["timing_s"]
    lines.append("timing  " + "  ".join(f"{key}={_fmt(value)}" for key, value in timing.items() if key != "shutdown"))
    lines.append("throughput  " + "  ".join(f"{k}={_fmt(v)}" for k, v in receipt["throughput"].items()))
    thread_cpu = receipt.get("thread_cpu_s") or {}
    if thread_cpu:
        top = ", ".join(f"{name} {seconds}" for name, seconds in (thread_cpu.get("writer_by_actor") or [])[:4])
        lines.append(
            f"writer cpu  >={_fmt(thread_cpu.get('writer_total_lower_bound'))}s  ({top})"
            f"  process={_fmt(thread_cpu.get('process_total'))}s unattributed={_fmt(thread_cpu.get('unattributed'))}s"
        )
    writer = receipt.get("writer") or {}
    lines.append(
        f"writer  busy_share_to_promotion={_fmt(writer.get('busy_share_to_promotion'))}"
        f"  busy_s={_fmt(writer.get('busy_s_to_promotion'))}  queue_max={_fmt(writer.get('queue_depth_max'))}"
    )
    for actor, stats in (writer.get("by_actor") or [])[:10]:
        lines.append(f"    {stats['hold_s']:9.2f}s x{int(stats['count']):<5} {actor}")
    tree = receipt.get("process_tree") or {}
    lines.append(
        f"process  rss_peak={tree.get('rss_peak_bytes', 0) / 2**20:,.0f} MiB"
        f"  cpu_to_promotion={_fmt(tree.get('cpu_seconds_to_promotion'))}s"
        f"  mean_cores={_fmt(tree.get('mean_cores_to_promotion'))}"
        f"  io_write={tree.get('io_write_bytes', 0) / 2**20:,.0f} MiB"
    )
    stages = receipt.get("stages") or {}
    lines.append("stage buckets (sum over batches, s)")
    for key, value in stages.get("stage_buckets_seconds") or []:
        lines.append(f"    {value:9.2f}  {key}")
    lines.append("top stages (s)")
    for key, value in (stages.get("stage_seconds") or [])[:15]:
        lines.append(f"    {value:9.2f}  {key}")
    lines.append("by source")
    for key, value in (receipt.get("by_source") or {}).items():
        lines.append(f"    {key:<16} files={_fmt(int(value['files']))} seconds={_fmt(value['seconds'])}")
    proj = receipt.get("projection") or {}
    if proj.get("by_origin"):
        lines.append(
            f"projection  intake_wall={_fmt(proj.get('intake_wall_s'))}s  wall_scale={_fmt(proj.get('wall_scale'))}"
            f"  projected_intake_hours={_fmt(proj.get('projected_intake_hours'))}"
        )
        for origin, row in proj["by_origin"].items():
            lines.append(
                f"    {origin:<12} {row['s_per_mib']:.3f} s/MiB  sample={row['sample_mib']} MiB"
                f"  population={row.get('population_mib', '-')} MiB  projected={row.get('projected_s', '-')} s"
            )
    problems = receipt.get("warnings_and_errors") or {}
    if problems:
        lines.append("warnings/errors")
        for key, value in list(problems.items())[:10]:
            lines.append(f"    {value:6d}  {key}")
    failed_checks = [name for name, ok in (receipt.get("checks") or {}).items() if not ok]
    lines.append(f"checks  {'all pass' if not failed_checks else 'FAILED: ' + ', '.join(failed_checks)}")
    for name, row in (receipt.get("budgets") or {}).items():
        lines.append(
            f"budget  {name} observed={_fmt(row['observed'])} limit={_fmt(row['limit'])}"
            f" {'pass' if row['pass'] else 'FAIL'}"
        )
    final = receipt.get("final_observation") or {}
    lines.append(
        "final  "
        + "  ".join(f"{key}={_fmt(value)}" for key, value in final.items() if key not in {"debt_by_stage", "readiness"})
    )
    if final.get("debt_by_stage"):
        lines.append(f"    open debt by stage: {final['debt_by_stage']}")
    if final.get("readiness"):
        lines.append(f"    readiness: {final['readiness']}")
    fingerprint = receipt.get("output_fingerprint")
    if fingerprint:
        lines.append(f"output digest {fingerprint['digest'][:16]}")
    return "\n".join(lines)


def comparability_problems(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Why two receipts' numbers or outputs may not be set side by side."""
    problems = []
    if before["corpus"]["digest"] != after["corpus"]["digest"]:
        problems.append("different corpora")
    if before["config"]["digest"] != after["config"]["digest"]:
        problems.append("different run configurations (profile, overrides or budgets)")
    if before.get("benchmark_implementation_sha256") != after.get("benchmark_implementation_sha256"):
        problems.append("different benchmark implementation")
    # Version and GIL mode alone equate two builds of one version (PGO or
    # not); the build string and the resolved executable tell them apart.
    for key in ("python", "gil_enabled", "python_build", "python_executable"):
        if before["environment"].get(key) != after["environment"].get(key):
            problems.append(f"different interpreter ({key})")
    # Host capacity and storage explain wall, CPU and throughput deltas on
    # their own; a comparison is a controlled one only on the same kind of host.
    # The effective limits are what the daemon sizes its pools and budgets
    # from: two runs on one host under different cgroup quotas differ.
    for key in (
        "machine",
        "cpu_model",
        "host_cpu_count",
        "host_mem_total_kib",
        "effective_cpus",
        "effective_memory_bytes",
        "work_filesystem",
        "work_device",
    ):
        if before["environment"].get(key) != after["environment"].get(key):
            problems.append(f"different host ({key})")
    for side, receipt in (("before", before), ("after", after)):
        # Integrity failures mean the declared inputs or the measurement
        # evidence are invalid; no waiver makes such a run's numbers or
        # output digests admissible.
        for check in INTEGRITY_CHECKS:
            if (receipt.get("checks") or {}).get(check) is False:
                problems.append(f"{side} run failed integrity check {check}")
        if not receipt.get("qualified"):
            problems.append(f"{side} run is not qualified (outcome {receipt.get('outcome')})")
    return problems


#: Checks whose failure invalidates a receipt rather than leaving its build
#: unsettled: the candidate or corpus changed under the run, or the event log
#: the receipt reduces lost events. ``--allow-unqualified`` never waives them.
INTEGRITY_CHECKS: Final = ("candidate_unchanged", "corpus_unchanged", "events_lossless")


def compare(before: dict[str, Any], after: dict[str, Any], *, allow_unqualified: bool = False) -> tuple[bool, str]:
    """Render the deltas; the flag says whether the comparison is admissible.

    Different corpora or configurations are never admissible, nor is a run
    that failed an integrity check (:data:`INTEGRITY_CHECKS`). An otherwise
    unqualified run is admissible only when asked for (``allow_unqualified``),
    for example to read a promoted index's output digests from a build whose
    derived phase did not settle; the verdict line says so.
    """
    problems = comparability_problems(before, after)
    # Only a run that promoted (and so has an output to compare) may have its
    # qualification waived; one that stalled before promotion never can.
    promoted = all((receipt.get("checks") or {}).get("promoted") is True for receipt in (before, after))
    blocking = [
        problem for problem in problems if "not qualified" not in problem or not allow_unqualified or not promoted
    ]
    lines = [f"NOT COMPARABLE: {problem}" for problem in blocking]
    lines += [f"WARNING: {problem}" for problem in problems if problem not in blocking]
    lines.append(
        f"before {before['label']} {before['candidate']['git_sha'][:12]}  ->  after {after['label']}"
        f" {after['candidate']['git_sha'][:12]}"
    )

    def row(name: str, a: Any, b: Any) -> None:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a:
            lines.append(f"  {name:<40} {_fmt(a):>14} {_fmt(b):>14}  {b / a:6.3f}x")
        else:
            lines.append(f"  {name:<40} {_fmt(a):>14} {_fmt(b):>14}")

    for key in ("promotion", "terminal", "derived_after_promotion", "first_chunk_done", "wall"):
        row(f"timing.{key}", before["timing_s"].get(key), after["timing_s"].get(key))
    for key in ("mib_per_s_to_promotion", "mib_per_s_to_terminal"):
        row(f"throughput.{key}", before["throughput"].get(key), after["throughput"].get(key))
    row(
        "writer.busy_s_to_promotion",
        (before.get("writer") or {}).get("busy_s_to_promotion"),
        (after.get("writer") or {}).get("busy_s_to_promotion"),
    )
    for key in ("rss_peak_bytes", "cpu_seconds_to_promotion", "io_write_bytes"):
        row(f"process.{key}", (before.get("process_tree") or {}).get(key), (after.get("process_tree") or {}).get(key))
    a_buckets = dict((before.get("stages") or {}).get("stage_buckets_seconds") or [])
    b_buckets = dict((after.get("stages") or {}).get("stage_buckets_seconds") or [])
    for key in sorted(set(a_buckets) | set(b_buckets)):
        row(f"stage.{key}", a_buckets.get(key), b_buckets.get(key))
    a_fp, b_fp = before.get("output_fingerprint"), after.get("output_fingerprint")
    if blocking:
        lines.append("output: not compared (receipts are not comparable)")
    elif a_fp and b_fp:
        if a_fp["digest"] == b_fp["digest"]:
            lines.append("output: IDENTICAL (per-table digests match)")
        else:
            lines.append("output: DIFFERENT")
            for table in sorted(set(a_fp["tables"]) | set(b_fp["tables"])):
                left, right = a_fp["tables"].get(table), b_fp["tables"].get(table)
                if left != right:
                    lines.append(f"    {table}: {left} != {right}")
    else:
        lines.append("output: not compared (fingerprint missing)")
    return not blocking, "\n".join(lines)


def refresh(receipt_path: Path) -> dict[str, Any]:
    """Recompute a receipt's derived sections from its run directory.

    The event log and ops ledger are the evidence; the receipt is a view of
    them. Refreshing lets a finished run be read with a newer report without
    rerunning the build. Identity, environment, the terminal time and the
    output fingerprint are kept as recorded; everything derived from the
    re-read milestones is recomputed.
    """
    receipt: dict[str, Any] = json.loads(receipt_path.read_text(encoding="utf-8"))
    work = receipt_path.parent
    events = analyse_events(work / "events.jsonl", origin_unix=receipt["started_at_unix"])
    milestones = events.get("milestones_s", {})
    if milestones.get("promoted_s") != receipt["timing_s"].get("promotion"):
        # CPU to promotion was cut from the full sample series, which the
        # receipt does not retain; a moved milestone leaves it unknown, not stale.
        tree = receipt.get("process_tree") or {}
        for key in ("cpu_seconds_to_promotion", "mean_cores_to_promotion"):
            if key in tree:
                tree[key] = None
    receipt["stages"] = analyse_batches(work / "archive" / "ops.db")
    for key in ("writer", "chunks", "intake_pages_by_class", "by_source", "warnings_and_errors"):
        receipt[key] = events.get(key)
    receipt["timing_s"].update(
        {
            "promotion": milestones.get("promoted_s"),
            "first_chunk_done": milestones.get("first_chunk_done_s"),
            "last_chunk_done": milestones.get("last_chunk_done_s"),
            "preparation_done": milestones.get("preparation_done_s"),
        }
    )
    _derive_dependents(receipt)
    # The projection's inputs are sealed into the receipt itself; the
    # corpus directory may be gone by the time a newer report refreshes it.
    corpus = receipt["corpus"]
    manifest = {"by_origin": corpus.get("by_origin"), "parameters": {"population": corpus.get("population")}}
    receipt["projection"] = projection(
        manifest,
        events.get("by_source") or {},
        (milestones.get("last_chunk_done_s") or 0) - (milestones.get("preparation_done_s") or 0) or None,
    )
    # The refreshed sections were reduced by this implementation, the rest by
    # the recording one: the identity names both, so a refreshed receipt
    # compares only with receipts refreshed the same way.
    receipt["benchmark_implementation_sha256"] = (
        f"{receipt.get('benchmark_implementation_sha256')}+refresh:{benchmark_implementation_sha256()}"
    )
    # Atomic: the receipt is replaced only by a complete document.
    staging = receipt_path.with_name(receipt_path.name + ".tmp")
    staging.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    staging.replace(receipt_path)
    return receipt
