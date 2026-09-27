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
from contextlib import closing
from datetime import datetime
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
    (r"^full\.index\.", "index"),
    (r"^full\.index_parsed_write$", "index_total"),
    (r"graph_resolve|lineage|delegation|hook_paste|paste", "derived_inline"),
)

_TS_FORMAT: Final = "%Y-%m-%dT%H:%M:%S.%fZ"


def _ts(value: str) -> float:
    return datetime.strptime(value, _TS_FORMAT).timestamp()


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


_SOURCE_GROUP: Final = re.compile(r"batch ingested (\S+) — (\d+) in ([0-9.]+)s")


def analyse_events(path: Path) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not events:
        return {"event_count": 0}
    start = next((_ts(e["ts"]) for e in events if e.get("event") == "daemon.run.start"), _ts(events[0]["ts"]))

    def rel(event: dict[str, Any]) -> float:
        return round(_ts(event["ts"]) - start, 3)

    writer: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "hold_s": 0.0, "wait_s": 0.0, "max_hold_s": 0.0}
    )
    holds: list[tuple[float, float]] = []
    queued: list[int] = []
    chunks: list[dict[str, Any]] = []
    pages: dict[str, dict[str, float]] = defaultdict(lambda: {"pages": 0, "files": 0, "bytes": 0, "seconds": 0.0})
    by_source: dict[str, dict[str, float]] = defaultdict(lambda: {"groups": 0, "files": 0, "seconds": 0.0})
    problems: Counter[str] = Counter()
    milestones: dict[str, float] = {}
    preparation: list[float] = []
    for event in events:
        name = event.get("event", "")
        if name == "daemon.writer.released":
            actor = str(event.get("actor"))
            hold = float(event.get("hold_ms") or 0) / 1000
            entry = writer[actor]
            entry["count"] += 1
            entry["hold_s"] += hold
            entry["wait_s"] += float(event.get("wait_ms") or 0) / 1000
            entry["max_hold_s"] = max(entry["max_hold_s"], hold)
            released = rel(event)
            holds.append((released - hold, released))
            queued.append(int(event.get("queued") or 0))
        elif name == "live.ingest.chunk":
            chunks.append(
                {
                    "t": rel(event),
                    "files": int(event.get("files") or 0),
                    "bytes": int(event.get("bytes") or 0),
                    "seconds": float(event.get("duration_ms") or 0) / 1000,
                }
            )
            milestones.setdefault("first_chunk_done_s", rel(event))
        elif name == "daemon.intake.page":
            entry = pages[str(event.get("component"))]
            entry["pages"] += 1
            entry["files"] += int(event.get("files") or 0)
            entry["bytes"] += int(event.get("bytes") or 0)
            entry["seconds"] += float(event.get("duration_ms") or 0) / 1000
        elif name == "live.ingest.source_group":
            entry = by_source[str(event.get("source_name"))]
            entry["groups"] += 1
            entry["files"] += int(event.get("files") or 0)
            entry["seconds"] += float(event.get("duration_ms") or 0) / 1000
        elif name == "stdlib.record" and (match := _SOURCE_GROUP.search(str(event.get("error_detail") or ""))):
            entry = by_source[match.group(1)]
            entry["groups"] += 1
            entry["files"] += int(match.group(2))
            entry["seconds"] += float(match.group(3))
        elif name == "daemon.cold_build.preparation":
            preparation.append(rel(event))
        elif name == "daemon.cold_build.generation_created":
            milestones["generation_created_s"] = rel(event)
        elif name == "daemon.cold_build.generation_promoted":
            milestones["promoted_s"] = rel(event)
        if event.get("level") in {"warning", "error"}:
            problems[f"{name}:{event.get('reason')}"] += 1
    if preparation:
        milestones["preparation_done_s"] = max(preparation)
    # Intake ends at the last chunk before promotion. Chunks after it are the
    # promoted generation's own live passes (raw retention, re-offered
    # files), which belong to the derived phase, not to intake throughput.
    intake_chunks = [
        chunk["t"] for chunk in chunks if "promoted_s" not in milestones or chunk["t"] <= milestones["promoted_s"]
    ]
    if intake_chunks:
        milestones["last_chunk_done_s"] = max(intake_chunks)
    milestones["post_promotion_chunks"] = len(chunks) - len(intake_chunks)
    last = rel(events[-1])
    build_end = milestones.get("promoted_s", last)
    busy_build = sum(max(0.0, min(end, build_end) - max(begin, 0.0)) for begin, end in holds if begin < build_end)
    busy_total = sum(end - begin for begin, end in holds)
    return {
        "event_count": len(events),
        "milestones_s": milestones,
        "writer": {
            "busy_s_to_promotion": round(busy_build, 3),
            "busy_share_to_promotion": round(busy_build / build_end, 4) if build_end > 0 else None,
            "busy_s_total": round(busy_total, 3),
            "queue_depth_max": max(queued, default=0),
            "queue_depth_mean": round(sum(queued) / len(queued), 3) if queued else 0.0,
            # Lists, not objects: receipts are written with sorted keys, and
            # the order here is the ranking.
            "by_actor": [
                [actor, {key: round(value, 4) for key, value in stats.items()}]
                for actor, stats in sorted(writer.items(), key=lambda item: -item[1]["hold_s"])
            ],
        },
        "chunks": {
            "count": len(chunks),
            "files": sum(chunk["files"] for chunk in chunks),
            "bytes": sum(chunk["bytes"] for chunk in chunks),
            "seconds": round(sum(chunk["seconds"] for chunk in chunks), 3),
            "seconds_p50": round(_percentile([chunk["seconds"] for chunk in chunks], 0.5), 3),
            "seconds_max": round(max((chunk["seconds"] for chunk in chunks), default=0.0), 3),
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


def output_fingerprint(archive: Path, promoted_index: str, scratch: Path) -> dict[str, Any]:
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
            rows = 0
            for row in read.execute(f'SELECT {selected} FROM "{table}"'):
                batch.append((json.dumps(_fact_row(row), ensure_ascii=True, separators=(",", ":")),))
                rows += 1
                if len(batch) >= 5000:
                    spool.executemany("INSERT INTO facts VALUES (?)", batch)
                    batch.clear()
            if batch:
                spool.executemany("INSERT INTO facts VALUES (?)", batch)
            digest = hashlib.sha256()
            digest.update(json.dumps(columns).encode())
            for (payload,) in spool.execute("SELECT payload FROM facts ORDER BY payload"):
                digest.update(payload.encode())
                digest.update(b"\n")
            tables[table] = {"rows": rows, "sha256": digest.hexdigest()}
    spool_path.unlink(missing_ok=True)
    overall = hashlib.sha256(json.dumps(tables, sort_keys=True).encode()).hexdigest()
    return {"digest": overall, "tables": tables}


# ---------------------------------------------------------------------------
# receipt


def _tree_summary(samples: list[tuple[float, int, float, int, int, int]], build_end_s: float | None) -> dict[str, Any]:
    if not samples:
        return {}
    rss = [sample[1] for sample in samples]
    last = samples[-1]
    to_end = [sample for sample in samples if build_end_s is None or sample[0] <= build_end_s]
    cpu_to_end = to_end[-1][2] if to_end else 0.0
    return {
        "rss_peak_bytes": max(rss),
        "rss_p95_bytes": int(_percentile([float(value) for value in rss], 0.95)),
        "rss_final_bytes": rss[-1],
        "cpu_seconds_total": round(last[2], 2),
        "cpu_seconds_to_promotion": round(cpu_to_end, 2),
        "mean_cores_to_promotion": round(cpu_to_end / to_end[-1][0], 3) if to_end and to_end[-1][0] > 0 else None,
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
_SOURCE_ORIGIN = {"claude-code": "claude-code", "codex": "codex", "gemini-cli": "gemini-cli", "chatgpt": "chatgpt"}


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
    return {
        "by_origin": rows,
        "serial_sample_s": round(serial_sample, 1),
        "intake_wall_s": intake_wall_s,
        "wall_scale": round(scale, 4) if scale else None,
        "projected_intake_hours": round(serial_projected * scale / 3600, 2) if scale and serial_projected else None,
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


def _progress_timeline(observations: list[Any]) -> list[tuple[float, int, int, int, bool]]:
    rows: list[tuple[float, int, int, int, bool]] = []
    for observation in observations:
        row = (
            observation.t,
            observation.cursor_complete,
            observation.raw_rows,
            observation.open_debt,
            observation.promoted_index is not None,
        )
        if not rows or rows[-1][1:] != row[1:]:
            rows.append(row)
    return rows


def config_digest(config: Any) -> str:
    """What must match, besides the corpus, for two receipts to compare."""
    payload = {
        "profile": config.profile,
        "extra_env": sorted(config.extra_env),
        "daemon_argv": ["run", "--no-browser-capture", "--no-api", "--cold-build-index"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_receipt(
    *,
    config: Any,
    manifest: dict[str, Any],
    paths: dict[str, Path],
    identity: dict[str, Any],
    environment: dict[str, Any],
    command: list[str],
    daemon_env: dict[str, str],
    started_wall: float,
    wall_s: float,
    outcome: str,
    terminal_at: float | None,
    exit_code: int | None,
    shutdown_s: float,
    observations: list[Any],
    final: Any,
    tree_samples: list[tuple[float, int, float, int, int, int]],
) -> dict[str, Any]:
    events = analyse_events(paths["events"])
    batches = analyse_batches(paths["archive"] / "ops.db")
    milestones = events.get("milestones_s", {})
    promoted = milestones.get("promoted_s")
    census = archive_census(paths["archive"], final.promoted_index)
    fingerprint: dict[str, Any] | None = None
    # An interrupted run has seconds before its supervisor escalates; the
    # fingerprint is the one step that scales with the archive.
    if config.fingerprint and final.promoted_index is not None and outcome != "interrupted":
        fingerprint = output_fingerprint(paths["archive"], final.promoted_index, paths["work"] / "tmp")
    total_bytes = int(manifest["total_bytes"])
    tree = _tree_summary(tree_samples, promoted)
    checks = {
        "promoted": final.promoted_index is not None,
        "intake_complete": final.intake_complete,
        "raw_parse_failures_zero": final.raw_failed == 0,
        "memberships_settled": final.memberships_pending == 0,
        "debt_zero": final.open_debt == 0,
        "readiness_complete": final.readiness_complete,
        "fts_exact": census.get("messages_fts_rows") is not None
        and census.get("messages_fts_rows") == census.get("fts_indexable_rows"),
    }
    budgets = evaluate_budgets(
        dict(config.budgets),
        {
            "rss_peak_mib": tree["rss_peak_bytes"] / 2**20 if tree.get("rss_peak_bytes") else None,
            "promotion_s": promoted,
            "terminal_s": terminal_at,
        },
    )
    receipt: dict[str, Any] = {
        "format": RECEIPT_FORMAT,
        "label": config.label,
        "outcome": outcome,
        "candidate": identity,
        "environment": environment,
        "config": {
            "digest": config_digest(config),
            "argv": command[3:],
            "profile": config.profile,
            "overrides": sorted(
                key for key in daemon_env if key.startswith("POLYLOGUE_") and key not in _DRIVER_OWNED_ENV
            ),
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
            "derived_after_promotion": round(terminal_at - promoted, 3)
            if terminal_at is not None and promoted is not None
            else None,
            "shutdown": round(shutdown_s, 3),
        },
        "throughput": {
            "mib_per_s_to_promotion": round(total_bytes / 2**20 / promoted, 3) if promoted else None,
            "mib_per_s_to_terminal": round(total_bytes / 2**20 / terminal_at, 3) if terminal_at else None,
            "files_per_s_to_promotion": round(manifest["file_count"] / promoted, 3) if promoted else None,
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
        "budgets": budgets,
        # A qualified build finished, satisfied every terminal check and
        # stayed inside every asserted budget.
        "qualified": outcome == "terminal" and all(checks.values()) and all(row["pass"] for row in budgets.values()),
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
        # Compact progress timeline: (t, cursors complete, raw rows, open debt,
        # promoted), one row per change.
        "progress": _progress_timeline(observations),
        "started_at_unix": round(started_wall, 3),
    }
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


_DRIVER_OWNED_ENV: Final = frozenset(
    {
        "POLYLOGUE_ARCHIVE_ROOT",
        "POLYLOGUE_SINEX_MODE",
        "POLYLOGUE_LOG_FORMAT",
        "POLYLOGUE_LOG_FILE",
        "POLYLOGUE_BENCH_STACK_SAMPLES",
        "POLYLOGUE_BENCH_STACK_INTERVAL_S",
        "POLYLOGUE_CONFIG",
    }
)


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


def compare(before: dict[str, Any], after: dict[str, Any]) -> str:
    lines = []
    if before["corpus"]["digest"] != after["corpus"]["digest"]:
        lines.append("WARNING: different corpora; numbers are not comparable")
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
    if a_fp and b_fp:
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
    return "\n".join(lines)


def refresh(receipt_path: Path) -> dict[str, Any]:
    """Recompute a receipt's derived sections from its run directory.

    The event log and ops ledger are the evidence; the receipt is a view of
    them. Refreshing lets a finished run be read with a newer report without
    rerunning the build. Identity, environment, timing and the output
    fingerprint are kept as recorded.
    """
    from devtools.fresh_build_bench.corpus import load_manifest

    receipt: dict[str, Any] = json.loads(receipt_path.read_text(encoding="utf-8"))
    work = receipt_path.parent
    events = analyse_events(work / "events.jsonl")
    milestones = events.get("milestones_s", {})
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
    corpus_path = receipt["corpus"].get("path")
    if corpus_path and Path(corpus_path, "manifest.json").exists():
        manifest = load_manifest(Path(corpus_path))
        receipt["projection"] = projection(
            manifest,
            events.get("by_source") or {},
            (milestones.get("last_chunk_done_s") or 0) - (milestones.get("preparation_done_s") or 0) or None,
        )
    receipt_path.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return receipt
