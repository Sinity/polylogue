"""Build an executable degraded-archive self-healing proof."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

from devtools import repo_root
from devtools.daemon_workload_probe import probe
from polylogue.demo import seed_demo_archive, verify_demo_archive
from polylogue.storage.fts.dangling_repair import configure_bounded_repair_connection, repair_stale_fts_rows
from polylogue.storage.fts.freshness import STALE, record_fts_surface_state_sync
from polylogue.storage.fts.fts_lifecycle import message_fts_search_readiness_sync
from polylogue.storage.sqlite.maintenance import maybe_optimize_archive_tiers
from polylogue.storage.sqlite.wal_checkpoint import maybe_checkpoint_archive_wals


@dataclass(frozen=True, slots=True)
class DegradedArchiveProofResult:
    """Summary of one degraded-copy proof run."""

    ok: bool
    healing_driver: str
    degraded_inputs: tuple[str, ...]
    daemon_owned_primitives: tuple[str, ...]
    always_running_paths: tuple[str, ...]
    out_dir: str
    archive_root: str
    archive_preserved: bool
    seeded_sessions: int
    seeded_messages: int
    demo_verified: bool
    fts_ready_clean: bool
    fts_ready_degraded: bool
    fts_ready_after: bool
    wal_bytes_degraded: int
    wal_bytes_after: int
    wal_degraded_observed: bool
    checkpoint_modes: tuple[str, ...]
    checkpoint_errors: tuple[str, ...]
    optimize_ran: int
    optimize_errors: tuple[str, ...]
    fts_repair_success: bool
    fts_repair_detail: str
    artifact_json: str
    artifact_markdown: str

    def to_payload(self) -> dict[str, object]:
        """Return the stable JSON payload for devtools and demo artifacts."""

        return asdict(self)


def _index_db(root: Path) -> Path:
    return root / "index.db"


def _fts_ready(root: Path) -> bool:
    with sqlite3.connect(f"file:{_index_db(root)}?mode=ro", uri=True) as conn:
        return bool(message_fts_search_readiness_sync(conn)["ready"])


def _total_wal_bytes(root: Path) -> int:
    total = 0
    for name in ("source.db", "index.db", "embeddings.db", "user.db", "ops.db"):
        wal = root / f"{name}-wal"
        if wal.exists():
            total += wal.stat().st_size
    return total


def _delete_planner_stats(root: Path) -> None:
    for name in ("source.db", "index.db", "embeddings.db", "user.db", "ops.db"):
        db = root / name
        if not db.exists():
            continue
        with sqlite3.connect(db) as conn:
            try:
                conn.execute("DELETE FROM sqlite_stat1")
                conn.commit()
            except sqlite3.Error:
                conn.rollback()


def _mark_message_fts_stale(root: Path) -> None:
    with sqlite3.connect(_index_db(root)) as conn:
        record_fts_surface_state_sync(
            conn,
            surface="messages_fts",
            state=STALE,
            source_rows=999,
            indexed_rows=0,
            detail="devtools degraded archive proof",
        )
        conn.commit()


def _open_wal_degradation_connection(root: Path) -> sqlite3.Connection:
    """Create committed WAL activity and keep the connection open for proof."""

    conn = sqlite3.connect(_index_db(root))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("CREATE TABLE IF NOT EXISTS _devtools_degraded_wal_probe (id INTEGER PRIMARY KEY, payload BLOB)")
    conn.execute("INSERT INTO _devtools_degraded_wal_probe(payload) VALUES (zeroblob(65536))")
    conn.commit()
    conn.execute("DELETE FROM _devtools_degraded_wal_probe")
    conn.commit()
    return conn


def _repair_fts(root: Path) -> tuple[bool, str]:
    with sqlite3.connect(_index_db(root)) as conn:
        configure_bounded_repair_connection(conn)
        outcome = repair_stale_fts_rows(conn)
        conn.commit()
    return outcome.success, outcome.detail


def _write_markdown(path: Path, result: DegradedArchiveProofResult) -> None:
    lines = [
        "# Degraded Archive Proof",
        "",
        f"- ok: `{result.ok}`",
        f"- healing driver: `{result.healing_driver}`",
        f"- degraded inputs: `{', '.join(result.degraded_inputs)}`",
        f"- daemon-owned primitives: `{', '.join(result.daemon_owned_primitives)}`",
        f"- always-running paths: `{', '.join(result.always_running_paths)}`",
        f"- work archive: `{result.archive_root}`",
        f"- work archive preserved: `{result.archive_preserved}`",
        f"- seeded sessions/messages: `{result.seeded_sessions}` / `{result.seeded_messages}`",
        f"- FTS ready: clean `{result.fts_ready_clean}` -> degraded `{result.fts_ready_degraded}` -> after `{result.fts_ready_after}`",
        f"- WAL bytes: degraded `{result.wal_bytes_degraded}` -> after `{result.wal_bytes_after}`",
        f"- WAL degradation observed: `{result.wal_degraded_observed}`",
        f"- checkpoint modes: `{', '.join(result.checkpoint_modes) or 'none'}`",
        f"- optimize ran: `{result.optimize_ran}` tiers",
        f"- optimize errors: `{', '.join(result.optimize_errors) or 'none'}`",
        f"- FTS repair: `{result.fts_repair_success}` — {result.fts_repair_detail}",
        "",
        "This proof uses a deterministic demo archive copy, deliberately corrupts only rebuildable derived state, then runs the same bounded FTS repair, WAL checkpoint, and PRAGMA optimize primitives owned by daemon startup/convergence/periodic upkeep and direct ingest commit upkeep.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_degraded_archive_proof(
    out_dir: Path,
    *,
    force: bool = True,
    keep_archive: bool = False,
) -> DegradedArchiveProofResult:
    """Seed, degrade, self-heal, and report a deterministic archive copy."""

    out_dir = out_dir.expanduser().resolve()
    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_root = out_dir / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)

    seed = asyncio.run(seed_demo_archive(archive_root, force=True, with_overlays=True))
    verification = verify_demo_archive(archive_root, require_overlays=True)

    clean_probe = probe(_index_db(archive_root), exact_table_counts=True, exact_derived_counts=True)
    clean_fts_ready = _fts_ready(archive_root)

    _delete_planner_stats(archive_root)
    _mark_message_fts_stale(archive_root)
    wal_conn = _open_wal_degradation_connection(archive_root)
    try:
        degraded_probe = probe(_index_db(archive_root), exact_table_counts=True, exact_derived_counts=True)
        degraded_fts_ready = _fts_ready(archive_root)
        degraded_wal_bytes = _total_wal_bytes(archive_root)

        fts_success, fts_detail = _repair_fts(archive_root)
        checkpoint_observations = maybe_checkpoint_archive_wals(
            archive_root,
            reason="degraded_archive_proof",
            warn_bytes=1,
            truncate_bytes=1,
            timeout_s=5.0,
        )
    finally:
        wal_conn.close()

    optimize_observations = maybe_optimize_archive_tiers(archive_root, reason="degraded_archive_proof")
    after_probe = probe(_index_db(archive_root), exact_table_counts=True, exact_derived_counts=True)
    after_fts_ready = _fts_ready(archive_root)
    after_wal_bytes = _total_wal_bytes(archive_root)

    checkpoint_errors = tuple(str(obs.error) for obs in checkpoint_observations if obs.error)
    optimize_errors = tuple(str(obs.error) for obs in optimize_observations if obs.error)
    checkpoint_modes = tuple(obs.mode for obs in checkpoint_observations if obs.ran)
    ok = (
        verification.ok
        and clean_probe["ok"]
        and degraded_probe["ok"]
        and after_probe["ok"]
        and clean_fts_ready
        and not degraded_fts_ready
        and after_fts_ready
        and degraded_wal_bytes > 0
        and fts_success
        and not checkpoint_errors
        and not optimize_errors
    )

    artifact_json = out_dir / "degraded-archive-proof.json"
    artifact_markdown = out_dir / "degraded-archive-proof.md"
    result = DegradedArchiveProofResult(
        ok=ok,
        healing_driver="daemon_owned_upkeep_primitives",
        degraded_inputs=("messages_fts_freshness", "split_tier_wal", "split_tier_sqlite_stat1"),
        daemon_owned_primitives=(
            "repair_stale_fts_rows",
            "maybe_checkpoint_archive_wals",
            "maybe_optimize_archive_tiers",
        ),
        always_running_paths=(
            "daemon_startup_fts_readiness",
            "daemon_convergence_fts_surface_debt",
            "daemon_periodic_wal_checkpoint",
            "daemon_periodic_db_optimize",
            "direct_archive_ingest_post_commit_upkeep",
        ),
        out_dir=str(out_dir),
        archive_root=str(archive_root),
        archive_preserved=keep_archive,
        seeded_sessions=seed.session_count,
        seeded_messages=seed.message_count,
        demo_verified=verification.ok,
        fts_ready_clean=clean_fts_ready,
        fts_ready_degraded=degraded_fts_ready,
        fts_ready_after=after_fts_ready,
        wal_bytes_degraded=degraded_wal_bytes,
        wal_bytes_after=after_wal_bytes,
        wal_degraded_observed=degraded_wal_bytes > 0,
        checkpoint_modes=checkpoint_modes,
        checkpoint_errors=checkpoint_errors,
        optimize_ran=sum(1 for obs in optimize_observations if obs.ran),
        optimize_errors=optimize_errors,
        fts_repair_success=fts_success,
        fts_repair_detail=fts_detail,
        artifact_json=str(artifact_json),
        artifact_markdown=str(artifact_markdown),
    )
    artifact_json.write_text(json.dumps(result.to_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(artifact_markdown, result)
    if not keep_archive:
        shutil.rmtree(archive_root)
    return result


def _default_out_dir() -> Path:
    return repo_root() / ".local" / "degraded-archive-proof" / "current"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a degraded archive self-healing proof artifact.")
    parser.add_argument(
        "--out-dir", type=Path, default=_default_out_dir(), help="Output directory for proof artifacts."
    )
    parser.add_argument("--no-force", action="store_true", help="Do not clear an existing output directory first.")
    parser.add_argument(
        "--keep-archive", action="store_true", help="Keep the temporary degraded archive for debugging."
    )
    parser.add_argument("--json", action="store_true", help="Emit the proof payload as JSON.")
    return parser


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    stream = stdout
    started = time.perf_counter()
    result = run_degraded_archive_proof(args.out_dir, force=not args.no_force, keep_archive=args.keep_archive)
    payload: dict[str, Any] = result.to_payload()
    payload["duration_s"] = round(time.perf_counter() - started, 6)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True), file=stream)
    else:
        print(f"degraded archive proof: {'ok' if result.ok else 'failed'}", file=stream)
        print(f"  artifact: {result.artifact_markdown}", file=stream)
        print(f"  json: {result.artifact_json}", file=stream)
    return 0 if result.ok else 1


__all__ = ["DegradedArchiveProofResult", "main", "run_degraded_archive_proof"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
