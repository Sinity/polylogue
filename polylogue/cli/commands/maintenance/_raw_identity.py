"""Raw-identity repair commands: missing cursors, quarantined/duplicate/mismatched raws."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

import click

from polylogue.archive.raw_materialization import source_path_native_id_candidates
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.paths import archive_root, render_root


def _raw_blob_path_for_hash(root: Path, blob_hash: bytes | str) -> Path | None:
    hex_hash = blob_hash.hex() if isinstance(blob_hash, bytes) else str(blob_hash).lower()
    if len(hex_hash) != 64 or any(char not in "0123456789abcdef" for char in hex_hash):
        return None
    return root / "blob" / hex_hash[:2] / hex_hash[2:]


def _missing_raw_blob_cursor_candidates(root: Path, *, limit: int | None = None) -> list[dict[str, object]]:
    source_db = root / "source.db"
    index_db = root / "index.db"
    ops_db = root / "ops.db"
    if not source_db.exists() or not index_db.exists() or not ops_db.exists():
        return []
    conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    ops_conn = sqlite3.connect(f"file:{ops_db}?mode=ro", uri=True)
    ops_conn.row_factory = sqlite3.Row
    try:
        conn.execute("ATTACH DATABASE ? AS index_tier", (str(index_db),))
        rows = conn.execute(
            """
            SELECT
                r.raw_id,
                r.origin,
                r.native_id,
                r.source_path,
                r.blob_hash,
                r.blob_size,
                r.validation_status,
                r.parse_error
            FROM raw_sessions AS r
            LEFT JOIN index_tier.sessions AS s_by_raw ON s_by_raw.raw_id = r.raw_id
            LEFT JOIN index_tier.sessions AS s_by_native
              ON r.native_id IS NOT NULL
             AND s_by_native.origin = r.origin
             AND s_by_native.native_id = r.native_id
            WHERE r.blob_hash IS NOT NULL
              AND r.source_path IS NOT NULL
              AND s_by_raw.raw_id IS NULL
              AND s_by_native.native_id IS NULL
              AND NOT (
                r.validation_status = 'skipped'
                AND r.parsed_at_ms IS NOT NULL
                AND r.parse_error IS NULL
              )
            ORDER BY r.origin, r.blob_size DESC, r.raw_id
            """
        ).fetchall()
        candidates: list[dict[str, object]] = []
        seen_paths: set[str] = set()
        for row in rows:
            source_path = str(row["source_path"] or "")
            if not source_path or source_path in seen_paths:
                continue
            blob_path = _raw_blob_path_for_hash(root, row["blob_hash"])
            if blob_path is None or blob_path.exists() or not Path(source_path).exists():
                continue
            if _raw_materialized_by_source_path_candidate(conn, row):
                continue
            cursor = ops_conn.execute(
                "SELECT stat_size, byte_offset, updated_at_ms FROM ingest_cursor WHERE source_path = ?",
                (source_path,),
            ).fetchone()
            if cursor is None:
                continue
            candidates.append(
                {
                    "source_path": source_path,
                    "raw_id": str(row["raw_id"]),
                    "origin": str(row["origin"] or ""),
                    "native_id": str(row["native_id"] or ""),
                    "blob_path": str(blob_path),
                    "blob_size": int(row["blob_size"] or 0),
                    "cursor_stat_size": int(cursor["stat_size"] or 0),
                    "cursor_byte_offset": int(cursor["byte_offset"] or 0),
                    "cursor_updated_at_ms": int(cursor["updated_at_ms"] or 0),
                }
            )
            seen_paths.add(source_path)
            if limit is not None and len(candidates) >= limit:
                break
        return candidates
    finally:
        ops_conn.close()
        conn.close()


def _raw_materialized_by_source_path_candidate(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    origin = str(row["origin"] or "")
    if not origin:
        return False
    for native_id in source_path_native_id_candidates(str(row["source_path"] or "")):
        existing = conn.execute(
            """
            SELECT 1
            FROM index_tier.sessions
            WHERE origin = ?
              AND native_id = ?
            LIMIT 1
            """,
            (origin, native_id),
        ).fetchone()
        if existing is not None:
            return True
    return False


@click.command("missing-raw-blob-cursors")
@click.option("--apply", "apply_changes", is_flag=True, help="Delete matching rebuildable live cursor rows.")
@click.option("--limit", "-l", type=int, default=None, help="Limit the number of candidate source paths.")
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
@click.pass_obj
def missing_raw_blob_cursors_command(
    env: AppEnv,
    apply_changes: bool,
    limit: int | None,
    output_format: str,
) -> None:
    """Invalidate cursors hiding missing raw-blob re-acquisition debt.

    This command only touches ``ops.db.ingest_cursor`` rows. It leaves
    source-tier raw rows, source files, blobs, index rows, and user state
    intact so the next daemon catch-up can re-acquire through the normal
    ingestion path.
    """
    del env
    root = archive_root()
    candidates = _missing_raw_blob_cursor_candidates(root, limit=limit)
    deleted = 0
    if apply_changes and candidates:
        ops_db = root / "ops.db"
        with sqlite3.connect(ops_db) as conn:
            for candidate in candidates:
                deleted += conn.execute(
                    "DELETE FROM ingest_cursor WHERE source_path = ?",
                    (str(candidate["source_path"]),),
                ).rowcount
            conn.commit()

    payload = {
        "archive_root": str(root),
        "mode": "apply" if apply_changes else "dry-run",
        "candidate_count": len(candidates),
        "deleted_cursor_count": deleted,
        "candidates": candidates,
        "next_action": "restart or run polylogued catch-up" if apply_changes and deleted else None,
    }
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    action = "Deleted" if apply_changes else "Would delete"
    click.echo(f"{action} {deleted if apply_changes else len(candidates)} live cursor row(s)")
    for candidate in candidates[:10]:
        click.echo(
            f"  {candidate['origin']} {candidate['source_path']} "
            f"raw={candidate['raw_id']} blob_size={candidate['blob_size']}"
        )
    if len(candidates) > 10:
        click.echo(f"  ... {len(candidates) - 10} more")
    if apply_changes and deleted:
        click.echo("Next: restart or run polylogued catch-up.")


@click.command("quarantined-accepted-raws")
@click.option("--raw-id", "raw_ids", multiple=True, required=True, help="Exact retained raw SHA-256 id (repeatable).")
@click.option("--apply", "apply_changes", is_flag=True, help="Apply only after every target passes exact proof.")
@click.option("--proof-digest", help="Exact aggregate digest emitted by the matching dry-run.")
@click.option(
    "--receipt",
    "receipt_path",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Required append-only operator recovery receipt path for --apply.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
)
@click.pass_obj
def quarantined_accepted_raws_command(
    env: AppEnv,
    raw_ids: tuple[str, ...],
    apply_changes: bool,
    proof_digest: str | None,
    receipt_path: Path | None,
    output_format: str,
) -> None:
    """Repair a typed accepted full raw whose byte authority stayed quarantined.

    The actuator revalidates the existing immutable selected-baseline receipt,
    retained blob bytes, parser-normalized session identity/content, and the
    current accepted head. It never changes the index head or its receipt.
    Apply additionally writes an exclusive, fsynced planned→applied operator
    receipt so the source-only refinement is crash-resumable and auditable.
    """
    del env
    if apply_changes and receipt_path is None:
        raise click.UsageError("--apply requires --receipt PATH")
    if apply_changes and proof_digest is None:
        raise click.UsageError("--apply requires --proof-digest from the exact dry-run")
    root = archive_root()
    config = Config(archive_root=root, render_root=render_root(), sources=[], db_path=root / "index.db")
    from polylogue.storage.repair import repair_quarantined_accepted_raws

    try:
        report = repair_quarantined_accepted_raws(
            config,
            list(raw_ids),
            apply=apply_changes,
            receipt_path=receipt_path,
            proof_digest=proof_digest,
        )
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    payload = asdict(report)
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    click.echo(
        f"{report.mode}: requested={report.requested_count} eligible={report.eligible_count} "
        f"already_repaired={report.already_repaired_count} repaired={report.repaired_count} "
        f"ineligible={report.ineligible_count}"
    )
    click.echo(f"Proof digest: {report.proof_digest}")
    for item in report.items:
        click.echo(f"  {item.raw_id} {item.status} proof={item.proof_digest or 'unavailable'}: {item.reason}")
    if report.receipt_path is not None:
        click.echo(f"Receipt: {report.receipt_path}")


@click.command("browser-capture-origin-mismatches")
@click.option("--raw-id", "raw_ids", multiple=True, required=True, help="Exact mismatched raw id (repeatable).")
@click.option("--apply", "apply_changes", is_flag=True, help="Copy forward only after every target passes proof.")
@click.option("--proof-digest", help="Exact aggregate digest emitted by the matching dry-run.")
@click.option(
    "--receipt",
    "receipt_path",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Required append-only operator recovery receipt path for --apply.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
)
@click.pass_obj
def browser_capture_origin_mismatches_command(
    env: AppEnv,
    raw_ids: tuple[str, ...],
    apply_changes: bool,
    proof_digest: str | None,
    receipt_path: Path | None,
    output_format: str,
) -> None:
    """Copy mismatched browser captures forward under parsed origin authority.

    The old raw, blob, membership, byte head, and application receipts remain
    immutable.  Apply creates a new raw reference to the exact retained blob,
    records a canonical head, and advances only the derived session raw pointer.
    """
    del env
    if apply_changes and receipt_path is None:
        raise click.UsageError("--apply requires --receipt PATH")
    if apply_changes and proof_digest is None:
        raise click.UsageError("--apply requires --proof-digest from the exact dry-run")
    root = archive_root()
    config = Config(archive_root=root, render_root=render_root(), sources=[], db_path=root / "index.db")
    from polylogue.storage.repair import repair_browser_capture_origin_mismatches

    try:
        report = repair_browser_capture_origin_mismatches(
            config,
            list(raw_ids),
            apply=apply_changes,
            receipt_path=receipt_path,
            proof_digest=proof_digest,
        )
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    payload = asdict(report)
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    click.echo(
        f"{report.mode}: requested={report.requested_count} eligible={report.eligible_count} "
        f"already_repaired={report.already_repaired_count} repaired={report.repaired_count} "
        f"ineligible={report.ineligible_count}"
    )
    click.echo(f"Proof digest: {report.proof_digest}")
    for item in report.items:
        click.echo(
            f"  {item.raw_id} {item.status} strategy={item.repair_strategy or 'unavailable'} "
            f"replacement={item.replacement_raw_id or 'unavailable'} "
            f"proof={item.proof_digest or 'unavailable'}: {item.reason}"
        )
    if report.receipt_path is not None:
        click.echo(f"Receipt: {report.receipt_path}")


@click.command("legacy-browser-capture-missing-native-id")
@click.option(
    "--raw-id", "raw_ids", multiple=True, required=True, help="Exact legacy raw id with native_id NULL (repeatable)."
)
@click.option(
    "--apply", "apply_changes", is_flag=True, help="Copy forward only after every legacy witness passes proof."
)
@click.option("--proof-digest", help="Exact aggregate digest emitted by the matching dry-run.")
@click.option(
    "--receipt",
    "receipt_path",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Required append-only operator recovery receipt path for --apply.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
)
@click.pass_obj
def legacy_browser_capture_missing_native_id_command(
    env: AppEnv,
    raw_ids: tuple[str, ...],
    apply_changes: bool,
    proof_digest: str | None,
    receipt_path: Path | None,
    output_format: str,
) -> None:
    """Copy forward the narrow legacy browser shape whose native ID is NULL.

    The original raw, blob, memberships, head, and application receipts remain
    immutable. This is not an alternate mode of ordinary origin repair.
    """
    del env
    if apply_changes and receipt_path is None:
        raise click.UsageError("--apply requires --receipt PATH")
    if apply_changes and proof_digest is None:
        raise click.UsageError("--apply requires --proof-digest from the exact dry-run")
    root = archive_root()
    config = Config(archive_root=root, render_root=render_root(), sources=[], db_path=root / "index.db")
    from polylogue.storage.repair import repair_legacy_browser_capture_missing_native_ids

    try:
        report = repair_legacy_browser_capture_missing_native_ids(
            config,
            list(raw_ids),
            apply=apply_changes,
            receipt_path=receipt_path,
            proof_digest=proof_digest,
        )
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    payload = asdict(report)
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    click.echo(
        f"{report.mode}: requested={report.requested_count} eligible={report.eligible_count} "
        f"already_repaired={report.already_repaired_count} repaired={report.repaired_count} "
        f"ineligible={report.ineligible_count}"
    )
    click.echo(f"Proof digest: {report.proof_digest}")
    for item in report.items:
        click.echo(
            f"  {item.raw_id} {item.status} parsed_native_id={item.parser_derived_native_id or 'unavailable'} "
            f"proof={item.proof_digest or 'unavailable'}: {item.reason}"
        )
    if report.receipt_path is not None:
        click.echo(f"Receipt: {report.receipt_path}")


@click.command("browser-canonical-authority-conflicts")
@click.option(
    "--raw-id",
    "raw_ids",
    multiple=True,
    required=True,
    help="Exact unknown-export raw id a safe rekey refuses (repeatable).",
)
@click.option(
    "--record",
    "record_blockers",
    is_flag=True,
    help="Persist each conflict as a durable, non-injected user.db blocker candidate.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
)
@click.pass_obj
def browser_canonical_authority_conflicts_command(
    env: AppEnv,
    raw_ids: tuple[str, ...],
    record_blockers: bool,
    output_format: str,
) -> None:
    """Show why a byte-proven-rekey actuator refuses these browser-capture raws.

    Read-only by default: re-runs the ordinary rekey actuator's exact
    eligibility proof and, for every raw that stays ineligible, re-derives the
    competing-authority evidence it discards on its own reject path (competing
    head content hash/frontier kind/decision, any blocking membership row, and
    -- when both sides are single-session byte-frontier raws -- the first
    diverging message index). Never selects an authority between the two
    histories. Pass ``--record`` to additionally persist each conflict as one
    ``AssertionKind.BLOCKER`` candidate assertion in ``user.db`` (always
    ``status=candidate``/``inject:false``; an operator must judge it
    explicitly -- see ``polylogue mark`` / assertion judgment tooling).
    """
    del env
    root = archive_root()
    config = Config(archive_root=root, render_root=render_root(), sources=[], db_path=root / "index.db")
    from polylogue.storage.repair import (
        inspect_browser_canonical_authority_conflicts,
        record_browser_canonical_authority_conflict_blockers,
    )

    assertion_ids: tuple[str, ...] = ()
    try:
        if record_blockers:
            report, assertion_ids = record_browser_canonical_authority_conflict_blockers(config, list(raw_ids))
        else:
            report = inspect_browser_canonical_authority_conflicts(config, list(raw_ids))
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    payload: dict[str, Any] = asdict(report)
    if record_blockers:
        payload["assertion_ids"] = list(assertion_ids)
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    click.echo(f"requested={report.requested_count} conflicts={report.conflict_count} resolved={report.resolved_count}")
    for item in report.items:
        click.echo(
            f"  {item.raw_id} competing={item.competing_raw_id or 'none'} "
            f"frontier={item.competing_frontier_kind or 'unavailable'} "
            f"divergent_message_index={item.divergent_message_index if item.divergent_message_index is not None else 'unavailable'}: "
            f"{item.divergence_note or item.reason}"
        )
    if record_blockers:
        for assertion_id in assertion_ids:
            click.echo(f"Blocker: {assertion_id}")


@click.command("duplicate-raw-identity")
@click.option(
    "--pair",
    "pairs",
    multiple=True,
    required=True,
    metavar="STALE_RAW_ID:CANONICAL_RAW_ID",
    help="Stale (currently accepted) and canonical (post-fix, dangling) raw id pair (repeatable).",
)
@click.option("--apply", "apply_changes", is_flag=True, help="Apply only after every pair passes exact proof.")
@click.option("--proof-digest", help="Exact aggregate digest emitted by the matching dry-run.")
@click.option(
    "--receipt",
    "receipt_path",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Required append-only operator recovery receipt path for --apply.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
)
@click.pass_obj
def duplicate_raw_identity_command(
    env: AppEnv,
    pairs: tuple[str, ...],
    apply_changes: bool,
    proof_digest: str | None,
    receipt_path: Path | None,
    output_format: str,
) -> None:
    """Reconcile a pre-#2729 duplicate raw pair onto its post-fix accepted head.

    PR #2729 aligned new ingests on one deterministic raw-id scheme, but did
    not retroactively repair raw pairs that already duplicated under the OLD
    scheme: the accepted head stays bound to the stale raw while its
    post-fix twin sits orphaned. This actuator proves both raws are
    byte-identical retained duplicates of one logical session, then
    repoints the accepted head and session pointer to the canonical raw via
    the existing revision-application machinery. The stale raw's own row is
    never mutated or deleted. Apply additionally writes an exclusive,
    fsynced planned-then-applied operator receipt so the repair is
    crash-resumable and auditable.
    """
    del env
    if apply_changes and receipt_path is None:
        raise click.UsageError("--apply requires --receipt PATH")
    if apply_changes and proof_digest is None:
        raise click.UsageError("--apply requires --proof-digest from the exact dry-run")
    parsed_pairs: list[tuple[str, str]] = []
    for pair in pairs:
        stale_raw_id, sep, canonical_raw_id = pair.partition(":")
        if not sep:
            raise click.UsageError(f"--pair must be STALE_RAW_ID:CANONICAL_RAW_ID, got {pair!r}")
        parsed_pairs.append((stale_raw_id, canonical_raw_id))
    root = archive_root()
    config = Config(archive_root=root, render_root=render_root(), sources=[], db_path=root / "index.db")
    from polylogue.storage.repair import repair_duplicate_raw_identity

    try:
        report = repair_duplicate_raw_identity(
            config,
            parsed_pairs,
            apply=apply_changes,
            receipt_path=receipt_path,
            proof_digest=proof_digest,
        )
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    payload = asdict(report)
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    click.echo(
        f"{report.mode}: requested={report.requested_count} eligible={report.eligible_count} "
        f"already_repaired={report.already_repaired_count} repaired={report.repaired_count} "
        f"ineligible={report.ineligible_count}"
    )
    click.echo(f"Proof digest: {report.proof_digest}")
    for item in report.items:
        click.echo(
            f"  {item.stale_raw_id}->{item.canonical_raw_id} {item.status} "
            f"proof={item.proof_digest or 'unavailable'}: {item.reason}"
        )
    if report.receipt_path is not None:
        click.echo(f"Receipt: {report.receipt_path}")
