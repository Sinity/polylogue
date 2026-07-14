"""Excise command: the archive can forget on purpose (polylogue-27m).

Standalone/off mode is authoritative here: ``--mode standalone`` (the
default) plans/applies a real cross-tier removal. ``--mode mirror`` and
``--mode primary`` only create/inspect the durable local lifecycle-request
outbox row -- driving that request against a real Sinex confirmation is
explicitly out of this command's scope (polylogue-303r.6); see
``docs/security.md``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from polylogue.surfaces.payloads import MutationStatus

from polylogue.cli.shared.helpers import fail
from polylogue.cli.shared.types import AppEnv
from polylogue.paths import archive_root


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _emit(
    env: AppEnv,
    *,
    status: MutationStatus,
    session_id: str,
    affected_count: int,
    output_format: str | None,
    plain_message: str,
    detail: str | None = None,
) -> None:
    if output_format == "json":
        from polylogue.surfaces.payloads import MutationResultPayload

        click.echo(
            MutationResultPayload(
                status=status,
                operation="excise",
                session_id=session_id,
                affected_count=affected_count,
                detail=detail,
            ).to_json(exclude_none=True)
        )
        return
    env.ui.console.print(plain_message)


@click.command("excise")
@click.option("--session", "session_id", required=True, help="Session id to excise.")
@click.option("--reason", required=True, help="Why this content is being excised (recorded in the audit receipt).")
@click.option(
    "--mode",
    type=click.Choice(["standalone", "mirror", "primary"]),
    default="standalone",
    show_default=True,
    help=(
        "standalone: apply the local cross-tier removal now (authoritative). "
        "mirror/primary: create the durable lifecycle-request outbox row only "
        "-- see docs/security.md for why this command does not drive it "
        "against a real Sinex confirmation."
    ),
)
@click.option("--actor", default="user:local", show_default=True, help="Actor recorded on the audit receipt/request.")
@click.option("--dry-run", is_flag=True, help="Preview affected rows per tier without mutating anything.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
@click.option(
    "--json",
    "output_format",
    flag_value="json",
    default=None,
    help="Shortcut for --format json.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json"]),
    default=None,
    help="Output format. JSON emits a MutationResultPayload.",
)
@click.pass_obj
def excise_command(
    env: AppEnv,
    session_id: str,
    reason: str,
    mode: str,
    actor: str,
    dry_run: bool,
    yes: bool,
    output_format: str | None,
) -> None:
    """Excise a session: durable, cross-tier removal that ordinary re-ingest cannot resurrect.

    \b
    standalone (default): removes the session from index.db (cascading to
      messages/blocks/FTS/session_links), embeddings.db, and source.db
      (blob_refs + raw_sessions), records a durable removed-hash marker so
      re-ingest of unmodified source files cannot resurrect it, and writes
      one durable audit receipt to user.db.
    mirror/primary: creates a durable lifecycle-request outbox row in
      user.db (survives an ops.db reset) and stops there. Local content is
      NOT touched by this command in mirror/primary mode.
    """
    root = archive_root()

    if mode != "standalone":
        from polylogue.security.lifecycle import submit_lifecycle_request

        target_ref = f"session:{session_id}"
        if dry_run:
            _emit(
                env,
                status="preview",
                session_id=session_id,
                affected_count=0,
                output_format=output_format,
                plain_message=(
                    f"Would submit a {mode} lifecycle request for {target_ref} (reason: {reason!r}). "
                    "No local content would be touched until a real Sinex confirmation lands "
                    "(polylogue-303r.6)."
                ),
            )
            return
        if not yes:
            if output_format == "json" or env.ui.plain:
                _emit(
                    env,
                    status="aborted",
                    session_id=session_id,
                    affected_count=0,
                    output_format=output_format,
                    plain_message="Use --yes to confirm.",
                )
                return
            if not env.ui.confirm(
                f"Submit a {mode} lifecycle request for session {session_id!r}? "
                "This does NOT remove local content -- it only records a durable pending request.",
                default=False,
            ):
                env.ui.console.print("Aborted.")
                return

        import sqlite3

        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        user_db = root / "user.db"
        initialize_archive_database(user_db, ArchiveTier.USER)
        conn = sqlite3.connect(user_db)
        try:
            with conn:
                assertion_id = submit_lifecycle_request(
                    conn,
                    target_ref=target_ref,
                    mode=mode,  # type: ignore[arg-type]
                    reason=reason,
                    actor=actor,
                    now_ms=_now_ms(),
                )
        finally:
            conn.close()
        _emit(
            env,
            status="ok",
            session_id=session_id,
            affected_count=1,
            output_format=output_format,
            plain_message=(
                f"Recorded {mode} lifecycle request {assertion_id} for {target_ref}. "
                "Pending real Sinex confirmation (polylogue-303r.6); local content unchanged."
            ),
            detail=assertion_id,
        )
        return

    from polylogue.security.excision import apply_session_excision, plan_session_excision

    if dry_run:
        plan = plan_session_excision(root, session_id)
        if not plan.found:
            _emit(
                env,
                status="not_found",
                session_id=session_id,
                affected_count=0,
                output_format=output_format,
                plain_message=f"No session found for {session_id!r}.",
            )
            return
        if output_format == "json":
            click.echo(__import__("json").dumps({"status": "preview", "plan": plan.as_dict()}))
            return
        env.ui.summary(
            f"Would excise session {session_id}",
            [
                f"  source.db raw rows: {plan.source_raw_rows}",
                f"  source.db blob refs: {plan.source_blob_refs}",
                f"  index.db sessions: {plan.index_sessions}",
                f"  index.db messages: {plan.index_messages}",
                f"  index.db blocks: {plan.index_blocks}",
                f"  embeddings.db vectors: {plan.embeddings_vectors}",
                f"  user.db assertions: {plan.user_assertions}",
                *(
                    [f"  already excised blob hashes: {', '.join(plan.already_excised_blob_hashes)}"]
                    if plan.already_excised_blob_hashes
                    else []
                ),
            ],
        )
        return

    plan = plan_session_excision(root, session_id)
    if not plan.found:
        _emit(
            env,
            status="not_found",
            session_id=session_id,
            affected_count=0,
            output_format=output_format,
            plain_message=f"No session found for {session_id!r}.",
        )
        return

    if not yes:
        if output_format == "json" or env.ui.plain:
            _emit(
                env,
                status="aborted",
                session_id=session_id,
                affected_count=0,
                output_format=output_format,
                plain_message="Use --yes to confirm excision.",
            )
            return
        if not env.ui.confirm(
            f"Permanently excise session {session_id!r} ({plan.index_messages} message(s), "
            f"{plan.source_raw_rows} raw row(s))? This cannot be undone by re-ingest.",
            default=False,
        ):
            env.ui.console.print("Aborted.")
            return

    receipt = apply_session_excision(root, session_id, reason=reason, actor=actor)
    if not receipt.found:
        fail("excise", f"Session {session_id!r} disappeared between plan and apply.")
        return
    _emit(
        env,
        status="ok",
        session_id=session_id,
        affected_count=receipt.counts.get("index_sessions", 0),
        output_format=output_format,
        plain_message=(f"Excised session {session_id}: {receipt.counts} (receipt: {receipt.receipt_assertion_id})"),
        detail=receipt.receipt_assertion_id,
    )


__all__ = ["excise_command"]
