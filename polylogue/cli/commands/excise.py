"""Excise command: the archive can forget on purpose (polylogue-27m).

Standalone/off mode is authoritative here: ``--mode standalone`` (the
default) plans/applies a real cross-tier removal. ``--mode mirror`` and
``--mode primary`` only create/inspect the durable local lifecycle-request
outbox row -- driving that request against a real Sinex confirmation is
explicitly out of this command's scope (polylogue-303r.6); see
``docs/security.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import click

if TYPE_CHECKING:
    from collections.abc import Generator

    from polylogue.surfaces.payloads import MutationStatus

from polylogue.cli.shared.types import AppEnv


def _submit(env: AppEnv, operation: str, payload: dict[str, object]) -> dict[str, object]:
    from polylogue.cli.operation_kernel import OperationKernelError, configured_mutation_operation
    from polylogue.cli.shared.helpers import mutation_refusal

    try:
        return configured_mutation_operation(env.config, operation, payload)
    except OperationKernelError as exc:
        raise mutation_refusal(exc, operation) from exc


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


def _receipt_summary(session_id: str, domain_receipt: dict[str, object], reference: object) -> str:
    """Render scalar receipt facts without opening the complete domain document."""
    detail_message = (
        f"Excised session {session_id}: {domain_receipt.get('counts', {})} "
        f"(receipt: {domain_receipt.get('receipt_assertion_id', reference)})"
    )
    for field, description in (
        ("cascaded_session_ids_count", "lineage-dependent sessions also excised"),
        ("retained_hook_events_count", "hook events remain readable"),
        ("retained_source_containers_count", "source containers retain bytes for other live sessions"),
        ("shared_blob_hashes_count", "blobs retained for other live sessions"),
    ):
        count = domain_receipt.get(field)
        if type(count) is not int or count < 0:
            raise click.ClickException(f"Excision committed; invalid receipt summary field {field}.")
        if count:
            detail_message += f"; {count} {description}"
    if domain_receipt.get("complete") is False:
        detail_message += "; INCOMPLETE"
    return detail_message


def _read_excision_plan(env: AppEnv, session_id: str, *, cascade_lineage: bool) -> dict[str, Any]:
    from polylogue.cli.operation_kernel import OperationKernelError, configured_read_operation

    try:
        result = configured_read_operation(
            env.config,
            "session.excision.plan",
            {"session_id": session_id, "cascade_lineage": cascade_lineage},
        ).value
    except OperationKernelError as exc:
        from polylogue.cli.render.outcome import exit_for_read_failure

        exit_for_read_failure(exc)
    if not isinstance(result, dict):
        raise click.ClickException("session.excision.plan returned an invalid result")
    if result.get("refused"):
        detail = str(result.get("detail") or "lineage dependents prevent excision")
        raise _LineagePlanRefusalError(detail)
    plan = result.get("plan")
    if not isinstance(plan, dict):
        raise click.ClickException("session.excision.plan omitted its plan")
    return plan


class _LineagePlanRefusalError(ValueError):
    """The pinned planner found dependents that the selected cascade omits."""


def _emit_complete_receipt(env: AppEnv, result: dict[str, object], *, session_id: str, affected_count: int) -> None:
    """Verify the entire request-owned document before emitting any machine bytes."""
    import codecs
    import hashlib

    from polylogue.cli.operation_kernel import (
        OperationFailedError,
        OperationKernelError,
        iter_configured_operation_result,
    )
    from polylogue.core.staged_content import staged_binary_content
    from polylogue.surfaces.payloads import MutationResultPayload

    document = result.get("result_document")
    summary = result.get("result")
    reference = summary.get("receipt_assertion_id") if isinstance(summary, dict) else None
    reference = reference or result.get("receipt_ref")
    request_id = document.get("request_id") if isinstance(document, dict) else None
    effect = result.get("effect")
    emitting = False
    try:
        if not isinstance(document, dict):
            raise OperationFailedError("operation_result_delivery_failed", "missing complete receipt document")
        length = document.get("byte_length")
        expected_digest = document.get("sha256")
        if (
            not isinstance(request_id, str)
            or not request_id
            or type(length) is not int
            or length < 0
            or not isinstance(expected_digest, str)
            or len(expected_digest) != 64
        ):
            raise OperationFailedError("operation_result_delivery_failed", "invalid receipt document identity")
        with staged_binary_content() as stage:
            digest = hashlib.sha256()
            decoder = codecs.getincrementaldecoder("utf-8")("strict")
            received = 0
            chunks = cast("Generator[bytes, None, None]", iter_configured_operation_result(env.config, document))
            try:
                for chunk in chunks:
                    decoder.decode(chunk)
                    received += len(chunk)
                    digest.update(chunk)
                    stage.write(chunk)
            finally:
                chunks.close()
            decoder.decode(b"", final=True)
            if not received or received != length or digest.hexdigest() != expected_digest:
                raise OperationFailedError("operation_result_delivery_failed", "incomplete or corrupt receipt document")
            stage.seek(0)
            prefix = MutationResultPayload(
                status="ok",
                operation="excise",
                session_id=session_id,
                affected_count=affected_count,
                detail=str(reference) if reference else None,
            ).to_json(exclude_none=True)
            emitting = True
            click.echo(prefix[:-1].encode("utf-8") + b',"domain_receipt":', nl=False)
            while chunk := stage.read(64 * 1024):
                click.echo(chunk, nl=False)
            click.echo(b"}")
    except (OperationKernelError, OSError, ValueError, KeyboardInterrupt) as exc:
        if emitting:
            # stdout cannot be rolled back; never append an error document to a partial write.
            raise
        failure = OperationFailedError(
            "operation_result_delivery_cancelled"
            if isinstance(exc, KeyboardInterrupt)
            else "operation_result_delivery_failed",
            f"Excision completed with {effect} (receipt {reference}); complete receipt delivery failed. Do not repeat the excision.",
            {"reference": reference, "effect_committed": effect == "committed", "effect": effect},
            request_id=request_id if isinstance(request_id, str) else None,
        )
        raise failure from exc


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
    "--cascade-lineage",
    is_flag=True,
    help=(
        "Required when --session is a prefix-sharing lineage parent (see "
        "docs/security.md): excises the session AND every dependent that "
        "shares its prefix, together, so no dependent's composed transcript "
        "is left with a dangling branch point. Without this flag, excising "
        "a lineage parent is refused."
    ),
)
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
    cascade_lineage: bool,
    output_format: str | None,
) -> None:
    """Excise a session: durable, cross-tier removal that ordinary re-ingest cannot resurrect.

    \b
    standalone (default): removes the session from index.db (cascading to
      messages/blocks/FTS/session_links), embeddings.db, and source.db
      (blob_refs + raw_sessions), records a durable removed-hash marker so
      re-ingest of unmodified source files cannot resurrect it, and writes
      one durable audit receipt to user.db. If the session is a
      prefix-sharing lineage parent, this is refused unless
      --cascade-lineage is also passed (see docs/security.md).
    mirror/primary: creates a durable lifecycle-request outbox row in
      user.db (survives an ops.db reset) and stops there. Local content is
      NOT touched by this command in mirror/primary mode.
    """
    if mode != "standalone":
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

        result = _submit(
            env,
            "mutation.session.lifecycle-request",
            {"session_id": session_id, "mode": mode, "reason": reason, "actor": actor, "confirm": True},
        )
        detail_result = result.get("result")
        assertion_id = (
            str(detail_result.get("assertion_id"))
            if isinstance(detail_result, dict) and detail_result.get("assertion_id")
            else str(result.get("reference", ""))
        )
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

    try:
        plan = _read_excision_plan(env, session_id, cascade_lineage=cascade_lineage)
    except _LineagePlanRefusalError as exc:
        if dry_run:
            _emit(
                env,
                status="aborted",
                session_id=session_id,
                affected_count=0,
                output_format=output_format,
                plain_message=f"Refusing to excise {session_id!r}: {exc}",
                detail=str(exc),
            )
            return
        else:
            _emit(
                env,
                status="aborted",
                session_id=session_id,
                affected_count=0,
                output_format=output_format,
                plain_message=f"Refusing to excise {session_id!r}: {exc}",
                detail=str(exc),
            )
            return
    if dry_run:
        if not plan.get("found"):
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
            click.echo(__import__("json").dumps({"status": "preview", "plan": plan}))
            return
        env.ui.summary(
            f"Would excise session {session_id}",
            [
                f"  source.db raw rows: {plan['source_raw_rows']}"
                + (
                    f" (including {plan['source_fact_rows']} fact/plan snapshot row(s))"
                    if plan["source_fact_rows"]
                    else ""
                )
                + (
                    f" (including {plan['source_sidecar_rows']} tool-output sidecar row(s))"
                    if plan["source_sidecar_rows"]
                    else ""
                ),
                f"  source.db hook events: {plan['source_hook_events']}",
                f"  source.db container members: {plan['source_container_members']}"
                + (
                    f" (releasing {plan['source_container_items']} container item(s))"
                    if plan["source_container_items"]
                    else ""
                ),
                f"  source.db blob refs: {plan['source_blob_refs']}",
                f"  source.db marker carriers: {plan['source_marker_inputs_pending']} pending, "
                f"{plan['source_marker_inputs_accepted']} accepted",
                *(
                    [f"  marker carrier digests: {', '.join(plan['marker_input_digests'])}"]
                    if plan["marker_input_digests"]
                    else []
                ),
                f"  index.db sessions: {plan['index_sessions']}",
                f"  index.db messages: {plan['index_messages']}",
                f"  index.db blocks: {plan['index_blocks']}",
                f"  embeddings.db vectors: {plan['embeddings_vectors']}",
                f"  user.db assertions: {plan['user_assertions']}",
                *(
                    [
                        "  WARNING container(s) retained for other live sessions, still holding these "
                        f"bytes: {', '.join(plan['retained_source_containers'])}"
                    ]
                    if plan["retained_source_containers"]
                    else []
                ),
                *(
                    [f"  already excised blob hashes: {', '.join(plan['already_excised_blob_hashes'])}"]
                    if plan["already_excised_blob_hashes"]
                    else []
                ),
                *(
                    [
                        f"  WARNING lineage-dependent sessions ({len(plan['lineage_dependent_session_ids'])}) would "
                        "lose composed content unless --cascade-lineage is also passed: "
                        + ", ".join(plan["lineage_dependent_session_ids"])
                    ]
                    if plan["lineage_dependent_session_ids"]
                    else []
                ),
            ],
        )
        return

    if not plan.get("found"):
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
        confirm_message = (
            f"Permanently excise session {session_id!r} ({plan['index_messages']} message(s), "
            f"{plan['source_raw_rows']} raw row(s))? This cannot be undone by re-ingest."
        )
        if plan["lineage_dependent_session_ids"]:
            confirm_message += (
                f" This will ALSO permanently excise {len(plan['lineage_dependent_session_ids'])} "
                "lineage-dependent session(s) (--cascade-lineage)."
            )
        if not env.ui.confirm(confirm_message, default=False):
            env.ui.console.print("Aborted.")
            return

    # The daemon re-runs PREPARE/AUTHORIZE/EXECUTE against live state.  The
    # CLI only lowers the confirmed request and never opens a writer.
    result = _submit(
        env,
        "mutation.session.excision",
        {
            "session_id": session_id,
            "reason": reason,
            "actor": actor,
            "cascade_lineage": cascade_lineage,
            "confirm": True,
        },
    )
    domain_receipt = result.get("result")
    domain_receipt = domain_receipt if isinstance(domain_receipt, dict) else {}
    affected_raw = result.get("affected_count")
    affected_count = affected_raw if isinstance(affected_raw, int) else 0
    if output_format == "json":
        from polylogue.cli.operation_kernel import OperationFailedError
        from polylogue.cli.shared.machine_errors import MachineError

        try:
            _emit_complete_receipt(env, result, session_id=session_id, affected_count=affected_count)
        except OperationFailedError as exc:
            MachineError(
                code=exc.code,
                message=str(exc),
                command=("ops", "excise"),
                details={**exc.data, "request_id": exc.request_id, "operation": "operation.result"},
            ).emit()
        return
    detail_message = _receipt_summary(session_id, domain_receipt, result.get("receipt_ref"))
    _emit(
        env,
        status="ok",
        session_id=session_id,
        affected_count=affected_count,
        output_format=output_format,
        plain_message=detail_message,
        detail=cast(str | None, domain_receipt.get("receipt_assertion_id")) or str(result.get("receipt_ref", "")),
    )


__all__ = ["excise_command"]
