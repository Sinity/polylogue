"""``polylogue feedback`` — record and inspect learning corrections (#1131).

Corrections are user overrides applied to heuristic insights. They live
outside the content-hashed conversation payload, so recording one never
re-imports the underlying session; insight rebuilds consume the latest
corrections deterministically on the next materialization.

Subcommands:

- ``polylogue feedback record CONV KIND --value KEY=VAL [--note TEXT]``
- ``polylogue feedback list [CONV] [--kind KIND]``
- ``polylogue feedback clear CONV [--kind KIND]``
"""

from __future__ import annotations

import click

from polylogue.api.archive import ConversationNotFoundError
from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.shared.machine_errors import emit_success
from polylogue.cli.shared.types import AppEnv
from polylogue.insights.feedback import (
    CorrectionKind,
    LearningCorrection,
    UnknownCorrectionKindError,
)

_KIND_CHOICES = tuple(k.value for k in CorrectionKind)


def _serialize(correction: LearningCorrection) -> dict[str, object]:
    return {
        "conversation_id": correction.conversation_id,
        "kind": correction.kind.value,
        "payload": dict(correction.payload),
        "note": correction.note,
        "created_at": correction.created_at.isoformat(),
        "feedback_version": correction.feedback_version,
    }


def _parse_value_pairs(values: tuple[str, ...]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise click.UsageError(f"--value entries must be KEY=VAL, got: {raw!r}")
        key, _, value = raw.partition("=")
        key = key.strip()
        if not key:
            raise click.UsageError(f"--value entry has empty key: {raw!r}")
        payload[key] = value
    return payload


@click.group("feedback")
def feedback_command() -> None:
    """Record and inspect learning corrections for derived insights."""


@feedback_command.command("record")
@click.argument("conversation_id")
@click.argument("kind", type=click.Choice(_KIND_CHOICES))
@click.option(
    "--value",
    "values",
    multiple=True,
    help="One or more KEY=VAL pairs forming the correction payload",
)
@click.option("--note", type=str, default=None, help="Optional user-supplied reason text")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def record_correction_cmd(
    env: AppEnv,
    conversation_id: str,
    kind: str,
    values: tuple[str, ...],
    note: str | None,
    output_format: str | None,
) -> None:
    """Record a typed correction for ``conversation_id`` of ``kind``.

    \b
    Examples:
        polylogue feedback record conv:abc tag_reject --value tag=costly --note "wrong"
        polylogue feedback record conv:abc summary_override --value summary="Hand-written summary"
    """

    payload = _parse_value_pairs(values)
    if not payload:
        raise click.UsageError("at least one --value KEY=VAL must be provided")

    try:
        correction = run_coroutine_sync(env.polylogue.record_correction(conversation_id, kind, payload, note=note))
    except ConversationNotFoundError as exc:
        raise click.ClickException(f"Conversation not found: {exc}") from exc
    except UnknownCorrectionKindError as exc:
        raise click.ClickException(str(exc)) from exc

    if output_format == "json":
        emit_success({"correction": _serialize(correction)})
        return
    click.echo(
        f"recorded {correction.kind.value} for {correction.conversation_id}: "
        f"payload={correction.payload!r}" + (f" note={correction.note!r}" if correction.note else "")
    )


@feedback_command.command("list")
@click.argument("conversation_id", required=False)
@click.option("--kind", type=click.Choice(_KIND_CHOICES), default=None)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def list_corrections_cmd(
    env: AppEnv,
    conversation_id: str | None,
    kind: str | None,
    output_format: str | None,
) -> None:
    """List stored corrections, optionally filtered by session and/or kind."""

    try:
        corrections = run_coroutine_sync(env.polylogue.list_corrections(conversation_id=conversation_id, kind=kind))
    except ConversationNotFoundError as exc:
        raise click.ClickException(f"Conversation not found: {exc}") from exc
    except UnknownCorrectionKindError as exc:
        raise click.ClickException(str(exc)) from exc

    if output_format == "json":
        emit_success({"corrections": [_serialize(c) for c in corrections]})
        return
    if not corrections:
        click.echo("No corrections recorded.")
        return
    for correction in corrections:
        click.echo(
            f"{correction.conversation_id}  {correction.kind.value}  "
            f"{correction.payload!r}" + (f"  # {correction.note}" if correction.note else "")
        )


@feedback_command.command("clear")
@click.argument("conversation_id")
@click.option(
    "--kind",
    type=click.Choice(_KIND_CHOICES),
    default=None,
    help="Only delete this kind of correction; otherwise clear all",
)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def clear_corrections_cmd(
    env: AppEnv,
    conversation_id: str,
    kind: str | None,
    output_format: str | None,
) -> None:
    """Remove one or all corrections for ``conversation_id``."""

    try:
        if kind is None:
            removed = run_coroutine_sync(env.polylogue.clear_corrections(conversation_id))
            outcome = {"cleared": removed, "conversation_id": conversation_id}
        else:
            deleted = run_coroutine_sync(env.polylogue.delete_correction(conversation_id, kind))
            outcome = {
                "deleted": 1 if deleted else 0,
                "conversation_id": conversation_id,
                "kind": kind,
            }
    except ConversationNotFoundError as exc:
        raise click.ClickException(f"Conversation not found: {exc}") from exc
    except UnknownCorrectionKindError as exc:
        raise click.ClickException(str(exc)) from exc

    if output_format == "json":
        emit_success(outcome)
        return
    click.echo(str(outcome))


__all__ = ["feedback_command"]
