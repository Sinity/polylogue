"""Durable reader user-state commands."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

import click

from polylogue.cli.shared.machine_errors import emit_success
from polylogue.cli.shared.types import AppEnv

_MARK_TYPES = ("star", "pin", "archive")


def _run(coro: Any) -> Any:
    from polylogue.api.sync.bridge import run_coroutine_sync

    return run_coroutine_sync(coro)


def _json_or_plain(output_format: str | None, payload: dict[str, object], plain: str) -> None:
    if output_format == "json":
        emit_success(payload)
        return
    click.echo(plain)


def _canonical_query_json(query_json: str) -> str:
    from polylogue.archive.query.spec import ConversationQuerySpec

    try:
        query = json.loads(query_json)
    except json.JSONDecodeError as exc:
        raise click.ClickException("query-json must be valid JSON") from exc
    if not isinstance(query, dict):
        raise click.ClickException("query-json must encode an object")
    try:
        ConversationQuerySpec.from_params(query, strict=True)
    except Exception as exc:
        raise click.ClickException(
            f"query-json is not a valid ConversationQuerySpec: {type(exc).__name__}: {exc}"
        ) from exc
    return json.dumps(query, sort_keys=True, separators=(",", ":"))


def _default_saved_view_id(name: str, query_json: str) -> str:
    digest = sha256(f"{name}\0{query_json}".encode()).hexdigest()
    return f"saved-view-{digest[:16]}"


@click.group("user-state")
def user_state_command() -> None:
    """Manage durable reader marks, annotations, and saved views."""


@user_state_command.group("marks")
def marks_group() -> None:
    """Manage conversation and message marks."""


@marks_group.command("list")
@click.option("--mark-type", type=click.Choice(_MARK_TYPES), default=None)
@click.option("--conversation-id", default=None)
@click.option("--target-type", type=click.Choice(["conversation", "message"]), default=None)
@click.option("--target-id", default=None)
@click.option("--message-id", default=None)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def list_marks_command(
    env: AppEnv,
    mark_type: str | None,
    conversation_id: str | None,
    target_type: str | None,
    target_id: str | None,
    message_id: str | None,
    output_format: str | None,
) -> None:
    """List durable marks."""
    rows = _run(
        env.polylogue.list_marks(
            mark_type=mark_type,
            conversation_id=conversation_id,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )
    )
    if output_format == "json":
        emit_success({"items": rows, "total": len(rows)})
        return
    if not rows:
        click.echo("No marks found.")
        return
    for row in rows:
        click.echo(f"{row['mark_type']:<7} {row['target_type']}:{row['target_id']}")


@marks_group.command("add")
@click.argument("conversation_id")
@click.argument("mark_type", type=click.Choice(_MARK_TYPES))
@click.option("--target-type", type=click.Choice(["conversation", "message"]), default="conversation")
@click.option("--target-id", default=None)
@click.option("--message-id", default=None)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def add_mark_command(
    env: AppEnv,
    conversation_id: str,
    mark_type: str,
    target_type: str,
    target_id: str | None,
    message_id: str | None,
    output_format: str | None,
) -> None:
    """Add a durable mark."""
    created = bool(
        _run(
            env.polylogue.add_mark(
                conversation_id,
                mark_type,
                target_type=target_type,
                target_id=target_id,
                message_id=message_id,
            )
        )
    )
    _json_or_plain(
        output_format,
        {"status": "ok" if created else "unchanged", "conversation_id": conversation_id, "mark_type": mark_type},
        f"Mark {mark_type} on {conversation_id}: {'added' if created else 'unchanged'}",
    )


@marks_group.command("remove")
@click.argument("conversation_id")
@click.argument("mark_type", type=click.Choice(_MARK_TYPES))
@click.option("--target-type", type=click.Choice(["conversation", "message"]), default="conversation")
@click.option("--target-id", default=None)
@click.option("--message-id", default=None)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def remove_mark_command(
    env: AppEnv,
    conversation_id: str,
    mark_type: str,
    target_type: str,
    target_id: str | None,
    message_id: str | None,
    output_format: str | None,
) -> None:
    """Remove a durable mark."""
    deleted = bool(
        _run(
            env.polylogue.remove_mark(
                conversation_id,
                mark_type,
                target_type=target_type,
                target_id=target_id,
                message_id=message_id,
            )
        )
    )
    _json_or_plain(
        output_format,
        {"status": "ok" if deleted else "not_found", "conversation_id": conversation_id, "mark_type": mark_type},
        f"Mark {mark_type} on {conversation_id}: {'removed' if deleted else 'not found'}",
    )


@user_state_command.group("annotations")
def annotations_group() -> None:
    """Manage conversation and message annotations."""


@annotations_group.command("list")
@click.option("--conversation-id", default=None)
@click.option("--target-type", type=click.Choice(["conversation", "message"]), default=None)
@click.option("--target-id", default=None)
@click.option("--message-id", default=None)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def list_annotations_command(
    env: AppEnv,
    conversation_id: str | None,
    target_type: str | None,
    target_id: str | None,
    message_id: str | None,
    output_format: str | None,
) -> None:
    """List durable annotations."""
    rows = _run(
        env.polylogue.list_annotations(
            conversation_id=conversation_id,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )
    )
    if output_format == "json":
        emit_success({"items": rows, "total": len(rows)})
        return
    if not rows:
        click.echo("No annotations found.")
        return
    for row in rows:
        click.echo(f"{row['annotation_id']:<24} {row['target_type']}:{row['target_id']}  {row['note_text']}")


@annotations_group.command("save")
@click.argument("annotation_id")
@click.argument("conversation_id")
@click.argument("note_text")
@click.option("--target-type", type=click.Choice(["conversation", "message"]), default="conversation")
@click.option("--target-id", default=None)
@click.option("--message-id", default=None)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def save_annotation_command(
    env: AppEnv,
    annotation_id: str,
    conversation_id: str,
    note_text: str,
    target_type: str,
    target_id: str | None,
    message_id: str | None,
    output_format: str | None,
) -> None:
    """Create or update a durable annotation."""
    created = bool(
        _run(
            env.polylogue.save_annotation(
                annotation_id,
                conversation_id,
                note_text,
                target_type=target_type,
                target_id=target_id,
                message_id=message_id,
            )
        )
    )
    _json_or_plain(
        output_format,
        {"status": "ok", "annotation_id": annotation_id, "outcome": "added" if created else "updated"},
        f"Annotation {annotation_id}: {'added' if created else 'updated'}",
    )


@annotations_group.command("delete")
@click.argument("annotation_id")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def delete_annotation_command(env: AppEnv, annotation_id: str, output_format: str | None) -> None:
    """Delete a durable annotation."""
    deleted = bool(_run(env.polylogue.delete_annotation(annotation_id)))
    _json_or_plain(
        output_format,
        {"status": "deleted" if deleted else "not_found", "annotation_id": annotation_id},
        f"Annotation {annotation_id}: {'deleted' if deleted else 'not found'}",
    )


@user_state_command.group("saved-views")
def saved_views_group() -> None:
    """Manage saved query views."""


@saved_views_group.command("list")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def list_saved_views_command(env: AppEnv, output_format: str | None) -> None:
    """List saved views."""
    rows = _run(env.polylogue.list_views())
    if output_format == "json":
        emit_success({"items": rows, "total": len(rows)})
        return
    if not rows:
        click.echo("No saved views found.")
        return
    for row in rows:
        click.echo(f"{row['view_id']:<24} {row['name']}")


@saved_views_group.command("save")
@click.argument("name")
@click.argument("query_json")
@click.option("--view-id", default=None)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def save_saved_view_command(
    env: AppEnv,
    name: str,
    query_json: str,
    view_id: str | None,
    output_format: str | None,
) -> None:
    """Create or update a saved view from a canonical query JSON object."""
    canonical_query_json = _canonical_query_json(query_json)
    saved_id = view_id or _default_saved_view_id(name.strip(), canonical_query_json)
    created = bool(_run(env.polylogue.save_view(saved_id, name.strip(), canonical_query_json)))
    _json_or_plain(
        output_format,
        {"status": "ok", "view_id": saved_id, "outcome": "added" if created else "updated"},
        f"Saved view {saved_id}: {'added' if created else 'updated'}",
    )


@saved_views_group.command("delete")
@click.argument("view_id")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def delete_saved_view_command(env: AppEnv, view_id: str, output_format: str | None) -> None:
    """Delete a saved view."""
    deleted = bool(_run(env.polylogue.delete_view(view_id)))
    _json_or_plain(
        output_format,
        {"status": "deleted" if deleted else "not_found", "view_id": view_id},
        f"Saved view {view_id}: {'deleted' if deleted else 'not found'}",
    )


__all__ = ["user_state_command"]
