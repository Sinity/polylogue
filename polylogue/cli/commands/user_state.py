"""Durable reader state maintenance commands."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

import click

from polylogue.cli.commands.user_state_blackboard import blackboard_command
from polylogue.cli.commands.user_state_feedback import feedback_command
from polylogue.cli.commands.user_state_tags import tags_command
from polylogue.cli.shared.machine_errors import emit_success
from polylogue.cli.shared.types import AppEnv
from polylogue.core.user_state_targets import MARK_TYPE_NAMES, TARGET_KIND_NAMES, TARGET_SESSION
from polylogue.surfaces.payloads import AssertionClaimListPayload, serialize_surface_payload

_WORKSPACE_MODES = ("tabs", "stack", "compare", "timeline")
_TARGET_TYPES = list(TARGET_KIND_NAMES)


def _run(coro: Any) -> Any:
    from polylogue.api.sync.bridge import run_coroutine_sync

    return run_coroutine_sync(coro)


def _json_or_plain(output_format: str | None, payload: dict[str, object], plain: str) -> None:
    if output_format == "json":
        emit_success(payload)
        return
    click.echo(plain)


def _canonical_query_json(query_json: str) -> str:
    from polylogue.archive.query.spec import SessionQuerySpec

    try:
        query = json.loads(query_json)
    except json.JSONDecodeError as exc:
        raise click.ClickException("query-json must be valid JSON") from exc
    if not isinstance(query, dict):
        raise click.ClickException("query-json must encode an object")
    try:
        SessionQuerySpec.from_params(query, strict=True)
    except Exception as exc:
        raise click.ClickException(f"query-json is not a valid SessionQuerySpec: {type(exc).__name__}: {exc}") from exc
    return json.dumps(query, sort_keys=True, separators=(",", ":"))


def _canonical_json_object(raw_json: str, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"{label} must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise click.ClickException(f"{label} must encode an object")
    return payload


def _canonical_json_list(raw_json: str, *, label: str) -> list[object]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"{label} must be valid JSON") from exc
    if not isinstance(payload, list):
        raise click.ClickException(f"{label} must encode a list")
    return payload


def _default_saved_view_id(name: str, query_json: str) -> str:
    digest = sha256(f"{name}\0{query_json}".encode()).hexdigest()
    return f"saved-view-{digest[:16]}"


@click.group("state")
def state_command() -> None:
    """Manage durable reader state."""


state_command.add_command(blackboard_command)
state_command.add_command(feedback_command)
state_command.add_command(tags_command)


@state_command.group("candidates")
def candidates_group() -> None:
    """Review candidate assertions awaiting judgment."""


@candidates_group.command("list")
@click.option("--target-ref", default=None, help="Limit candidates to one target object ref.")
@click.option("--limit", "-l", type=int, default=50, show_default=True)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def list_candidates_command(env: AppEnv, target_ref: str | None, limit: int, output_format: str | None) -> None:
    """List candidate assertion claims."""
    items = _run(env.polylogue.list_assertion_candidates(target_ref=target_ref, limit=limit))
    if output_format == "json":
        payload = AssertionClaimListPayload(items=tuple(items), total=len(items), limit=limit, statuses=("candidate",))
        click.echo(serialize_surface_payload(payload, exclude_none=True))
        return
    if not items:
        click.echo("No candidate assertions found.")
        return
    for item in items:
        detail = item.body_text or (json.dumps(item.value, sort_keys=True) if item.value is not None else "")
        click.echo(f"{item.assertion_id:<32} {item.kind:<20} {item.target_ref} {detail}")


@candidates_group.command("accept")
@click.argument("candidate_ref")
@click.option("--reason", default=None)
@click.option("--actor-ref", default="user:local", show_default=True)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def accept_candidate_command(
    env: AppEnv,
    candidate_ref: str,
    reason: str | None,
    actor_ref: str,
    output_format: str | None,
) -> None:
    """Accept a candidate assertion into an active assertion."""
    _emit_candidate_judgment(
        env,
        candidate_ref=candidate_ref,
        decision="accept",
        reason=reason,
        actor_ref=actor_ref,
        output_format=output_format,
    )


@candidates_group.command("reject")
@click.argument("candidate_ref")
@click.option("--reason", required=True)
@click.option("--actor-ref", default="user:local", show_default=True)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def reject_candidate_command(
    env: AppEnv,
    candidate_ref: str,
    reason: str,
    actor_ref: str,
    output_format: str | None,
) -> None:
    """Reject a candidate assertion with a durable reason."""
    _emit_candidate_judgment(
        env,
        candidate_ref=candidate_ref,
        decision="reject",
        reason=reason,
        actor_ref=actor_ref,
        output_format=output_format,
    )


@candidates_group.command("defer")
@click.argument("candidate_ref")
@click.option("--reason", default=None)
@click.option("--actor-ref", default="user:local", show_default=True)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def defer_candidate_command(
    env: AppEnv,
    candidate_ref: str,
    reason: str | None,
    actor_ref: str,
    output_format: str | None,
) -> None:
    """Record a candidate assertion deferral without changing candidate status."""
    _emit_candidate_judgment(
        env,
        candidate_ref=candidate_ref,
        decision="defer",
        reason=reason,
        actor_ref=actor_ref,
        output_format=output_format,
    )


@candidates_group.command("supersede")
@click.argument("candidate_ref")
@click.option("--kind", "replacement_kind", required=True, help="Kind for the replacement active assertion.")
@click.option("--body", "replacement_body_text", required=True, help="Body text for the replacement assertion.")
@click.option("--reason", default=None)
@click.option("--actor-ref", default="user:local", show_default=True)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def supersede_candidate_command(
    env: AppEnv,
    candidate_ref: str,
    replacement_kind: str,
    replacement_body_text: str,
    reason: str | None,
    actor_ref: str,
    output_format: str | None,
) -> None:
    """Supersede a candidate with an explicit active assertion."""
    _emit_candidate_judgment(
        env,
        candidate_ref=candidate_ref,
        decision="supersede",
        reason=reason,
        actor_ref=actor_ref,
        output_format=output_format,
        replacement_kind=replacement_kind,
        replacement_body_text=replacement_body_text,
    )


def _emit_candidate_judgment(
    env: AppEnv,
    *,
    candidate_ref: str,
    decision: str,
    reason: str | None,
    actor_ref: str,
    output_format: str | None,
    replacement_kind: str | None = None,
    replacement_body_text: str | None = None,
) -> None:
    payload = _run(
        env.polylogue.judge_assertion_candidate(
            candidate_ref=candidate_ref,
            decision=decision,
            reason=reason,
            actor_ref=actor_ref,
            replacement_kind=replacement_kind,
            replacement_body_text=replacement_body_text,
        )
    )
    if output_format == "json":
        click.echo(serialize_surface_payload(payload, exclude_none=True))
        return
    result_ref = payload.judgment.resulting_assertion_ref or "no active assertion"
    click.echo(f"{decision}: {payload.candidate.assertion_id} -> {result_ref}")


@state_command.group("marks")
def marks_group() -> None:
    """Manage session and message marks."""


@marks_group.command("list")
@click.option("--mark-type", type=click.Choice(MARK_TYPE_NAMES), default=None)
@click.option("--session-id", default=None)
@click.option("--target-type", type=click.Choice(_TARGET_TYPES), default=None)
@click.option("--target-id", default=None)
@click.option("--message-id", default=None)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def list_marks_command(
    env: AppEnv,
    mark_type: str | None,
    session_id: str | None,
    target_type: str | None,
    target_id: str | None,
    message_id: str | None,
    output_format: str | None,
) -> None:
    """List durable marks."""
    rows = _run(
        env.polylogue.list_marks(
            mark_type=mark_type,
            session_id=session_id,
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
@click.argument("session_id")
@click.argument("mark_type", type=click.Choice(MARK_TYPE_NAMES))
@click.option("--target-type", type=click.Choice(_TARGET_TYPES), default=TARGET_SESSION)
@click.option("--target-id", default=None)
@click.option("--message-id", default=None)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def add_mark_command(
    env: AppEnv,
    session_id: str,
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
                session_id,
                mark_type,
                target_type=target_type,
                target_id=target_id,
                message_id=message_id,
            )
        )
    )
    _json_or_plain(
        output_format,
        {"status": "ok" if created else "unchanged", "session_id": session_id, "mark_type": mark_type},
        f"Mark {mark_type} on {session_id}: {'added' if created else 'unchanged'}",
    )


@marks_group.command("remove")
@click.argument("session_id")
@click.argument("mark_type", type=click.Choice(MARK_TYPE_NAMES))
@click.option("--target-type", type=click.Choice(_TARGET_TYPES), default=TARGET_SESSION)
@click.option("--target-id", default=None)
@click.option("--message-id", default=None)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def remove_mark_command(
    env: AppEnv,
    session_id: str,
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
                session_id,
                mark_type,
                target_type=target_type,
                target_id=target_id,
                message_id=message_id,
            )
        )
    )
    _json_or_plain(
        output_format,
        {"status": "ok" if deleted else "not_found", "session_id": session_id, "mark_type": mark_type},
        f"Mark {mark_type} on {session_id}: {'removed' if deleted else 'not found'}",
    )


@state_command.group("annotations")
def annotations_group() -> None:
    """Manage session and message annotations."""


@annotations_group.command("list")
@click.option("--session-id", default=None)
@click.option("--target-type", type=click.Choice(_TARGET_TYPES), default=None)
@click.option("--target-id", default=None)
@click.option("--message-id", default=None)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def list_annotations_command(
    env: AppEnv,
    session_id: str | None,
    target_type: str | None,
    target_id: str | None,
    message_id: str | None,
    output_format: str | None,
) -> None:
    """List durable annotations."""
    rows = _run(
        env.polylogue.list_annotations(
            session_id=session_id,
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
@click.argument("session_id")
@click.argument("note_text")
@click.option("--target-type", type=click.Choice(_TARGET_TYPES), default=TARGET_SESSION)
@click.option("--target-id", default=None)
@click.option("--message-id", default=None)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def save_annotation_command(
    env: AppEnv,
    annotation_id: str,
    session_id: str,
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
                session_id,
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


@state_command.group("saved-views")
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


@state_command.group("recall-packs")
def recall_packs_group() -> None:
    """Manage recall packs with explicit target evidence."""


@recall_packs_group.command("list")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def list_recall_packs_command(env: AppEnv, output_format: str | None) -> None:
    """List recall packs."""
    rows = _run(env.polylogue.list_recall_packs())
    if output_format == "json":
        emit_success({"items": rows, "total": len(rows)})
        return
    if not rows:
        click.echo("No recall packs found.")
        return
    for row in rows:
        click.echo(f"{row['pack_id']:<24} {row['label']}")


@recall_packs_group.command("save")
@click.argument("pack_id")
@click.argument("label")
@click.option("--item-json", "item_jsons", multiple=True, help="Recall-pack item JSON object.")
@click.option("--payload-json", default="{}", help="Additional recall-pack payload JSON object.")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def save_recall_pack_command(
    env: AppEnv,
    pack_id: str,
    label: str,
    item_jsons: tuple[str, ...],
    payload_json: str,
    output_format: str | None,
) -> None:
    """Create or update a recall pack."""
    payload = _canonical_json_object(payload_json, label="payload-json")
    payload_items = payload.get("items", [])
    if not isinstance(payload_items, list) or not all(isinstance(item, dict) for item in payload_items):
        raise click.ClickException("payload-json items must be objects")
    items = list(payload_items)
    items.extend(_canonical_json_object(raw, label="item-json") for raw in item_jsons)
    if not items:
        raise click.ClickException("at least one --item-json or payload-json item is required")
    payload["items"] = items
    created = bool(
        _run(
            env.polylogue.create_recall_pack(
                pack_id,
                label,
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
            )
        )
    )
    _json_or_plain(
        output_format,
        {"status": "ok", "pack_id": pack_id, "outcome": "added" if created else "updated"},
        f"Recall pack {pack_id}: {'added' if created else 'updated'}",
    )


@recall_packs_group.command("delete")
@click.argument("pack_id")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def delete_recall_pack_command(env: AppEnv, pack_id: str, output_format: str | None) -> None:
    """Delete a recall pack."""
    deleted = bool(_run(env.polylogue.delete_recall_pack(pack_id)))
    _json_or_plain(
        output_format,
        {"status": "deleted" if deleted else "not_found", "pack_id": pack_id},
        f"Recall pack {pack_id}: {'deleted' if deleted else 'not found'}",
    )


@state_command.group("workspaces")
def workspaces_group() -> None:
    """Manage durable reader workspaces."""


@workspaces_group.command("list")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def list_workspaces_command(env: AppEnv, output_format: str | None) -> None:
    """List reader workspaces."""
    rows = _run(env.polylogue.list_workspaces())
    if output_format == "json":
        emit_success({"items": rows, "total": len(rows)})
        return
    if not rows:
        click.echo("No reader workspaces found.")
        return
    for row in rows:
        click.echo(f"{row['workspace_id']:<24} {row['mode']:<8} {row['name']}")


@workspaces_group.command("save")
@click.argument("workspace_id")
@click.argument("name")
@click.option("--mode", type=click.Choice(_WORKSPACE_MODES), default="tabs")
@click.option("--open-targets-json", default="[]", help="Workspace open target JSON list.")
@click.option("--layout-json", default="{}", help="Workspace layout JSON object.")
@click.option("--active-target-json", default="{}", help="Workspace active target JSON object.")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def save_workspace_command(
    env: AppEnv,
    workspace_id: str,
    name: str,
    mode: str,
    open_targets_json: str,
    layout_json: str,
    active_target_json: str,
    output_format: str | None,
) -> None:
    """Create or update a reader workspace."""
    open_targets = _canonical_json_list(open_targets_json, label="open-targets-json")
    if not all(isinstance(item, dict) for item in open_targets):
        raise click.ClickException("open-targets-json items must be objects")
    layout = _canonical_json_object(layout_json, label="layout-json")
    active_target = _canonical_json_object(active_target_json, label="active-target-json")
    created = bool(
        _run(
            env.polylogue.save_workspace(
                workspace_id,
                name,
                mode,
                json.dumps(open_targets, sort_keys=True, separators=(",", ":")),
                json.dumps(layout, sort_keys=True, separators=(",", ":")),
                json.dumps(active_target, sort_keys=True, separators=(",", ":")),
            )
        )
    )
    _json_or_plain(
        output_format,
        {"status": "ok", "workspace_id": workspace_id, "outcome": "added" if created else "updated"},
        f"Workspace {workspace_id}: {'added' if created else 'updated'}",
    )


@workspaces_group.command("delete")
@click.argument("workspace_id")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def delete_workspace_command(env: AppEnv, workspace_id: str, output_format: str | None) -> None:
    """Delete a reader workspace."""
    deleted = bool(_run(env.polylogue.delete_workspace(workspace_id)))
    _json_or_plain(
        output_format,
        {"status": "deleted" if deleted else "not_found", "workspace_id": workspace_id},
        f"Workspace {workspace_id}: {'deleted' if deleted else 'not found'}",
    )


__all__ = ["state_command"]
