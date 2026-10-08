"""Reset command for clearing database and state.

Supports identity-preserving soft-delete: sessions can be tombstoned
rather than hard-deleted, preserving user metadata across reset cycles.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from polylogue.surfaces.payloads import MutationStatus

from polylogue.cli.shared.helpers import fail
from polylogue.cli.shared.types import AppEnv
from polylogue.paths import (
    archive_root,
    blob_store_root,
    cache_home,
    data_home,
    drive_cache_path,
    drive_token_path,
    state_home,
)

# Rebuildable tiers: replayed from preserved source evidence by maintenance.
# Deleting these is the supported "move aside and replay source.db" reset path.
# They are the only tiers a reset deletes: bootstrap recreates them, while an
# established archive missing source.db, user.db or audit.db refuses to open.
# ``embeddings.db`` is deliberately NOT here: bootstrap classifies it
# ``expensive_rebuild`` (storage/sqlite/archive_tiers/bootstrap.py) because
# nothing replays its vectors from source.db -- they are re-purchased from the
# embedding provider. Its reuse key (``vector_derivation_hash``) survives an
# index rebuild, so preserving the file is what makes the rebuild cheap.
_REBUILDABLE_ARCHIVE_DATABASES = (
    ("index database", "index.db"),
    ("ops database", "ops.db"),
)

#: Expensive-to-rebuild tier preserved by ``reset --database``.  Deleting it is
#: not a reset, it is a repurchase, so this file has no delete target at all and
#: the command names the preservation route instead.
_EMBEDDINGS_ARCHIVE_DATABASE = ("embeddings database", "embeddings.db")
_INDEX_ARCHIVE_DATABASE = ("index database", "index.db")


def _submit(env: AppEnv, operation: str, payload: dict[str, object]) -> dict[str, object]:
    from polylogue.cli.operation_kernel import OperationKernelError, configured_mutation_operation
    from polylogue.cli.shared.helpers import mutation_refusal

    try:
        return configured_mutation_operation(env.config, operation, payload)
    except OperationKernelError as exc:
        raise mutation_refusal(exc, operation) from exc


def _archive_root() -> Path:
    return archive_root()


def _index_db_path() -> Path:
    from polylogue.storage.archive_identity import ArchiveLocation

    return ArchiveLocation.resolve(_archive_root()).active_index_path


def _source_db_path() -> Path:
    return _archive_root() / "source.db"


def _user_db_path() -> Path:
    return _archive_root() / "user.db"


def _archive_database_targets() -> list[tuple[str, Path]]:
    """Resolve archive-tier files to delete for ``reset --database``.

    Only the derived tiers: index rebuilds replay preserved source evidence,
    including rows whose original source files have rotated away.
    """
    root = _archive_root()
    databases = _REBUILDABLE_ARCHIVE_DATABASES
    if (root / ".index-active-pointer").exists():
        raise click.ClickException(
            "reset --database is unsafe for a managed active generation; "
            "the pointer-managed index.db must not be deleted in place"
        )
    targets: list[tuple[str, Path]] = []
    for name, filename in databases:
        path = root / filename
        if path.exists():
            targets.append((name, path))
        for suffix in ("-wal", "-shm"):
            sidecar = path.with_name(f"{path.name}{suffix}")
            if sidecar.exists():
                targets.append((f"{name} {suffix}", sidecar))
    return targets


def _archive_index_targets() -> list[tuple[str, Path]]:
    """Resolve only the rebuildable index tier files for schema rebuilds."""
    root = _archive_root()
    name, _filename = _INDEX_ARCHIVE_DATABASE
    # A managed generation is not a disposable sibling of the configured
    # root.  Deleting it in place bypasses blue/green promotion and can leave
    # a second writable index beside the pointer (the July live incident).
    # The refusal reads the pointer's presence, never its target: a pointer
    # naming a path outside the root is precisely a generation this must not
    # touch, and resolving it first would raise instead of refusing.
    if (root / ".index-active-pointer").exists():
        raise click.ClickException(
            "reset --index is unsafe for a managed active generation; "
            "the pointer-managed index.db must not be deleted in place"
        )
    path = _index_db_path()
    targets: list[tuple[str, Path]] = []
    if path.exists():
        targets.append((name, path))
    for suffix in ("-wal", "-shm"):
        sidecar = path.with_name(f"{path.name}{suffix}")
        if sidecar.exists():
            targets.append((f"{name} {suffix}", sidecar))
    return targets


def _user_db_present() -> bool:
    return _user_db_path().exists()


def _source_db_present() -> bool:
    return _source_db_path().exists()


def _embeddings_db_present() -> bool:
    return (_archive_root() / _EMBEDDINGS_ARCHIVE_DATABASE[1]).exists()


def _identity_reset_targets(env: AppEnv, *, conv_id: str | None, source_path: Path | None) -> tuple[str, int, str]:
    """Retain one server-owned audited preview, returning no target collection."""
    payload: dict[str, object] = {"session": conv_id} if conv_id else {"source_path": str(source_path)}
    payload["reason"] = "reset --session" if conv_id else f"reset --source {source_path}"
    response = _submit(env, "mutation.identity-reset.preview", payload)
    reference = response.get("reference")
    request_id = reference.get("request_id") if isinstance(reference, dict) else None
    count = response.get("session_count")
    if not isinstance(request_id, str) or type(count) is not int or count < 0:
        raise click.ClickException("identity reset preview returned an invalid result")
    label = f"session {conv_id!r}" if conv_id else f"source {source_path}"
    return request_id, count, label


def _identity_reset_target_pages(env: AppEnv, preview_request_id: str, count: int) -> Iterator[list[str]]:
    from polylogue.cli.operation_kernel import OperationKernelError, configured_read_operation

    offset = 0
    while True:
        try:
            result = configured_read_operation(
                env.config,
                "session.identity-reset.targets",
                {"preview_request_id": preview_request_id, "offset": offset, "page_size": 256},
            ).value
        except OperationKernelError as exc:
            from polylogue.cli.render.outcome import exit_for_read_failure

            exit_for_read_failure(exc)
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("session_ids"), list)
            or result.get("total") != count
            or result.get("offset") != offset
        ):
            raise click.ClickException("identity reset targets returned an invalid page")
        ids = result["session_ids"]
        if len(ids) > 256 or not all(isinstance(value, str) for value in ids):
            raise click.ClickException("identity reset targets returned invalid IDs")
        next_offset = result.get("next_offset")
        end = offset + len(ids)
        if (
            end > count
            or (next_offset is None and end != count)
            or (next_offset is not None and (next_offset != end or end <= offset or end >= count))
        ):
            raise click.ClickException("identity reset targets returned an invalid continuation")
        yield ids
        if next_offset is None:
            break
        offset = end


def _emit_identity_reset_result(
    env: AppEnv,
    *,
    status: MutationStatus,
    preview_request_id: str,
    session_count: int,
    affected_count: int,
    output_format: str | None,
    plain_message: str,
    include_plain_targets: bool = False,
) -> None:
    if output_format == "json" or include_plain_targets:
        from polylogue.surfaces.payloads import MutationResultPayload

        # The immutable ordinal walk is staged, never collected into a client
        # target list. A malformed page or cancellation exposes no prefix.
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as staged:
            if output_format == "json":
                payload = MutationResultPayload(
                    status=status, operation="reset", session_count=session_count, affected_count=affected_count
                ).model_dump(mode="json", exclude_none=True)
                payload.pop("session_ids", None)
                staged.write(json.dumps(payload)[:-1] + ', "session_ids": [')
            else:
                staged.write(plain_message + ": ")
            written = 0
            for ids in _identity_reset_target_pages(env, preview_request_id, session_count):
                for session_id in ids:
                    if written:
                        staged.write(", " if output_format != "json" else ",")
                    staged.write(json.dumps(session_id) if output_format == "json" else session_id)
                    written += 1
            if output_format == "json":
                staged.write("]}\n")
            else:
                staged.write("\n" if written else "(no matching sessions)\n")
            staged.seek(0)
            while chunk := staged.read(64 * 1024):
                click.echo(chunk, nl=False)
        return
    env.ui.console.print(plain_message)


@click.command("reset")
@click.option(
    "--index",
    "index",
    is_flag=True,
    help="Delete the rebuildable index tier when polylogued next starts",
)
@click.option(
    "--database",
    is_flag=True,
    help="Delete the derived index.db and ops.db tiers when polylogued next starts; durable tiers are never deleted",
)
@click.option("--blob", is_flag=True, help="Delete the content-addressed blob store")
@click.option("--assets", is_flag=True, help="Delete archived assets/attachments")
@click.option("--cache", is_flag=True, help="Delete search indexes, schemas, and cache")
@click.option("--auth", is_flag=True, help="Delete Google Drive OAuth tokens")
@click.option("--all", "reset_all", is_flag=True, help="Reset everything")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--session", "conv_id", default=None, help="Tombstone a specific session by ID")
@click.option(
    "--source",
    "source_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Tombstone all sessions from a source path",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Retain an audited preview of --session/--source targets without changing sessions",
)
@click.option(
    "--json",
    "output_format",
    flag_value="json",
    default=None,
    help="Shortcut for --format json (applies to --session/--source).",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json"]),
    default=None,
    help="Output format for --session/--source identity resets. JSON emits a MutationResultPayload.",
)
@click.pass_obj
def reset_command(
    env: AppEnv,
    index: bool,
    database: bool,
    blob: bool,
    assets: bool,
    cache: bool,
    auth: bool,
    reset_all: bool,
    yes: bool,
    conv_id: str | None,
    source_path: Path | None,
    dry_run: bool,
    output_format: str | None,
) -> None:
    """Reset database, blob store, assets, cache, or auth state.

    By default, requires explicit flags to specify what to reset.
    Use --all to reset everything.

    \b
    Identity-preserving reset:
      --session ID  Tombstone a specific session (preserves user metadata)
      --source PATH      Tombstone all sessions from a source path
      --dry-run     Preview exact target rows before any tombstone write
    """
    if reset_all:
        database = blob = assets = cache = auth = True

    if dry_run and not (conv_id or source_path):
        raise click.UsageError("--dry-run is only supported with --session/--source.")
    if output_format and not (conv_id or source_path):
        raise click.UsageError("--format/--json is only supported with --session/--source.")

    # Identity-preserving soft-delete paths. Archive rows are rebuildable;
    # user suppressions are the durable tombstone. Targets are resolved once
    # up front so the dry-run preview and the real mutation act on the
    # identical id set (#jnj.5).
    if conv_id or source_path:
        preview_request_id, count, label = _identity_reset_targets(env, conv_id=conv_id, source_path=source_path)

        if dry_run:
            _emit_identity_reset_result(
                env,
                status="preview",
                preview_request_id=preview_request_id,
                session_count=count,
                affected_count=0,
                output_format=output_format,
                plain_message=f"Would tombstone {count} session(s) for {label}",
                include_plain_targets=True,
            )
            return

        if count == 0:
            _emit_identity_reset_result(
                env,
                status="ok",
                preview_request_id=preview_request_id,
                session_count=0,
                affected_count=0,
                output_format=output_format,
                plain_message=f"No sessions found for {label}.",
            )
            return

        if not yes:
            if output_format == "json" or env.ui.plain:
                _emit_identity_reset_result(
                    env,
                    status="aborted",
                    preview_request_id=preview_request_id,
                    session_count=count,
                    affected_count=0,
                    output_format=output_format,
                    plain_message="Use --yes to confirm deletion.",
                )
                return
            if not env.ui.confirm(
                f"Tombstone {count} session(s) for {label}? "
                "This suppresses them and deletes their (rebuildable) archive rows",
                default=False,
            ):
                env.ui.console.print("Aborted.")
                return

        authorized = _submit(
            env,
            "mutation.identity-reset.authorize",
            {"preview_request_id": preview_request_id, "confirm": True},
        )
        reference = authorized.get("reference")
        if not isinstance(reference, dict) or not isinstance(reference.get("request_id"), str):
            raise click.ClickException("identity reset authorization returned an invalid reference")
        result = _submit(
            env,
            "mutation.identity-reset",
            {"authorization_request_id": reference["request_id"]},
        )
        result_payload = result.get("result")
        result_payload = result_payload if isinstance(result_payload, dict) else {}
        suppressed_value = result_payload.get("suppressed_count", result.get("affected_count", 0))
        suppressed = suppressed_value if isinstance(suppressed_value, int) else 0
        deleted_value = result_payload.get("deleted_archive_rows", 0)
        deleted = deleted_value if isinstance(deleted_value, int) else 0
        if conv_id:
            plain_message = (
                f"Tombstoned session {conv_id}: {suppressed} suppression(s), {deleted} archive row(s) deleted."
            )
        else:
            plain_message = (
                f"Tombstoned {count} session(s) from {source_path}: "
                f"{suppressed} suppression(s), {deleted} archive row(s) deleted."
            )
        _emit_identity_reset_result(
            env,
            status="ok",
            preview_request_id=preview_request_id,
            session_count=count,
            affected_count=suppressed,
            output_format=output_format,
            plain_message=plain_message,
        )
        return

    if not (index or database or blob or assets or cache or auth):
        fail(
            "reset",
            "Specify at least one target (e.g., --index, --database, --assets, --cache, --auth) or use --all",
        )

    targets = []
    if index:
        targets.extend(_archive_index_targets())
    if database:
        targets.extend(_archive_database_targets())
        if _source_db_present():
            env.ui.console.print(
                "Preserving source.db (durable acquired evidence). Rebuild index.db from it with `polylogued run`; "
                "ordinary convergence replays source.db into index.db."
            )
        if _embeddings_db_present():
            env.ui.console.print(
                "Preserving embeddings.db (expensive to rebuild: vectors are re-purchased from the embedding "
                "provider, never replayed from source.db). Its vectors are keyed by content, so they are "
                "reused after an index rebuild."
            )
        if _user_db_present():
            env.ui.console.print(
                "Preserving user.db (irreplaceable: tags, annotations, marks, saved views, "
                "notes). Rebuild index.db from preserved source evidence with `polylogued run`."
            )
    if blob:
        _blob_root = blob_store_root()
        if _blob_root.exists():
            targets.append(("blob store", _blob_root))
    if assets:
        assets_dir = data_home() / "assets"
        if assets_dir.exists():
            targets.append(("assets", assets_dir))
    if cache:
        if cache_home().exists():
            targets.append(("cache/indexes", cache_home()))
        schemas_dir = data_home() / "schemas"
        if schemas_dir.exists():
            targets.append(("inferred schemas", schemas_dir))
        _drive_cache = drive_cache_path()
        if _drive_cache.exists():
            targets.append(("drive cache", _drive_cache))
    if auth and drive_token_path().exists():
        targets.append(("OAuth token", drive_token_path()))
    if reset_all:
        last_source = state_home() / "last-source.json"
        if last_source.exists():
            targets.append(("last-source state", last_source))
    targets = _dedupe_targets(targets)

    if not targets and not yes:
        env.ui.console.print("Nothing to reset (no files exist for selected targets).")
        return

    # Show what will be deleted
    lines = [f"  {name}: {path}" for name, path in targets]
    env.ui.summary("Will delete", lines)

    # Confirm unless --yes
    if not yes:
        if env.ui.plain:
            env.ui.console.print("Use --yes to confirm deletion.")
            return
        if not env.ui.confirm("Delete these files/directories?", default=False):
            env.ui.console.print("Reset cancelled.")
            return

    # The resident daemon re-resolves and deletes these targets. The CLI only
    # displays its read-only preview and lowers the confirmed intent.
    from polylogue.operations.daemon_mutations import reset_confirmation_paths

    result = _submit(
        env,
        "maintenance.reset",
        {
            "index": index,
            "database": database,
            "blob": blob,
            "assets": assets,
            "cache": cache,
            "auth": auth,
            "reset_all": reset_all,
            "confirm": True,
            "expected_targets": list(reset_confirmation_paths(targets)),
        },
    )
    deleted_value = result.get("affected_count", 0)
    deleted = deleted_value if isinstance(deleted_value, int) else 0
    result_body = result.get("result")
    result_body = result_body if isinstance(result_body, dict) else {}
    target_names = result_body.get("targets", [])
    target_names = target_names if isinstance(target_names, list) else []
    if result_body.get("state") == "staged":
        # The daemon holds index.db and ops.db open, so it records the
        # authorized reset and deletes it at its next start, before any tier
        # opens. Nothing is deleted yet.
        for name in target_names:
            env.ui.console.print(f"  Staged {name}")
        env.ui.console.print(
            f"\nReset staged: {len(target_names)} item(s) are deleted when polylogued next starts, "
            "before it opens any archive database. Nothing has been deleted yet."
        )
        env.ui.console.print(
            "Next: restart polylogued (for example `systemctl --user restart polylogued`); "
            "it applies the reset, then rebuilds index.db from source.db."
        )
        return
    for name in target_names:
        env.ui.console.print(f"  Deleted {name}")

    env.ui.console.print(f"\nReset complete: {deleted} item(s) deleted.")


def _dedupe_targets(targets: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    seen: set[Path] = set()
    deduped: list[tuple[str, Path]] = []
    for name, path in targets:
        resolved = path.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append((name, path))
    return deduped


__all__ = ["reset_command"]
