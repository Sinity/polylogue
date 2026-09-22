"""Typed annotation batch import commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

import click

from polylogue.annotations.join import (
    AnnotationGroupDimension,
    AnnotationStructuralJoinError,
    AnnotationStructuralJoinRequest,
    AnnotationStructuralJoinResult,
    join_typed_annotations,
)
from polylogue.api import Polylogue
from polylogue.cli.shared.helpers import fail
from polylogue.cli.shared.types import AppEnv
from polylogue.core.enums import AssertionStatus
from polylogue.paths import archive_root

#: Mirrors ``polylogue.annotations.importer.MAX_ANNOTATION_IMPORT_BYTES`` and
#: the ``jsonl`` bound on ``mutation.annotation.import_batch``'s request
#: contract. Duplicated as a literal rather than imported: importing that
#: module from ``polylogue/cli`` is exactly the direct substrate-driving
#: import the mutation-authority layering rule (docs/plans/layering.yaml)
#: disallows for this package (polylogue-gjwto / polylogue-r29bv AC3), and
#: this bound is not the enforcement -- it only keeps a CLI process from
#: buffering more than the operation would ever accept before sending it.
_MAX_ANNOTATION_JSONL_READ_BYTES = 1_048_576


@click.group("annotations")
def annotations_command() -> None:
    """Import and inspect typed annotation batches."""


@annotations_command.command("import")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--batch-id", required=True)
@click.option("--schema-id", required=True)
@click.option("--schema-version", required=True, type=click.IntRange(min=1))
@click.option("--target-ref", required=True)
@click.option("--source-result-ref", required=True)
@click.option("--actor-ref", required=True)
@click.option("--model-ref", required=True)
@click.option("--prompt-ref", required=True)
@click.option("--metadata-json", default="{}", show_default=True, help="Batch provenance metadata JSON object.")
@click.pass_obj
def import_annotations_command(
    env: AppEnv,
    path: Path,
    batch_id: str,
    schema_id: str,
    schema_version: int,
    target_ref: str,
    source_result_ref: str,
    actor_ref: str,
    model_ref: str,
    prompt_ref: str,
    metadata_json: str,
) -> None:
    """Import bounded JSONL labels as candidate assertions.

    ``user.db`` is the archive's one irreplaceable tier and the daemon is its
    sole writer, so this lowers to the declared
    ``mutation.annotation.import_batch`` operation instead of driving
    ``OperationExecutor`` against ``user.db`` from the CLI process
    (polylogue-gjwto / polylogue-r29bv). With no daemon the command refuses
    rather than becoming a second writer.
    """

    try:
        with path.open("rb") as source:
            raw_jsonl = source.read(_MAX_ANNOTATION_JSONL_READ_BYTES + 1)
        if len(raw_jsonl) > _MAX_ANNOTATION_JSONL_READ_BYTES:
            raise ValueError(f"annotation JSONL exceeds {_MAX_ANNOTATION_JSONL_READ_BYTES} byte limit")
        metadata = json.loads(metadata_json)
        if not isinstance(metadata, dict):
            raise ValueError("--metadata-json must decode to a JSON object")
        jsonl_text = raw_jsonl.decode("utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        fail("annotations import", str(exc))

    from polylogue.cli.archive_query import submit_cli_mutation

    written = submit_cli_mutation(
        env,
        "mutation.annotation.import_batch",
        {
            "jsonl": jsonl_text,
            "batch_id": batch_id,
            "schema_id": schema_id,
            "schema_version": schema_version,
            "target_ref": target_ref,
            "source_result_ref": source_result_ref,
            "actor_ref": actor_ref,
            "model_ref": model_ref,
            "prompt_ref": prompt_ref,
            "metadata": metadata,
        },
    )
    result = written.get("result")
    if not isinstance(result, dict):
        raise click.ClickException("daemon accepted the annotation batch but returned no result")
    click.echo(json.dumps(result, ensure_ascii=False, sort_keys=True))


@annotations_command.command("join")
@click.option("--schema-id", required=True)
@click.option("--schema-version", required=True, type=click.IntRange(min=1))
@click.option(
    "--status",
    "statuses",
    required=True,
    multiple=True,
    type=click.Choice(("active", "candidate", "accepted", "rejected", "deferred", "superseded")),
)
@click.option("--target-kind", default=None)
@click.option("--group-by", multiple=True, type=click.Choice(("repo", "model", "time", "origin")))
@click.option("--limit", "-l", default=500, show_default=True, type=click.IntRange(min=1, max=1_000))
@click.option("--offset", default=0, show_default=True, type=click.IntRange(min=0))
def join_annotations_command(
    schema_id: str,
    schema_version: int,
    statuses: tuple[str, ...],
    target_kind: str | None,
    group_by: tuple[str, ...],
    limit: int,
    offset: int,
) -> None:
    """Join typed labels to exact structural targets without fanout."""

    try:
        request = AnnotationStructuralJoinRequest(
            schema_id=schema_id,
            schema_version=schema_version,
            statuses=tuple(AssertionStatus.from_string(status) for status in statuses),
            target_kind=target_kind,
            group_by=cast(tuple[AnnotationGroupDimension, ...], group_by),
            limit=limit,
            offset=offset,
        )

        async def run() -> AnnotationStructuralJoinResult:
            async with Polylogue(archive_root=archive_root()) as poly:
                return await join_typed_annotations(poly, request)

        result = asyncio.run(run())
    except (AnnotationStructuralJoinError, ValueError) as exc:
        fail("annotations join", str(exc))
    click.echo(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))


__all__ = ["annotations_command", "import_annotations_command", "join_annotations_command"]
