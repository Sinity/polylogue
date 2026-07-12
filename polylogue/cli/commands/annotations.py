"""Typed annotation batch import commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click

from polylogue.annotations.importer import (
    MAX_ANNOTATION_IMPORT_BYTES,
    AnnotationBatchImportError,
    AnnotationBatchImportRequest,
    AnnotationBatchImportResult,
    import_annotation_batch,
)
from polylogue.api import Polylogue
from polylogue.cli.shared.helpers import fail
from polylogue.paths import archive_root


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
def import_annotations_command(
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
    """Import bounded JSONL labels as candidate assertions."""

    try:
        with path.open("rb") as source:
            raw_jsonl = source.read(MAX_ANNOTATION_IMPORT_BYTES + 1)
        if len(raw_jsonl) > MAX_ANNOTATION_IMPORT_BYTES:
            raise AnnotationBatchImportError(f"annotation JSONL exceeds {MAX_ANNOTATION_IMPORT_BYTES} byte limit")
        metadata = json.loads(metadata_json)
        if not isinstance(metadata, dict):
            raise AnnotationBatchImportError("--metadata-json must decode to a JSON object")
        request = AnnotationBatchImportRequest(
            jsonl=raw_jsonl.decode("utf-8"),
            batch_id=batch_id,
            schema_id=schema_id,
            schema_version=schema_version,
            target_ref=target_ref,
            source_result_ref=source_result_ref,
            actor_ref=actor_ref,
            model_ref=model_ref,
            prompt_ref=prompt_ref,
            metadata=metadata,
        )

        async def run() -> AnnotationBatchImportResult:
            async with Polylogue(archive_root=archive_root()) as poly:
                return await import_annotation_batch(poly, request)

        result = asyncio.run(run())
    except (OSError, UnicodeError, AnnotationBatchImportError, ValueError) as exc:
        fail("annotations import", str(exc))
    click.echo(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))


__all__ = ["annotations_command", "import_annotations_command"]
