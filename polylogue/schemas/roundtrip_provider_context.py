"""Context/setup helpers for per-provider roundtrip proof."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from polylogue.config import Config, Source
from polylogue.lib.raw_payload import build_raw_payload_envelope
from polylogue.schemas.operator_models import SchemaPayloadResolveRequest
from polylogue.schemas.operator_workflow import resolve_schema_payload
from polylogue.schemas.roundtrip_models import RoundtripStageReport, stage_ok
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.synthetic.models import SyntheticGenerationBatch, SyntheticSchemaSelection
from polylogue.showcase.workspace import create_verification_workspace
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository


@dataclass(frozen=True)
class RoundtripProviderContext:
    provider: str
    selection: SyntheticSchemaSelection
    workspace: object
    provider_dir: Path
    config: Config
    batch: SyntheticGenerationBatch
    extension: str
    backend: SQLiteBackend | None = None
    repository: ConversationRepository | None = None


def create_roundtrip_context(
    provider: str,
    *,
    selection: SyntheticSchemaSelection,
    count: int,
    style: str,
    seed: int,
) -> tuple[RoundtripProviderContext, RoundtripStageReport, RoundtripStageReport]:
    """Create workspace/backend context and synthetic artifacts for a provider."""
    workspace = create_verification_workspace(prefix=f"polylogue-roundtrip-{provider}-")
    provider_dir = workspace.fixture_dir / provider
    provider_dir.mkdir(parents=True, exist_ok=True)
    config = Config(
        archive_root=workspace.archive_root,
        render_root=workspace.render_root,
        sources=[Source(name=provider, path=provider_dir)],
    )
    corpus = SyntheticCorpus.from_selection(selection)
    batch = corpus.generate_batch(
        count=count,
        messages_per_conversation=range(4, 9),
        seed=seed,
        style=style,
    )
    extension = ".json" if selection.wire_format.encoding == "json" else ".jsonl"
    for index, artifact in enumerate(batch.artifacts):
        (provider_dir / f"roundtrip-{index:02d}{extension}").write_bytes(artifact.raw_bytes)

    synthetic_stage = stage_ok(
        "synthetic",
        f"Generated {batch.report.generated_count} synthetic artifact(s)",
        generated_artifacts=batch.report.generated_count,
        requested_artifacts=batch.report.requested_count,
        wire_encoding=batch.report.wire_encoding,
        style=batch.report.style,
    )

    first_artifact = batch.artifacts[0]
    source_path = str(provider_dir / f"roundtrip-00{extension}")
    envelope = build_raw_payload_envelope(
        first_artifact.raw_bytes,
        source_path=source_path,
        fallback_provider=provider,
    )
    resolution_result = resolve_schema_payload(
        SchemaPayloadResolveRequest(
            provider=envelope.provider,
            payload=envelope.payload,
            source_path=source_path,
        )
    )
    if resolution_result.resolution is None:
        raise ValueError("Runtime registry could not resolve the synthetic payload")
    resolved = resolution_result.resolution
    if resolved.package_version != selection.package_version:
        raise ValueError(
            f"Resolved package {resolved.package_version} does not match selected {selection.package_version}"
        )
    if (resolved.element_kind or selection.element_kind) != selection.element_kind:
        raise ValueError(
            f"Resolved element {resolved.element_kind} does not match selected {selection.element_kind}"
        )

    selection_stage = stage_ok(
        "selection",
        f"Selected {selection.package_version}/{selection.element_kind or 'default'}",
        package_version=selection.package_version,
        element_kind=selection.element_kind,
        wire_encoding=selection.wire_format.encoding,
        resolution_reason=resolved.reason,
    )

    return (
        RoundtripProviderContext(
            provider=provider,
            selection=selection,
            workspace=workspace,
            provider_dir=provider_dir,
            config=config,
            batch=batch,
            extension=extension,
        ),
        selection_stage,
        synthetic_stage,
    )


def bind_roundtrip_storage(context: RoundtripProviderContext) -> RoundtripProviderContext:
    """Bind backend/repository after the workspace env override is active."""
    backend = SQLiteBackend()
    repository = ConversationRepository(backend=backend)
    return replace(context, backend=backend, repository=repository)


__all__ = [
    "RoundtripProviderContext",
    "bind_roundtrip_storage",
    "create_roundtrip_context",
]
