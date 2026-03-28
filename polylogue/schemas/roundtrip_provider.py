"""Per-provider schema-and-evidence roundtrip proof execution."""

from __future__ import annotations

from polylogue.schemas.roundtrip_models import (
    ProviderRoundtripProofReport,
    RoundtripStageReport,
)
from polylogue.schemas.roundtrip_models import (
    finalize_stages as _finalize_stages,
)
from polylogue.schemas.roundtrip_models import (
    stage_error as _stage_error,
)
from polylogue.schemas.roundtrip_models import (
    stage_ok as _stage_ok,
)
from polylogue.schemas.roundtrip_provider_context import (
    bind_roundtrip_storage,
    create_roundtrip_context,
)
from polylogue.schemas.roundtrip_provider_stages import (
    recover_verification_stages,
    run_acquisition_stage,
    run_artifact_proof_stage,
    run_corpus_verification_stage,
    run_parse_dispatch_stage,
    run_prepare_persist_stage,
    run_validation_stage,
)
from polylogue.schemas.synthetic.selection import select_synthetic_schema
from polylogue.showcase.workspace import override_workspace_env


async def _prove_provider_roundtrip(
    provider: str,
    *,
    count: int,
    style: str,
    seed: int,
) -> ProviderRoundtripProofReport:
    stages: dict[str, RoundtripStageReport] = {}
    selection = None
    current_stage = "selection"
    context = None
    try:
        selection = select_synthetic_schema(provider)
        stages["selection"] = _stage_ok(
            "selection",
            f"Selected {selection.package_version}/{selection.element_kind or 'default'}",
            package_version=selection.package_version,
            element_kind=selection.element_kind,
            wire_encoding=selection.wire_format.encoding,
        )

        current_stage = "synthetic"
        context, stages["selection"], stages["synthetic"] = create_roundtrip_context(
            provider,
            selection=selection,
            count=count,
            style=style,
            seed=seed,
        )

        with override_workspace_env(context.workspace.env_vars):
            context = bind_roundtrip_storage(context)
            current_stage = "acquisition"
            stages["acquisition"], acquire_result = await run_acquisition_stage(context)

            current_stage = "validation"
            stages["validation"], validation_result = await run_validation_stage(context, acquire_result)

            current_stage = "parse_dispatch"
            stages["parse_dispatch"], parsed_items = await run_parse_dispatch_stage(context, validation_result)

            current_stage = "prepare_persist"
            stages["prepare_persist"] = await run_prepare_persist_stage(context, parsed_items)

            current_stage = "corpus_verification"
            stages["corpus_verification"] = run_corpus_verification_stage(context)

            current_stage = "artifact_proof"
            stages["artifact_proof"] = run_artifact_proof_stage(context)
    except Exception as exc:
        stages[current_stage] = _stage_error(current_stage, exc)
        if context is not None and "acquisition" in stages:
            recover_verification_stages(
                provider,
                db_path=context.backend.db_path,
                stages=stages,
            )
    finally:
        if context is not None:
            await context.repository.close()

    return ProviderRoundtripProofReport(
        provider=provider,
        package_version=selection.package_version if selection is not None else "unknown",
        element_kind=selection.element_kind if selection is not None else None,
        wire_encoding=selection.wire_format.encoding if selection is not None else "unknown",
        stages=_finalize_stages(stages),
    )


__all__ = ["_prove_provider_roundtrip"]
