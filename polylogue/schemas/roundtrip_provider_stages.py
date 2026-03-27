"""Execution stages for per-provider roundtrip proof."""

from __future__ import annotations

from typing import Any

from polylogue.pipeline.prepare import prepare_records
from polylogue.pipeline.services.acquisition import AcquisitionService
from polylogue.pipeline.services.parsing import ParsingService
from polylogue.pipeline.services.validation import ValidationService
from polylogue.schemas.operator_workflow import (
    list_artifact_observations,
    run_artifact_proof,
    run_schema_verification,
)
from polylogue.schemas.roundtrip_models import RoundtripStageReport, stage_error, stage_ok
from polylogue.schemas.roundtrip_provider_context import RoundtripProviderContext
from polylogue.schemas.verification_requests import (
    ArtifactObservationQuery,
    ArtifactProofRequest,
    SchemaVerificationRequest,
)


async def run_acquisition_stage(context: RoundtripProviderContext) -> tuple[RoundtripStageReport, Any]:
    assert context.backend is not None
    acquire_result = await AcquisitionService(backend=context.backend).acquire_sources(context.config.sources)
    if acquire_result.counts["errors"]:
        raise ValueError(f"Acquisition recorded {acquire_result.counts['errors']} error(s)")
    if acquire_result.counts["acquired"] != context.batch.report.generated_count:
        raise ValueError(
            f"Acquired {acquire_result.counts['acquired']} raw record(s) for "
            f"{context.batch.report.generated_count} generated artifact(s)"
        )
    return (
        stage_ok(
            "acquisition",
            f"Acquired {acquire_result.counts['acquired']} raw record(s)",
            acquired=acquire_result.counts["acquired"],
            skipped=acquire_result.counts["skipped"],
            raw_ids=list(acquire_result.raw_ids),
        ),
        acquire_result,
    )


async def run_validation_stage(
    context: RoundtripProviderContext,
    acquire_result: Any,
) -> tuple[RoundtripStageReport, Any]:
    assert context.backend is not None
    validation_result = await ValidationService(backend=context.backend).validate_raw_ids(
        raw_ids=acquire_result.raw_ids,
    )
    if validation_result.counts["errors"] or validation_result.counts["invalid"]:
        raise ValueError(
            "Validation failed: "
            f"errors={validation_result.counts['errors']}, "
            f"invalid={validation_result.counts['invalid']}"
        )
    return (
        stage_ok(
            "validation",
            f"Validated {len(validation_result.parseable_raw_ids)} parseable raw record(s)",
            validated=validation_result.counts["validated"],
            drift=validation_result.counts["drift"],
            skipped_no_schema=validation_result.counts["skipped_no_schema"],
            parseable_raw_ids=list(validation_result.parseable_raw_ids),
        ),
        validation_result,
    )


async def run_parse_dispatch_stage(
    context: RoundtripProviderContext,
    validation_result: Any,
) -> tuple[RoundtripStageReport, list[tuple[Any, str, str, Any]]]:
    assert context.repository is not None
    parsing_service = ParsingService(
        repository=context.repository,
        archive_root=context.workspace.archive_root,
        config=context.config,
    )
    raw_records = await context.repository.get_raw_conversations_batch(validation_result.parseable_raw_ids)
    parsed_items: list[tuple[Any, str, str, Any]] = []
    parsed_message_count = 0
    for raw_record in raw_records:
        parsed_conversations = await parsing_service._parse_raw_record(raw_record)
        for parsed_conversation in parsed_conversations:
            parsed_items.append(
                (
                    parsed_conversation,
                    raw_record.source_name or raw_record.source_path,
                    raw_record.raw_id,
                    raw_record.payload_provider,
                )
            )
            parsed_message_count += len(parsed_conversation.messages)
    if not parsed_items:
        raise ValueError("Parser produced no conversations from validated raw payloads")
    return (
        stage_ok(
            "parse_dispatch",
            f"Parsed {len(parsed_items)} conversation(s) / {parsed_message_count} message(s)",
            parsed_conversations=len(parsed_items),
            parsed_messages=parsed_message_count,
            parseable_raw_ids=list(validation_result.parseable_raw_ids),
        ),
        parsed_items,
    )


async def run_prepare_persist_stage(
    context: RoundtripProviderContext,
    parsed_items: list[tuple[Any, str, str, Any]],
) -> RoundtripStageReport:
    assert context.backend is not None
    assert context.repository is not None
    persisted_conversations = 0
    persisted_messages = 0
    persisted_attachments = 0
    touched_raw_ids: set[str] = set()
    for parsed_conversation, source_name, raw_id, payload_provider in parsed_items:
        persisted = await prepare_records(
            parsed_conversation,
            source_name,
            archive_root=context.workspace.archive_root,
            backend=context.backend,
            repository=context.repository,
            raw_id=raw_id,
        )
        persisted_conversations += persisted.counts["conversations"]
        persisted_messages += persisted.counts["messages"]
        persisted_attachments += persisted.counts["attachments"]
        touched_raw_ids.add(raw_id)
        await context.repository.mark_raw_parsed(raw_id, payload_provider=payload_provider)
    return stage_ok(
        "prepare_persist",
        f"Persisted {persisted_conversations} conversation(s) / {persisted_messages} message(s)",
        persisted_conversations=persisted_conversations,
        persisted_messages=persisted_messages,
        persisted_attachments=persisted_attachments,
        parsed_raw_ids=sorted(touched_raw_ids),
    )


def run_corpus_verification_stage(context: RoundtripProviderContext) -> RoundtripStageReport:
    assert context.backend is not None
    schema_report = run_schema_verification(
        SchemaVerificationRequest(providers=[context.provider]),
        db_path=context.backend.db_path,
    )
    provider_schema_report = schema_report.providers.get(context.provider)
    if provider_schema_report is None or provider_schema_report.invalid_records or provider_schema_report.decode_errors:
        raise ValueError(
            f"Corpus verification failed for {context.provider}: "
            f"{provider_schema_report.to_dict() if provider_schema_report else 'missing provider stats'}"
        )
    return stage_ok(
        "corpus_verification",
        f"Corpus verification passed for {context.provider}",
        total_records=schema_report.total_records,
        provider_stats=provider_schema_report.to_dict(),
    )


def run_artifact_proof_stage(context: RoundtripProviderContext) -> RoundtripStageReport:
    assert context.backend is not None
    proof_result = run_artifact_proof(
        ArtifactProofRequest(providers=[context.provider]),
        db_path=context.backend.db_path,
    ).report
    observation_rows = list_artifact_observations(
        ArtifactObservationQuery(
            providers=[context.provider],
            support_statuses=["supported_parseable"],
        ),
        db_path=context.backend.db_path,
    ).rows
    matching_rows = [
        row
        for row in observation_rows
        if row.resolved_package_version == context.selection.package_version
        and row.resolved_element_kind == context.selection.element_kind
    ]
    if not proof_result.is_clean:
        raise ValueError(f"Artifact proof reported unresolved issues for {context.provider}")
    if not matching_rows:
        raise ValueError(
            f"No supported artifact observations resolved to "
            f"{context.selection.package_version}/{context.selection.element_kind}"
        )
    return stage_ok(
        "artifact_proof",
        f"Artifact proof resolved {len(matching_rows)} matching observation(s)",
        total_records=proof_result.total_records,
        contract_backed_records=proof_result.contract_backed_records,
        matching_observations=len(matching_rows),
        package_versions=proof_result.package_versions,
        element_kinds=proof_result.element_kinds,
    )


def recover_verification_stages(
    provider: str,
    *,
    db_path,
    stages: dict[str, RoundtripStageReport],
) -> None:
    """Best-effort verification/proof capture after an earlier stage failed."""
    if "corpus_verification" not in stages:
        try:
            schema_report = run_schema_verification(
                SchemaVerificationRequest(providers=[provider]),
                db_path=db_path,
            )
        except Exception as verification_exc:
            stages["corpus_verification"] = stage_error("corpus_verification", verification_exc)
        else:
            provider_schema_report = schema_report.providers.get(provider)
            if provider_schema_report is None:
                stages["corpus_verification"] = stage_error(
                    "corpus_verification",
                    "Missing provider stats after verification",
                )
            else:
                stages["corpus_verification"] = stage_ok(
                    "corpus_verification",
                    f"Corpus verification completed for {provider}",
                    total_records=schema_report.total_records,
                    provider_stats=provider_schema_report.to_dict(),
                )

    if "artifact_proof" not in stages:
        try:
            proof_result = run_artifact_proof(
                ArtifactProofRequest(providers=[provider]),
                db_path=db_path,
            ).report
        except Exception as proof_exc:
            stages["artifact_proof"] = stage_error("artifact_proof", proof_exc)
        else:
            stages["artifact_proof"] = stage_ok(
                "artifact_proof",
                f"Artifact proof completed for {provider}",
                total_records=proof_result.total_records,
                contract_backed_records=proof_result.contract_backed_records,
                package_versions=proof_result.package_versions,
                element_kinds=proof_result.element_kinds,
            )


__all__ = [
    "recover_verification_stages",
    "run_acquisition_stage",
    "run_artifact_proof_stage",
    "run_corpus_verification_stage",
    "run_parse_dispatch_stage",
    "run_prepare_persist_stage",
    "run_validation_stage",
]
