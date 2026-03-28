"""Per-provider schema-and-evidence roundtrip proof execution."""

from __future__ import annotations

from typing import Any

from polylogue.config import Config, Source
from polylogue.lib.raw_payload import build_raw_payload_envelope
from polylogue.pipeline.prepare import prepare_records
from polylogue.pipeline.services.acquisition import AcquisitionService
from polylogue.pipeline.services.parsing import ParsingService
from polylogue.pipeline.services.validation import ValidationService
from polylogue.schemas.operator_models import SchemaPayloadResolveRequest
from polylogue.schemas.operator_workflow import (
    list_artifact_observations,
    resolve_schema_payload,
    run_artifact_proof,
    run_schema_verification,
)
from polylogue.schemas.roundtrip_models import (
    ProviderRoundtripProofReport,
    RoundtripStageReport,
    finalize_stages as _finalize_stages,
    stage_error as _stage_error,
    stage_ok as _stage_ok,
)
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.synthetic.selection import select_synthetic_schema
from polylogue.schemas.verification_requests import (
    ArtifactObservationQuery,
    ArtifactProofRequest,
    SchemaVerificationRequest,
)
from polylogue.showcase.workspace import create_verification_workspace, override_workspace_env
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository


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
    repository: ConversationRepository | None = None
    backend: SQLiteBackend | None = None
    try:
        selection = select_synthetic_schema(provider)
        stages["selection"] = _stage_ok(
            "selection",
            f"Selected {selection.package_version}/{selection.element_kind or 'default'}",
            package_version=selection.package_version,
            element_kind=selection.element_kind,
            wire_encoding=selection.wire_format.encoding,
        )

        workspace = create_verification_workspace(prefix=f"polylogue-roundtrip-{provider}-")
        provider_dir = workspace.fixture_dir / provider
        provider_dir.mkdir(parents=True, exist_ok=True)
        config = Config(
            archive_root=workspace.archive_root,
            render_root=workspace.render_root,
            sources=[Source(name=provider, path=provider_dir)],
        )

        with override_workspace_env(workspace.env_vars):
            backend = SQLiteBackend()
            repository = ConversationRepository(backend=backend)
            current_stage = "synthetic"
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
            stages["synthetic"] = _stage_ok(
                "synthetic",
                f"Generated {batch.report.generated_count} synthetic artifact(s)",
                generated_artifacts=batch.report.generated_count,
                requested_artifacts=batch.report.requested_count,
                wire_encoding=batch.report.wire_encoding,
                style=batch.report.style,
            )

            first_artifact = batch.artifacts[0]
            envelope = build_raw_payload_envelope(
                first_artifact.raw_bytes,
                source_path=str(provider_dir / f"roundtrip-00{extension}"),
                fallback_provider=provider,
            )
            resolution = resolve_schema_payload(
                SchemaPayloadResolveRequest(
                    provider=envelope.provider,
                    payload=envelope.payload,
                    source_path=str(provider_dir / f"roundtrip-00{extension}"),
                )
            )
            if resolution.resolution is None:
                raise ValueError("Runtime registry could not resolve the synthetic payload")
            resolved = resolution.resolution
            if resolved.package_version != selection.package_version:
                raise ValueError(
                    f"Resolved package {resolved.package_version} does not match selected {selection.package_version}"
                )
            if (resolved.element_kind or selection.element_kind) != selection.element_kind:
                raise ValueError(
                    f"Resolved element {resolved.element_kind} does not match selected {selection.element_kind}"
                )
            stages["selection"] = _stage_ok(
                "selection",
                f"Selected {selection.package_version}/{selection.element_kind or 'default'}",
                package_version=selection.package_version,
                element_kind=selection.element_kind,
                wire_encoding=selection.wire_format.encoding,
                resolution_reason=resolved.reason,
            )

            current_stage = "acquisition"
            acquire_result = await AcquisitionService(backend=backend).acquire_sources(config.sources)
            if acquire_result.counts["errors"]:
                raise ValueError(f"Acquisition recorded {acquire_result.counts['errors']} error(s)")
            if acquire_result.counts["acquired"] != batch.report.generated_count:
                raise ValueError(
                    f"Acquired {acquire_result.counts['acquired']} raw record(s) for "
                    f"{batch.report.generated_count} generated artifact(s)"
                )
            stages["acquisition"] = _stage_ok(
                "acquisition",
                f"Acquired {acquire_result.counts['acquired']} raw record(s)",
                acquired=acquire_result.counts["acquired"],
                skipped=acquire_result.counts["skipped"],
                raw_ids=list(acquire_result.raw_ids),
            )

            current_stage = "validation"
            validation_result = await ValidationService(backend=backend).validate_raw_ids(
                raw_ids=acquire_result.raw_ids,
            )
            if validation_result.counts["errors"] or validation_result.counts["invalid"]:
                raise ValueError(
                    "Validation failed: "
                    f"errors={validation_result.counts['errors']}, "
                    f"invalid={validation_result.counts['invalid']}"
                )
            stages["validation"] = _stage_ok(
                "validation",
                f"Validated {len(validation_result.parseable_raw_ids)} parseable raw record(s)",
                validated=validation_result.counts["validated"],
                drift=validation_result.counts["drift"],
                skipped_no_schema=validation_result.counts["skipped_no_schema"],
                parseable_raw_ids=list(validation_result.parseable_raw_ids),
            )

            current_stage = "parse_dispatch"
            parsing_service = ParsingService(
                repository=repository,
                archive_root=workspace.archive_root,
                config=config,
            )
            raw_records = await repository.get_raw_conversations_batch(validation_result.parseable_raw_ids)
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
            stages["parse_dispatch"] = _stage_ok(
                "parse_dispatch",
                f"Parsed {len(parsed_items)} conversation(s) / {parsed_message_count} message(s)",
                parsed_conversations=len(parsed_items),
                parsed_messages=parsed_message_count,
                parseable_raw_ids=list(validation_result.parseable_raw_ids),
            )

            current_stage = "prepare_persist"
            persisted_conversations = 0
            persisted_messages = 0
            persisted_attachments = 0
            touched_raw_ids: set[str] = set()
            for parsed_conversation, source_name, raw_id, payload_provider in parsed_items:
                persisted = await prepare_records(
                    parsed_conversation,
                    source_name,
                    archive_root=workspace.archive_root,
                    backend=backend,
                    repository=repository,
                    raw_id=raw_id,
                )
                persisted_conversations += persisted.counts["conversations"]
                persisted_messages += persisted.counts["messages"]
                persisted_attachments += persisted.counts["attachments"]
                touched_raw_ids.add(raw_id)
                await repository.mark_raw_parsed(raw_id, payload_provider=payload_provider)
            stages["prepare_persist"] = _stage_ok(
                "prepare_persist",
                f"Persisted {persisted_conversations} conversation(s) / {persisted_messages} message(s)",
                persisted_conversations=persisted_conversations,
                persisted_messages=persisted_messages,
                persisted_attachments=persisted_attachments,
                parsed_raw_ids=sorted(touched_raw_ids),
            )

            current_stage = "corpus_verification"
            schema_report = run_schema_verification(
                SchemaVerificationRequest(providers=[provider]),
                db_path=backend.db_path,
            )
            provider_schema_report = schema_report.providers.get(provider)
            if provider_schema_report is None or provider_schema_report.invalid_records or provider_schema_report.decode_errors:
                raise ValueError(
                    f"Corpus verification failed for {provider}: "
                    f"{provider_schema_report.to_dict() if provider_schema_report else 'missing provider stats'}"
                )
            stages["corpus_verification"] = _stage_ok(
                "corpus_verification",
                f"Corpus verification passed for {provider}",
                total_records=schema_report.total_records,
                provider_stats=provider_schema_report.to_dict(),
            )

            current_stage = "artifact_proof"
            proof_result = run_artifact_proof(
                ArtifactProofRequest(providers=[provider]),
                db_path=backend.db_path,
            ).report
            observation_rows = list_artifact_observations(
                ArtifactObservationQuery(
                    providers=[provider],
                    support_statuses=["supported_parseable"],
                ),
                db_path=backend.db_path,
            ).rows
            matching_rows = [
                row
                for row in observation_rows
                if row.resolved_package_version == selection.package_version
                and row.resolved_element_kind == selection.element_kind
            ]
            if not proof_result.is_clean:
                raise ValueError(f"Artifact proof reported unresolved issues for {provider}")
            if not matching_rows:
                raise ValueError(
                    f"No supported artifact observations resolved to "
                    f"{selection.package_version}/{selection.element_kind}"
                )
            stages["artifact_proof"] = _stage_ok(
                "artifact_proof",
                f"Artifact proof resolved {len(matching_rows)} matching observation(s)",
                total_records=proof_result.total_records,
                contract_backed_records=proof_result.contract_backed_records,
                matching_observations=len(matching_rows),
                package_versions=proof_result.package_versions,
                element_kinds=proof_result.element_kinds,
            )
    except Exception as exc:
        stages[current_stage] = _stage_error(current_stage, exc)
        if backend is not None and "acquisition" in stages:
            if "corpus_verification" not in stages:
                try:
                    schema_report = run_schema_verification(
                        SchemaVerificationRequest(providers=[provider]),
                        db_path=backend.db_path,
                    )
                except Exception as verification_exc:
                    stages["corpus_verification"] = _stage_error("corpus_verification", verification_exc)
                else:
                    provider_schema_report = schema_report.providers.get(provider)
                    if provider_schema_report is None:
                        stages["corpus_verification"] = _stage_error(
                            "corpus_verification",
                            "Missing provider stats after verification",
                        )
                    else:
                        stages["corpus_verification"] = _stage_ok(
                            "corpus_verification",
                            f"Corpus verification completed for {provider}",
                            total_records=schema_report.total_records,
                            provider_stats=provider_schema_report.to_dict(),
                        )

            if "artifact_proof" not in stages:
                try:
                    proof_result = run_artifact_proof(
                        ArtifactProofRequest(providers=[provider]),
                        db_path=backend.db_path,
                    ).report
                except Exception as proof_exc:
                    stages["artifact_proof"] = _stage_error("artifact_proof", proof_exc)
                else:
                    stages["artifact_proof"] = _stage_ok(
                        "artifact_proof",
                        f"Artifact proof completed for {provider}",
                        total_records=proof_result.total_records,
                        contract_backed_records=proof_result.contract_backed_records,
                        package_versions=proof_result.package_versions,
                        element_kinds=proof_result.element_kinds,
                    )
    finally:
        if repository is not None:
            await repository.close()

    return ProviderRoundtripProofReport(
        provider=provider,
        package_version=selection.package_version if selection is not None else "unknown",
        element_kind=selection.element_kind if selection is not None else None,
        wire_encoding=selection.wire_format.encoding if selection is not None else "unknown",
        stages=_finalize_stages(stages),
    )


__all__ = ["_prove_provider_roundtrip"]
