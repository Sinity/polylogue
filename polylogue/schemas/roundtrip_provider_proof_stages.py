"""Verification/proof stages for per-provider roundtrip proof."""

from __future__ import annotations

from polylogue.schemas.operator_workflow import (
    list_artifact_observations,
    run_artifact_proof,
    run_schema_verification,
)
from polylogue.schemas.roundtrip_models import stage_error, stage_ok
from polylogue.schemas.verification_requests import (
    ArtifactObservationQuery,
    ArtifactProofRequest,
    SchemaVerificationRequest,
)


def run_corpus_verification_stage(context) -> object:
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


def run_artifact_proof_stage(context) -> object:
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
    stages: dict[str, object],
) -> None:
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
    "run_artifact_proof_stage",
    "run_corpus_verification_stage",
]
