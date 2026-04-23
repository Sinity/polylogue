from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from polylogue.cli.schema_command_support import (
    _privacy_config_payload,
    _privacy_level,
    build_schema_privacy_config,
)
from polylogue.cli.schema_rendering_explain import render_explain_verbose, render_schema_explain_result
from polylogue.cli.schema_rendering_results import (
    render_schema_audit_result,
    render_schema_compare_result,
    render_schema_generate_result,
    render_schema_list_result,
    render_schema_promote_result,
)
from polylogue.lib.outcomes import OutcomeStatus
from polylogue.scenarios import CorpusProfile, CorpusSpec, build_corpus_scenarios
from polylogue.schemas.audit_models import AuditCheck, AuditReport
from polylogue.schemas.generation_models import GenerationResult
from polylogue.schemas.operator_models import (
    JSONDocument,
    SchemaAnnotationSummary,
    SchemaCompareResult,
    SchemaCoverageSummary,
    SchemaExplainResult,
    SchemaInferResult,
    SchemaListResult,
    SchemaPromoteResult,
    SchemaProviderSnapshot,
    SchemaReviewProof,
    SchemaRoleAssignment,
    SchemaRoleProofEntry,
)
from polylogue.schemas.packages import SchemaElementManifest, SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.privacy_config import PrivacyConfig
from polylogue.schemas.redaction_report import SchemaReport
from polylogue.schemas.tooling_models import ClusterManifest, SchemaCluster, SchemaDiff


def _package(*, version: str = "v1") -> SchemaVersionPackage:
    return SchemaVersionPackage(
        provider="chatgpt",
        version=version,
        anchor_kind="conversation_document",
        default_element_kind="conversation_document",
        first_seen="2026-04-01T00:00:00Z",
        last_seen="2026-04-02T00:00:00Z",
        bundle_scope_count=2,
        sample_count=12,
        anchor_profile_family_id="cluster-a",
        profile_family_ids=["cluster-a", "cluster-b"],
        elements=[
            SchemaElementManifest(
                element_kind="conversation_document",
                schema_file="conversation_document.schema.json",
                sample_count=12,
                artifact_count=12,
                bundle_scope_count=2,
                profile_family_ids=["cluster-a", "cluster-b"],
                first_seen="2026-04-01T00:00:00Z",
                last_seen="2026-04-02T00:00:00Z",
            )
        ],
        orphan_adjunct_counts={"attachment": 2},
    )


def _manifest() -> ClusterManifest:
    return ClusterManifest(
        provider="chatgpt",
        default_version="v1",
        artifact_counts={"conversation_document": 12},
        clusters=[
            SchemaCluster(
                cluster_id="cluster-a",
                provider="chatgpt",
                sample_count=12,
                first_seen="2026-04-01T00:00:00Z",
                last_seen="2026-04-02T00:00:00Z",
                representative_paths=["/tmp/source.json"],
                dominant_keys=["id", "messages"],
                confidence=0.91,
                artifact_kind="conversation_document",
                promoted_package_version="v1",
            )
        ],
    )


def _corpus_specs() -> tuple[CorpusSpec, ...]:
    return (
        CorpusSpec.for_provider(
            "chatgpt",
            package_version="v1",
            element_kind="conversation_document",
            count=2,
            messages_min=3,
            messages_max=6,
            profile=CorpusProfile(
                family_ids=("cluster-a",),
                artifact_kind="conversation_document",
                observed_sample_count=12,
            ),
        ),
        CorpusSpec.for_provider(
            "chatgpt",
            package_version="v1",
            element_kind="conversation_document",
            count=3,
            messages_min=2,
            messages_max=5,
            profile=CorpusProfile(
                family_ids=("cluster-b",),
                artifact_kind="conversation_document",
                observed_sample_count=7,
            ),
        ),
    )


def _annotation_summary() -> SchemaAnnotationSummary:
    return SchemaAnnotationSummary(
        semantic_count=2,
        format_count=1,
        values_count=1,
        total_enum_values=6,
        roles=[
            SchemaRoleAssignment(
                path="$.messages[*].role",
                role="message_role",
                confidence=0.95,
                evidence={"signal": "name-match"},
            )
        ],
        coverage=SchemaCoverageSummary(
            total_fields=10,
            with_format=3,
            with_values=1,
            with_role=2,
        ),
    )


def _schema_payload() -> JSONDocument:
    return {
        "$id": "schema://chatgpt/v1",
        "title": "ChatGPT Conversation",
        "description": "Observed ChatGPT payload schema",
        "x-polylogue-version": "v1",
        "x-polylogue-generated-at": "2026-04-01T00:00:00Z",
        "x-polylogue-registered-at": "2026-04-02T00:00:00Z",
        "x-polylogue-promoted-at": "2026-04-03T00:00:00Z",
        "x-polylogue-sample-count": 12,
        "x-polylogue-sample-granularity": "conversation",
        "x-polylogue-anchor-profile-family-id": "cluster-a",
        "x-polylogue-profile-family-ids": ["cluster-a", "cluster-b"],
        "x-polylogue-package-profile-family-ids": ["cluster-a", "cluster-b"],
        "x-polylogue-observed-artifact-count": 12,
        "x-polylogue-evidence-confidence": 0.91,
        "x-polylogue-foreign-keys": [{"path": "$.messages[*].parent_id"}],
        "x-polylogue-time-deltas": [{"path": "$.messages[*].created_at"}],
        "x-polylogue-mutually-exclusive": [{"paths": ["$.a", "$.b"]}],
        "properties": {
            "role": {
                "type": "string",
                "x-polylogue-semantic-role": "message_role",
                "x-polylogue-format": "enum",
                "x-polylogue-frequency": 1.0,
                "x-polylogue-values": ["user", "assistant", "system", "tool", "other", "extra"],
            },
            "messages": {
                "type": "array",
            },
        },
    }


def _explain_result(*, review_proof: SchemaReviewProof | None = None) -> SchemaExplainResult:
    return SchemaExplainResult(
        provider="chatgpt",
        version="v1",
        element_kind="conversation_document",
        package=_package(),
        schema=_schema_payload(),
        annotations=_annotation_summary(),
        review_proof=review_proof,
    )


def test_schema_command_support_builds_payloads_from_privacy_inputs(tmp_path: Path) -> None:
    config = PrivacyConfig(
        level="strict",
        field_overrides={"$.id": "deny"},
        allow_value_patterns=["safe*"],
        deny_value_patterns=["secret*"],
        safe_enum_max_length=12,
        high_entropy_min_length=14,
        cross_conv_min_count=5,
        cross_conv_proportional=True,
    )

    assert _privacy_level("strict") == "strict"
    assert _privacy_level("permissive") == "permissive"
    assert _privacy_level("bogus") == "standard"
    assert _privacy_config_payload(config) == {
        "level": "strict",
        "safe_enum_max_length": 12,
        "high_entropy_min_length": 14,
        "cross_conv_min_count": 5,
        "cross_conv_proportional": True,
        "field_overrides": {"$.id": "deny"},
        "allow_value_patterns": ["safe*"],
        "deny_value_patterns": ["secret*"],
    }

    privacy_path = tmp_path / "polylogue-schemas.toml"
    privacy_path.write_text("", encoding="utf-8")
    with patch(
        "polylogue.schemas.privacy_config.load_privacy_config",
        return_value=config,
    ) as load_privacy_config:
        loaded = build_schema_privacy_config(privacy="strict", privacy_config_path=privacy_path)

    load_privacy_config.assert_called_once_with(
        cli_overrides={"level": "strict"},
        project_path=tmp_path,
    )
    assert loaded == _privacy_config_payload(config)
    assert build_schema_privacy_config(privacy="bogus", privacy_config_path=None) == {
        "level": "standard",
        "safe_enum_max_length": 50,
        "high_entropy_min_length": 10,
        "cross_conv_min_count": 3,
        "cross_conv_proportional": False,
    }
    assert build_schema_privacy_config(privacy=None, privacy_config_path=None) is None


def test_render_schema_explain_result_json_and_verbose_text_paths() -> None:
    result = _explain_result()

    with patch("polylogue.cli.schema_rendering_explain.emit_success") as emit_success:
        render_schema_explain_result(result=result, json_output=True, verbose=False)

    emit_success.assert_called_once_with(result.to_dict())

    with (
        patch("click.echo") as echo,
        patch("polylogue.cli.schema_rendering_explain.render_explain_verbose") as render_verbose,
    ):
        render_schema_explain_result(result=result, json_output=False, verbose=True)

    echoed = [call.args[0] for call in echo.call_args_list if call.args]
    assert "Schema: chatgpt v1 [conversation_document]" in echoed
    assert any("Package anchor=conversation_document" in line for line in echoed)
    assert any("Properties (2):" in line for line in echoed)
    assert any("role: string" in line for line in echoed)
    assert any("x-polylogue-foreign-keys:" in line for line in echoed)
    render_verbose.assert_called_once_with(result)


def test_render_explain_verbose_and_review_proof_surfaces() -> None:
    result = _explain_result()

    with patch("click.echo") as echo:
        render_explain_verbose(result)

    echoed = [call.args[0] for call in echo.call_args_list if call.args]
    assert "  Semantic Roles:" in echoed
    assert any("message_role -> $.messages[*].role" in line for line in echoed)
    assert any("Annotation Coverage (10 fields):" in line for line in echoed)

    proof_result = _explain_result(
        review_proof=SchemaReviewProof(
            artifact_kind="conversation_document",
            eligible_roles=["message_role", "message_text"],
            ineligible_roles=["message_tokens"],
            roles=[
                SchemaRoleProofEntry(
                    role="message_role",
                    chosen_path="$.messages[*].role",
                    chosen_score=0.97,
                    competing=[{"path": "$.role", "score": 0.40}],
                    evidence={"name_similarity": 1.0},
                    abstained=False,
                ),
                SchemaRoleProofEntry(
                    role="message_text",
                    chosen_path=None,
                    chosen_score=0.0,
                    competing=[],
                    evidence={},
                    abstained=True,
                    abstain_reason="no plausible path",
                ),
            ],
        )
    )

    with patch("click.echo") as echo:
        render_schema_explain_result(result=proof_result, json_output=False, verbose=False)

    proof_lines = [call.args[0] for call in echo.call_args_list if call.args]
    assert "Schema Review Proof: chatgpt v1" in proof_lines
    assert any("message_role: $.messages[*].role" in line for line in proof_lines)
    assert any("competing (1):" in line for line in proof_lines)
    assert any("message_text: ABSTAINED" in line for line in proof_lines)
    assert any("reason: no plausible path" in line for line in proof_lines)


def test_render_schema_generate_result_covers_plain_and_json_paths(tmp_path: Path) -> None:
    corpus_specs = _corpus_specs()
    corpus_scenarios = build_corpus_scenarios(corpus_specs)
    generation = GenerationResult(
        provider="chatgpt",
        schema={"type": "object"},
        sample_count=12,
        cluster_count=1,
        package_count=2,
        versions=["v1", "v2"],
        default_version="v1",
        artifact_counts={"conversation_document": 12},
        redaction_report=SchemaReport(
            provider="chatgpt",
            privacy_level="standard",
            total_fields=4,
            total_values_considered=10,
            total_included=3,
            total_rejected=1,
            rejection_reasons={"identifier_field": 1},
        ),
    )
    result = SchemaInferResult(
        generation=generation,
        manifest=_manifest(),
        manifest_path=tmp_path / "manifest.json",
        corpus_specs=corpus_specs,
        corpus_scenarios=corpus_scenarios,
    )

    with patch("polylogue.cli.schema_rendering_results.emit_success") as emit_success:
        render_schema_generate_result(provider="chatgpt", result=result, json_output=True, report=False)

    emit_payload = emit_success.call_args.args[0]
    assert emit_payload["provider"] == "chatgpt"
    assert emit_payload["manifest_path"] == str(tmp_path / "manifest.json")
    assert len(emit_payload["corpus_specs"]) == 2
    assert len(emit_payload["corpus_scenarios"]) == 1

    with (
        patch("click.echo") as echo,
        patch("pathlib.Path.write_text") as write_text,
    ):
        render_schema_generate_result(provider="chatgpt", result=result, json_output=False, report=True)

    write_text.assert_called_once()
    assert write_text.call_args.kwargs == {"encoding": "utf-8"}
    assert write_text.call_args.args[0].startswith("# Schema Redaction Report: chatgpt")
    echoed = [call.args[0] for call in echo.call_args_list if call.args]
    assert any(line.startswith("schema[chatgpt]: 4 fields, 10 samples") for line in echoed)
    assert any("Redaction report: chatgpt-redaction-report.md" in line for line in echoed)
    assert any("Generated schema package set for chatgpt" in line for line in echoed)
    assert any("Suggested synthetic scenarios:" in line for line in echoed)
    assert any("Suggested synthetic corpus specs:" in line for line in echoed)


def test_render_schema_list_result_covers_selected_global_json_and_empty_paths() -> None:
    corpus_specs = _corpus_specs()
    corpus_scenarios = build_corpus_scenarios(corpus_specs)
    snapshot = SchemaProviderSnapshot(
        provider="chatgpt",
        versions=["v1", "v2"],
        catalog=SchemaPackageCatalog(
            provider="chatgpt",
            packages=[_package()],
            default_version="v1",
            latest_version="v2",
            recommended_version="v1",
            orphan_adjunct_counts={"attachment": 2},
        ),
        manifest=_manifest(),
        latest_age_days=3,
        corpus_specs=corpus_specs,
        corpus_scenarios=corpus_scenarios,
    )

    with patch("click.echo") as echo:
        render_schema_list_result(
            provider="chatgpt",
            result=SchemaListResult(provider="chatgpt", selected=snapshot),
            json_output=False,
        )

    echoed = [call.args[0] for call in echo.call_args_list if call.args]
    assert "Provider: chatgpt" in echoed
    assert any("Versions: v1, v2" in line for line in echoed)
    assert any("Orphan adjunct evidence: attachment=2" in line for line in echoed)
    assert any("Evidence manifest (1 clusters):" in line for line in echoed)
    assert any("Suggested synthetic corpus specs:" in line for line in echoed)
    assert any("Suggested synthetic scenarios:" in line for line in echoed)

    with patch("polylogue.cli.schema_rendering_results.emit_success") as emit_success:
        render_schema_list_result(
            provider=None,
            result=SchemaListResult(provider=None, providers=[snapshot]),
            json_output=True,
        )

    emit_success.assert_called_once_with(
        {
            "providers": [
                {
                    "provider": "chatgpt",
                    "versions": ["v1", "v2"],
                    "package_count": 1,
                    "default_version": "v1",
                    "latest_version": "v2",
                    "cluster_count": 1,
                    "corpus_spec_count": 2,
                    "corpus_scenario_count": 1,
                }
            ]
        }
    )

    with patch("click.echo") as echo:
        render_schema_list_result(
            provider=None,
            result=SchemaListResult(provider=None, providers=[]),
            json_output=False,
        )

    echo.assert_called_once_with("No schemas found.")


def test_render_schema_compare_promote_and_audit_result_variants() -> None:
    compare_result = SchemaCompareResult(
        diff=SchemaDiff(
            provider="chatgpt",
            version_a="v1",
            version_b="v2",
            changed_properties=["messages"],
        )
    )
    promote_result = SchemaPromoteResult(
        provider="chatgpt",
        cluster_id="cluster-a",
        package_version="v2",
        package=_package(version="v2"),
        schema={"type": "object"},
        versions=["v1", "v2"],
    )
    audit_report = AuditReport(
        provider="chatgpt",
        checks=[AuditCheck(name="schema_bundle", status=OutcomeStatus.OK, summary="bundle OK", provider="chatgpt")],
    )

    with patch("polylogue.cli.schema_rendering_results.emit_success") as emit_success:
        render_schema_compare_result(result=compare_result, json_output=True, md_output=False)
        render_schema_promote_result(result=promote_result, json_output=True)
        render_schema_audit_result(report=audit_report, json_output=True)

    assert emit_success.call_args_list[0].args[0]["provider"] == "chatgpt"
    assert emit_success.call_args_list[1].args[0]["package_version"] == "v2"
    assert emit_success.call_args_list[2].args[0]["summary"] == {"passed": 1, "warned": 0, "failed": 0}

    with patch("click.echo") as echo:
        render_schema_compare_result(result=compare_result, json_output=False, md_output=True)
        render_schema_compare_result(result=compare_result, json_output=False, md_output=False)
        render_schema_promote_result(result=promote_result, json_output=False)
        render_schema_audit_result(report=audit_report, json_output=False)

    echoed = [call.args[0] for call in echo.call_args_list if call.args]
    assert any(line.startswith("# Schema Diff: chatgpt") for line in echoed)
    assert any(line.startswith("Schema diff: chatgpt v1 -> v2") for line in echoed)
    assert "Promoted cluster cluster-a -> package v2" in echoed
    assert "Schema package registered for chatgpt as v2" in echoed
    assert "Available versions: v1, v2" in echoed
    assert any(line.startswith("Schema Audit (chatgpt): 1 pass, 0 warn, 0 fail") for line in echoed)
