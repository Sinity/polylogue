from __future__ import annotations

from polylogue.proof.catalog import build_verification_catalog
from polylogue.proof.diffing import (
    build_affected_obligation_report,
    classify_changed_paths,
    diff_obligation_ids,
    render_affected_obligations,
    route_affected_obligations,
)


def test_parser_change_routes_to_provider_capability_obligations() -> None:
    catalog = build_verification_catalog()
    changes = classify_changed_paths(("polylogue/sources/parsers/codex.py",), catalog=catalog)

    assert changes[0].kind == "parser"
    assert "provider.capability.codex" in changes[0].subject_ids

    affected = route_affected_obligations(changes, catalog=catalog)

    assert {item.claim_id for item in affected} == {
        "provider.capability.identity_bridge",
        "provider.capability.partial_coverage_declared",
    }
    assert all(item.subject_id == "provider.capability.codex" for item in affected)


def test_schema_change_routes_to_schema_annotation_obligations() -> None:
    catalog = build_verification_catalog()
    schema_subject = next(subject for subject in catalog.subjects if subject.kind == "schema.annotation")
    assert schema_subject.source_span is not None

    changes = classify_changed_paths((schema_subject.source_span.path,), catalog=catalog)
    affected = route_affected_obligations(changes, catalog=catalog)

    assert changes[0].kind == "schema.annotation"
    assert schema_subject.id in changes[0].subject_ids
    assert {item.subject_id for item in affected} >= {schema_subject.id}
    assert all(item.claim_id.startswith("schema.") for item in affected)


def test_command_change_routes_to_cli_obligations() -> None:
    catalog = build_verification_catalog()

    changes = classify_changed_paths(("polylogue/cli/command_inventory.py",), catalog=catalog)
    affected = route_affected_obligations(changes, catalog=catalog)

    assert changes[0].kind == "command"
    assert {"cli.command.help", "cli.command.no_traceback", "cli.command.plain_mode"}.issubset(
        {item.claim_id for item in affected}
    )
    assert "cli.command.json_envelope" in {item.claim_id for item in affected}


def test_generated_surface_change_routes_to_workflow_claim() -> None:
    catalog = build_verification_catalog()

    report = build_affected_obligation_report(
        ("docs/verification-catalog.md",),
        catalog=catalog,
        base_obligation_ids=(obligation.id for obligation in catalog.obligations),
        head_obligation_ids=(obligation.id for obligation in catalog.obligations),
    )

    assert report.change_subjects[0].kind == "generated_surface"
    assert report.change_subjects[0].surface_names == ("verification-catalog",)
    assert {item.claim_id for item in report.affected_obligations} == {"assurance.coverage.item_declared"}
    assert [check.rendered_command for check in report.inner_loop_checks] == ["devtools render-all --check"]
    assert [check.rendered_command for check in report.pr_gates] == ["devtools verify --quick", "devtools verify"]
    assert [check.rendered_command for check in report.deployment_gates] == [
        "devtools build-package",
        "nix flake check",
    ]


def test_architecture_change_routes_to_structural_obligation() -> None:
    catalog = build_verification_catalog()

    changes = classify_changed_paths(("docs/plans/layering.yaml",), catalog=catalog)
    affected = route_affected_obligations(changes, catalog=catalog)

    assert changes[0].kind == "architecture"
    assert "architecture.layering.import_rules" in changes[0].subject_ids
    assert "architecture.manifest.consistency" in changes[0].subject_ids
    assert "architecture.layering.import_rules_enforced" in {item.claim_id for item in affected}


def test_coverage_manifest_change_routes_known_gap_obligations() -> None:
    catalog = build_verification_catalog()

    changes = classify_changed_paths(("docs/plans/docs-media-coverage.yaml",), catalog=catalog)
    affected = route_affected_obligations(changes, catalog=catalog)

    assert changes[0].kind == "coverage_manifest"
    assert "architecture.manifest.consistency" in changes[0].subject_ids
    assert any(
        subject_id.startswith("assurance.coverage_manifest.docs-media-coverage")
        for subject_id in changes[0].subject_ids
    )
    assert {"assurance.coverage.manifest_structured", "assurance.coverage.gap_has_closure_path"}.issubset(
        {item.claim_id for item in affected}
    )


def test_docs_media_surface_change_routes_through_manifest_subject() -> None:
    catalog = build_verification_catalog()

    changes = classify_changed_paths(("README.md",), catalog=catalog)
    affected = route_affected_obligations(changes, catalog=catalog)

    assert "assurance.coverage_item.docs-media-coverage.surfaces.readme" in changes[0].subject_ids
    assert "assurance.coverage.item_declared" in {item.claim_id for item in affected}


def test_schema_change_routes_to_roundtrip_obligation() -> None:
    catalog = build_verification_catalog()
    schema_subject = next(subject for subject in catalog.subjects if subject.kind == "schema.annotation")
    assert schema_subject.source_span is not None

    changes = classify_changed_paths((schema_subject.source_span.path,), catalog=catalog)
    affected = route_affected_obligations(changes, catalog=catalog)

    assert "schema.roundtrip.provider_packages" in changes[0].subject_ids
    assert "schema.roundtrip.inference_validation" in {item.claim_id for item in affected}


def test_obligation_diff_buckets_new_dropped_stale_and_suppressed() -> None:
    diff = diff_obligation_ids(
        base_ids=("stable", "dropped"),
        head_ids=("stable", "new"),
        affected_ids=("stable", "new"),
        suppressed=("test:tests/unit/example.py",),
    )

    assert diff.new == ("new",)
    assert diff.dropped == ("dropped",)
    assert diff.stale_evidence == ("stable",)
    assert diff.now_failing == ()
    assert diff.now_passing == ()
    assert diff.suppressed == ("test:tests/unit/example.py",)


def test_human_report_explains_selected_obligations() -> None:
    catalog = build_verification_catalog()
    report = build_affected_obligation_report(
        ("polylogue/sources/parsers/codex.py",),
        catalog=catalog,
        base_obligation_ids=(),
        head_obligation_ids=(obligation.id for obligation in catalog.obligations),
    )

    rendered = render_affected_obligations(report)

    assert "Affected Obligations" in rendered
    assert "provider parser semantics can change normalized archive facts" in rendered
    assert "provider.capability.codex" in rendered
    assert "Recommended Inner-Loop Checks" in rendered
