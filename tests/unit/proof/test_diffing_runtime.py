# mypy: disable-error-code="no-untyped-def,arg-type,call-arg,attr-defined"

from __future__ import annotations

import json
from subprocess import CompletedProcess
from unittest.mock import patch

from polylogue.proof.catalog import build_verification_catalog
from polylogue.proof.diffing import (
    AffectedObligationReport,
    ChangeSubject,
    ObligationDiff,
    RecommendedCheck,
    _checks_for_change,
    _classify_path,
    _normalize_path,
    _obligation_reason,
    _operations_by_code_path,
    _path_for_code_ref,
    _provider_for_parser_path,
    _provider_for_schema_path,
    _render_checks,
    _subject_ids_for_path,
    _surface_names_for_path,
    build_affected_obligation_report,
    changed_paths_between_refs,
    diff_obligation_ids,
    obligation_ids_for_ref,
    render_affected_obligations,
    route_affected_obligations,
)
from polylogue.proof.models import SourceSpan, SubjectRef


def test_changed_paths_between_refs_normalizes_and_sorts_subprocess_output() -> None:
    with patch("polylogue.proof.diffing.subprocess.run") as run:
        run.return_value = CompletedProcess(
            args=("git",),
            returncode=0,
            stdout="b.py\n\n./a.py\n",
        )

        paths = changed_paths_between_refs("master", "HEAD")

    assert paths == ("./a.py", "b.py")


def test_build_affected_obligation_report_handles_unclassified_and_suppressed_changes() -> None:
    report = build_affected_obligation_report(
        ("tests/unit/example.py", " ./notes.txt "), catalog=build_verification_catalog()
    )

    assert report.changed_paths == ("notes.txt", "tests/unit/example.py")
    assert report.obligation_diff.suppressed == ("test:tests/unit/example.py", "unknown:notes.txt")
    assert report.pr_gates[0].rendered_command == "devtools verify --quick"
    assert report.deployment_gates[0].rendered_command == "devtools build-package"


def test_route_affected_obligations_expands_proof_catalog_changes_to_all_subjects() -> None:
    catalog = build_verification_catalog()
    change = ChangeSubject(
        id="proof_catalog:polylogue/proof/diffing.py",
        path="polylogue/proof/diffing.py",
        kind="proof_catalog",
        reason="proof changed",
    )

    affected = route_affected_obligations((change,), catalog=catalog)

    assert affected
    assert all(item.change_subject_ids == (change.id,) for item in affected[:3])


def test_diff_obligation_ids_without_base_marks_every_affected_id_as_stale() -> None:
    diff = diff_obligation_ids(base_ids=(), head_ids=("a", "b"), affected_ids=("b", "a"))

    assert diff.new == ()
    assert diff.dropped == ()
    assert diff.stable_affected == ("a", "b")


def test_obligation_ids_for_ref_uses_current_catalog_for_head_aliases() -> None:
    catalog = build_verification_catalog()

    ids = obligation_ids_for_ref("HEAD")

    assert ids == tuple(obligation.id for obligation in catalog.obligations)


def test_obligation_ids_for_ref_reads_json_payload_from_detached_worktree() -> None:
    payload = json.dumps({"obligations": [{"id": "b"}, {"id": "a"}, {"id": 3}, "skip"]})
    responses = [
        CompletedProcess(args=("git",), returncode=0, stdout="", stderr=""),
        CompletedProcess(args=("python",), returncode=0, stdout=payload, stderr=""),
        CompletedProcess(args=("git",), returncode=0, stdout="", stderr=""),
    ]

    with patch("polylogue.proof.diffing.subprocess.run", side_effect=responses) as run:
        ids = obligation_ids_for_ref("origin/master")

    assert ids == ("a", "b")
    assert run.call_args_list[0].args[0][:3] == ("git", "worktree", "add")
    assert run.call_args_list[-1].args[0][:3] == ("git", "worktree", "remove")


def test_render_affected_obligations_handles_empty_sets_and_truncates_large_lists() -> None:
    large = tuple(
        type(
            "Obligation",
            (),
            {"obligation_id": f"obligation-{index}", "reasons": (f"reason-{index}",)},
        )()
        for index in range(32)
    )
    report = AffectedObligationReport(
        base_ref="base",
        head_ref="head",
        changed_paths=("polylogue/proof/diffing.py",),
        change_subjects=(
            ChangeSubject(
                id="command:polylogue/cli/query.py",
                path="polylogue/cli/query.py",
                kind="command",
                reason="command changed",
                operation_names=("query.search",),
                surface_names=("cli-reference",),
            ),
        ),
        affected_obligations=large,
        obligation_diff=ObligationDiff(),
        inner_loop_checks=(),
        pr_gates=(RecommendedCheck(command=("devtools", "verify"), scope="pr_gate", reason="gate"),),
        deployment_gates=(),
    )

    rendered = render_affected_obligations(report)

    assert "operations=query.search; surfaces=cli-reference" in rendered
    assert "- ... 2 more" in rendered
    assert "Deployment Gates:" in rendered
    assert "- none" in rendered


def test_internal_diffing_helpers_cover_classification_and_reason_paths() -> None:
    catalog = build_verification_catalog()
    subject = SubjectRef(kind="workflow.claim", id="workflow.generated", attrs={"claim_family": "generated-surfaces"})
    subject_ids = _subject_ids_for_path(
        "docs/verification-catalog.md",
        kind="generated_surface",
        subjects_by_path={},
        subjects=(subject,),
    )

    assert subject_ids == ("workflow.generated",)
    assert _classify_path("devtools/render_cli_reference.py") == "generated_surface"
    assert _classify_path("polylogue/proof/diffing.py") == "proof_catalog"
    assert _classify_path("notes.txt") == "unknown"
    assert _surface_names_for_path("devtools/render_cli_reference.py")
    assert _provider_for_parser_path("polylogue/sources/parsers/claude.py") == "claude-code"
    assert _provider_for_schema_path("polylogue/schemas/providers/codex/messages.json") == "codex"
    assert _provider_for_schema_path("polylogue/schemas/messages.json") is None
    assert _normalize_path(".\\polylogue\\cli\\query.py") == "polylogue/cli/query.py"
    assert _path_for_code_ref("polylogue.cli.commands.run.run_command") == "polylogue/cli/commands/run.py"
    assert _operations_by_code_path(
        (
            type("Spec", (), {"name": "run", "code_refs": ("polylogue.cli.commands.run.run_command",)})(),
            type("Spec", (), {"name": "missing", "code_refs": ("no.such.module",)})(),
        )
    )["polylogue/cli/commands/run.py"] == ("run",)
    assert _checks_for_change("workflow", path="AGENTS.md")[0].rendered_command == (
        "pytest tests/unit/devtools/test_command_catalog.py"
    )
    assert _render_checks(()) == ["- none"]
    assert _obligation_reason(
        ChangeSubject(id="proof", path="polylogue/proof/diffing.py", kind="proof_catalog", reason="proof"),
        catalog.subjects[0],
    ).endswith("compiler or runner vocabulary")
    assert (
        _obligation_reason(
            ChangeSubject(id="command", path="polylogue/cli/click_app.py", kind="command", reason="command"),
            SubjectRef(
                kind="cli.command", id="polylogue", attrs={}, source_span=SourceSpan("polylogue/cli/click_app.py")
            ),
        )
        == "polylogue/cli/click_app.py is the source span for subject polylogue"
    )
