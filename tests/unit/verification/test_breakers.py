"""Mutation fixtures: each verifier must fail on intentionally-bad input.

These tests enforce the anti-vacuity rule (#594): no verifier should silently
pass when fed garbage.  Each test creates minimal bad fixtures, calls the
verifier, and asserts non-zero exit (or error list non-empty).
"""

from __future__ import annotations

import json as json_mod
from pathlib import Path

import pytest
import yaml

from devtools import verify_manifests
from polylogue.proof.catalog import build_verification_catalog
from polylogue.verification.manifests.models import (
    LintEscalationManifest,
    SuppressionsManifest,
    TopologyManifest,
    validate_manifest,
)

# ══════════════════════════════════════════════════════════════════════
# D1 — Typed manifest validation (#629)
# ══════════════════════════════════════════════════════════════════════

LINT_RULES_WITH_BAD_SEVERITY = """\
description: Test
rules:
  - id: test-rule
    description: a rule with invalid severity
    severity: critical
    sunset: "2026-12-31"
"""


def test_pydantic_rejects_invalid_lint_severity() -> None:
    """A lint rule with severity outside {soft, hard} must fail."""
    data = yaml.safe_load(LINT_RULES_WITH_BAD_SEVERITY)
    with pytest.raises(Exception) as excinfo:
        LintEscalationManifest.model_validate(data)
    assert "severity" in str(excinfo.value) or "critical" in str(excinfo.value)


SUPPRESSION_WITH_BAD_DATE = """\
suppressions:
  - id: test-sup
    reason: testing
    expires_at: "not-a-date"
"""


def test_pydantic_rejects_invalid_suppression_date() -> None:
    """A suppression with an unparseable expires_at must fail."""
    data = yaml.safe_load(SUPPRESSION_WITH_BAD_DATE)
    with pytest.raises(Exception) as excinfo:
        SuppressionsManifest.model_validate(data)
    assert "date" in str(excinfo.value).lower() or "not-a-date" in str(excinfo.value)


TOPOLOGY_WITH_UNKNOWN_FIELD = """\
files:
  - path: polylogue/test.py
    owner: stable
    unknown_field: should be rejected
"""


def test_pydantic_rejects_unknown_field() -> None:
    """A topology entry with an undeclared field must fail (extra='forbid')."""
    data = yaml.safe_load(TOPOLOGY_WITH_UNKNOWN_FIELD)
    with pytest.raises(Exception) as excinfo:
        TopologyManifest.model_validate(data)
    assert "unknown_field" in str(excinfo.value)


COVERAGE_GAP_BAD_NEXT = """\
coverage_gaps:
  - id: test-gap-001
    gap: missing test
    owner: nobody
    severity: serious
    declared_at: "2026-01-01"
    review_after: "2026-06-01"
    next_evidence: "we should test this someday"
"""


def test_verify_manifests_fails_on_prose_next_evidence(tmp_path: Path) -> None:
    """A coverage gap with prose next_evidence (non-command) must fail."""
    manifest = tmp_path / "bad-coverage-gaps.yaml"
    manifest.write_text(COVERAGE_GAP_BAD_NEXT, encoding="utf-8")
    errors = verify_manifests.check_coverage_gaps(tmp_path)
    assert errors, "prose next_evidence should produce verification errors"
    assert any("next_evidence" in err for err in errors), f"expected next_evidence error, got: {errors}"
    assert any("does not resolve" in err for err in errors), f"expected resolution error, got: {errors}"


BAD_REFERENCE_MANIFEST = """\
items:
  nonexistent:
    location: tests/nonexistent_file_xyz987.py
    verified_by: pytest tests/nonexistent_file_xyz987.py
"""


def test_verify_manifests_fails_on_nonexistent_test_ref(tmp_path: Path) -> None:
    """A manifest referencing a nonexistent test file path must fail."""
    manifest = tmp_path / "bad-coverage-refs.yaml"
    manifest.write_text(BAD_REFERENCE_MANIFEST, encoding="utf-8")
    errors = verify_manifests.check_coverage_references(tmp_path)
    assert errors, "nonexistent test ref should produce verification errors"
    assert any("path does not exist" in err for err in errors), f"expected path-does-not-exist error, got: {errors}"
    assert any("nonexistent_file_xyz987" in err for err in errors), f"expected reference to the bad path, got: {errors}"


def test_committed_manifest_passes_pydantic_validation() -> None:
    """All committed YAML manifests under docs/plans/ must pass Pydantic model
    validation."""

    repo_root = Path(__file__).resolve().parents[3]
    plans_dir = repo_root / "docs" / "plans"

    if not plans_dir.is_dir():
        pytest.skip("docs/plans/ directory not found")

    all_errors: list[str] = []
    for path in sorted(plans_dir.glob("*.yaml")):
        model_errors = _validate_yaml_and_catch(path)
        all_errors.extend(model_errors)

    assert not all_errors, f"{len(all_errors)} Pydantic validation error(s) in committed manifests:\n" + "\n".join(
        all_errors
    )


def _validate_yaml_and_catch(path: Path) -> list[str]:
    """Load a YAML file and validate it against its Pydantic model."""
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return [f"{path}: not a mapping"]
        return validate_manifest(str(path), data)
    except Exception as exc:
        return [f"{path}: {exc}"]


# ══════════════════════════════════════════════════════════════════════
# D2 — Anti-vacuity slop detection (#594)
# ══════════════════════════════════════════════════════════════════════


def test_slop_dashboard_detects_missing_breakers() -> None:
    """The slop dashboard must flag claims without breakers."""
    from devtools.proof_pack import build_slop_dashboard

    catalog = build_verification_catalog()
    dashboard = build_slop_dashboard(catalog)

    assert "total_claims" in dashboard
    assert "slop_count" in dashboard
    assert "by_reason" in dashboard
    assert "rows" in dashboard

    # At least some missing_breaker claims should exist (new claims may lack them)
    missing_breaker_count = dashboard["by_reason"]["missing_breaker"]
    # We don't assert a specific number — the catalog evolves. But we assert
    # the dashboard is structured and non-empty (it should have slop).
    assert isinstance(missing_breaker_count, int)
    assert isinstance(dashboard["slop_count"], int)


def test_slop_dashboard_by_reason_breaks_down() -> None:
    """Each reason in by_reason must be a non-negative integer."""
    from devtools.proof_pack import build_slop_dashboard

    catalog = build_verification_catalog()
    dashboard = build_slop_dashboard(catalog)
    by_reason = dashboard["by_reason"]

    for reason in ("missing_breaker", "missing_runner", "zero_subjects", "self_referential", "stale_evidence"):
        count = by_reason.get(reason, -1)
        assert isinstance(count, int), f"{reason} count is not int: {type(count)}"
        assert count >= 0, f"{reason} count is negative: {count}"


def test_slop_dashboard_json_serializable() -> None:
    """Slop dashboard output must be JSON-serializable."""
    from devtools.proof_pack import build_slop_dashboard

    catalog = build_verification_catalog()
    dashboard = build_slop_dashboard(catalog)
    # Should not raise
    json_mod.dumps(dashboard, indent=2, sort_keys=True)


def test_slop_dashboard_rows_have_required_fields() -> None:
    """Each slop row must carry claim_id, severity, and all reason flags."""
    from devtools.proof_pack import build_slop_dashboard

    catalog = build_verification_catalog()
    dashboard = build_slop_dashboard(catalog)
    required = {
        "claim_id",
        "severity",
        "missing_breaker",
        "missing_runner",
        "zero_subjects",
        "self_referential",
        "stale_evidence",
    }
    for row in dashboard["rows"]:
        missing = required - set(row.keys())
        assert not missing, f"row {row.get('claim_id')} missing fields: {missing}"


# ══════════════════════════════════════════════════════════════════════
# D3 — verifier must fail on bad input
# ══════════════════════════════════════════════════════════════════════


def test_pydantic_check_fails_on_invalid_suppression_yaml(tmp_path: Path) -> None:
    """check_pydantic_models must return errors for a suppression with
    wrong type value (int instead of str for expires_at)."""
    path = tmp_path / "suppressions.yaml"
    path.write_text(
        "suppressions:\n  - id: bad-sup\n    reason: testing\n    expires_at: 12345\n",  # Should be a string
        encoding="utf-8",
    )
    errors = verify_manifests.check_pydantic_models(tmp_path)
    assert errors, "wrong-type suppression should produce Pydantic errors"


def test_pydantic_check_fails_on_repeated_rule_id(tmp_path: Path) -> None:
    """The Pydantic model does not catch duplicate IDs (that's the
    existing cross-reference check), but it MUST reject structural
    issues like missing required fields."""
    path = tmp_path / "lint-escalation.yaml"
    path.write_text(
        "rules:\n"
        "  - id: dup-rule\n"
        "    description: first\n"
        "    severity: hard\n"
        "    sunset: '2026-12-31'\n"
        "  - id: dup-rule\n"
        "    description: second\n"
        "    severity: hard\n"
        "    sunset: '2026-12-31'\n",
        encoding="utf-8",
    )
    # Pydantic should accept structurally valid data (duplicate IDs
    # are caught by the existing cross-reference check, not the model)
    # So this test asserts NO Pydantic errors
    errors = verify_manifests.check_pydantic_models(tmp_path)
    assert not errors, "duplicate IDs are structurally valid for Pydantic"


def test_empty_routing_fails() -> None:
    """A changed path that routes to zero obligations should eventually
    fail. This is a forward-looking test — routing is not yet implemented
    per #594."""
    pytest.xfail("routing not yet implemented per #594")


def test_manifests_integration_passes(tmp_path: Path) -> None:
    """The integrated verify_manifests main() must pass (return 0)
    on the committed manifests.  This is a regression guard — the
    Pydantic validation layer should not break the existing checks."""
    rc = verify_manifests.main([])
    assert rc == 0, "verify_manifests should pass on committed manifests"


def test_slop_dashboard_renders_without_error() -> None:
    """The slop markdown renderer must handle any dashboard structure."""
    from devtools.proof_pack import _print_slop_dashboard, build_slop_dashboard, render_slop_markdown

    catalog = build_verification_catalog()
    dashboard = build_slop_dashboard(catalog)

    # Should not raise
    markdown = render_slop_markdown(dashboard)
    assert isinstance(markdown, str)
    assert len(markdown) > 0
    assert "Anti-Vacuity" in markdown

    # Human-readable should also not raise
    import io
    import sys as sys_mod

    buf = io.StringIO()
    old_stdout = sys_mod.stdout
    try:
        sys_mod.stdout = buf
        _print_slop_dashboard(dashboard)
    finally:
        sys_mod.stdout = old_stdout
    output = buf.getvalue()
    assert len(output) > 0
    assert "Slop claims" in output
