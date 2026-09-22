"""The schema audit must ask the declaration before calling a package missing.

``polylogue.core.schema_subjects`` decides which subjects owe a committed
package.  The audit walked ``observation.PROVIDERS`` instead and reported a hard
error for every subject without a package directory, so an adjudicated absence
(``requires_package=False`` with a recorded reason) failed the gate.

Anti-vacuity: delete the declaration lookup from
``polylogue.schemas.audit.workflow.audit_provider`` and
``test_declared_absence_is_not_an_audit_error`` goes red with
``schema_exists`` at ``ERROR``.  The two direction tests go red the other way:
if the lookup ever degrades into "a missing package is always acceptable", both
an undeclared subject and a subject that genuinely owes a package would stop
failing.
"""

from __future__ import annotations

import dataclasses

import pytest

from polylogue.core.outcomes import OutcomeStatus
from polylogue.core.schema_subjects import SCHEMA_SUBJECT_BY_TOKEN
from polylogue.schemas.audit import workflow
from polylogue.schemas.audit.models import AuditCheck, AuditReport
from polylogue.schemas.audit.walkers import _load_committed_schema


def _existence_checks(report: AuditReport) -> list[AuditCheck]:
    return [check for check in report.checks if isinstance(check, AuditCheck) and check.name == "schema_exists"]


def _schema_exists(report: AuditReport) -> AuditCheck:
    checks = _existence_checks(report)
    assert len(checks) == 1, "audit_provider must record exactly one schema_exists outcome"
    return checks[0]


def test_grok_is_declared_without_a_package() -> None:
    """The premise the other tests rest on, asserted rather than assumed."""
    subject = SCHEMA_SUBJECT_BY_TOKEN["grok"]
    assert subject.requires_package is False
    assert subject.package_not_required_reason
    assert _load_committed_schema("grok") is None


def test_declared_absence_is_not_an_audit_error() -> None:
    report = workflow.audit_provider("grok")
    check = _schema_exists(report)

    assert check.status is OutcomeStatus.SKIP
    assert report.all_passed
    assert SCHEMA_SUBJECT_BY_TOKEN["grok"].package_not_required_reason in check.details


def test_undeclared_subject_without_a_package_fails() -> None:
    report = workflow.audit_provider("not-a-declared-subject")
    check = _schema_exists(report)

    assert check.status is OutcomeStatus.ERROR
    assert not report.all_passed


def test_required_package_absent_still_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A subject that genuinely owes a package must keep failing the audit."""
    subject = "declared-and-required"
    spec = dataclasses.replace(
        SCHEMA_SUBJECT_BY_TOKEN["grok"],
        token=subject,
        package_dir=subject,
        provider=subject,
        requires_package=True,
        package_not_required_reason=None,
    )
    monkeypatch.setitem(SCHEMA_SUBJECT_BY_TOKEN, subject, spec)

    report = workflow.audit_provider(subject)
    check = _schema_exists(report)

    assert check.status is OutcomeStatus.ERROR
    assert not report.all_passed


def test_full_audit_has_no_missing_package_error() -> None:
    """The gate's own route: every declared provider reconciles without an error."""
    report = workflow.audit_all_providers()

    existence = _existence_checks(report)
    assert [c.provider for c in existence if c.status is OutcomeStatus.ERROR] == []
    assert [c.provider for c in existence if c.status is OutcomeStatus.SKIP] == ["grok"]
