"""Automated schema quality audit checks."""

from __future__ import annotations

from polylogue.lib.outcomes import OutcomeCheck as CheckResult
from polylogue.schemas.audit_checks import (
    check_annotation_coverage,
    check_cross_provider_consistency,
    check_privacy_guards,
    check_semantic_roles,
)
from polylogue.schemas.audit_models import AuditReport
from polylogue.schemas.audit_workflow import audit_all_providers, audit_provider

__all__ = [
    "AuditReport",
    "CheckResult",
    "audit_all_providers",
    "audit_provider",
    "check_annotation_coverage",
    "check_cross_provider_consistency",
    "check_privacy_guards",
    "check_semantic_roles",
]
