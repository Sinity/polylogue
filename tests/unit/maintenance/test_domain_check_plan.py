from __future__ import annotations

import pytest

from polylogue.maintenance.domain_check_plan import (
    DomainCheckDeclaration,
    DomainCheckPlanError,
    compile_domain_check_plan,
)


def _declaration(**changes: object) -> DomainCheckDeclaration:
    values: dict[str, object] = {
        "identity": "lineage",
        "version": 1,
        "owner_operation": "candidate-build",
        "phase": "candidate",
        "denominator": "index.db.session_links",
        "target_bindings": ("reindex-index-candidate",),
        "production_route": "index graph materialization",
        "oracle_reference": "test_lineage_route",
    }
    values.update(changes)
    return DomainCheckDeclaration(**values)  # type: ignore[arg-type]


def test_compilation_is_order_independent_and_digest_binds_members() -> None:
    first = _declaration(identity="z-check")
    second = _declaration(identity="a-check")
    left = compile_domain_check_plan([first, second], phase="candidate")
    right = compile_domain_check_plan([second, first], phase="candidate")
    assert left.to_dict() == right.to_dict()
    assert left.member_identities == ("a-check@1", "z-check@1")
    assert len(left.digest) == 64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("owner_operation", "missing-operation", "unknown owning operation"),
        ("denominator", "", "denominator"),
        ("target_bindings", (), "target"),
        ("production_route", "", "production owner"),
        ("oracle_reference", "", "production owner"),
    ],
)
def test_weak_declarations_fail_closed(field: str, value: object, message: str) -> None:
    with pytest.raises(DomainCheckPlanError, match=message):
        compile_domain_check_plan([_declaration(**{field: value})], phase="candidate")


def test_duplicate_identity_version_and_phase_mismatch_fail_closed() -> None:
    with pytest.raises(DomainCheckPlanError, match="duplicate"):
        compile_domain_check_plan([_declaration(), _declaration()], phase="candidate")
    with pytest.raises(DomainCheckPlanError, match="phase/target"):
        compile_domain_check_plan([_declaration(phase="source")], phase="candidate")


def test_not_applicable_candidate_declaration_is_not_a_plan_member() -> None:
    plan = compile_domain_check_plan([_declaration(candidate_applicability="not-applicable")], phase="candidate")
    assert plan.rows == ()
