from __future__ import annotations

import json

import pytest

from devtools import verify_durable_enum_checks
from polylogue.core.enums import Origin
from polylogue.storage.sqlite.archive_tiers.common import check, nullable_check


def test_reintroduced_enum_check_is_rejected() -> None:
    """Anti-vacuity: pasting `CHECK({check("origin", Origin)})` back into a
    durable table must be refused. The offending DDL is constructed here, so
    the test never depends on corrupting the real tier DDL."""
    offending_ddl = f"""
    CREATE TABLE IF NOT EXISTS raw_sessions (
        raw_id   TEXT PRIMARY KEY,
        origin   TEXT NOT NULL CHECK({check("origin", Origin)})
    ) STRICT;
    """
    violations = verify_durable_enum_checks.scan_ddl_for_enum_membership_checks(offending_ddl, tier="source")
    assert [(v.column, v.enum_name) for v in violations] == [("origin", "Origin")]


def test_reintroduced_nullable_enum_check_is_rejected() -> None:
    """`nullable_check()` renders the same membership list plus an IS NULL arm;
    the nullable spelling must not be an escape hatch."""
    offending_ddl = f"""
    CREATE TABLE IF NOT EXISTS raw_sessions (
        origin TEXT CHECK({nullable_check("origin", Origin)})
    ) STRICT;
    """
    violations = verify_durable_enum_checks.scan_ddl_for_enum_membership_checks(offending_ddl, tier="source")
    assert [v.enum_name for v in violations] == ["Origin"]


def test_handwritten_structural_literal_list_stays_legal() -> None:
    """A closed vocabulary an author typed is not an enum mirror, so it must
    pass -- these lists are deliberately kept in durable DDL."""
    fixture_ddl = """
    CREATE TABLE IF NOT EXISTS widgets (
        revision_kind TEXT NOT NULL CHECK(revision_kind IN ('full', 'append', 'unknown')),
        durability    TEXT NOT NULL CHECK(durability IN ('durable', 'derived', 'disposable', 'external'))
    ) STRICT;
    """
    assert verify_durable_enum_checks.scan_ddl_for_enum_membership_checks(fixture_ddl, tier="fixture") == []


def test_integer_and_non_literal_membership_lists_are_ignored() -> None:
    """`retryable IN (0, 1)` and a subquery carry no string literals and are
    not vocabulary CHECKs."""
    fixture_ddl = """
    CREATE TABLE IF NOT EXISTS widgets (
        retryable INTEGER CHECK(retryable IN (0, 1))
    ) STRICT;
    """
    assert verify_durable_enum_checks.scan_ddl_for_enum_membership_checks(fixture_ddl, tier="fixture") == []


def test_grandfathered_waiver_is_keyed_on_its_exact_member_set() -> None:
    """A waived column that changes vocabulary loses its waiver, so the column
    is re-examined instead of silently re-widened."""
    members = sorted(member.value for member in Origin)
    rendered = ", ".join(f"'{value}'" for value in members)
    waived = ("source", "previous_revision_authority", frozenset({"asserted", "byte_proven", "quarantined"}))
    assert waived in verify_durable_enum_checks.GRANDFATHERED
    # The same column carrying a different enum's value set is still refused.
    ddl = f"x TEXT CHECK(previous_revision_authority IN ({rendered}))"
    violations = verify_durable_enum_checks.scan_ddl_for_enum_membership_checks(ddl, tier="source")
    assert [v.enum_name for v in violations] == ["Origin"]


def test_waiver_does_not_leak_across_tiers() -> None:
    """A source-tier waiver must not excuse the same list in user or audit."""
    ddl = "previous_revision_authority TEXT CHECK(previous_revision_authority IN ('asserted', 'byte_proven', 'quarantined'))"
    assert verify_durable_enum_checks.scan_ddl_for_enum_membership_checks(ddl, tier="source") == []
    assert verify_durable_enum_checks.scan_ddl_for_enum_membership_checks(ddl, tier="audit") != []


def test_durable_tier_ddl_carries_no_enum_checks(capsys: pytest.CaptureFixture[str]) -> None:
    """The real source/user/audit DDL must currently pass: PR #4495 moved every
    enum membership CHECK to write-boundary validation."""
    assert verify_durable_enum_checks.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["violations"] == []


def test_main_reports_violation_and_nonzero_exit(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        verify_durable_enum_checks,
        "_collect_durable_tier_violations",
        lambda: [
            verify_durable_enum_checks.EnumCheckViolation(
                tier="source", column="origin", enum_name="Origin", members=("a", "b")
            )
        ],
    )
    assert verify_durable_enum_checks.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["violations"] == [{"tier": "source", "column": "origin", "enum": "Origin", "members": ["a", "b"]}]
