"""Canonical six-tier string-vocabulary inventory contract."""

from __future__ import annotations

from typing import Literal, cast

from polylogue.core import types as core_types
from polylogue.storage.sqlite.archive_tiers import vocabulary_inventory
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

build_inventory = vocabulary_inventory.build_inventory


def test_inventory_covers_all_string_membership_checks_and_declares_exclusions() -> None:
    inventory = build_inventory()

    # Runtime canonical DDL, not a remembered source-text count, is the
    # denominator: numeric and NOT IN checks are intentionally excluded.
    assert inventory.denominator == 90
    assert inventory.durable_exclusions == 38
    assert inventory.unknown_ownership == inventory.unverified_owner_bindings
    assert inventory.unknown_ownership > 0
    assert {item.tier.value for item in inventory.checks} == {
        "source",
        "index",
        "embeddings",
        "user",
        "audit",
        "ops",
    }


def test_derived_rows_have_a_domain_owner_or_remain_unknown() -> None:
    inventory = build_inventory()
    derived = [item for item in inventory.checks if item.disposition != "DURABLE_WRITE_BOUNDARY"]

    assert derived
    assert all(item.reason for item in derived)
    assert all(
        item.disposition in {"OWNER_CANDIDATE", "GENERATED_FROM_OWNER", "STORAGE_LOCAL", "UNKNOWN_OWNERSHIP"}
        for item in derived
    )
    assert any(item.disposition == "OWNER_CANDIDATE" for item in derived)
    assert all(item.disposition != "UNKNOWN_OWNERSHIP" for item in derived)


def test_durable_rows_are_excluded_even_when_a_domain_owner_matches() -> None:
    inventory = build_inventory()
    durable = [item for item in inventory.checks if item.tier.value in {"source", "user", "audit"}]

    assert durable
    assert all(item.disposition == "DURABLE_WRITE_BOUNDARY" for item in durable)
    assert all("write boundary" in item.reason for item in durable)
    assert all(item.lifecycle == "durable" for item in durable)


def test_inventory_records_lifecycle_for_every_row_and_serializes_it() -> None:
    inventory = build_inventory()

    assert {item.lifecycle for item in inventory.checks} == {"durable", "derived"}
    assert all(
        item.lifecycle == ("durable" if item.tier.value in {"source", "user", "audit"} else "derived")
        for item in inventory.checks
    )
    payload = inventory.to_payload()
    assert all("lifecycle" in row for row in cast("list[dict[str, object]]", payload["checks"]))


def test_equivalent_python_vocabularies_are_reported_as_one_owner_group() -> None:
    """DDL ownership is keyed by members, so aliases cannot form new lists."""
    inventory = build_inventory()

    # The frontier kind is deliberately declared once in ``types.py`` and
    # consumed by both revision tables.  The inventory's owner projection
    # remains useful if a future enum/Literal alias is introduced: it reports
    # the collapsed group rather than silently choosing two owners.
    frontier = [item for item in inventory.checks if item.column == "accepted_frontier_kind"]
    assert len(frontier) == 2
    assert {item.owner for item in frontier} == {"polylogue.storage.sqlite.archive_tiers.types.RevisionFrontierKind"}


def test_detaching_a_derived_check_from_its_owner_reports_unknown_ownership(monkeypatch) -> None:
    before = build_inventory()
    ddl = dict(vocabulary_inventory.ARCHIVE_DDL_BY_TIER)
    ddl[ArchiveTier.INDEX] = ddl[ArchiveTier.INDEX].replace(
        "identity_source IN ('native', 'content')",
        "identity_source IN ('native', 'detached')",
        1,
    )
    monkeypatch.setattr(vocabulary_inventory, "ARCHIVE_DDL_BY_TIER", ddl)

    inventory = build_inventory()
    detached = [item for item in inventory.checks if item.ref == "index.messages.identity_source"]
    assert len(detached) == 1
    assert detached[0].disposition == "UNKNOWN_OWNERSHIP"
    assert inventory.unknown_ownership == before.unknown_ownership
    assert inventory.unverified_owner_bindings == before.unverified_owner_bindings - 1
    assert (
        sum(item.disposition == "UNKNOWN_OWNERSHIP" for item in inventory.checks)
        == sum(item.disposition == "UNKNOWN_OWNERSHIP" for item in before.checks) + 1
    )


def test_omitting_a_canonical_tier_invalidates_the_inventory(monkeypatch) -> None:
    ddl = {tier: sql for tier, sql in vocabulary_inventory.ARCHIVE_DDL_BY_TIER.items() if tier != ArchiveTier.OPS}
    monkeypatch.setattr(vocabulary_inventory, "ARCHIVE_DDL_BY_TIER", ddl)

    try:
        build_inventory()
    except ValueError as exc:
        assert "ops" in str(exc)
    else:
        raise AssertionError("inventory accepted a canonical tier omission")


def test_reintroducing_a_duplicate_owner_makes_ownership_ambiguous(monkeypatch) -> None:
    before = build_inventory()
    monkeypatch.setattr(core_types, "DuplicateMessageIdentitySource", Literal["content", "native"], raising=False)

    inventory = build_inventory()

    identity = [item for item in inventory.checks if item.ref == "index.messages.identity_source"]
    assert len(identity) == 1
    assert identity[0].disposition == "UNKNOWN_OWNERSHIP"
    assert inventory.unknown_ownership == before.unknown_ownership
    assert inventory.unverified_owner_bindings == before.unverified_owner_bindings - 1
    assert (
        sum(item.disposition == "UNKNOWN_OWNERSHIP" for item in inventory.checks)
        == sum(item.disposition == "UNKNOWN_OWNERSHIP" for item in before.checks) + 1
    )
