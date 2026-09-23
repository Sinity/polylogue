"""Canonical six-tier string-vocabulary inventory contract."""

from __future__ import annotations

from polylogue.storage.sqlite.archive_tiers.vocabulary_inventory import build_inventory


def test_inventory_covers_all_string_membership_checks_and_declares_exclusions() -> None:
    inventory = build_inventory()

    # Runtime canonical DDL, not a remembered source-text count, is the
    # denominator: numeric and NOT IN checks are intentionally excluded.
    assert inventory.denominator == 95
    assert inventory.durable_exclusions == 41
    assert inventory.unknown_ownership == 0
    assert {item.tier.value for item in inventory.checks} == {
        "source",
        "index",
        "embeddings",
        "user",
        "audit",
        "ops",
    }


def test_derived_rows_have_a_domain_owner_or_explicit_storage_local_reason() -> None:
    inventory = build_inventory()
    derived = [item for item in inventory.checks if item.disposition != "DURABLE_WRITE_BOUNDARY"]

    assert derived
    assert all(item.reason for item in derived)
    assert all(item.disposition in {"GENERATED_FROM_OWNER", "STORAGE_LOCAL"} for item in derived)
    assert any(item.disposition == "GENERATED_FROM_OWNER" for item in derived)
    assert any(item.disposition == "STORAGE_LOCAL" for item in derived)


def test_durable_rows_are_excluded_even_when_a_domain_owner_matches() -> None:
    inventory = build_inventory()
    durable = [item for item in inventory.checks if item.tier.value in {"source", "user", "audit"}]

    assert durable
    assert all(item.disposition == "DURABLE_WRITE_BOUNDARY" for item in durable)
    assert all("write boundary" in item.reason for item in durable)


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
