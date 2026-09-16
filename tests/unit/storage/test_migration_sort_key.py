"""polylogue-tgq71: migration discovery orders by numeric version, not filename."""

from __future__ import annotations

from polylogue.storage.sqlite.migration_runner import migration_sort_key


def test_migration_discovery_orders_1000_after_999() -> None:
    names = ["1000_later.sql", "999_earlier.sql", "045_drop_census.sql", "README.md"]
    ordered = sorted(names, key=migration_sort_key)
    # Anti-vacuity: a lexicographic sort yields ["1000_later.sql", ...] first.
    assert sorted(names)[0] == "045_drop_census.sql" and sorted(names)[1] == "1000_later.sql"
    assert ordered == ["README.md", "045_drop_census.sql", "999_earlier.sql", "1000_later.sql"]
