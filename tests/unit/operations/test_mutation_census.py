"""Enforcement test for docs/plans/mutation-census.yaml (polylogue-kwsb.2 AC1).

Every destructive operation classified in the census must use the closed
status vocabulary, and rows that name a ``spec_name`` must point at a real
``OperationSpec`` whose ``executor_status`` agrees with the census row --
so the census and the OperationSpec catalog cannot silently drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from polylogue.operations.specs import build_declared_operation_catalog

_CENSUS_PATH = Path(__file__).resolve().parents[3] / "docs" / "plans" / "mutation-census.yaml"

_VALID_STATUSES = {"executor-routed", "declared-not-routed", "typed-exemption"}


def _load_rows() -> list[dict[str, Any]]:
    document = cast("dict[str, Any]", yaml.safe_load(_CENSUS_PATH.read_text(encoding="utf-8")))
    return cast("list[dict[str, Any]]", document["rows"])


def test_census_file_exists_and_parses() -> None:
    document = cast("dict[str, Any]", yaml.safe_load(_CENSUS_PATH.read_text(encoding="utf-8")))
    assert document["schema_version"] == 1
    rows = document["rows"]
    assert isinstance(rows, list)
    assert len(rows) > 0


def test_every_row_has_a_valid_status() -> None:
    for row in _load_rows():
        assert row["status"] in _VALID_STATUSES, row


def test_typed_exemptions_declare_a_reason() -> None:
    for row in _load_rows():
        if row["status"] == "typed-exemption":
            assert row.get("reason"), row


def test_rows_naming_a_spec_agree_with_the_operation_spec_catalog() -> None:
    catalog = build_declared_operation_catalog().by_name()
    for row in _load_rows():
        spec_name = row.get("spec_name")
        if spec_name is None:
            continue
        assert spec_name in catalog, f"census row {row['operation']!r} names unknown spec {spec_name!r}"
        spec = catalog[spec_name]
        assert spec.executor_status == row["status"], (
            f"census row {row['operation']!r} declares status {row['status']!r} but "
            f"OperationSpec {spec_name!r} declares executor_status {spec.executor_status!r}"
        )


def test_named_t46_9_routes_are_executor_routed() -> None:
    rows = {row["operation"]: row for row in _load_rows()}
    for operation in ("mutate-delete-session", "mutate-session-excision", "mutate-identity-reset"):
        assert rows[operation]["status"] == "executor-routed"
