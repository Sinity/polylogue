from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from devtools.incident_coverage_ledger import (
    CAMPAIGN_GRAPH_PATH,
    LEDGER_PATH,
    IncidentCoverageLedgerError,
    load_beads_forcing_export,
    load_campaign_graph,
    load_ledger,
    resolve_default_incident_coverage,
    resolve_incident_coverage,
)


def _ledger() -> dict[str, object]:
    return deepcopy(load_ledger())


def _graph() -> dict[str, object]:
    return deepcopy(load_campaign_graph())


def _rows(ledger: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], ledger["rows"])


def _mutated_beads(tmp_path: Path, mutation: Callable[[dict[str, dict[str, object]]], None]) -> Path:
    records = deepcopy(load_beads_forcing_export(_graph()))
    mutation(records)
    path = tmp_path / "issues.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records.values()) + "\n", encoding="utf-8")
    return path


def test_real_campaign_graph_resolves_all_current_forcing_dependencies() -> None:
    result = resolve_default_incident_coverage()

    assert result.target_bead_id == "polylogue-818fy"
    assert result.ledger_row_count == 39
    assert len(result.forcing_dependency_ids) == 39
    assert set(result.successor_backed_ids) == {
        "polylogue-0qfy",
        "polylogue-2hwl",
        "polylogue-2qrx",
        "polylogue-5iz4",
        "polylogue-6753s",
        "polylogue-foee",
        "polylogue-ix5r",
        "polylogue-xofj",
    }
    assert LEDGER_PATH.is_file()
    assert CAMPAIGN_GRAPH_PATH.is_file()


def test_default_resolution_uses_the_graphs_committed_beads_export() -> None:
    result = resolve_default_incident_coverage()

    assert result.ledger_row_count == len(result.forcing_dependency_ids)


def test_deleting_a_forcing_row_is_blocking() -> None:
    ledger = _ledger()
    _rows(ledger).pop()

    with pytest.raises(IncidentCoverageLedgerError, match="ledger rows do not match forcing dependencies"):
        resolve_incident_coverage(ledger, _graph())


def test_duplicate_forcing_row_is_blocking() -> None:
    ledger = _ledger()
    _rows(ledger).append(deepcopy(_rows(ledger)[0]))

    with pytest.raises(IncidentCoverageLedgerError, match="duplicate rows"):
        resolve_incident_coverage(ledger, _graph())


def test_replacing_a_forcing_row_with_an_unknown_bead_is_blocking() -> None:
    ledger = _ledger()
    row = _rows(ledger)[0]
    row["bead_id"] = "polylogue-not-a-forcing-dependency"
    cast(dict[str, object], row["incident"])["bead_id"] = "polylogue-not-a-forcing-dependency"

    with pytest.raises(IncidentCoverageLedgerError, match="ledger rows do not match forcing dependencies"):
        resolve_incident_coverage(ledger, _graph())


def test_current_beads_jsonl_removing_a_forcing_dependency_is_blocking(tmp_path: Path) -> None:
    def remove_dependency(records: dict[str, dict[str, object]]) -> None:
        target = records["polylogue-818fy"]
        dependencies = cast(list[dict[str, object]], target["dependencies"])
        dependencies.pop()

    beads_path = _mutated_beads(tmp_path, remove_dependency)

    with pytest.raises(IncidentCoverageLedgerError, match="current Beads forcing dependencies") as error:
        resolve_incident_coverage(_ledger(), _graph(), beads_path=beads_path)
    assert error.value.diagnostic["extra_ids"] == ["polylogue-xselt"]


def test_current_beads_jsonl_adding_a_p0_forcing_blocker_is_blocking(tmp_path: Path) -> None:
    def add_dependency(records: dict[str, dict[str, object]]) -> None:
        target = records["polylogue-818fy"]
        dependencies = cast(list[dict[str, object]], target["dependencies"])
        dependencies.append(
            {
                "issue_id": "polylogue-818fy",
                "depends_on_id": "polylogue-dudtn",
                "type": "blocks",
            }
        )

    beads_path = _mutated_beads(tmp_path, add_dependency)

    with pytest.raises(IncidentCoverageLedgerError, match="current Beads forcing dependencies") as error:
        resolve_incident_coverage(_ledger(), _graph(), beads_path=beads_path)
    assert error.value.diagnostic["missing_ids"] == ["polylogue-dudtn"]


def test_current_beads_jsonl_dependency_kind_change_is_blocking(tmp_path: Path) -> None:
    def change_dependency_kind(records: dict[str, dict[str, object]]) -> None:
        target = records["polylogue-818fy"]
        dependencies = cast(list[dict[str, object]], target["dependencies"])
        dependencies[0]["type"] = "relates-to"

    beads_path = _mutated_beads(tmp_path, change_dependency_kind)

    with pytest.raises(IncidentCoverageLedgerError, match="current Beads forcing dependencies") as error:
        resolve_incident_coverage(_ledger(), _graph(), beads_path=beads_path)
    assert error.value.diagnostic["extra_ids"] == ["polylogue-0qfy"]


def test_unknown_dependency_kind_is_structured_and_blocking(tmp_path: Path) -> None:
    def change_dependency_kind(records: dict[str, dict[str, object]]) -> None:
        target = records["polylogue-818fy"]
        dependencies = cast(list[dict[str, object]], target["dependencies"])
        dependencies[0]["type"] = "invented-kind"

    beads_path = _mutated_beads(tmp_path, change_dependency_kind)

    with pytest.raises(IncidentCoverageLedgerError, match="unknown dependency kind") as error:
        resolve_incident_coverage(_ledger(), _graph(), beads_path=beads_path)
    assert error.value.diagnostic["error"] == "unknown_dependency_kind"
    assert error.value.diagnostic["dependency_kind"] == "invented-kind"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row["red_mutation"].__setitem__("fixture_id", "deleted-fixture"), "unknown fixture"),
        (lambda row: row["registry_checks"].append("deleted-check"), "unknown checks"),
        (lambda row: row["expected_snapshot"].__setitem__("snapshot_id", "deleted-snapshot"), "unknown snapshot"),
    ],
)
def test_deleted_fixture_check_or_snapshot_is_blocking(
    mutation: Callable[[dict[str, object]], None], message: str
) -> None:
    ledger = _ledger()
    row = _rows(ledger)[0]
    mutation(row)

    with pytest.raises(IncidentCoverageLedgerError, match=message):
        resolve_incident_coverage(ledger, _graph())


def test_deleting_a_named_successor_is_blocking() -> None:
    ledger = _ledger()
    successor = cast(dict[str, object], _rows(ledger)[0]["residual_successor"])
    cast(dict[str, object], ledger["successors"]).pop(str(successor["bead_id"]))

    with pytest.raises(IncidentCoverageLedgerError, match="unknown successor"):
        resolve_incident_coverage(ledger, _graph())


def test_dangling_receipt_reference_is_blocking() -> None:
    ledger = _ledger()
    row = next(row for row in _rows(ledger) if row["bead_id"] == "polylogue-5xxmc")
    cast(list[str], row["receipts"]).append("deleted-receipt")

    with pytest.raises(IncidentCoverageLedgerError, match="unknown receipts"):
        resolve_incident_coverage(ledger, _graph())


def test_fixture_source_must_resolve_to_a_committed_file() -> None:
    ledger = _ledger()
    fixtures = cast(dict[str, dict[str, object]], ledger["fixtures"])
    fixtures["campaign-corpus"]["source"] = "tests/fixtures/reindex_incident_coverage/deleted.json"

    with pytest.raises(IncidentCoverageLedgerError, match="not a committed file"):
        resolve_incident_coverage(ledger, _graph())


def test_receipt_owner_must_match_the_row_that_uses_it() -> None:
    ledger = _ledger()
    receipts = cast(dict[str, dict[str, object]], ledger["receipts"])
    receipts["live-proof-5xxmc"]["owner_bead_id"] = "polylogue-7zp4"

    with pytest.raises(IncidentCoverageLedgerError, match="owned by polylogue-7zp4"):
        resolve_incident_coverage(ledger, _graph())


def test_successor_from_another_parent_is_blocking() -> None:
    ledger = _ledger()
    successor = cast(dict[str, object], _rows(ledger)[0]["residual_successor"])
    successor["bead_id"] = "polylogue-active-leaf-live-proof"

    with pytest.raises(IncidentCoverageLedgerError, match="not a named child"):
        resolve_incident_coverage(ledger, _graph())


def test_closed_implementation_without_live_proof_or_child_is_blocking() -> None:
    ledger = _ledger()
    row = next(row for row in _rows(ledger) if row["bead_id"] == "polylogue-5xxmc")
    row["receipts"] = []
    row["residual_successor"] = None

    with pytest.raises(IncidentCoverageLedgerError, match="closed implementation bead polylogue-5xxmc"):
        resolve_incident_coverage(ledger, _graph())


def test_closed_implementation_with_named_child_remains_explicitly_blocked() -> None:
    result = resolve_default_incident_coverage()

    assert "polylogue-0qfy" in result.closed_implementation_ids
    assert "polylogue-0qfy" in result.successor_backed_ids
