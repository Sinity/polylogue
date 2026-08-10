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
    load_beads_jsonl,
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
    records = deepcopy(load_beads_jsonl())
    mutation(records)
    path = tmp_path / "issues.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records.values()) + "\n", encoding="utf-8")
    return path


def test_real_campaign_graph_resolves_the_current_forcing_set() -> None:
    result = resolve_default_incident_coverage()

    assert result.target_bead_id == "polylogue-818fy"
    assert result.forcing_dependency_ids[:3] == (
        "polylogue-a7xr.25",
        "polylogue-reindex-source-remediation",
        "polylogue-xselt",
    )
    assert len(result.forcing_dependency_ids) == 99
    assert result.ledger_row_count == 99
    assert LEDGER_PATH.is_file()
    assert CAMPAIGN_GRAPH_PATH.is_file()


def test_deleting_a_ledger_row_emits_machine_readable_missing_id() -> None:
    ledger = _ledger()
    _rows(ledger)[:] = [row for row in _rows(ledger) if row["bead_id"] != "polylogue-xselt"]

    with pytest.raises(IncidentCoverageLedgerError) as error:
        resolve_incident_coverage(ledger, _graph())

    assert error.value.diagnostic["error"] == "forcing_set_mismatch"
    assert "polylogue-xselt" in error.value.diagnostic["missing_ids"]


def test_duplicate_forcing_row_is_blocking() -> None:
    ledger = _ledger()
    _rows(ledger).append(deepcopy(_rows(ledger)[0]))

    with pytest.raises(IncidentCoverageLedgerError, match="duplicate rows"):
        resolve_incident_coverage(ledger, _graph())


def test_current_beads_jsonl_removing_a_forcing_dependency_is_blocking(tmp_path: Path) -> None:
    def remove_dependency(records: dict[str, dict[str, object]]) -> None:
        target = records["polylogue-818fy"]
        dependencies = cast(list[dict[str, object]], target["dependencies"])
        dependencies.pop()

    beads_path = _mutated_beads(tmp_path, remove_dependency)

    with pytest.raises(IncidentCoverageLedgerError) as error:
        resolve_incident_coverage(_ledger(), _graph(), beads_path=beads_path)

    assert error.value.diagnostic["error"] == "campaign_graph_mismatch"
    assert "polylogue-xselt" in error.value.diagnostic["extra_ids"]


def test_current_beads_jsonl_adding_a_p0_forcing_blocker_is_blocking(tmp_path: Path) -> None:
    def add_dependency(records: dict[str, dict[str, object]]) -> None:
        target = records["polylogue-818fy"]
        dependencies = cast(list[dict[str, object]], target["dependencies"])
        dependencies.append(
            {
                "issue_id": "polylogue-818fy",
                "depends_on_id": "polylogue-jdesf",
                "type": "blocks",
            }
        )

    beads_path = _mutated_beads(tmp_path, add_dependency)

    with pytest.raises(IncidentCoverageLedgerError) as error:
        resolve_incident_coverage(_ledger(), _graph(), beads_path=beads_path)

    assert "polylogue-jdesf" in error.value.diagnostic["missing_ids"]


def test_current_beads_jsonl_dependency_kind_change_is_blocking(tmp_path: Path) -> None:
    def change_dependency_kind(records: dict[str, dict[str, object]]) -> None:
        target = records["polylogue-818fy"]
        dependencies = cast(list[dict[str, object]], target["dependencies"])
        dependencies[0]["type"] = "relates-to"

    beads_path = _mutated_beads(tmp_path, change_dependency_kind)

    with pytest.raises(IncidentCoverageLedgerError) as error:
        resolve_incident_coverage(_ledger(), _graph(), beads_path=beads_path)

    assert "polylogue-a7xr.25" in error.value.diagnostic["extra_ids"]


def test_transitive_forcing_closure_is_load_bearing() -> None:
    result = resolve_default_incident_coverage()

    assert "polylogue-dudtn" in result.forcing_dependency_ids
    assert "polylogue-818fy" not in result.forcing_dependency_ids
    assert result.ledger_row_count == len(result.forcing_dependency_ids)


def test_route_entrypoint_must_be_registered() -> None:
    ledger = _ledger()
    cast(dict[str, object], _rows(ledger)[0]["route"])["entrypoint"] = "deleted-route"

    with pytest.raises(IncidentCoverageLedgerError) as error:
        resolve_incident_coverage(ledger, _graph())

    assert error.value.diagnostic["error"] == "unknown_route_entrypoint"


def test_red_mutation_must_be_declared_by_its_fixture() -> None:
    ledger = _ledger()
    cast(dict[str, object], _rows(ledger)[0]["red_mutation"])["mutation_id"] = "deleted-mutation"

    with pytest.raises(IncidentCoverageLedgerError) as error:
        resolve_incident_coverage(ledger, _graph())

    assert error.value.diagnostic["error"] == "unknown_mutation"


def test_unknown_dependency_kind_is_structured_and_blocking(tmp_path: Path) -> None:
    def change_dependency_kind(records: dict[str, dict[str, object]]) -> None:
        target = records["polylogue-818fy"]
        dependencies = cast(list[dict[str, object]], target["dependencies"])
        dependencies[0]["type"] = "invented-kind"

    beads_path = _mutated_beads(tmp_path, change_dependency_kind)

    with pytest.raises(IncidentCoverageLedgerError) as error:
        resolve_incident_coverage(_ledger(), _graph(), beads_path=beads_path)

    assert error.value.diagnostic["error"] == "unknown_dependency_kind"
    assert error.value.diagnostic["dependency_kind"] == "invented-kind"


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    [
        (lambda row: row["red_mutation"].__setitem__("fixture_id", "deleted-fixture"), "unknown_fixture"),
        (lambda row: row["registry_checks"].append("deleted-check"), "unknown_checks"),
        (lambda row: row["expected_snapshot"].__setitem__("snapshot_id", "deleted-snapshot"), "unknown_snapshot"),
    ],
)
def test_deleted_fixture_check_or_snapshot_is_blocking(
    mutation: Callable[[dict[str, object]], None], diagnostic: str
) -> None:
    ledger = _ledger()
    mutation(_rows(ledger)[0])

    with pytest.raises(IncidentCoverageLedgerError) as error:
        resolve_incident_coverage(ledger, _graph())

    assert error.value.diagnostic["error"] == diagnostic


def test_catalog_source_must_resolve_to_a_committed_file() -> None:
    ledger = _ledger()
    fixtures = cast(dict[str, dict[str, object]], ledger["fixtures"])
    fixtures["campaign-corpus"]["source"] = "tests/fixtures/reindex_incident_coverage/deleted.json"

    with pytest.raises(IncidentCoverageLedgerError) as error:
        resolve_incident_coverage(ledger, _graph())

    assert error.value.diagnostic["error"] == "unresolved_source_reference"


def test_receipt_owner_must_match_the_row_that_uses_it() -> None:
    ledger = _ledger()
    receipts = cast(dict[str, dict[str, object]], ledger["receipts"])
    receipts["live-proof-5xxmc"]["owner_bead_id"] = "polylogue-7zp4"
    cast(list[str], _rows(ledger)[0]["receipts"]).append("live-proof-5xxmc")

    with pytest.raises(IncidentCoverageLedgerError) as error:
        resolve_incident_coverage(ledger, _graph())

    assert error.value.diagnostic["error"] == "receipt_owner_mismatch"
    assert error.value.diagnostic["expected_owner"] == "polylogue-a7xr.25"


def test_successor_source_and_parent_are_resolved() -> None:
    ledger = _ledger()
    graph = _graph()
    successor_id = "polylogue-claude-vintage-live-proof"
    cast(list[str], graph["known_child_bead_ids"]).append(successor_id)
    dependency = cast(list[dict[str, object]], graph["forcing_dependencies"])[0]
    dependency["child_bead_ids"] = [successor_id]
    _rows(ledger)[0]["residual_successor"] = {
        "bead_id": successor_id,
        "kind": "named-child-bead",
    }

    result = resolve_incident_coverage(ledger, graph)

    assert result.successor_backed_ids == ("polylogue-a7xr.25",)


def test_missing_successor_catalog_entry_is_blocking() -> None:
    ledger = _ledger()
    graph = _graph()
    successor_id = "polylogue-claude-vintage-live-proof"
    cast(list[str], graph["known_child_bead_ids"]).append(successor_id)
    cast(list[dict[str, object]], graph["forcing_dependencies"])[0]["child_bead_ids"] = [successor_id]
    _rows(ledger)[0]["residual_successor"] = {
        "bead_id": successor_id,
        "kind": "named-child-bead",
    }
    cast(dict[str, object], ledger["successors"]).pop(successor_id)

    with pytest.raises(IncidentCoverageLedgerError) as error:
        resolve_incident_coverage(ledger, graph)

    assert error.value.diagnostic["error"] == "unknown_successor"
