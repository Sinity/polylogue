from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import cast

import pytest

from devtools.incident_coverage_ledger import (
    CAMPAIGN_GRAPH_PATH,
    LEDGER_PATH,
    IncidentCoverageLedgerError,
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


def test_real_campaign_graph_resolves_all_current_forcing_dependencies() -> None:
    result = resolve_default_incident_coverage()

    assert result.target_bead_id == "polylogue-818fy"
    assert result.ledger_row_count == 40
    assert len(result.forcing_dependency_ids) == 40
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
