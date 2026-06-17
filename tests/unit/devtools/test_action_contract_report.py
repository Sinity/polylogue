from __future__ import annotations

from devtools.action_contract_report import render_action_contract_report
from devtools.render_cli_output_schemas import SCHEMAS
from polylogue.operations.action_contracts import ACTION_CONTRACTS


def test_report_is_derived_from_action_contracts() -> None:
    rendered = render_action_contract_report()

    assert "## Public Action Contracts" in rendered
    assert "ACTION_CONTRACTS" in rendered
    for contract in ACTION_CONTRACTS:
        path = f"`polylogue {' '.join(contract.path)}`"
        assert path in rendered
        assert f"`{contract.effect}`" in rendered
        assert f"`{contract.input_unit}`" in rendered
        assert f"`{contract.machine_envelope}`" in rendered
        for output_format in contract.formats:
            assert f"`{output_format}`" in rendered
        for guard in contract.guards:
            assert f"`{guard}`" in rendered


def test_report_is_derived_from_published_output_schemas() -> None:
    rendered = render_action_contract_report()

    assert "## Published Machine Output Schemas" in rendered
    assert "render_cli_output_schemas.SCHEMAS" in rendered
    for schema in SCHEMAS:
        assert f"`{schema.name}`" in rendered
        assert f"`{schema.model.__name__}`" in rendered
        for surface in schema.surfaces:
            assert f"`{surface}`" in rendered
