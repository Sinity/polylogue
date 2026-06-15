"""Output-assurance matrix coverage tests (#1272).

Every primary `polylogue` subcommand must:

* be declared in the ``polylogue.cli.output_assurance`` registry, and
* satisfy at least one of the assurance gates (``snapshot`` or
  ``json_contract``).

These tests fail when a new primary subcommand lands without an
assurance declaration, so the matrix in
``docs/cli-reference.md`` stays grounded in the live CLI tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.cli.action_contracts import ACTION_CONTRACTS
from polylogue.cli.click_app import cli
from polylogue.cli.command_inventory import iter_command_paths
from polylogue.cli.output_assurance import (
    ASSURANCE_BY_COMMAND,
    ASSURANCE_REGISTRY,
    FAMILY_ORDER,
    OutputAssurance,
    action_contract_coverage,
    render_matrix_markdown,
)

# Virtual commands have entries in the assurance registry but are not
# Click subcommands — they are implicit modes of the root group. The
# default `search` mode is the canonical example (`polylogue <terms>`
# without any verb dispatches to the search/show path).
VIRTUAL_COMMANDS: frozenset[str] = frozenset({"search"})


# Some Click commands exist as utility surfaces but are not "primary
# subcommands" in the #1272 sense — they are hidden helpers, internal
# trampolines, or experimental flags we intentionally do not contract
# with external consumers about. Keep this list small and grounded.
EXCLUDED_TOP_LEVEL_COMMANDS: frozenset[str] = frozenset(
    {
        # `qa` is the showcase / QA harness; output is fixture-only.
        "qa",
        # `mcp` launches an MCP server, output is JSON-RPC, not CLI-shaped.
        "mcp",
        # `polylogue-hook` companion entrypoint, exposed as `hook`.
        "hook",
        # `polylogued`-companion runtime trampoline.
        "run",
        # site renderer; output is files on disk, not stdout.
        "site",
    }
)


def _top_level_command_names() -> set[str]:
    names: set[str] = set()
    for entry in iter_command_paths(cli):
        if len(entry.path) != 1:
            continue
        names.add(entry.path[0])
    return names


def test_every_primary_command_is_in_assurance_registry() -> None:
    """Every top-level primary command must declare an assurance contract."""
    discovered = _top_level_command_names() - EXCLUDED_TOP_LEVEL_COMMANDS
    declared = set(ASSURANCE_BY_COMMAND.keys())
    missing = sorted(discovered - declared)
    assert not missing, (
        "These primary CLI commands have no entry in "
        "polylogue.cli.output_assurance.ASSURANCE_REGISTRY (#1272). "
        "Declare an OutputAssurance for each, then re-run "
        "`devtools render-cli-reference` to refresh the matrix.\n"
        f"Missing: {missing}"
    )


def test_assurance_registry_has_no_dead_entries() -> None:
    """Every assurance entry must map to a live primary command or virtual mode."""
    discovered = _top_level_command_names() | VIRTUAL_COMMANDS
    declared = set(ASSURANCE_BY_COMMAND.keys())
    stale = sorted(declared - discovered)
    assert not stale, (
        "These assurance entries no longer correspond to live CLI commands. "
        "Remove them from ASSURANCE_REGISTRY.\n"
        f"Stale: {stale}"
    )


@pytest.mark.parametrize("entry", ASSURANCE_REGISTRY, ids=lambda e: e.command)
def test_each_entry_satisfies_some_gate(entry: OutputAssurance) -> None:
    """Every primary command must have either a JSON contract or a snapshot."""
    assert entry.json_contract or entry.snapshot, (
        f"Command `{entry.command}` declares neither a JSON contract nor a "
        "snapshot test. Output drift would go undetected. Add one of the two "
        "gates to the assurance registry."
    )


@pytest.mark.parametrize("entry", ASSURANCE_REGISTRY, ids=lambda e: e.command)
def test_entry_family_is_recognized(entry: OutputAssurance) -> None:
    """Every entry's family must appear in FAMILY_ORDER."""
    assert entry.family in FAMILY_ORDER, (
        f"Command `{entry.command}` declares family `{entry.family}` which is "
        "not in FAMILY_ORDER. Add the family there (or fix the entry)."
    )


def test_matrix_renders_non_empty_markdown() -> None:
    """The matrix renders a Markdown table including every primary command."""
    rendered = render_matrix_markdown()
    assert "## Output Assurance Matrix" in rendered
    assert "### Public Action Contract Coverage" in rendered
    for entry in ASSURANCE_REGISTRY:
        assert f"`{entry.command}`" in rendered, f"Rendered matrix is missing command `{entry.command}`."


def test_action_contract_coverage_is_derived_from_action_contracts() -> None:
    """The output assurance report must expose every public action contract."""
    rows = action_contract_coverage()
    assert [row.contract for row in rows] == list(ACTION_CONTRACTS)
    assert all(row.assurance is not None for row in rows)


def test_rendered_matrix_reflects_action_contract_metadata() -> None:
    """Contract effect/format/envelope metadata should come from ACTION_CONTRACTS."""
    rendered = render_matrix_markdown()
    for contract in ACTION_CONTRACTS:
        path_cell = "`" + " ".join(contract.path) + "`"
        assert path_cell in rendered
        assert f"`{contract.effect}`" in rendered
        assert f"`{contract.machine_envelope}`" in rendered
        for output_format in contract.formats:
            assert f"`{output_format}`" in rendered


def test_virtual_find_contract_maps_to_search_assurance_row() -> None:
    """`find` is virtual grammar, but the assurance report must still cover it."""
    by_path = {row.contract.path: row for row in action_contract_coverage()}
    find_row = by_path[("find",)]
    assert find_row.assurance_command == "search"
    assert find_row.assurance is ASSURANCE_BY_COMMAND["search"]


def test_streaming_implies_json_contract() -> None:
    """Any command that advertises NDJSON streaming must have a JSON contract."""
    for entry in ASSURANCE_REGISTRY:
        if entry.streaming:
            assert entry.json_contract, (
                f"Command `{entry.command}` advertises NDJSON streaming "
                "without a stable JSON contract — streaming what?"
            )


def test_schema_field_references_existing_file() -> None:
    """Any entry with a schema must point at a published schema file."""
    schemas_dir = Path("docs/schemas/cli-output")
    for entry in ASSURANCE_REGISTRY:
        if entry.schema is None:
            continue
        target = schemas_dir / entry.schema
        assert target.is_file(), (
            f"Command `{entry.command}` references schema `{entry.schema}` "
            f"but {target} does not exist. Run "
            "`devtools render-cli-output-schemas` (or add the entry to "
            "devtools.render_cli_output_schemas.SCHEMAS)."
        )
