"""Declaration-driven interaction coverage for the CLI test plane.

This module is deliberately test infrastructure, not a second CLI registry.
The Click tree, query metadata, output contracts, and visual tape inventory are
the product declarations; this module joins them into an auditable matrix and
gives every dimension a named owning oracle.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product

from polylogue.archive.query.metadata import query_unit_descriptors
from polylogue.cli.click_app import cli
from polylogue.cli.command_inventory import iter_command_paths
from polylogue.cli.query_output import MACHINE_OUTPUT_FORMATS
from polylogue.cli.verb_names import VERB_NAMES

SUPPORTED_SHELLS: tuple[str, ...] = ("bash", "zsh", "fish")
QUERY_CAPABILITIES: tuple[str, ...] = (
    "argv",
    "query",
    "completion",
    "projection",
    "pagination",
)
RENDERERS: tuple[str, ...] = ("plain", *sorted(MACHINE_OUTPUT_FORMATS))
RESULT_STATES: tuple[str, ...] = (
    "empty",
    "data",
    "error",
    "cancelled",
    "daemon-unavailable",
    "progress",
)
INTERACTION_TRANSITIONS: tuple[str, ...] = (
    "edit",
    "resize",
    "interrupt",
    "fuzzy",
    "progress",
    "daemon-loss",
    "non-tty",
)

# Query-first verbs do not appear as Click command paths.  They are still
# public grammar capabilities and therefore must have coverage ownership.
DECLARED_QUERY_COMMANDS: frozenset[str] = frozenset({"find", *VERB_NAMES})

DECLARED_COMPLETION_SOURCES: tuple[str, ...] = (
    "action",
    "actions",
    "argv",
    "assertion",
    "assertions",
    "block",
    "blocks",
    "context-snapshot",
    "context-snapshots",
    "delegation",
    "delegations",
    "field",
    "file",
    "files",
    "message",
    "messages",
    "observed-event",
    "observed-events",
    "run",
    "runs",
    "tool-episode",
    "tool-episodes",
    "value",
)


# Dimension declarations are explicit too.  A new renderer, state, token, or
# transition must first choose an oracle owner; otherwise the structural gate
# reports an uncovered cell instead of silently multiplying the matrix.
def _owners(values: Iterable[str], owner: str) -> dict[str, str]:
    return dict.fromkeys(values, owner)


DIMENSION_OWNERS: dict[str, dict[str, str]] = {
    "capability": _owners(QUERY_CAPABILITIES, "tests/unit/cli/test_interaction_oracles.py"),
    "renderer": _owners(RENDERERS, "tests/unit/cli/test_terminal_cells.py"),
    "result-state": _owners(RESULT_STATES, "tests/unit/cli/test_terminal_cells.py"),
    "transition": _owners(INTERACTION_TRANSITIONS, "tests/unit/cli/test_terminal_cells.py"),
    "completion-source": _owners(DECLARED_COMPLETION_SOURCES, "tests/unit/cli/test_completion_matrix.py"),
}

# This is the only intentionally explicit coverage declaration.  A new root
# command must choose an owning oracle here; deriving this set from the live
# Click tree would make an accidentally-added command self-authorizing.
COMMAND_ROOT_OWNERS: dict[str, str] = {
    "agent": "tests/unit/cli/test_interactive_cli.py",
    "agents": "tests/unit/cli/test_interactive_cli.py",
    "analyze": "tests/unit/cli/test_cli_action_contracts.py",
    "annotations": "tests/unit/cli/test_annotation_join_command.py",
    "compare": "tests/unit/cli/test_compare_command.py",
    "config": "tests/unit/cli/test_completions_contract.py",
    "context": "tests/unit/cli/test_context_view.py",
    "continue": "tests/unit/cli/test_continue_absorption.py",
    "dashboard": "tests/unit/cli/test_dashboard_command.py",
    "delete": "tests/unit/cli/test_query_verbs_runtime.py",
    "demo": "tests/unit/cli/test_demo_command.py",
    "facets": "tests/unit/cli/test_facets.py",
    "find": "tests/unit/cli/test_query_exec_laws.py",
    "hooks": "tests/unit/cli/test_hooks.py",
    "import": "tests/unit/cli/test_import.py",
    "init": "tests/unit/cli/test_init.py",
    "judge": "tests/unit/cli/test_judge_command.py",
    "manual": "tests/unit/cli/test_manual_command.py",
    "mark": "tests/unit/cli/test_mark_note_identity.py",
    "note": "tests/unit/cli/test_note.py",
    "ops": "tests/unit/cli/test_status.py",
    "read": "tests/unit/cli/test_query_set_read.py",
    "select": "tests/unit/cli/test_interactive_cli.py",
    "setting": "tests/unit/cli/test_setting_command.py",
    "status": "tests/unit/cli/test_status.py",
    "tutorial": "tests/unit/cli/test_tutorial.py",
}


@dataclass(frozen=True, slots=True)
class CoverageCell:
    """One declared interaction fact and its owning test layer."""

    command: str
    capability: str
    completion_source: str
    renderer: str
    result_state: str
    transition: str
    owner: str

    @property
    def key(self) -> tuple[str, ...]:
        return (
            self.command,
            self.capability,
            self.completion_source,
            self.renderer,
            self.result_state,
            self.transition,
        )


def _command_names() -> tuple[str, ...]:
    paths = iter_command_paths(cli)
    names = {" ".join(path.path) for path in paths}
    names.update(DECLARED_QUERY_COMMANDS)
    return tuple(sorted(names))


def completion_sources() -> tuple[str, ...]:
    """Return dynamic sources from the query metadata registry."""
    sources = {"argv", "field", "value", "action"}
    for descriptor in query_unit_descriptors(terminal_supported=True):
        sources.update(descriptor.source_aliases)
    values = tuple(sorted(sources))
    return values


def coverage_matrix() -> tuple[CoverageCell, ...]:
    """Generate the complete interaction matrix from product declarations."""
    cells: list[CoverageCell] = []
    for command, capability, source, renderer, state, transition in product(
        _command_names(),
        QUERY_CAPABILITIES,
        completion_sources(),
        RENDERERS,
        RESULT_STATES,
        INTERACTION_TRANSITIONS,
    ):
        root = command.split(" ", 1)[0]
        owner = COMMAND_ROOT_OWNERS.get(root, "")
        cells.append(
            CoverageCell(
                command=command,
                capability=capability,
                completion_source=source,
                renderer=renderer,
                result_state=state,
                transition=transition,
                owner=owner,
            )
        )
    return tuple(cells)


def coverage_gaps(
    *,
    command_paths: Iterable[str] | None = None,
    owners: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Return uncovered declarations without mutating the live registries.

    ``command_paths`` and ``owners`` are injectable specifically so the
    anti-vacuity test can add a fake production command and prove that the
    matrix turns red instead of silently expanding its own declaration.
    """
    effective_owners = COMMAND_ROOT_OWNERS if owners is None else owners
    names = tuple(_command_names() if command_paths is None else command_paths)
    gaps: set[str] = set()
    for command in names:
        root = command.split(" ", 1)[0]
        if root not in effective_owners or not effective_owners[root]:
            gaps.add(f"command:{command}")
    for dimension, values in (
        ("capability", QUERY_CAPABILITIES),
        ("completion-source", completion_sources()),
        ("renderer", RENDERERS),
        ("result-state", RESULT_STATES),
        ("transition", INTERACTION_TRANSITIONS),
    ):
        if not values:
            gaps.add(f"empty:{dimension}")
        declared_owners = DIMENSION_OWNERS[dimension]
        for value in values:
            if value not in declared_owners or not declared_owners[value]:
                gaps.add(f"{dimension}:{value}")
    return tuple(sorted(gaps))


def assert_matrix_complete() -> tuple[CoverageCell, ...]:
    """Validate structural coverage and return the generated matrix."""
    gaps = coverage_gaps()
    if gaps:
        raise AssertionError("CLI interaction coverage gaps: " + ", ".join(gaps))
    matrix = coverage_matrix()
    expected = len(_command_names()) * len(QUERY_CAPABILITIES) * len(completion_sources())
    expected *= len(RENDERERS) * len(RESULT_STATES) * len(INTERACTION_TRANSITIONS)
    if len(matrix) != expected or len({cell.key for cell in matrix}) != expected:
        raise AssertionError("CLI interaction matrix has duplicate or missing cells")
    return matrix


__all__ = [
    "COMMAND_ROOT_OWNERS",
    "CoverageCell",
    "DECLARED_QUERY_COMMANDS",
    "INTERACTION_TRANSITIONS",
    "QUERY_CAPABILITIES",
    "RENDERERS",
    "RESULT_STATES",
    "SUPPORTED_SHELLS",
    "assert_matrix_complete",
    "completion_sources",
    "coverage_gaps",
    "coverage_matrix",
]
