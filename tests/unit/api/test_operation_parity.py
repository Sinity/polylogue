"""Operation-parity contract for the CLI/MCP/Python semantic-operation matrix.

Anti-vacuity: each test names the mutation that turns it red -- a new
unclassified facade callable, a classified name that no longer exists, a CLI
binding whose command was renamed, an MCP binding to an undeclared tool, an
absence without a reason, or a new MCP tool with no parity row.
"""

from __future__ import annotations

import pytest

from polylogue.api import parity
from polylogue.mcp.declarations.registry import declared_tool_names


def test_live_matrix_has_no_findings() -> None:
    """The committed matrix is total and every binding resolves.

    Anti-vacuity: adding a public method to ``Polylogue`` without classifying
    it makes this red with ``unclassified_callable``.
    """

    findings = parity.validate_parity()
    assert findings == (), [f"{item.code}: {item.subject}: {item.message}" for item in findings]


def test_every_declared_mcp_tool_has_a_parity_row() -> None:
    """MCP rows are derived, so a new tool cannot silently escape the matrix.

    Anti-vacuity: adding an ``_ToolRow`` to the MCP registry without a CLI
    binding entry makes this (and ``validate_parity``) red.
    """

    mcp_targets = {
        binding.target
        for operation in parity.semantic_operations()
        for binding in operation.bindings
        if binding.surface == "mcp" and binding.bound
    }
    assert declared_tool_names() <= mcp_targets


def test_unclassified_public_callable_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A facade callable that is neither bound nor excluded fails the gate."""

    live = (*parity.public_facade_callables(), "brand_new_public_method")
    monkeypatch.setattr(parity, "public_facade_callables", lambda: live)
    codes = {finding.code for finding in parity.validate_parity()}
    assert "unclassified_callable" in codes


def test_stale_classification_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A classified name that left the facade fails the gate."""

    live = tuple(name for name in parity.public_facade_callables() if name != "query_units")
    monkeypatch.setattr(parity, "public_facade_callables", lambda: live)
    codes = {finding.code for finding in parity.validate_parity()}
    assert "stale_classification" in codes


def test_unknown_cli_binding_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CLI binding naming a command that does not exist fails the gate."""

    bindings = dict(parity._CLI_BINDINGS)
    bindings["read"] = parity.SurfaceBinding("cli", target="command-that-does-not-exist")
    monkeypatch.setattr(parity, "_CLI_BINDINGS", bindings)
    findings = {finding.code for finding in parity.validate_parity()}
    assert "unknown_cli_command" in findings


def test_unknown_mcp_binding_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """An MCP binding naming an undeclared tool fails the gate."""

    rows = (
        parity._ExtraOperationRow(
            "api.fake",
            "Synthetic row.",
            f"{parity.FACADE_SYMBOL}.stats",
            parity.SurfaceBinding("cli", target="status"),
            parity.SurfaceBinding("mcp", target="tool-that-does-not-exist"),
        ),
    )
    monkeypatch.setattr(parity, "_EXTRA_OPERATION_ROWS", rows)
    codes = {finding.code for finding in parity.validate_parity()}
    assert "unknown_mcp_tool" in codes


def test_unjustified_absence_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """An absence with no recorded reason fails the gate."""

    bindings = dict(parity._CLI_BINDINGS)
    bindings["read"] = parity.SurfaceBinding("cli")
    monkeypatch.setattr(parity, "_CLI_BINDINGS", bindings)
    codes = {finding.code for finding in parity.validate_parity()}
    assert "unjustified_absence" in codes


def test_unresolved_python_binding_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Python binding that no longer resolves fails the gate."""

    overrides = dict(parity._PYTHON_OVERRIDES)
    overrides["status"] = parity.SurfaceBinding("python", target=f"{parity.FACADE_SYMBOL}.method_that_vanished")
    monkeypatch.setattr(parity, "_PYTHON_OVERRIDES", overrides)
    codes = {finding.code for finding in parity.validate_parity()}
    assert "unresolved_python_binding" in codes


def test_classification_is_a_partition() -> None:
    """No callable is both bound by an operation and listed as an exclusion."""

    excluded = [item.name for item in parity.EXCLUSIONS]
    assert len(excluded) == len(set(excluded))
    assert not set(excluded) & set(parity.operation_python_names())


def test_every_exclusion_carries_a_reason() -> None:
    assert all(item.category and item.reason for item in parity.EXCLUSIONS)
