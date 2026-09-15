"""Every adopted declaration family is actually enforced by the gate.

Anti-vacuity: drop a family from ``verify_declaration_bindings.REGISTRIES`` and
``test_every_adopted_family_is_gated`` fails; make the gate stop reporting
findings and ``test_a_broken_registry_fails_the_gate`` fails.
"""

from __future__ import annotations

from pathlib import Path

from devtools import verify_declaration_bindings as gate

#: Families that have adopted the shared declaration kernel (polylogue-o21.3
#: for query/marker/maintenance; t46.8.1 and the daemon slice for the rest).
ADOPTED_FAMILIES = frozenset({"mcp", "daemon-route", "query", "marker", "maintenance"})


def test_every_adopted_family_is_gated() -> None:
    assert set(gate.REGISTRIES) == ADOPTED_FAMILIES


def test_live_run_is_clean_for_every_registry() -> None:
    report = gate.run()
    assert set(report) == ADOPTED_FAMILIES
    assert {name: lines for name, lines in report.items() if lines} == {}


def test_main_exits_zero_at_head() -> None:
    assert gate.main([]) == 0


def test_a_broken_registry_fails_the_gate(tmp_path: Path) -> None:
    """A root with no source tree makes every declared owner path unresolvable."""

    report = gate.run(root=tmp_path)
    assert all(lines for lines in report.values()), report
    for lines in report.values():
        assert any("unresolved_owner_path" in line for line in lines)
        assert any("repair: `" in line for line in lines)


def test_query_domain_diagnostics_are_included() -> None:
    """The query family's own binding resolution runs inside the gate."""

    entry = gate.REGISTRIES["query"]
    assert entry.domain is not None
    assert entry.domain() == ()
