"""polylogue-lb39z (Phase 1, item 4): static frontier-state executability lint.

Proves this lint catches the exact defect class polylogue-w32w found (a
dispatched actuator paired with a non-executable state) statically, from
source alone -- without ever constructing a ``RawAuthorityFrontierItem`` --
so it fails at review time even for a branch no test exercises. Also proves
the live repo's real ``raw_reconciler.py`` currently passes (anti-regression:
the fixed state post-#3466).
"""

from __future__ import annotations

from pathlib import Path

from devtools import verify_raw_authority_frontier_executability as lint


def test_live_repo_raw_reconciler_has_no_unreachable_actuator_pairing() -> None:
    """Anti-regression: the real, current raw_reconciler.py passes clean."""
    report = lint.compute_executability_report()
    assert report.ok
    assert report.violations == ()
    # Sanity: this lint actually found real construction sites, not zero
    # (a lint that silently matches nothing would trivially "pass").
    assert len(report.pairs) > 10


def test_detects_dispatched_actuator_paired_with_non_executable_state(tmp_path: Path) -> None:
    """Anti-vacuity: reproduce the exact polylogue-w32w defect shape in a fixture.

    A synthetic module pairing REFINE_QUARANTINE (a dispatched actuator) with
    UNRESOLVED_PROVENANCE (not in _EXECUTABLE_STATES) -- the precise
    pre-#3466 shape -- must be flagged as a violation.
    """
    fixture = tmp_path / "fixture_reconciler.py"
    fixture.write_text(
        "\n".join(
            [
                "from polylogue.storage.raw_reconciler import RawAuthorityActuator, RawAuthorityFrontierState",
                "",
                "def _classify(row):",
                "    return _item(",
                "        state=RawAuthorityFrontierState.UNRESOLVED_PROVENANCE,",
                "        actuator=RawAuthorityActuator.REFINE_QUARANTINE,",
                "        row=row,",
                "        reason='pre-w32w regression shape',",
                "    )",
                "",
            ]
        ),
        encoding="utf-8",
    )
    report = lint.compute_executability_report(fixture)
    assert not report.ok
    assert len(report.violations) == 1
    violation = report.violations[0]
    assert violation.state == "UNRESOLVED_PROVENANCE"
    assert violation.actuator == "REFINE_QUARANTINE"
    assert violation.callee == "_item"


def test_safely_rekeyable_pairing_with_same_actuator_passes(tmp_path: Path) -> None:
    """Control: the SAME dispatched actuator paired with an executable state is fine."""
    fixture = tmp_path / "fixture_reconciler_ok.py"
    fixture.write_text(
        "\n".join(
            [
                "from polylogue.storage.raw_reconciler import RawAuthorityActuator, RawAuthorityFrontierState",
                "",
                "def _classify(row):",
                "    return _item(",
                "        state=RawAuthorityFrontierState.SAFELY_REKEYABLE,",
                "        actuator=RawAuthorityActuator.REFINE_QUARANTINE,",
                "        row=row,",
                "        reason='fixed shape',",
                "    )",
                "",
            ]
        ),
        encoding="utf-8",
    )
    report = lint.compute_executability_report(fixture)
    assert report.ok
    assert len(report.pairs) == 1


def test_non_dispatched_actuator_paired_with_non_executable_state_is_fine(tmp_path: Path) -> None:
    """Control: RawAuthorityActuator.NONE (never dispatched) is always safe to pair
    with a non-executable state -- most terminal/informational states use exactly
    this shape (CORRUPT, PROVEN_CURRENT, SUPERSEDED, UNRESOLVED_PROVENANCE)."""
    fixture = tmp_path / "fixture_reconciler_none.py"
    fixture.write_text(
        "\n".join(
            [
                "from polylogue.storage.raw_reconciler import RawAuthorityActuator, RawAuthorityFrontierState",
                "",
                "def _classify(row):",
                "    return _item(",
                "        state=RawAuthorityFrontierState.UNRESOLVED_PROVENANCE,",
                "        actuator=RawAuthorityActuator.NONE,",
                "        row=row,",
                "        reason='terminal, no actuator',",
                "    )",
                "",
            ]
        ),
        encoding="utf-8",
    )
    report = lint.compute_executability_report(fixture)
    assert report.ok


def test_dynamic_forwarding_site_is_reported_but_never_fails(tmp_path: Path) -> None:
    """A state/actuator sourced from a variable (not a literal enum attribute)
    cannot be statically resolved -- reported as informational, not a violation
    (mirrors the real _item(state=strategy_override.state, ...) forwarding call)."""
    fixture = tmp_path / "fixture_reconciler_dynamic.py"
    fixture.write_text(
        "\n".join(
            [
                "def _classify(row, strategy_override):",
                "    return _item(",
                "        state=strategy_override.state,",
                "        actuator=strategy_override.actuator,",
                "        row=row,",
                "        reason='forwarded override',",
                "    )",
                "",
            ]
        ),
        encoding="utf-8",
    )
    report = lint.compute_executability_report(fixture)
    assert report.ok
    assert report.pairs == ()
    assert len(report.dynamic_sites) == 1
    assert report.dynamic_sites[0].callee == "_item"


def test_unknown_enum_member_name_raises_instead_of_silently_passing(tmp_path: Path) -> None:
    """Fail closed: an unrecognized state/actuator name (e.g. a rename this lint's
    own imports haven't caught up with) must not be silently treated as 'no violation'."""
    fixture = tmp_path / "fixture_reconciler_unknown.py"
    fixture.write_text(
        "\n".join(
            [
                "from polylogue.storage.raw_reconciler import RawAuthorityActuator, RawAuthorityFrontierState",
                "",
                "def _classify(row):",
                "    return _item(",
                "        state=RawAuthorityFrontierState.NOT_A_REAL_MEMBER,",
                "        actuator=RawAuthorityActuator.NONE,",
                "        row=row,",
                "        reason='typo',",
                "    )",
                "",
            ]
        ),
        encoding="utf-8",
    )
    try:
        lint.compute_executability_report(fixture)
    except ValueError as exc:
        assert "NOT_A_REAL_MEMBER" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown enum member name")
