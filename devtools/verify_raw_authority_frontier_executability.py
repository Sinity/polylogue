"""Statically verify every raw-authority frontier state has a reachable actuator.

Background
----------

``polylogue.storage.raw_reconciler`` classifies every accepted raw-authority
head into one of a small closed set of ``RawAuthorityFrontierState`` values,
each paired with a ``RawAuthorityActuator``. Only actuators with a real
``apply()`` dispatch branch (``_APPLY_DISPATCHED_ACTUATORS``) promise
"something automatically executes this"; only states in ``_EXECUTABLE_STATES``
are ever selected by the daemon or the operator break-glass path
(``item.executable``). polylogue-w32w found a state (``UNRESOLVED_PROVENANCE``)
paired with a dispatched actuator (``REFINE_QUARANTINE``) that was NOT in
``_EXECUTABLE_STATES`` -- 4,174 blockers demanded an actuator no path could
ever select, and the gap accumulated silently for weeks because nothing
checked the pairing except live production data eventually noticing the
backlog never drained.

``RawAuthorityFrontierItem.__post_init__`` now raises if a *constructed*
instance has this shape (PR #3466) -- but that is a runtime assertion: it
only fires on whichever (state, actuator) pairs a test happens to construct.
A future contributor adding a new frontier state, or re-pairing an existing
one, can ship a classification branch that is never exercised by any test
fixture; the constructor guard stays silent until that branch runs against
real archive data, which is exactly the failure mode that let the original
defect accumulate for weeks undetected. This lint closes that gap: it
statically enumerates every (state, actuator) pair
``polylogue/storage/raw_reconciler.py``'s classification code can literally
construct -- independent of whether any test ever exercises that branch --
and re-validates the same invariant the constructor enforces, so the CHECK
fails at review time, not months later against live data.

What this lint checks
----------------------

Parses ``polylogue/storage/raw_reconciler.py`` and finds every call site that
constructs a frontier item's ``state``/``actuator`` pairing with **literal**
enum-attribute arguments:

* ``_item(state=RawAuthorityFrontierState.X, actuator=RawAuthorityActuator.Y, ...)``
  -- the sole ``RawAuthorityFrontierItem`` builder.
* ``_StrategyOverride(state=RawAuthorityFrontierState.X,
  actuator=RawAuthorityActuator.Y, ...)`` -- overrides that later flow into
  ``_item`` via ``_item(state=strategy_override.state,
  actuator=strategy_override.actuator, ...)``; that forwarding call site's
  arguments are not literal (they read a variable), so this lint checks the
  override's own literal construction instead -- the same (state, actuator)
  pair reaches ``RawAuthorityFrontierItem.__post_init__`` either way.

For each literal pair found, re-checks the exact invariant
``RawAuthorityFrontierItem.__post_init__`` enforces at runtime: an actuator in
``_APPLY_DISPATCHED_ACTUATORS`` must only ever be paired with a state in
``_EXECUTABLE_STATES``. Both sets are imported directly from
``polylogue.storage.raw_reconciler`` (not re-declared here), so this lint
never drifts out of sync with the real executability gate.

A ``state=``/``actuator=`` argument that is not a literal
``RawAuthorityFrontierState.X`` / ``RawAuthorityActuator.Y`` attribute access
(e.g. a bare variable) cannot be resolved statically and is reported
separately as "dynamic" -- informational only, never a failure, since every
current dynamic pairing (the ``_item(state=strategy_override.state, ...)``
forwarding call) is already covered by checking its override's own literal
construction site. A future dynamic pairing with no literal source anywhere
in this file would not be caught by this lint; it would still be caught by
the runtime constructor guard the first time a test or live classification
constructs it.

Wired standalone via ``devtools lab policy raw-authority-frontier-executability``
(like ``schema-versioning``): static, archive-independent, sub-second.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from devtools import repo_root as _get_root
from polylogue.storage.raw_reconciler import (
    _APPLY_DISPATCHED_ACTUATORS,
    _EXECUTABLE_STATES,
    RawAuthorityActuator,
    RawAuthorityFrontierState,
)

ROOT = _get_root()
RECONCILER_PATH = ROOT / "polylogue" / "storage" / "raw_reconciler.py"

_TARGET_CALLEES: tuple[str, ...] = ("_item", "_StrategyOverride")

_EXECUTABLE_STATE_NAMES = {state.name for state in _EXECUTABLE_STATES}
_APPLY_DISPATCHED_ACTUATOR_NAMES = {actuator.name for actuator in _APPLY_DISPATCHED_ACTUATORS}


@dataclass(frozen=True, slots=True)
class FrontierPair:
    callee: str
    lineno: int
    state: str
    actuator: str


@dataclass(frozen=True, slots=True)
class DynamicSite:
    callee: str
    lineno: int
    detail: str


@dataclass(frozen=True, slots=True)
class ExecutabilityReport:
    pairs: tuple[FrontierPair, ...]
    dynamic_sites: tuple[DynamicSite, ...]
    violations: tuple[FrontierPair, ...]

    @property
    def ok(self) -> bool:
        return not self.violations


def _literal_enum_attr(node: ast.expr, *, enum_name: str) -> str | None:
    """Return ``X`` for an ``EnumName.X`` attribute access node, else ``None``."""
    if not isinstance(node, ast.Attribute):
        return None
    value = node.value
    if not isinstance(value, ast.Name) or value.id != enum_name:
        return None
    return node.attr


def collect_frontier_pairs(path: Path = RECONCILER_PATH) -> tuple[tuple[FrontierPair, ...], tuple[DynamicSite, ...]]:
    """Statically enumerate every literal (state, actuator) construction pair."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pairs: list[FrontierPair] = []
    dynamic: list[DynamicSite] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Name) or func.id not in _TARGET_CALLEES:
            continue
        state_arg: ast.expr | None = None
        actuator_arg: ast.expr | None = None
        for keyword in node.keywords:
            if keyword.arg == "state":
                state_arg = keyword.value
            elif keyword.arg == "actuator":
                actuator_arg = keyword.value
        if state_arg is None or actuator_arg is None:
            # Every real call site names both explicitly by keyword; a call
            # missing either is not this lint's concern (it would fail at
            # import/call time as a TypeError against _item's/StrategyOverride's
            # own required signature).
            continue
        state_name = _literal_enum_attr(state_arg, enum_name="RawAuthorityFrontierState")
        actuator_name = _literal_enum_attr(actuator_arg, enum_name="RawAuthorityActuator")
        if state_name is None or actuator_name is None:
            dynamic.append(
                DynamicSite(
                    callee=func.id,
                    lineno=node.lineno,
                    detail="state/actuator argument is not a literal EnumName.MEMBER attribute access",
                )
            )
            continue
        pairs.append(FrontierPair(callee=func.id, lineno=node.lineno, state=state_name, actuator=actuator_name))

    return tuple(pairs), tuple(dynamic)


def compute_executability_report(path: Path = RECONCILER_PATH) -> ExecutabilityReport:
    pairs, dynamic_sites = collect_frontier_pairs(path)
    # Fail closed on an unknown name: a rename that outpaces this lint's own
    # enum imports must not silently pass as "no violation found".
    for pair in pairs:
        if pair.state not in {state.name for state in RawAuthorityFrontierState}:
            raise ValueError(f"{path}:{pair.lineno}: unknown RawAuthorityFrontierState member {pair.state!r}")
        if pair.actuator not in {actuator.name for actuator in RawAuthorityActuator}:
            raise ValueError(f"{path}:{pair.lineno}: unknown RawAuthorityActuator member {pair.actuator!r}")
    violations = tuple(
        pair
        for pair in pairs
        if pair.actuator in _APPLY_DISPATCHED_ACTUATOR_NAMES and pair.state not in _EXECUTABLE_STATE_NAMES
    )
    return ExecutabilityReport(pairs=pairs, dynamic_sites=dynamic_sites, violations=violations)


def _format_report(report: ExecutabilityReport, *, path: Path) -> str:
    rel = path.relative_to(ROOT) if path.is_absolute() else path
    lines = [
        f"frontier (state, actuator) construction sites checked: {len(report.pairs)}",
        f"dynamic (unresolvable) sites, informational only: {len(report.dynamic_sites)}",
        f"unreachable-actuator violations: {len(report.violations)}",
    ]
    if report.violations:
        lines.append("")
        lines.append(
            "Frontier states pairing a dispatched actuator with a non-executable "
            "state -- no path (daemon or operator) would ever select these:"
        )
        for pair in report.violations:
            lines.append(
                f"  {rel}:{pair.lineno}: {pair.callee}(state={pair.state}, actuator={pair.actuator}) -- "
                f"{pair.actuator} has an apply() dispatch branch but {pair.state} is not in _EXECUTABLE_STATES"
            )
        lines.append(
            "  Fix: either add the state to _EXECUTABLE_STATES (and prove the daemon/operator "
            "path can safely select it), or pair this classification with a non-dispatched "
            "actuator (RawAuthorityActuator.NONE, REACQUIRE, or REQUEST_JUDGMENT)."
        )
    if report.dynamic_sites:
        lines.append("")
        lines.append("Dynamic sites (state/actuator not a literal enum attribute -- not checked here):")
        for site in report.dynamic_sites:
            lines.append(f"  {rel}:{site.lineno}: {site.callee}(...) -- {site.detail}")
    if report.ok:
        lines.append("")
        lines.append("Raw-authority frontier executability policy intact.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    report = compute_executability_report()

    if args.json:
        print(
            json.dumps(
                {
                    "pairs_checked": len(report.pairs),
                    "dynamic_sites": [
                        {"callee": site.callee, "lineno": site.lineno, "detail": site.detail}
                        for site in report.dynamic_sites
                    ],
                    "violations": [
                        {"callee": pair.callee, "lineno": pair.lineno, "state": pair.state, "actuator": pair.actuator}
                        for pair in report.violations
                    ],
                    "ok": report.ok,
                },
                indent=2,
            )
        )
    else:
        print(_format_report(report, path=RECONCILER_PATH))

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
