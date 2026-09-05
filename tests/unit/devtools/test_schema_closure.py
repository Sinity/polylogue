"""The derived-identity closure command must consult the real closure.

Anti-vacuity: these assertions are pinned to files whose membership was
measured against the live import graph, and they straddle a directory boundary
in both directions. Replacing the closure computation with a hardcoded list, an
allowlist, or any path-prefix rule turns at least one of them red -- a prefix
rule on ``polylogue/daemon/`` cannot admit ``write_coordinator.py`` while
excluding ``convergence.py``.
"""

from __future__ import annotations

import json

import pytest

from devtools import schema_closure
from polylogue.sources.origin_specs import derived_identity_source_closure, in_derived_identity_closure

#: Measured against the live closure. Each pair shares a package with its
#: opposite, so no directory rule reproduces both columns.
IN_CLOSURE = (
    "polylogue/daemon/write_coordinator.py",
    "polylogue/core/degraded.py",
    "polylogue/storage/sqlite/archive_tiers/write.py",
    "polylogue/storage/sqlite/connection_profile.py",
)
OUTSIDE_CLOSURE = (
    "polylogue/daemon/convergence.py",
    "polylogue/daemon/cli.py",
    "polylogue/sources/live/watcher.py",
)


@pytest.mark.parametrize("path", IN_CLOSURE)
def test_member_is_in_closure(path: str) -> None:
    assert in_derived_identity_closure(path) is True


@pytest.mark.parametrize("path", OUTSIDE_CLOSURE)
def test_non_member_is_outside_closure(path: str) -> None:
    assert in_derived_identity_closure(path) is False


def test_closure_spans_directories_in_both_directions() -> None:
    """A path-prefix implementation cannot satisfy this."""
    daemon_members = {path for path in (*IN_CLOSURE, *OUTSIDE_CLOSURE) if path.startswith("polylogue/daemon/")}
    inside = {path for path in daemon_members if in_derived_identity_closure(path)}
    assert inside and inside != daemon_members


def test_command_reports_membership(capsys: pytest.CaptureFixture[str]) -> None:
    assert schema_closure.main(["--json", *IN_CLOSURE, *OUTSIDE_CLOSURE]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "polylogue.derived-identity-closure"
    reported = {result["path"]: result["in_closure"] for result in payload["results"]}
    assert all(reported[path] for path in IN_CLOSURE)
    assert not any(reported[path] for path in OUTSIDE_CLOSURE)


def test_command_lists_the_whole_closure(capsys: pytest.CaptureFixture[str]) -> None:
    assert schema_closure.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == len(derived_identity_source_closure())
    assert "polylogue/storage/sqlite/archive_tiers/write.py" in payload["members"]
