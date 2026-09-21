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
from pathlib import Path

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
    "polylogue/sources/live/watcher.py",
)
OUTSIDE_CLOSURE = (
    "polylogue/daemon/convergence.py",
    "polylogue/daemon/cli.py",
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


def test_the_cli_package_is_identity_neutral() -> None:
    """No file under ``polylogue/cli/`` may feed the derived schema identity.

    The CLI is a surface adapter: an ordinary CLI edit must not move the
    derived identity and demand an archive reconvergence. The single edge that
    used to exist was a function-local
    ``from polylogue.cli.shared.machine_errors import emit_success`` in
    ``analysis/registry.py`` -- ``_import_bases`` walks the AST, so a deferred
    import written to break a runtime cycle is still a full closure edge.

    Anti-vacuity: point any closure member's import back at
    ``polylogue.cli.*`` -- deferred or not -- and this goes red.
    """
    members = {Path(member).as_posix() for member in derived_identity_source_closure()}
    cli_members = sorted(member for member in members if "polylogue/cli/" in member)
    assert cli_members == []


def test_command_lists_the_whole_closure(capsys: pytest.CaptureFixture[str]) -> None:
    assert schema_closure.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == len(derived_identity_source_closure())
    assert "polylogue/storage/sqlite/archive_tiers/write.py" in payload["members"]


class TestClosureRatchetGate:
    """The closure may shrink freely; growth must be declared in the baseline.

    Anti-vacuity: make ``verify_schema_closure.main`` compare counts instead of
    membership, and ``test_growth_is_blocking`` still passes while
    ``test_a_swap_that_keeps_the_count_is_still_growth`` goes red. Make the
    gate symmetric (fail on removal too) and ``test_shrinking_is_free`` goes
    red. Delete the checked-in baseline and
    ``test_the_checked_in_baseline_matches_the_live_closure`` goes red.
    """

    def test_the_gate_is_registered_in_the_quick_path(self) -> None:
        from devtools.gate import GATES_BY_NAME, quick_gates

        assert "schema-closure" in {gate.name for gate in quick_gates()}
        assert GATES_BY_NAME["schema-closure"].blocking is True

    def test_the_checked_in_baseline_matches_the_live_closure(self) -> None:
        """A green gate on master means the baseline is the real membership."""
        from devtools import verify_schema_closure

        baseline = verify_schema_closure.load_baseline(verify_schema_closure.ROOT / verify_schema_closure.BASELINE_PATH)
        assert set(baseline) == set(verify_schema_closure.measure_closure())

    def test_growth_is_blocking(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from devtools import verify_schema_closure

        self._with(monkeypatch, tmp_path, baseline=("a.py",), observed=("a.py", "b.py"))
        assert verify_schema_closure.main(["--json"]) == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["added"] == ["b.py"]

    def test_shrinking_is_free(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from devtools import verify_schema_closure

        self._with(monkeypatch, tmp_path, baseline=("a.py", "b.py"), observed=("a.py",))
        assert verify_schema_closure.main(["--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["removed"] == ["b.py"]
        assert payload["added"] == []

    def test_a_swap_that_keeps_the_count_is_still_growth(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The ratchet is over membership, not over a size integer."""
        from devtools import verify_schema_closure

        self._with(monkeypatch, tmp_path, baseline=("a.py", "b.py"), observed=("a.py", "c.py"))
        assert verify_schema_closure.main(["--json"]) == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["added"] == ["c.py"]
        assert payload["observed_count"] == payload["baseline_count"]

    def test_a_missing_baseline_fails_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from devtools import verify_schema_closure

        self._with(monkeypatch, tmp_path, baseline=None, observed=("a.py",))
        assert verify_schema_closure.main(["--json"]) == 1

    @staticmethod
    def _with(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        baseline: tuple[str, ...] | None,
        observed: tuple[str, ...],
    ) -> None:
        from devtools import verify_schema_closure

        root = Path(tmp_path)
        if baseline is not None:
            verify_schema_closure.write_baseline(root / verify_schema_closure.BASELINE_PATH, baseline)
        monkeypatch.setattr(verify_schema_closure, "ROOT", root)
        monkeypatch.setattr(verify_schema_closure, "measure_closure", lambda **_kwargs: observed)
