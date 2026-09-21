"""The derived-identity closure command must consult the real closure.

Closure membership is the scheduling fact every change has to know before it
knows its landing window, so these assertions are about the *property* --
membership follows the live import graph and the command reports it
faithfully -- and never about which specific file happens to be a member.

This file used to pin a hand-written ``IN_CLOSURE`` roster. The closure
legitimately shrank (#5350, 558 -> 555 members) and ``daemon/write_coordinator
.py`` left it, so three tests here went red without anything being wrong
(polylogue-pxzms). A roster that a legitimate closure change turns red is a
roster that will go stale again on the next one, and it erodes trust in
exactly the check nobody can afford to distrust. Every sample below is
therefore derived from the live closure at run time.

Anti-vacuity, and this is what the derivation must not give up: replacing the
closure computation with a hardcoded list, an allowlist, or any path-prefix
rule still turns at least one test here red, because
``test_no_path_prefix_rule_reproduces_membership`` proves over *every* file
under ``polylogue/`` that 57-odd directories hold members and non-members
side by side, and ``test_the_command_agrees_with_the_library_in_both
_directions`` fails the moment the command reports a membership the library
disagrees with -- in either direction, so a command stubbed to answer "IN"
(or "out") for everything is red.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path, PurePosixPath

import pytest

from devtools import schema_closure
from polylogue.sources.origin_specs import derived_identity_source_closure, in_derived_identity_closure

REPO_ROOT = Path(__file__).resolve().parents[3]


def _live_members() -> frozenset[str]:
    """Repo-relative posix paths of every current closure member."""
    members: set[str] = set()
    for member in derived_identity_source_closure():
        path = Path(member)
        try:
            members.add(path.relative_to(REPO_ROOT).as_posix())
        except ValueError:
            members.add(path.as_posix())
    return frozenset(members)


def _package_sources() -> frozenset[str]:
    return frozenset(path.relative_to(REPO_ROOT).as_posix() for path in (REPO_ROOT / "polylogue").rglob("*.py"))


def _spread_sample(paths: frozenset[str], limit: int = 8) -> tuple[str, ...]:
    """A deterministic sample holding at most one path per directory.

    One path per directory keeps the sample spread across packages instead of
    clustering in whichever one sorts first, so a command that got membership
    right for a single subtree cannot pass.
    """
    by_directory: dict[str, str] = {}
    for path in sorted(paths):
        by_directory.setdefault(str(PurePosixPath(path).parent), path)
    return tuple(sorted(by_directory.values())[:limit])


def test_the_closure_is_non_trivial_in_both_directions() -> None:
    """Neither "everything" nor "nothing" is a closure, so both samples exist."""
    members = _live_members()
    sources = _package_sources()
    assert members, "the derived-identity closure is empty"
    assert members < sources, "every package source is a closure member, which no import graph produces"


def test_no_path_prefix_rule_reproduces_membership() -> None:
    """Membership follows the import graph, so no directory rule reproduces it.

    Derived over every file under ``polylogue/`` rather than over a pair of
    hand-picked siblings: any directory holding a member beside a non-member
    refutes a prefix rule, and this asserts such directories exist.
    """
    members = _live_members()
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for path in _package_sources():
        counts[str(PurePosixPath(path).parent)][0 if path in members else 1] += 1
    mixed = {directory for directory, (inside, outside) in counts.items() if inside and outside}
    assert mixed, "every directory is wholly in or wholly out of the closure, which a prefix rule would reproduce"


def test_the_command_agrees_with_the_library_in_both_directions(capsys: pytest.CaptureFixture[str]) -> None:
    """The command's verdict must equal the library's for members and non-members.

    This is the assertion the campaign leans on: ``devtools schema closure``
    is how a lane learns whether its files are IN. Both samples are derived,
    so a legitimate closure change cannot turn this red -- only a command that
    reports the wrong membership can.
    """
    members = _live_members()
    inside = _spread_sample(members)
    outside = _spread_sample(_package_sources() - members)
    assert inside and outside

    # The library predicate is the reference, and it is checked against the
    # enumerated closure first so a predicate that always agreed with the
    # command could not carry both sides of this test.
    expected = {path: in_derived_identity_closure(path) for path in (*inside, *outside)}
    assert expected == {**dict.fromkeys(inside, True), **dict.fromkeys(outside, False)}

    assert schema_closure.main(["--json", *inside, *outside]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "polylogue.derived-identity-closure"
    reported = {result["path"]: result["in_closure"] for result in payload["results"]}
    assert reported == expected
    assert payload["closure_size"] == len(members)


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
