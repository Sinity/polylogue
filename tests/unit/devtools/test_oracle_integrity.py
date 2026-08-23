"""Oracle-integrity lint: it must fail on real defects and stay quiet otherwise.

polylogue-4v2d3 AC3 requires that a controlled dead-symbol mutation makes
reachability fail and a controlled path escape makes hermeticity fail. Those
are the two ``*_mutation_*`` tests below; the rest pin the calibration
decisions that keep the lint believable on the real corpus, because a lint
that cries dead code is either switched off or -- far worse -- trusted by a
deletion sweep.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from devtools.oracle_integrity import (
    OracleAllowlistEntry,
    ReachabilityVerdict,
    build_import_closure,
    check_oracle_integrity,
    classify_test_module,
    module_import_edges,
    scan_hermeticity,
    scan_import_time_home_capture,
)

_REPO_ROOT = Path(__file__).parents[3]


def _parsed(path: Path) -> tuple[ast.Module, list[str]]:
    source = path.read_text(encoding="utf-8")
    return ast.parse(source), source.splitlines()


def _write(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _fake_repo(tmp_path: Path, *, include_dead: bool) -> Path:
    """A miniature package with one live chain and optionally one dead module."""
    _write(tmp_path, "pyproject.toml", '[project.scripts]\nfake = "polylogue.cli:main"\n')
    _write(tmp_path, "polylogue/__init__.py", "")
    _write(tmp_path, "polylogue/cli.py", "from polylogue.live import serve\n\n\ndef main() -> None:\n    serve()\n")
    _write(tmp_path, "polylogue/live.py", "def serve() -> None:\n    return None\n")
    if include_dead:
        _write(tmp_path, "polylogue/orphan.py", "def never_called() -> None:\n    return None\n")
    return tmp_path


# ---------------------------------------------------------------------------
# AC3: controlled dead-symbol mutation
# ---------------------------------------------------------------------------


def test_dead_symbol_mutation_makes_reachability_fail(tmp_path: Path) -> None:
    """A test importing only an unreachable module is reported as certifying dead code."""
    root = _fake_repo(tmp_path, include_dead=True)
    _write(root, "tests/unit/test_orphan.py", "from polylogue.orphan import never_called\n")

    report = check_oracle_integrity(root, baseline=frozenset())
    dead = [finding for finding in report.findings if finding.code == "certifies_dead_code"]
    assert [finding.path for finding in dead] == ["tests/unit/test_orphan.py"]
    assert "polylogue.orphan" in dead[0].detail


def test_live_symbol_is_not_reported(tmp_path: Path) -> None:
    """The control case: importing a reachable module is silent."""
    root = _fake_repo(tmp_path, include_dead=True)
    _write(root, "tests/unit/test_live.py", "from polylogue.live import serve\n")

    report = check_oracle_integrity(root, baseline=frozenset())
    assert [finding for finding in report.findings if finding.code == "certifies_dead_code"] == []


# ---------------------------------------------------------------------------
# AC3: controlled path escape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected_code"),
    [
        ('def test_x() -> None:\n    read("~/.codex/sessions")\n', "ambient_path_literal"),
        ('def test_x() -> None:\n    read("/realm/state/polylogue")\n', "ambient_path_literal"),
        ("from pathlib import Path\n\n\ndef test_x() -> None:\n    Path.home()\n", "ambient_path_call"),
        ("import os\n\n\ndef test_x() -> None:\n    os.path.expanduser('~')\n", "ambient_path_call"),
    ],
)
def test_path_escape_mutation_makes_hermeticity_fail(tmp_path: Path, source: str, expected_code: str) -> None:
    path = _write(tmp_path, "tests/unit/test_escape.py", source)
    findings = scan_hermeticity(Path("tests/unit/test_escape.py"), *_parsed(path))
    assert [finding.code for finding in findings] == [expected_code]


def test_hermetic_test_is_not_reported(tmp_path: Path) -> None:
    """The control: a tmp_path-scoped test names no ambient location."""
    source = "from pathlib import Path\n\n\ndef test_x(tmp_path: Path) -> None:\n    (tmp_path / 'a').write_text('x')\n"
    path = _write(tmp_path, "tests/unit/test_hermetic.py", source)
    assert scan_hermeticity(Path("x.py"), *_parsed(path)) == ()


def test_docstrings_naming_ambient_paths_are_not_escapes(tmp_path: Path) -> None:
    """Prose is not a filesystem read.

    Regression guard: the first implementation flagged every module docstring
    that merely *explained* the ``~/.claude`` precedence ladder it tested.
    """
    source = '"""Explains ~/.codex and /realm/state resolution."""\n\n\ndef test_x() -> None:\n    return None\n'
    path = _write(tmp_path, "tests/unit/test_doc.py", source)
    assert scan_hermeticity(Path("x.py"), *_parsed(path)) == ()


# ---------------------------------------------------------------------------
# Calibration decisions
# ---------------------------------------------------------------------------


def test_package_relative_imports_resolve_against_the_package(tmp_path: Path) -> None:
    """``polylogue/ui/__init__.py``'s ``from .facade import X`` is ``polylogue.ui.facade``.

    Getting this wrong resolved it to ``polylogue.facade`` and made every
    module reached only through a package ``__init__`` look unreachable --
    measured at 5 extra false "dead code" verdicts on the real corpus.
    """
    tree = ast.parse("from .facade import ConsoleFacade\n")
    runtime, _type_only = module_import_edges(tree, "polylogue.ui", is_package=True)
    assert "polylogue.ui.facade" in runtime
    assert "polylogue.facade" not in runtime


def test_type_checking_imports_are_reported_separately_not_as_dead(tmp_path: Path) -> None:
    """TYPE_CHECKING-only reachability is its own verdict, never a failure.

    An ``if TYPE_CHECKING:`` import is erased at runtime, so it cannot make a
    symbol execute -- but calling it dead overstates what a static pass knows.
    polylogue-4v2d3's notes record exactly this ambiguity against
    ``storage/sqlite/queries/session_links.py``.
    """
    root = _fake_repo(tmp_path, include_dead=True)
    _write(
        root,
        "polylogue/cli.py",
        "from typing import TYPE_CHECKING\n\nfrom polylogue.live import serve\n\n"
        "if TYPE_CHECKING:\n    from polylogue.orphan import never_called\n\n\n"
        "def main() -> None:\n    serve()\n",
    )
    _write(root, "tests/unit/test_orphan.py", "from polylogue.orphan import never_called\n")

    report = check_oracle_integrity(root, baseline=frozenset())
    assert [finding for finding in report.findings if finding.code == "certifies_dead_code"] == []
    assert report.type_only_modules == ("tests/unit/test_orphan.py",)


def test_allowlist_entries_must_carry_a_reason() -> None:
    """A bare path is not an accepted exemption anywhere in this lint."""
    from devtools.oracle_integrity import HERMETICITY_ALLOWLIST, REACHABILITY_ALLOWLIST

    for entry in (*REACHABILITY_ALLOWLIST, *HERMETICITY_ALLOWLIST):
        assert isinstance(entry, OracleAllowlistEntry)
        assert entry.reason.strip()
        assert len(entry.reason) > 40, f"{entry.path} needs a real reason, not a label"


def test_allowlist_suppresses_only_the_named_subtree(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path, include_dead=True)
    _write(root, "tests/infra/test_helper.py", "from polylogue.orphan import never_called\n")
    _write(root, "tests/unit/test_orphan.py", "from polylogue.orphan import never_called\n")

    report = check_oracle_integrity(
        root,
        baseline=frozenset(),
        reachability_allowlist=(OracleAllowlistEntry("tests/infra", "harness code, not a production certification"),),
    )
    assert [finding.path for finding in report.findings if finding.code == "certifies_dead_code"] == [
        "tests/unit/test_orphan.py"
    ]


# ---------------------------------------------------------------------------
# Real-corpus contract
# ---------------------------------------------------------------------------


def test_repository_is_clean_against_its_baseline() -> None:
    """The gate is green today; only NEW violations fail."""
    report = check_oracle_integrity(_REPO_ROOT)
    assert report.ok, report.to_json()
    assert report.scanned_modules > 500
    assert report.reachable_modules > 500


#: Pinned so regenerating the ratchet is a VISIBLE diff, never a silent grow.
#: A cold review widened the baseline 34 -> 35 by appending a new ambient-path
#: read to an already-baselined file; with the count pinned and the key
#: including the finding detail, that can no longer pass unnoticed.
#:
#: The key includes the line number, which is a deliberate trade-off: editing
#: a file ABOVE a baselined finding shifts its detail and forces a
#: regeneration. Dropping the line would make the key stable but would
#: re-open the exact hole above -- a second ``~/.codex`` read in an
#: already-baselined file would collapse onto the existing key and stay
#: exempt. For a ratchet, "never silently widens" beats "never churns", and
#: the churn is visible in ``--write-baseline``'s delta output.
EXPECTED_BASELINE_ENTRIES = 33


def test_baseline_entry_count_is_pinned() -> None:
    payload = json.loads((_REPO_ROOT / "docs/plans/oracle-integrity-baseline.json").read_text(encoding="utf-8"))
    assert len(payload["entries"]) == EXPECTED_BASELINE_ENTRIES, (
        "baseline size changed; update EXPECTED_BASELINE_ENTRIES in the same commit "
        "so growing the ratchet is reviewable"
    )


def test_baseline_key_includes_the_finding_detail(tmp_path: Path) -> None:
    """A new finding in an already-baselined file must still fail.

    Regression guard for the ratchet hole cold review found: keying exemptions
    on ``(code, path)`` exempted the whole FILE, so appending a second ambient
    read to a baselined test stayed green. Exemptions are now per-finding
    fingerprints, which keep that property while surviving unrelated edits.
    """
    root = _fake_repo(tmp_path, include_dead=False)
    _write(root, "tests/unit/test_two.py", 'def test_x() -> None:\n    read("~/.codex/a")\n    read("~/.claude/b")\n')
    every = check_oracle_integrity(root, baseline=frozenset())
    assert len(every.findings) == 2

    exempt_first = frozenset({every.findings[0].fingerprint})
    remaining = check_oracle_integrity(root, baseline=exempt_first)
    assert [finding.detail for finding in remaining.findings] == [every.findings[1].detail]


def test_baseline_entries_are_structured_and_current() -> None:
    """Every baseline entry names a real finding code and an existing file."""
    payload = json.loads((_REPO_ROOT / "docs/plans/oracle-integrity-baseline.json").read_text(encoding="utf-8"))
    entries = payload["entries"]
    assert entries, "an empty baseline should be deleted, not kept"
    known_codes = {
        "certifies_dead_code",
        "ambient_path_literal",
        "ambient_path_call",
        "import_time_home_capture",
    }
    for entry in entries:
        assert entry["code"] in known_codes
        assert (_REPO_ROOT / entry["path"]).is_file(), f"stale baseline entry: {entry['path']}"
        assert entry["detail"].strip()


def test_registry_roots_include_lazy_click_commands() -> None:
    """Click's deferred commands are roots, or every CLI test looks dead."""
    from devtools.oracle_integrity import PUBLIC_SURFACE_ROOTS, _packaging_entrypoints, _registry_roots
    from devtools.production_reachability import _parse_modules

    modules = _parse_modules(_REPO_ROOT, (_REPO_ROOT / "polylogue",))
    roots = _registry_roots(_REPO_ROOT, [m.tree for m in modules], [m.name for m in modules])
    assert any(root.startswith("polylogue.cli.commands.") for root in roots)
    closure = build_import_closure(_REPO_ROOT, [*_packaging_entrypoints(_REPO_ROOT), *PUBLIC_SURFACE_ROOTS, *roots])
    assert "polylogue.cli.commands.demo" in closure.reachable


def test_no_targets_module_is_not_a_violation(tmp_path: Path) -> None:
    """A test importing no production code certifies nothing and fails nothing."""
    root = _fake_repo(tmp_path, include_dead=True)
    path = _write(root, "tests/unit/test_pure.py", "def test_x() -> None:\n    assert 1 == 1\n")
    closure = build_import_closure(root, ["polylogue.cli.main"])
    verdict, _symbols = classify_test_module(
        ast.parse(path.read_text(encoding="utf-8")), "tests.unit.test_pure", closure=closure
    )
    assert verdict is ReachabilityVerdict.NO_TARGETS


# ---------------------------------------------------------------------------
# Root classes cold review reproduced as false-DEAD channels
# ---------------------------------------------------------------------------


def test_python_m_invoked_module_is_a_root(tmp_path: Path) -> None:
    """``if __name__ == "__main__":`` is an entrypoint no import edge records.

    Live cases this dissolved: ``security/precommit_scan`` (run from
    ``.githooks/pre-commit``) and ``schemas/promotion_audit`` (run from
    ``devtools/verify.py``, three lines from where this gate inserts itself).
    """
    root = _fake_repo(tmp_path, include_dead=True)
    _write(
        root,
        "polylogue/orphan.py",
        'def never_called() -> None:\n    return None\n\n\nif __name__ == "__main__":\n    never_called()\n',
    )
    _write(root, "tests/unit/test_orphan.py", "from polylogue.orphan import never_called\n")

    report = check_oracle_integrity(root, baseline=frozenset())
    assert [f for f in report.findings if f.code == "certifies_dead_code"] == []


def test_ancestor_packages_of_a_reachable_module_are_reachable(tmp_path: Path) -> None:
    """Importing ``pkg.sub`` executes ``pkg/__init__.py`` -- that is import semantics.

    Omitting this reported 53 live parent packages (``polylogue.core`` among
    them) as dead, and accounted for the lint's only ``type_only`` module.
    """
    root = _fake_repo(tmp_path, include_dead=False)
    _write(root, "polylogue/pkg/__init__.py", "")
    _write(root, "polylogue/pkg/leaf.py", "def leaf() -> None:\n    return None\n")
    _write(root, "polylogue/live.py", "from polylogue.pkg.leaf import leaf\n\n\ndef serve() -> None:\n    leaf()\n")
    _write(root, "tests/unit/test_pkg.py", "import polylogue.pkg\n")

    closure = build_import_closure(root, ["polylogue.cli.main"])
    assert "polylogue.pkg" in closure.reachable
    assert "polylogue.pkg.leaf" in closure.reachable


def test_symbol_reexported_through_an_unreachable_facade_is_not_dead(tmp_path: Path) -> None:
    """A live symbol reached via a pass-through facade certifies live code."""
    root = _fake_repo(tmp_path, include_dead=False)
    _write(root, "polylogue/facade.py", 'from polylogue.live import serve\n\n__all__ = ["serve"]\n')
    _write(root, "tests/unit/test_facade.py", "from polylogue.facade import serve\n")

    report = check_oracle_integrity(root, baseline=frozenset())
    assert [f for f in report.findings if f.code == "certifies_dead_code"] == []


def test_facade_upgrade_is_symbol_precise_not_module_wide(tmp_path: Path) -> None:
    """Importing a DEAD symbol from a facade is still dead code.

    Guards the tightening: a module-level "imports something reachable" rule
    would upgrade nearly every dead module in the tree, since almost all of
    them import some live enum -- trading false positives for false negatives,
    and a false negative here is a deletion sweep never hearing about real
    dead code.
    """
    root = _fake_repo(tmp_path, include_dead=False)
    _write(root, "polylogue/dead_impl.py", "def dead_fn() -> None:\n    return None\n")
    _write(
        root,
        "polylogue/facade.py",
        "from polylogue.dead_impl import dead_fn\nfrom polylogue.live import serve\n",
    )
    _write(root, "tests/unit/test_facade_dead.py", "from polylogue.facade import dead_fn\n")

    report = check_oracle_integrity(root, baseline=frozenset())
    assert [f.path for f in report.findings if f.code == "certifies_dead_code"] == ["tests/unit/test_facade_dead.py"]


def test_literal_container_registry_seeds_check_functions() -> None:
    """``ARCHIVE_VERIFICATION_CHECKS``' tuple-of-calls shape must be walked.

    Cold review measured the literal-container seeding at ZERO net
    contribution: the non-recursive pass saw only the ``ast.Call`` elements and
    never reached the ``_check_*`` reference inside each call's arguments.
    """
    from devtools.oracle_integrity import _literal_container_roots
    from devtools.production_reachability import _parse_modules

    modules = _parse_modules(_REPO_ROOT, (_REPO_ROOT / "polylogue",))
    roots = _literal_container_roots([m.tree for m in modules], [m.name for m in modules])
    assert any("_check_" in root for root in roots), "registry check functions are not seeded"


# ---------------------------------------------------------------------------
# Production-side import-time home capture
# ---------------------------------------------------------------------------


def test_module_level_home_capture_is_flagged(tmp_path: Path) -> None:
    """The PROVIDERS shape: an ambient location captured into a constant.

    This is the guard-removal twin. ``polylogue/schemas/observation_models.py``'s
    ``PROVIDERS`` is guarded in production (an opt-in fallback flag, a row_seen
    check, and an autouse conftest fixture that repoints every provider
    session_dir), so it is baselined rather than reported as an open escape.
    What must never happen is the DETECTOR failing to see the pattern -- a
    guarded escape that the lint cannot detect stops being guarded the moment
    someone edits the guard.
    """
    source = (
        'from pathlib import Path\n\nPROVIDERS = {\n    "codex": {"session_dir": Path.home() / ".codex/sessions"},\n}\n'
    )
    path = _write(tmp_path, "polylogue/schemas/models.py", source)
    findings = scan_import_time_home_capture(Path("polylogue/schemas/models.py"), *_parsed(path))
    assert [finding.code for finding in findings] == ["import_time_home_capture"]
    assert "Path.home()" in findings[0].detail


def test_call_time_home_use_is_not_flagged(tmp_path: Path) -> None:
    """The control, and the whole point of the check.

    The identical call inside a function body is evaluated per call, so it
    observes whatever environment is patched at that moment. Flagging it would
    report ~109 sites in this package and mean nothing.
    """
    source = 'from pathlib import Path\n\n\ndef resolve() -> Path:\n    return Path.home() / ".codex/sessions"\n'
    path = _write(tmp_path, "polylogue/sources/resolver.py", source)
    assert scan_import_time_home_capture(Path("polylogue/sources/resolver.py"), *_parsed(path)) == ()


def test_lambda_body_is_call_time_not_import_time(tmp_path: Path) -> None:
    """A lambda inside a module-level assignment is deferred, like a function."""
    source = "from pathlib import Path\n\nRESOLVER = lambda: Path.home() / '.codex'\n"
    path = _write(tmp_path, "polylogue/sources/lazy.py", source)
    assert scan_import_time_home_capture(Path("polylogue/sources/lazy.py"), *_parsed(path)) == ()


def test_real_repository_has_exactly_one_import_time_capture_module() -> None:
    """Calibration against the live tree (y9106/hhg58 forensics agree).

    Independent AST forensics found exactly one member of this class repo-wide.
    If this count moves, either a new unguarded capture landed or the detector
    changed scope -- both worth a human look, which is why it is pinned.
    """
    report = check_oracle_integrity(_REPO_ROOT, baseline=frozenset())
    captures = [f for f in report.findings if f.code == "import_time_home_capture"]
    assert {f.path for f in captures} == {"polylogue/schemas/observation_models.py"}


# ---------------------------------------------------------------------------
# Ratchet keying
# ---------------------------------------------------------------------------


def test_fingerprint_survives_an_unrelated_edit_above_the_finding(tmp_path: Path) -> None:
    """The reanchoring smell: an edit above a finding must not re-key it.

    Two separate PRs had baselined entries invalidated by edits that had nothing
    to do with them, because the key embedded a line number.
    """
    root = _fake_repo(tmp_path, include_dead=False)
    body = 'def test_x() -> None:\n    read("~/.codex/sessions")\n'
    path = _write(root, "tests/unit/test_shift.py", body)
    before = scan_hermeticity(Path("tests/unit/test_shift.py"), *_parsed(path))

    path.write_text("# an unrelated comment added above\n\n" + body, encoding="utf-8")
    after = scan_hermeticity(Path("tests/unit/test_shift.py"), *_parsed(path))

    assert before[0].fingerprint == after[0].fingerprint
    assert before[0].detail != after[0].detail, "the human detail should still show the new line"


def test_a_second_identical_read_gets_its_own_fingerprint(tmp_path: Path) -> None:
    """Stability must not be bought by collapsing distinct findings.

    Dropping the line entirely would make two ``~/.codex`` reads in one file
    share a key, so a newly added one would inherit the existing exemption --
    the ratchet hole cold review already found once.
    """
    root = _fake_repo(tmp_path, include_dead=False)
    path = _write(
        root,
        "tests/unit/test_twice.py",
        'def test_x() -> None:\n    read("~/.codex/a")\n    read("~/.codex/a")\n',
    )
    findings = scan_hermeticity(Path("tests/unit/test_twice.py"), *_parsed(path))
    assert len(findings) == 2
    assert findings[0].fingerprint != findings[1].fingerprint


def test_dead_code_fingerprint_changes_when_the_target_set_grows(tmp_path: Path) -> None:
    """A new dead import in a baselined module must not inherit its exemption."""
    root = _fake_repo(tmp_path, include_dead=True)
    _write(root, "polylogue/orphan2.py", "def also_dead() -> None:\n    return None\n")
    _write(root, "tests/unit/test_orphan.py", "from polylogue.orphan import never_called\n")
    first = check_oracle_integrity(root, baseline=frozenset()).findings

    _write(
        root,
        "tests/unit/test_orphan.py",
        "from polylogue.orphan import never_called\nfrom polylogue.orphan2 import also_dead\n",
    )
    second = check_oracle_integrity(root, baseline=frozenset()).findings
    assert first[0].fingerprint != second[0].fingerprint


def test_baseline_notes_mark_non_worklist_entries() -> None:
    """Annotated entries are judgments, not deletion-sweep work."""
    payload = json.loads((_REPO_ROOT / "docs/plans/oracle-integrity-baseline.json").read_text(encoding="utf-8"))
    annotated = [entry for entry in payload["entries"] if entry.get("note")]
    assert annotated, "the guarded PROVIDERS rows and the codex-804 false positive should be annotated"
    for entry in annotated:
        assert len(entry["note"]) > 80, f"{entry['path']} note must explain, not label"
    assert any("FALSE POSITIVE" in entry["note"] for entry in annotated)
    assert any("GUARDED" in entry["note"] for entry in annotated)
