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
)

_REPO_ROOT = Path(__file__).parents[3]


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
    findings = scan_hermeticity(Path("tests/unit/test_escape.py"), ast.parse(path.read_text(encoding="utf-8")))
    assert [finding.code for finding in findings] == [expected_code]


def test_hermetic_test_is_not_reported(tmp_path: Path) -> None:
    """The control: a tmp_path-scoped test names no ambient location."""
    source = "from pathlib import Path\n\n\ndef test_x(tmp_path: Path) -> None:\n    (tmp_path / 'a').write_text('x')\n"
    path = _write(tmp_path, "tests/unit/test_hermetic.py", source)
    assert scan_hermeticity(Path("x.py"), ast.parse(path.read_text(encoding="utf-8"))) == ()


def test_docstrings_naming_ambient_paths_are_not_escapes(tmp_path: Path) -> None:
    """Prose is not a filesystem read.

    Regression guard: the first implementation flagged every module docstring
    that merely *explained* the ``~/.claude`` precedence ladder it tested.
    """
    source = '"""Explains ~/.codex and /realm/state resolution."""\n\n\ndef test_x() -> None:\n    return None\n'
    path = _write(tmp_path, "tests/unit/test_doc.py", source)
    assert scan_hermeticity(Path("x.py"), ast.parse(path.read_text(encoding="utf-8"))) == ()


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


def test_baseline_entries_are_structured_and_current() -> None:
    """Every baseline entry names a real finding code and an existing file."""
    payload = json.loads((_REPO_ROOT / "docs/plans/oracle-integrity-baseline.json").read_text(encoding="utf-8"))
    entries = payload["entries"]
    assert entries, "an empty baseline should be deleted, not kept"
    known_codes = {"certifies_dead_code", "ambient_path_literal", "ambient_path_call"}
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
