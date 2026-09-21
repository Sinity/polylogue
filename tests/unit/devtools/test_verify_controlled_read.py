"""The controlled-read census must refuse an archive open nobody justified.

``polylogue-vclez`` claimed "six production ``ArchiveStore.open_existing()``
reads outside the controlled boundary" and no artifact said *which* call sites
those were.  The population is not small and it grows: 26 direct opens at this
head.  A prose list of them would be wrong the first time somebody adds one,
which is why the rule is a gate over a checked-in declaration rather than a
paragraph.

Anti-vacuity, stated as the mutations that must turn these tests red:

* Stop reporting undeclared sites and
  ``test_an_undeclared_open_is_reported_by_file_and_line`` goes green -- that
  is the whole census, and it is the mutation the bead names.
* Take ``classification: read_boundary`` on trust rather than checking it
  against ``read_boundary_owners`` and
  ``test_a_new_module_cannot_declare_itself_the_read_boundary`` goes green: any
  new uncontrolled read could then license itself by claiming to be the
  boundary.
* Drop the explicit-write-mode check and
  ``test_a_writer_that_lost_its_explicit_write_mode_is_reported`` goes green.
  That is the realistic silent regression: the declaration still reads
  "writer" while the call became a read with no admission, snapshot pin,
  cancellation or receipt.
* Let the declaration keep entries that no longer reproduce and
  ``test_a_declaration_that_no_longer_reproduces_is_reported`` goes green,
  which is how a ratchet stops ratcheting.
* Delete the "a boundary owner must exist" guard and
  ``test_a_tree_with_no_read_boundary_at_all_is_reported`` goes green over a
  tree whose controlled reader was deleted outright -- a clean census with no
  boundary left.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from devtools.gate import GATES_BY_NAME
from devtools.verify_controlled_read import (
    CLASSIFICATION_VOCABULARY,
    DECLARATION_PATH,
    WRITER_ROLE_VOCABULARY,
    collect_violations,
    load_declaration,
    main,
    observe_sites,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

_BOUNDARY_MODULE = '''
"""Synthetic read control plane."""

from store import ArchiveStore


class ReadControl:
    def run(self, root):
        store = ArchiveStore.open_existing(root)
        store.begin_read_snapshot()
        return store
'''

_WRITER_MODULE = '''
"""Synthetic writer-lease open."""

from store import ArchiveStore


def apply_mutation(root):
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        return archive.delete_sessions(())
'''

_UNCONTROLLED_MODULE = '''
"""Synthetic surface reading the archive with no control plane."""

from store import ArchiveStore


def count_rows(root):
    with ArchiveStore.open_existing(root) as archive:
        return archive.count_sessions()
'''


def _declaration(
    sites: list[dict[str, object]],
    *,
    owners: list[str] | None = None,
) -> str:
    import yaml

    return yaml.safe_dump(
        {
            "package": "pkg",
            "read_boundary_owners": owners if owners is not None else ["pkg/control.py"],
            "sites": sites,
        }
    )


def _tree(tmp_path: Path, modules: dict[str, str]) -> Path:
    for relative, body in modules.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return tmp_path


def _base_modules() -> dict[str, str]:
    return {
        "pkg/__init__.py": "",
        "pkg/control.py": _BOUNDARY_MODULE,
        "pkg/writer.py": _WRITER_MODULE,
    }


def _boundary_site() -> dict[str, object]:
    return {
        "site": "pkg/control.py::ReadControl.run",
        "classification": "read_boundary",
        "reason": "pins the read snapshot and installs the cancellation guard",
    }


def _writer_site() -> dict[str, object]:
    return {
        "site": "pkg/writer.py::apply_mutation",
        "classification": "writer",
        "role": "daemon_write",
        "reason": "applies a declared mutation under the writer lease",
    }


def _run(
    tmp_path: Path,
    modules: dict[str, str],
    sites: list[dict[str, object]],
    *,
    owners: list[str] | None = None,
) -> list[dict[str, object]]:
    root = _tree(tmp_path, modules)
    declaration = root / "census.yaml"
    declaration.write_text(_declaration(sites, owners=owners), encoding="utf-8")
    violations, _observed, _declaration_obj = collect_violations(repo_root=root, declaration_path=declaration)
    return violations


def _rules(violations: list[dict[str, object]]) -> set[str]:
    return {str(violation["rule"]) for violation in violations}


def test_gate_is_registered_against_its_module() -> None:
    gate = GATES_BY_NAME["controlled-read"]
    assert gate.args == ("devtools.verify_controlled_read", "--json")
    assert gate.kind == "module"
    assert gate.in_quick is True
    assert gate.blocking is True


def test_a_fully_declared_tree_reports_nothing(tmp_path: Path) -> None:
    assert _run(tmp_path, _base_modules(), [_boundary_site(), _writer_site()]) == []


def test_an_undeclared_open_is_reported_by_file_and_line(tmp_path: Path) -> None:
    """The bead's own anti-vacuity: one unjustified call site makes the gate red."""

    modules = {**_base_modules(), "pkg/surface.py": _UNCONTROLLED_MODULE}
    violations = _run(tmp_path, modules, [_boundary_site(), _writer_site()])

    assert _rules(violations) == {"controlled_read_site_undeclared"}
    reported = violations[0]
    assert reported["site"] == "pkg/surface.py::count_rows"
    # A census whose verdict is "it exists" is not a finding; this one names
    # the file and the line, which is what makes it auditable.
    assert reported["file"] == "pkg/surface.py:8"


def test_a_new_module_cannot_declare_itself_the_read_boundary(tmp_path: Path) -> None:
    """Which module owns a controlled read is policy, not a census row's claim."""

    modules = {**_base_modules(), "pkg/surface.py": _UNCONTROLLED_MODULE}
    violations = _run(
        tmp_path,
        modules,
        [
            _boundary_site(),
            _writer_site(),
            {
                "site": "pkg/surface.py::count_rows",
                "classification": "read_boundary",
                "reason": "it reads, so surely it is the reader",
            },
        ],
    )

    assert _rules(violations) == {"controlled_read_boundary_not_an_owner"}


def test_a_writer_that_lost_its_explicit_write_mode_is_reported(tmp_path: Path) -> None:
    """A declared writer that opens read-only is an uncontrolled read."""

    modules = dict(_base_modules())
    modules["pkg/writer.py"] = _WRITER_MODULE.replace(", read_only=False", "")
    violations = _run(tmp_path, modules, [_boundary_site(), _writer_site()])

    assert _rules(violations) == {"controlled_read_writer_is_not_explicit"}
    assert violations[0]["file"] == "pkg/writer.py:8"


def test_a_declaration_that_no_longer_reproduces_is_reported(tmp_path: Path) -> None:
    violations = _run(
        tmp_path,
        _base_modules(),
        [
            _boundary_site(),
            _writer_site(),
            {
                "site": "pkg/gone.py::vanished",
                "classification": "writer",
                "role": "daemon_write",
                "reason": "deleted in a previous change",
            },
        ],
    )

    assert _rules(violations) == {"controlled_read_census_stale"}


def test_a_tree_with_no_read_boundary_at_all_is_reported(tmp_path: Path) -> None:
    """A census over a tree whose controlled reader is gone must not read clean."""

    modules = {key: value for key, value in _base_modules().items() if key != "pkg/control.py"}
    violations = _run(tmp_path, modules, [_writer_site()], owners=[])

    assert "controlled_read_boundary_absent" in _rules(violations)


def test_an_unknown_classification_or_role_is_reported(tmp_path: Path) -> None:
    violations = _run(
        tmp_path,
        _base_modules(),
        [
            _boundary_site(),
            {**_writer_site(), "role": "because_i_said_so"},
        ],
    )

    assert _rules(violations) == {"controlled_read_unknown_writer_role"}

    violations = _run(
        tmp_path,
        _base_modules(),
        [_boundary_site(), {**_writer_site(), "classification": "probably_fine"}],
    )
    assert _rules(violations) == {"controlled_read_unknown_classification"}


def test_a_declared_owner_path_that_no_longer_exists_is_reported(tmp_path: Path) -> None:
    """A rename must not silently empty the policy half of the declaration."""

    violations = _run(
        tmp_path,
        _base_modules(),
        [_boundary_site(), _writer_site()],
        owners=["pkg/control.py", "pkg/renamed_away.py"],
    )

    assert "controlled_read_owner_missing" in _rules(violations)


def test_the_real_census_is_clean_and_names_its_two_boundary_owners() -> None:
    """The census at this head, as an assertion rather than a claim in a bead.

    Anti-vacuity: the observed-count assertion -- a census over an empty
    population would report no violations for the wrong reason.
    """

    declaration = load_declaration(REPO_ROOT / DECLARATION_PATH)
    observed = observe_sites(REPO_ROOT, package=declaration.package)
    violations, _observed, _declaration = collect_violations(repo_root=REPO_ROOT)

    assert violations == [], violations
    assert len(observed) >= 20, "the census must cover the real population, not an empty tree"
    assert set(declaration.owners) == {
        "polylogue/archive/query/execution_control.py",
        "polylogue/operations/operation_context.py::open_operation_read",
    }
    classifications = {entry.classification for entry in declaration.entries.values()}
    assert classifications <= set(CLASSIFICATION_VOCABULARY)
    roles = {entry.role for entry in declaration.entries.values() if entry.role}
    assert roles <= set(WRITER_ROLE_VOCABULARY)


def test_the_gate_prints_the_census_and_the_json_payload(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The census is an output, not only a pass/fail bit."""

    assert main(["--json"]) == 0
    payload = cast(dict[str, object], json.loads(capsys.readouterr().out))
    assert payload["count"] == 0
    assert payload["violations"] == []
    assert len(cast(list[str], payload["read_boundary"])) == 3

    assert main(["--census"]) == 0
    census = capsys.readouterr().out
    assert "polylogue/operations/operation_context.py:" in census
    assert "read_boundary" in census
    assert "writer/daemon_write" in census
