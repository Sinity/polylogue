from __future__ import annotations

from pathlib import Path

from devtools.gate import GATES_BY_NAME
from devtools.verify_atlas import Finding, declared_gate_names, inspect


def _write_atlas(root: Path, *, citation: str) -> None:
    page = root / "docs" / "atlas"
    page.mkdir(parents=True)
    (page / "example.md").write_text(
        f"# Example\n\n## Evidence\n\nSee `{citation}`.\n",
        encoding="utf-8",
    )


def test_inspect_reports_a_citation_whose_file_is_gone(tmp_path: Path) -> None:
    # Anti-vacuity: drop the `cited.is_file()` refusal in ``inspect`` and this
    # goes red -- a citation naming a deleted file would leave the queue empty.
    (tmp_path / "README").write_text("fixture\n", encoding="utf-8")
    _write_atlas(tmp_path, citation="missing.py:2")
    findings = inspect(tmp_path)
    assert Finding("docs/atlas/example.md", "Evidence", "missing-file", "missing.py") in findings


def test_inspect_reports_a_line_range_past_end_of_file(tmp_path: Path) -> None:
    # Anti-vacuity: widen the bounds check to accept any end line and this goes
    # red -- a citation pointing past EOF would be reported as resolvable.
    (tmp_path / "source.py").write_text("line\n", encoding="utf-8")
    _write_atlas(tmp_path, citation="source.py:1-9")
    findings = inspect(tmp_path)
    assert [finding.kind for finding in findings] == ["missing-anchor"]
    assert "file has 1 lines" in findings[0].detail


def test_inspect_accepts_a_citation_that_resolves(tmp_path: Path) -> None:
    # Anti-vacuity: report every citation unconditionally and this goes red --
    # the gate would block on correct anchors, which is what made the stale
    # check unusable.
    (tmp_path / "source.py").write_text("line\n", encoding="utf-8")
    _write_atlas(tmp_path, citation="source.py:1")
    assert inspect(tmp_path) == []


def test_inspect_does_not_require_a_verification_stamp(tmp_path: Path) -> None:
    # Anti-vacuity: reinstate a ``missing-verification`` finding for pages with
    # no ``verified:`` footer and this goes red. The stamp is gone: an anchor
    # that resolves is evidence on its own, and the stamp could never be
    # accurate in the PR that moved the cited code.
    (tmp_path / "source.py").write_text("line\n", encoding="utf-8")
    _write_atlas(tmp_path, citation="source.py:1")
    assert inspect(tmp_path) == []
    assert "verified:" not in (tmp_path / "docs" / "atlas" / "example.md").read_text(encoding="utf-8")


def _unknown_gate_name() -> str:
    """A name proven absent from the live registry, not assumed absent."""

    candidate = "no-such-gate"
    assert candidate not in GATES_BY_NAME
    return candidate


def _known_gate_name() -> str:
    """A name taken from the live registry, so this test cannot name a retired gate."""

    return sorted(GATES_BY_NAME)[0]


def _write_owner_sheet(root: Path, *, declaration: str) -> None:
    page = root / "docs" / "atlas"
    page.mkdir(parents=True, exist_ok=True)
    (page / "doctrine.md").write_text(
        f"# Doctrine\n\n## Time\n\n**Owning gate**: {declaration}\n",
        encoding="utf-8",
    )


def test_inspect_reports_an_owning_gate_absent_from_the_registry(tmp_path: Path) -> None:
    # Anti-vacuity: drop the ``name not in GATES_BY_NAME`` resolution from
    # ``inspect`` and this goes red -- a doctrine naming a gate that does not
    # exist would leave the queue empty, which is the whole defect (cpf.9).
    unknown = _unknown_gate_name()
    _write_owner_sheet(tmp_path, declaration=f"`{unknown}`")
    findings = inspect(tmp_path)
    assert Finding("docs/atlas/doctrine.md", "Time", "unknown-gate", unknown) in findings


def test_inspect_accepts_an_owning_gate_that_resolves(tmp_path: Path) -> None:
    # Anti-vacuity: report every declaration unconditionally and this goes red
    # -- the gate would block on a correctly declared owner.
    _write_owner_sheet(tmp_path, declaration=f"`{_known_gate_name()}`")
    assert inspect(tmp_path) == []


def test_inspect_accepts_the_runnable_spelling_of_a_declared_gate(tmp_path: Path) -> None:
    # Anti-vacuity: stop stripping the ``devtools gate`` prefix and this goes
    # red -- the whole command string would be looked up as a gate name.
    _write_owner_sheet(tmp_path, declaration=f"`devtools gate {_known_gate_name()}`")
    assert inspect(tmp_path) == []
    _write_owner_sheet(tmp_path, declaration=f"`devtools gate {_unknown_gate_name()}`")
    assert [finding.kind for finding in inspect(tmp_path)] == ["unknown-gate"]


def test_inspect_accepts_an_explicit_no_gate_declaration(tmp_path: Path) -> None:
    # Anti-vacuity: treat ``none`` as a gate name and this goes red -- an
    # invariant with no gate could not state that honestly.
    _write_owner_sheet(tmp_path, declaration="none -- enforced at the write boundary")
    assert inspect(tmp_path) == []


def test_inspect_refuses_a_declaration_it_cannot_parse(tmp_path: Path) -> None:
    # Anti-vacuity: return an empty name list instead of ``None`` for an
    # unrecognised value and this goes red -- prose naming an owner would pass
    # unchecked, which is exactly the bypass a marker form must not leave open.
    _write_owner_sheet(tmp_path, declaration="the layering gate, probably")
    findings = inspect(tmp_path)
    assert [finding.kind for finding in findings] == ["unparsable-owning-gate"]
    assert findings[0].section == "Time"


def test_inspect_refuses_an_unbackticked_name_beside_a_resolvable_one(tmp_path: Path) -> None:
    # Anti-vacuity: drop the leftover-text check in ``declared_gate_names`` and
    # this goes red -- an unchecked name could hide next to a real gate.
    _write_owner_sheet(tmp_path, declaration=f"`{_known_gate_name()}` and writer-ownership")
    assert [finding.kind for finding in inspect(tmp_path)] == ["unparsable-owning-gate"]


def test_declared_gate_names_parses_a_list(tmp_path: Path) -> None:
    known = _known_gate_name()
    assert declared_gate_names(f"`{known}`, `atlas`") == (known, "atlas")
    assert declared_gate_names("none") == ()
    assert declared_gate_names("layering") is None
