from __future__ import annotations

from pathlib import Path

from devtools.gate import GATES_BY_NAME
from devtools.verify_atlas import Finding, audit, declared_gate_names, inspect


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


#: ``anchored_owner`` is declared at line 10 and its body runs to line 16, far
#: enough from lines 1-3 that the slack window cannot reach it.  The padding is
#: load-bearing, so ``_SOURCE_LAYOUT`` proves it instead of trusting this note.
_SOURCE = """\
def unrelated_helper() -> int:
    value = 0
    return value






def anchored_owner() -> str:
    first = "a"
    second = "b"
    third = "c"
    fourth = "d"
    fifth = "e"
    return first + second + third + fourth + fifth
"""
_SOURCE_LAYOUT = _SOURCE.splitlines()
assert _SOURCE_LAYOUT[9].startswith("def anchored_owner"), "the owner must sit on line 10"
assert len(_SOURCE_LAYOUT) == 16, "the owner's body must end on line 16"
assert not any("anchored_owner" in line for line in _SOURCE_LAYOUT[10:]), "the body must not repeat the name"


def _write_claim_page(root: Path, *, prose: str) -> None:
    page = root / "docs" / "atlas"
    page.mkdir(parents=True, exist_ok=True)
    (root / "source.py").write_text(_SOURCE, encoding="utf-8")
    (page / "example.md").write_text(f"# Example\n\n## Evidence\n\n{prose}\n", encoding="utf-8")


def test_reports_an_anchor_missing_the_symbol_prose_names(tmp_path: Path) -> None:
    # Anti-vacuity: delete the ``misaimed-anchor`` branch from ``audit`` and
    # this goes red. ``source.py:1-3`` is inside the file and inside its line
    # count, so both older refusals pass it -- that is exactly the anchor the
    # gate certified without checking (polylogue-72cbt).
    _write_claim_page(tmp_path, prose="`anchored_owner` joins the parts (`source.py:1-3`).")
    findings = inspect(tmp_path)
    assert [finding.kind for finding in findings] == ["misaimed-anchor"]
    assert "anchored_owner" in findings[0].detail


def test_accepts_an_anchor_containing_the_symbol_prose_names(tmp_path: Path) -> None:
    # Anti-vacuity: refuse every citation carrying a symbol claim and this goes
    # red. A blanket refusal would "catch" every misaimed anchor and be useless.
    _write_claim_page(tmp_path, prose="`anchored_owner` joins the parts (`source.py:10-11`).")
    assert inspect(tmp_path) == []


def test_accepts_a_range_inside_the_definition_prose_names(tmp_path: Path) -> None:
    # Anti-vacuity: drop the definition-span fallback in ``_symbol_is_anchored``
    # and this goes red -- ``source.py:15-16`` is the tail of ``anchored_owner``
    # and never repeats the name, which is the ordinary shape of a citation
    # into a method body.
    _write_claim_page(tmp_path, prose="`anchored_owner` joins the parts (`source.py:15-16`).")
    assert inspect(tmp_path) == []


def test_does_not_refuse_a_citation_that_names_no_symbol(tmp_path: Path) -> None:
    # Anti-vacuity: treat a claimless citation as misaimed and this goes red.
    # Most atlas prose names no backticked identifier; refusing those would
    # make the gate unusable rather than more honest.
    _write_claim_page(tmp_path, prose="The helper returns early (`source.py:1-3`).")
    assert inspect(tmp_path) == []


def test_coverage_counts_an_unexamined_citation(tmp_path: Path) -> None:
    # Anti-vacuity: fold the unchecked bundles into ``checked`` (or drop the
    # counters) and this goes red. Reporting success over citations the check
    # never examined is the defect this check exists to remove, so the gate has
    # to state its own coverage.
    _write_claim_page(
        tmp_path,
        prose=(
            "`anchored_owner` joins the parts (`source.py:10-11`).\n\n"
            "The helper returns early (`source.py:1-3`).\n\n"
            "`absent_symbol` is declared elsewhere (`source.py:1-3`).\n"
        ),
    )
    coverage = audit(tmp_path).coverage
    assert coverage.citations == 3
    assert coverage.checked == 1
    assert coverage.misaimed == 0
    assert coverage.no_symbol_claim == 1
    assert coverage.claim_absent_from_target == 1
    assert coverage.unchecked == 2
    assert "1 carry a checkable symbol claim" in coverage.summary()
    assert "1 name no backticked symbol" in coverage.summary()


def test_accepts_a_bundle_anchored_in_one_of_its_ranges(tmp_path: Path) -> None:
    # Anti-vacuity: require the symbol in *every* cited range and this goes red.
    # A run of citations is one evidence bundle for one sentence, and its
    # clauses are routinely anchored separately.
    _write_claim_page(tmp_path, prose="`anchored_owner` joins the parts (`source.py:1-3`; `source.py:10-11`).")
    assert inspect(tmp_path) == []
