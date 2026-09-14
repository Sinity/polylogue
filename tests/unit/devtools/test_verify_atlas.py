from __future__ import annotations

from pathlib import Path

from devtools.verify_atlas import Finding, inspect


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
