from __future__ import annotations

from pathlib import Path

from devtools.verify_atlas import Finding, inspect


def _write_atlas(root: Path, *, citation: str, stamp: str) -> None:
    page = root / "docs" / "atlas"
    page.mkdir(parents=True)
    (page / "example.md").write_text(
        f"# Example\n\n## Evidence\n\nSee `{citation}`.\n\nverified: {stamp} 2026-08-25\n",
        encoding="utf-8",
    )


def test_inspect_reports_missing_file_and_anchor(tmp_path: Path, monkeypatch) -> None:
    # The test would pass vacuously if citation parsing or anchor bounds were
    # removed: both malformed references must enter the re-verification queue.
    (tmp_path / "README").write_text("fixture\n", encoding="utf-8")
    _write_atlas(tmp_path, citation="missing.py:2", stamp="a" * 40)
    monkeypatch.setattr("devtools.verify_atlas._git", lambda *_args: (0, "HEAD", ""))
    findings = inspect(tmp_path)
    assert Finding("docs/atlas/example.md", "Evidence", "missing-file", "missing.py") in findings


def test_inspect_reports_stale_cited_file(tmp_path: Path, monkeypatch) -> None:
    cited = tmp_path / "source.py"
    cited.write_text("line\n", encoding="utf-8")
    _write_atlas(tmp_path, citation="source.py:1", stamp="a" * 40)
    calls: list[tuple[str, ...]] = []

    def fake_git(_root: Path, *args: str):
        calls.append(args)
        if args[:2] == ("cat-file", "-e"):
            return 0, "", ""
        return 0, "changed-commit", ""

    monkeypatch.setattr("devtools.verify_atlas._git", fake_git)
    findings = inspect(tmp_path)
    assert any(finding.kind == "stale" for finding in findings)
    assert any(args[0] == "log" for args in calls)
