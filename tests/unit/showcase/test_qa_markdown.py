"""Tests for stable showcase Markdown reports and artifact manifests."""

from __future__ import annotations

import json
from pathlib import Path

from polylogue.core.json import JSONDocument, JSONDocumentList, json_document_list
from polylogue.showcase.exercise_models import Exercise
from polylogue.showcase.report_files import generate_manifest
from polylogue.showcase.runner import ExerciseResult, ShowcaseResult
from polylogue.showcase.showcase_report_text import generate_showcase_markdown

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exercise(
    name: str = "ex",
    group: str = "structural",
    tier: int = 1,
    description: str = "A test exercise",
) -> Exercise:
    return Exercise(name=name, group=group, tier=tier, description=description, output_ext=".txt")


def _make_result(
    name: str = "ex",
    group: str = "structural",
    passed: bool = True,
    skipped: bool = False,
    error: str | None = None,
    output: str = "output text",
) -> ExerciseResult:
    return ExerciseResult(
        exercise=_make_exercise(name=name, group=group),
        passed=passed,
        exit_code=0 if passed else 1,
        output=output,
        error=error,
        duration_ms=42.0,
        skipped=skipped,
        skip_reason="dep missing" if skipped else None,
    )


def _make_showcase(results: list[ExerciseResult], output_dir: Path | None = None) -> ShowcaseResult:
    sr = ShowcaseResult()
    sr.results = results
    sr.total_duration_ms = sum(r.duration_ms for r in results)
    sr.output_dir = output_dir
    return sr


def _manifest_entries(manifest: JSONDocument) -> JSONDocumentList:
    return json_document_list(manifest["entries"])


# ---------------------------------------------------------------------------
# generate_showcase_markdown tests
# ---------------------------------------------------------------------------


class TestGenerateQaMarkdown:
    """generate_showcase_markdown produces stable, diffable output."""

    def test_produces_markdown_header(self) -> None:
        sr = _make_showcase([_make_result()])
        md = generate_showcase_markdown(sr)
        assert md.startswith("# Showcase QA Session")

    def test_includes_git_sha_when_provided(self) -> None:
        sr = _make_showcase([_make_result()])
        md = generate_showcase_markdown(sr, git_sha="abc123")
        assert "`abc123`" in md

    def test_no_git_sha_when_not_provided(self) -> None:
        sr = _make_showcase([_make_result()])
        md = generate_showcase_markdown(sr)
        assert "Git SHA" not in md

    def test_no_timestamps_in_body(self) -> None:
        sr = _make_showcase(
            [
                _make_result(passed=True),
                _make_result(name="f", passed=False, error="boom"),
            ]
        )
        md = generate_showcase_markdown(sr)
        # No ISO timestamp patterns in the body
        import re

        # Match typical ISO timestamps like 2026-03-15T12:34:56
        timestamps = re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", md)
        assert len(timestamps) == 0, f"Found timestamps in body: {timestamps}"

    def test_summary_table_present(self) -> None:
        sr = _make_showcase(
            [
                _make_result(passed=True),
                _make_result(name="f", passed=False, error="err"),
            ]
        )
        md = generate_showcase_markdown(sr)
        assert "## Summary" in md
        assert "| Total | 2 |" in md
        assert "| Passed | 1 |" in md
        assert "| Failed | 1 |" in md

    def test_pass_fail_skip_markers(self) -> None:
        sr = _make_showcase(
            [
                _make_result(name="p", passed=True),
                _make_result(name="f", passed=False, error="err"),
                _make_result(name="s", skipped=True),
            ]
        )
        md = generate_showcase_markdown(sr)
        assert "[PASS]" in md
        assert "[FAIL]" in md
        assert "[SKIP]" in md

    def test_failed_exercise_shows_error(self) -> None:
        sr = _make_showcase(
            [
                _make_result(name="bad", passed=False, error="exit code 1, expected 0"),
            ]
        )
        md = generate_showcase_markdown(sr)
        assert "exit code 1, expected 0" in md

    def test_failed_exercise_shows_truncated_output(self) -> None:
        long_output = "\n".join(f"line {i}" for i in range(20))
        sr = _make_showcase(
            [
                _make_result(name="verbose", passed=False, error="failed", output=long_output),
            ]
        )
        md = generate_showcase_markdown(sr)
        assert "line 0" in md
        assert "line 9" in md
        assert "10 more lines" in md

    def test_exercises_grouped_by_group(self) -> None:
        sr = _make_showcase(
            [
                _make_result(name="a", group="structural"),
                _make_result(name="b", group="query-read"),
            ]
        )
        md = generate_showcase_markdown(sr)
        assert "### structural" in md
        assert "### query-read" in md

    def test_stable_output_across_calls(self) -> None:
        """Same input produces identical output (no randomness/timestamps)."""
        results = [
            _make_result(name="a", passed=True),
            _make_result(name="b", passed=False, error="err"),
        ]
        md1 = generate_showcase_markdown(_make_showcase(results), git_sha="abc")
        md2 = generate_showcase_markdown(_make_showcase(results), git_sha="abc")
        assert md1 == md2


# ---------------------------------------------------------------------------
# generate_manifest tests
# ---------------------------------------------------------------------------


class TestGenerateManifest:
    """generate_manifest produces expected structure."""

    def test_includes_expected_keys(self) -> None:
        sr = _make_showcase([])
        manifest = generate_manifest(sr)
        assert "schema_version" in manifest
        assert "entry_count" in manifest
        assert "entries" in manifest

    def test_schema_version_is_1(self) -> None:
        sr = _make_showcase([])
        manifest = generate_manifest(sr)
        assert manifest["schema_version"] == 1

    def test_empty_when_no_output_dir(self) -> None:
        sr = _make_showcase([], output_dir=None)
        manifest = generate_manifest(sr)
        assert manifest["entry_count"] == 0
        assert manifest["entries"] == []

    def test_captures_files_with_hashes(self, tmp_path: Path) -> None:
        # Create some files in the output dir
        (tmp_path / "report.json").write_text('{"a": 1}')
        (tmp_path / "summary.txt").write_text("pass")

        sr = _make_showcase([], output_dir=tmp_path)
        manifest = generate_manifest(sr, include_hashes=True)

        assert manifest["entry_count"] == 2
        for entry in _manifest_entries(manifest):
            assert "relative_path" in entry
            assert "size_bytes" in entry
            assert "sha256" in entry
            sha256 = entry["sha256"]
            assert isinstance(sha256, str)
            assert len(sha256) == 64  # SHA-256 hex length

    def test_without_hashes(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("hello")

        sr = _make_showcase([], output_dir=tmp_path)
        manifest = generate_manifest(sr, include_hashes=False)

        assert manifest["entry_count"] == 1
        assert "sha256" not in _manifest_entries(manifest)[0]

    def test_manifest_serializes_to_json(self, tmp_path: Path) -> None:
        (tmp_path / "data.json").write_text("{}")

        sr = _make_showcase([], output_dir=tmp_path)
        manifest = generate_manifest(sr)
        raw = json.dumps(manifest, indent=2)
        loaded = json.loads(raw)
        assert loaded["entry_count"] == manifest["entry_count"]
