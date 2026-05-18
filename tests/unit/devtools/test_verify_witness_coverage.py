"""Tests for ``devtools/verify_witness_coverage.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from devtools import verify_witness_coverage as vwc


def _pr(
    number: int,
    *,
    title: str = "fix: handle null cursor",
    files: list[str] | None = None,
    labels: list[str] | None = None,
    merged_at: str = "2026-05-01T12:00:00Z",
) -> dict[str, Any]:
    return {
        "number": number,
        "title": title,
        "mergedAt": merged_at,
        "url": f"https://github.com/example/repo/pull/{number}",
        "labels": [{"name": label} for label in (labels or [])],
        "files": [{"path": p} for p in (files or [])],
    }


class TestIsFixPR:
    def test_conventional_fix_subject_matches(self) -> None:
        assert vwc._is_fix_pr("fix: pagination off-by-one", [])

    def test_scoped_fix_subject_matches(self) -> None:
        assert vwc._is_fix_pr("fix(cli): broken --json output", [])

    def test_non_fix_subject_does_not_match(self) -> None:
        assert not vwc._is_fix_pr("feat: add new flag", [])

    def test_bug_label_matches_even_without_prefix(self) -> None:
        assert vwc._is_fix_pr("refactor: tidy up", ["bug"])

    def test_label_match_is_case_insensitive(self) -> None:
        assert vwc._is_fix_pr("chore: tweak", ["Bug"])


class TestAuditPRs:
    def test_flags_fix_pr_touching_source_without_witness(self) -> None:
        prs = [
            _pr(
                101,
                files=["polylogue/cli/query.py"],
            )
        ]
        result = vwc.audit_prs(prs, suppressions=set(), window_days=90)
        assert result.examined == 1
        assert len(result.flagged) == 1
        flagged = result.flagged[0]
        assert flagged.number == 101
        assert flagged.source_files == ("polylogue/cli/query.py",)

    def test_does_not_flag_when_witness_added(self) -> None:
        prs = [
            _pr(
                102,
                files=[
                    "polylogue/cli/query.py",
                    "tests/witnesses/query-null.witness.json",
                ],
            )
        ]
        result = vwc.audit_prs(prs, suppressions=set(), window_days=90)
        assert result.examined == 1
        assert result.flagged == ()

    def test_does_not_flag_when_no_source_touched(self) -> None:
        prs = [_pr(103, files=["docs/foo.md", "tests/unit/test_x.py"])]
        result = vwc.audit_prs(prs, suppressions=set(), window_days=90)
        assert result.examined == 1
        assert result.flagged == ()

    def test_ignores_non_fix_prs(self) -> None:
        prs = [
            _pr(
                104,
                title="feat: new endpoint",
                files=["polylogue/cli/query.py"],
            )
        ]
        result = vwc.audit_prs(prs, suppressions=set(), window_days=90)
        assert result.examined == 0
        assert result.flagged == ()

    def test_suppressions_skip_pr_before_examination(self) -> None:
        prs = [_pr(105, files=["polylogue/cli/query.py"])]
        result = vwc.audit_prs(prs, suppressions={105}, window_days=90)
        assert result.examined == 0
        assert result.flagged == ()
        assert result.suppressed == (105,)

    def test_bug_labeled_pr_without_fix_prefix_is_audited(self) -> None:
        prs = [
            _pr(
                106,
                title="refactor: rework storage layer",
                files=["polylogue/storage/sqlite/schema.py"],
                labels=["bug"],
            )
        ]
        result = vwc.audit_prs(prs, suppressions=set(), window_days=90)
        assert result.examined == 1
        assert len(result.flagged) == 1
        assert result.flagged[0].number == 106

    def test_unknown_field_shapes_are_tolerated(self) -> None:
        prs: list[dict[str, Any]] = [
            {"number": "not-an-int"},
            {"number": 107, "title": None, "labels": None, "files": None},
        ]
        result = vwc.audit_prs(prs, suppressions=set(), window_days=90)
        assert result.examined == 0
        assert result.flagged == ()


class TestLoadSuppressions:
    def test_returns_empty_set_when_file_missing(self, tmp_path: Path) -> None:
        assert vwc.load_suppressions(tmp_path / "nope.yaml") == set()

    def test_parses_pr_numbers_from_dict_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "supp.yaml"
        path.write_text(
            "suppressions:\n"
            "  - pr: 42\n"
            "    reason: doc-only fix mislabeled\n"
            "  - pr: 99\n"
            "    reason: covered by contract test\n"
        )
        assert vwc.load_suppressions(path) == {42, 99}

    def test_accepts_bare_int_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "supp.yaml"
        path.write_text("suppressions:\n  - 7\n  - 8\n")
        assert vwc.load_suppressions(path) == {7, 8}

    def test_malformed_yaml_returns_empty_set(self, tmp_path: Path) -> None:
        path = tmp_path / "supp.yaml"
        path.write_text("suppressions: [unterminated\n")
        assert vwc.load_suppressions(path) == set()


class TestMain:
    def test_main_skips_cleanly_when_gh_missing_in_soft_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(vwc, "_gh_available", lambda: False)
        rc = vwc.main(["--soft", "--json"])
        assert rc == 0

    def test_main_fails_when_gh_missing_in_strict_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(vwc, "_gh_available", lambda: False)
        rc = vwc.main(["--json"])
        assert rc == 1

    def test_main_flags_missing_witness_in_strict_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(vwc, "_gh_available", lambda: True)
        monkeypatch.setattr(
            vwc,
            "fetch_merged_prs",
            lambda *, since_iso, limit: [_pr(201, files=["polylogue/x.py"])],
        )
        monkeypatch.setattr(vwc, "load_suppressions", lambda: set())
        rc = vwc.main(["--json"])
        assert rc == 1

    def test_main_returns_zero_when_no_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(vwc, "_gh_available", lambda: True)
        monkeypatch.setattr(
            vwc,
            "fetch_merged_prs",
            lambda *, since_iso, limit: [
                _pr(
                    202,
                    files=[
                        "polylogue/x.py",
                        "tests/witnesses/x.witness.json",
                    ],
                )
            ],
        )
        monkeypatch.setattr(vwc, "load_suppressions", lambda: set())
        rc = vwc.main(["--json"])
        assert rc == 0

    def test_main_returns_zero_in_soft_mode_even_when_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(vwc, "_gh_available", lambda: True)
        monkeypatch.setattr(
            vwc,
            "fetch_merged_prs",
            lambda *, since_iso, limit: [_pr(203, files=["polylogue/x.py"])],
        )
        monkeypatch.setattr(vwc, "load_suppressions", lambda: set())
        rc = vwc.main(["--soft", "--json"])
        assert rc == 0

    def test_main_handles_gh_runtime_error_strict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(vwc, "_gh_available", lambda: True)

        def _raise(*, since_iso: str, limit: int) -> list[dict[str, Any]]:
            raise RuntimeError("gh exploded")

        monkeypatch.setattr(vwc, "fetch_merged_prs", _raise)
        monkeypatch.setattr(vwc, "load_suppressions", lambda: set())
        rc = vwc.main(["--json"])
        assert rc == 1


class TestCommandCatalogRegistration:
    def test_verify_witness_coverage_command_is_registered(self) -> None:
        from devtools.command_catalog import COMMANDS

        assert "verify-witness-coverage" in COMMANDS
        spec = COMMANDS["verify-witness-coverage"]
        assert spec.module == "devtools.verify_witness_coverage"
        assert spec.category == "verification"
