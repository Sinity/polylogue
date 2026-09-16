"""polylogue-n61h5: a committed package must describe its own subject."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from polylogue.schemas.provider_identity_audit import audit_committed_provider_identity


def _write_element(root: Path, package: str, version: str, subject: str) -> Path:
    element_dir = root / package / "versions" / version / "elements"
    element_dir.mkdir(parents=True, exist_ok=True)
    path = element_dir / "session_document.schema.json.gz"
    document = {
        "$id": f"polylogue://schemas/{subject}/{version}/session_document",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(document, handle)
    return path


class TestAuditCommittedProviderIdentity:
    def test_matching_subject_passes(self, tmp_path: Path) -> None:
        _write_element(tmp_path, "grok", "v1", "grok")

        report = audit_committed_provider_identity(tmp_path)

        assert report.elements_checked == 1
        assert report.all_passed is True

    def test_cross_provider_swap_is_refused(self, tmp_path: Path) -> None:
        """Anti-vacuity: this is the exact shape that shipped -- a claude-ai
        element under ``providers/grok``. Remove the ``$id`` comparison in
        ``audit_committed_provider_identity`` and this goes green.
        """
        _write_element(tmp_path, "grok", "v2", "claude-ai")

        report = audit_committed_provider_identity(tmp_path)

        assert report.all_passed is False
        assert [v.declared_subject for v in report.violations] == ["claude-ai"]
        assert report.violations[0].package_dir == "grok"
        assert "raises no drift" in report.format_text()

    def test_a_non_default_version_is_audited_too(self, tmp_path: Path) -> None:
        """The registry resolves one default version per package; the audit
        reads the tree, so a mislabelled older version cannot hide behind a
        correct default."""
        _write_element(tmp_path, "chatgpt", "v1", "claude-ai")
        _write_element(tmp_path, "chatgpt", "v2", "chatgpt")

        report = audit_committed_provider_identity(tmp_path)

        assert report.elements_checked == 2
        assert len(report.violations) == 1
        assert "v1" in report.violations[0].element_path


class TestCommittedTreeIsClean:
    def test_the_real_committed_packages_name_their_own_subjects(self) -> None:
        report = audit_committed_provider_identity()

        assert report.elements_checked > 0
        assert report.all_passed is True, report.format_text()
