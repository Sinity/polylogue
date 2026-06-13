"""Tests for the public-surface stale-vocabulary audit (#1850).

The audit (``tools/cleanup/polylogue_public_surface_audit.sh``) fails when
stale schema-v1 vocabulary (``conversation`` / ``conversation_id`` /
``provider_meta`` / the legacy ``/c/{id}`` reader route) reappears on a public
surface, while allowing it in provider-wire parsers, internal storage/core,
tests, and historical/tombstone docs.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_SCRIPT = REPO_ROOT / "tools" / "cleanup" / "polylogue_public_surface_audit.sh"

pytestmark = pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep (rg) not available")


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    # The script anchors its scan to its own location's repo root, so a seeded
    # tree must invoke the *copied* script under ``root`` (not the canonical one).
    script = root / "tools" / "cleanup" / AUDIT_SCRIPT.name
    return subprocess.run(
        ["bash", str(script)],
        cwd=root,
        capture_output=True,
        text=True,
    )


def test_audit_script_exists_and_is_executable() -> None:
    assert AUDIT_SCRIPT.is_file(), "audit script must exist at the issue-specified path"


def test_audit_passes_on_current_tree() -> None:
    """The live repository must be clean of stale public-surface vocabulary."""
    result = _run(REPO_ROOT)
    assert result.returncode == 0, f"audit unexpectedly failed:\n{result.stderr}"
    assert "clean" in result.stdout


def _seed_minimal_tree(root: Path) -> None:
    """Create the directory shape the audit scans plus the script itself."""
    (root / "tools" / "cleanup").mkdir(parents=True)
    shutil.copy(AUDIT_SCRIPT, root / "tools" / "cleanup" / AUDIT_SCRIPT.name)
    (root / "docs").mkdir()
    (root / "polylogue" / "daemon").mkdir(parents=True)
    (root / "polylogue" / "sources" / "parsers").mkdir(parents=True)


def test_audit_fails_on_stale_conversation_id_in_public_doc(tmp_path: Path) -> None:
    _seed_minimal_tree(tmp_path)
    (tmp_path / "docs" / "api.md").write_text("Fetch via `conversation_id` query param.\n")
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "conversation_id" in result.stderr
    assert "session_id" in result.stderr  # replacement guidance is named


def test_audit_fails_on_legacy_route_literal(tmp_path: Path) -> None:
    _seed_minimal_tree(tmp_path)
    (tmp_path / "polylogue" / "daemon" / "web.py").write_text("href = '/c/' + sid\n")
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "/s/{session_id}" in result.stderr


def test_audit_fails_on_provider_meta_in_public_surface(tmp_path: Path) -> None:
    _seed_minimal_tree(tmp_path)
    (tmp_path / "docs" / "schema.md").write_text("The `provider_meta` field carries vendor data.\n")
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "provider_meta" in result.stderr


def test_audit_allows_stale_terms_in_provider_wire_parsers(tmp_path: Path) -> None:
    """Provider-wire parsers read external vendor field names verbatim."""
    _seed_minimal_tree(tmp_path)
    (tmp_path / "polylogue" / "sources" / "parsers" / "vendor.py").write_text(
        "conv_id = payload.get('conversation_id')\nprovider_meta = payload.get('provider_meta')\n"
    )
    result = _run(tmp_path)
    assert result.returncode == 0, f"provider-wire paths must be allowlisted:\n{result.stderr}"


def test_audit_honors_inline_allow_marker(tmp_path: Path) -> None:
    """A policed line may opt out when it documents an external vendor format."""
    _seed_minimal_tree(tmp_path)
    (tmp_path / "docs" / "providers-note.md").write_text(
        "Antigravity stores `conversation` blobs.  <!-- polylogue-audit: allow external antigravity format -->\n"
    )
    result = _run(tmp_path)
    assert result.returncode == 0, f"inline allow marker must suppress the hit:\n{result.stderr}"
