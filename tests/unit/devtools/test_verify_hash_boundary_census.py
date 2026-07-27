"""Tests for the hash-boundary registry lint (polylogue-okpn).

The lint is the register-or-fail mechanism the 2026-07-09 hash-boundary
census (docs/audits/2026-07-09-hash-boundary-census.md) deferred as a
follow-up: a new hashlib.*/core.hashing call site must be registered in
docs/plans/hash-boundary-registry.yaml, or the gate fails. These tests prove
it catches a new unregistered call site, accepts a registered one, and
rejects a stale registry entry, using synthetic fixture trees (not the real
repo, so a future edit to real source can't accidentally make this test
vacuous).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from devtools import verify_hash_boundary_census


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def _run_json(root: Path, capsys: pytest.CaptureFixture[str], *, registry: Path | None = None) -> dict[str, Any]:
    args = ["--json", "--root", str(root)]
    if registry is not None:
        args += ["--registry", str(registry)]
    rc = verify_hash_boundary_census.main(args)
    payload: dict[str, Any] = json.loads(capsys.readouterr().out)
    payload["_rc"] = rc
    return payload


def test_flags_new_unregistered_hashlib_call(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(
        tmp_path,
        "polylogue/storage/example_write.py",
        """
        import hashlib


        def compute_digest(value):
            return hashlib.sha256(value.encode()).hexdigest()
        """,
    )

    payload = _run_json(tmp_path, capsys)

    assert payload["_rc"] == 1
    assert payload["ok"] is False
    violations = payload["violations"]
    assert len(violations) == 1
    assert violations[0]["path"] == "polylogue/storage/example_write.py"
    assert violations[0]["function"] == "<module>.compute_digest"
    assert violations[0]["call"] == "hashlib.sha256"


def test_flags_new_unregistered_hashing_helper_call(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(
        tmp_path,
        "polylogue/storage/example_ids.py",
        """
        from polylogue.core.hashing import hash_text


        def make_id(value):
            return hash_text(value)
        """,
    )

    payload = _run_json(tmp_path, capsys)

    assert payload["_rc"] == 1
    violations = payload["violations"]
    assert len(violations) == 1
    assert violations[0]["call"] == "hash_text"


def test_registered_site_passes_with_matching_entry(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(
        tmp_path,
        "polylogue/storage/example_write.py",
        """
        import hashlib


        def compute_digest(value):
            return hashlib.sha256(value.encode()).hexdigest()
        """,
    )
    registry = tmp_path / "docs" / "plans" / "hash-boundary-registry.yaml"
    _write(
        tmp_path,
        "docs/plans/hash-boundary-registry.yaml",
        """
        entries:
        - path: polylogue/storage/example_write.py
          function: <module>.compute_digest
          call: hashlib.sha256
          occurrence: 0
          classification: identifier
          note: 'test fixture registration'
        """,
    )

    payload = _run_json(tmp_path, capsys, registry=registry)

    assert payload["_rc"] == 0
    assert payload["ok"] is True
    assert payload["registered"] == 1
    assert payload["violations"] == []
    assert payload["stale_registry_entries"] == []


def test_stale_registry_entry_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A registry entry with no matching call site (the call was removed, or
    edited) must fail the gate too -- otherwise the registry only ever grows
    and stops meaning anything."""
    _write(
        tmp_path,
        "polylogue/storage/example_write.py",
        """
        def compute_digest(value):
            return value
        """,
    )
    registry = tmp_path / "docs" / "plans" / "hash-boundary-registry.yaml"
    _write(
        tmp_path,
        "docs/plans/hash-boundary-registry.yaml",
        """
        entries:
        - path: polylogue/storage/example_write.py
          function: <module>.compute_digest
          call: hashlib.sha256
          occurrence: 0
          classification: identifier
          note: 'no longer matches any call site in this function'
        """,
    )

    payload = _run_json(tmp_path, capsys, registry=registry)

    assert payload["_rc"] == 1
    assert payload["ok"] is False
    assert len(payload["stale_registry_entries"]) == 1
    assert payload["stale_registry_entries"][0]["function"] == "<module>.compute_digest"


def test_unknown_classification_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(
        tmp_path,
        "polylogue/storage/example_write.py",
        """
        import hashlib


        def compute_digest(value):
            return hashlib.sha256(value.encode()).hexdigest()
        """,
    )
    registry = tmp_path / "docs" / "plans" / "hash-boundary-registry.yaml"
    _write(
        tmp_path,
        "docs/plans/hash-boundary-registry.yaml",
        """
        entries:
        - path: polylogue/storage/example_write.py
          function: <module>.compute_digest
          call: hashlib.sha256
          occurrence: 0
          classification: not-a-real-tag
          note: 'bad classification'
        """,
    )

    payload = _run_json(tmp_path, capsys, registry=registry)

    assert payload["_rc"] == 1
    assert payload["ok"] is False
    assert len(payload["malformed_registry_entries"]) == 1


def test_hashing_definition_file_own_hashlib_calls_are_excluded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """core/hashing.py's own hashlib calls implement the helpers themselves
    and must not be double-counted as separate producer sites."""
    _write(
        tmp_path,
        "polylogue/core/hashing.py",
        """
        import hashlib


        def hash_text(text):
            return hashlib.sha256(text.encode()).hexdigest()
        """,
    )

    payload = _run_json(tmp_path, capsys)

    assert payload["_rc"] == 0
    assert payload["sites_scanned"] == 0


def test_files_under_a_test_directory_component_are_excluded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Matches the scanner's ``"test" in path.parts`` exclusion (same rule as
    verify_degrade_loudly.py) -- a path component that is exactly ``test``."""
    _write(
        tmp_path,
        "polylogue/storage/test/fixture.py",
        """
        import hashlib


        def make_fixture_digest():
            return hashlib.sha256(b"x").hexdigest()
        """,
    )

    payload = _run_json(tmp_path, capsys)

    assert payload["_rc"] == 0
    assert payload["sites_scanned"] == 0


def test_real_repo_registry_is_internally_consistent(capsys: pytest.CaptureFixture[str]) -> None:
    """The committed docs/plans/hash-boundary-registry.yaml must currently
    match the real repo exactly -- no unregistered sites, no stale entries.
    This is the gate that runs in ``devtools verify --quick``."""
    assert verify_hash_boundary_census.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["violations"] == []
    assert payload["stale_registry_entries"] == []
    assert payload["malformed_registry_entries"] == []
