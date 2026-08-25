"""Tests for composable provider-shaped source-package primitives."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.infra.source_builders import provider_source_package


def test_provider_source_package_identity_is_wire_and_inventory_bound(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = first_root / "session.jsonl"
    second = second_root / "session.jsonl"
    first.write_bytes(b'{"id":"law-owned-source"}\n')
    second.write_bytes(first.read_bytes())

    package = provider_source_package(
        "codex",
        (first,),
        generator_id="law-owned-candidate-v1",
        schema_inputs=("codex-schema:test",),
        attachment_bytes=(b"attachment",),
        schedule_digest="schedule:append-v1",
    )
    equivalent = provider_source_package(
        "codex",
        (second,),
        generator_id="law-owned-candidate-v1",
        schema_inputs=("codex-schema:test",),
        attachment_bytes=(b"attachment",),
        schedule_digest="schedule:append-v1",
    )

    assert package.identity == equivalent.identity
    assert package.admitted_sources()[0].name == "codex"
    assert package.admitted_sources()[0].path == first
    assert package.inventory[0] == ("files", 1)
    assert not hasattr(package, "expected_sessions")

    first.write_bytes(b'{"id":"changed-source"}\n')
    changed = provider_source_package("codex", (first,), generator_id="law-owned-candidate-v1")
    assert changed.identity != package.identity


def test_provider_source_package_rejects_missing_wire_material(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="files must already exist"):
        provider_source_package("codex", (tmp_path / "missing.jsonl",))
