"""Every runner must carry evidence trust and freshness metadata."""

from __future__ import annotations

import pytest

from polylogue.proof.catalog import build_verification_catalog

pytestmark = pytest.mark.proof_law


def test_runner_trust_metadata_is_complete() -> None:
    catalog = build_verification_catalog()
    missing: list[str] = []
    for runner in catalog.runner_bindings:
        trust = runner.trust
        required = {
            "producer": trust.producer,
            "reviewed_at": trust.reviewed_at,
            "code_revision": trust.code_revision,
            "schema_version": trust.schema_version,
            "environment_fingerprint": trust.environment_fingerprint,
            "runner_version": trust.runner_version,
            "freshness": trust.freshness,
            "origin": trust.origin,
        }
        absent = [name for name, value in required.items() if value is None or value == ""]
        if absent:
            missing.append(f"{runner.id}: {', '.join(absent)}")
    assert not missing, f"runner trust metadata incomplete: {missing}"
