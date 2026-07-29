"""Promotion-audit contracts for staged provider schema artifacts."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from polylogue.schemas.promotion_audit import audit_schema_artifacts


def _write_gzip_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)


def test_promotion_audit_blocks_leak_channels_without_misclassifying_review_values(tmp_path: Path) -> None:
    _write_gzip_json(
        tmp_path / "provider" / "versions" / "v1" / "elements" / "session.schema.json.gz",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "What should happen after approval?": {"type": "string"},
                "model": {
                    "type": "string",
                    "x-polylogue-values": ["gpt-5.6-terra", "Europe/Warsaw", "person@example.com"],
                },
            },
        },
    )
    (tmp_path / "provider" / "versions" / "v1" / "package.json").write_text(
        json.dumps(
            {
                "representative_paths": ["/home/operator/.claude/session.jsonl"],
                "bundle_scopes": ["session-123"],
                "profile_tokens": ["child:mapping:2f5a7f5d-a809-469a-a79a-8f032618fa92"],
            }
        ),
        encoding="utf-8",
    )

    report = audit_schema_artifacts(tmp_path)

    assert {(item.category, item.value) for item in report.blockers} == {
        ("raw_local_provenance", "field=bundle_scopes;value_count=1"),
        ("raw_local_provenance", "field=representative_paths;value_count=1"),
        ("unsafe_property_name", "What should happen after approval?"),
        ("unsafe_structural_identifier", "child:mapping:2f5a7f5d-a809-469a-a79a-8f032618fa92"),
    }
    review = {(item.category, item.value) for item in report.review_items}
    assert ("email_or_account", "person@example.com") in review
    assert ("approved_readable_value", "gpt-5.6-terra") in review
    assert ("approved_readable_value", "Europe/Warsaw") in review


def test_promotion_audit_redacts_credential_material_and_rejects_invalid_artifacts(tmp_path: Path) -> None:
    secret = "github_pat_abcdefghijklmnopqrstuvwxyz123456"
    _write_gzip_json(
        tmp_path / "provider" / "versions" / "v1" / "elements" / "session.schema.json.gz",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "definitely-not-a-json-schema-type",
            "x-polylogue-values": [secret],
        },
    )
    (tmp_path / "broken.json").write_text("{not-json", encoding="utf-8")

    report = audit_schema_artifacts(tmp_path)

    categories = {item.category for item in report.blockers}
    assert categories == {"github_token", "invalid_json_schema", "malformed_artifact"}
    rendered = json.dumps(report.to_payload())
    assert secret not in rendered
    assert "sha256:" in rendered


def test_promotion_audit_groups_repeated_review_values_without_dropping_inventory(tmp_path: Path) -> None:
    for version in ("v1", "v2"):
        _write_gzip_json(
            tmp_path / "provider" / "versions" / version / "elements" / "session.schema.json.gz",
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "x-polylogue-values": ["gpt-5.6-terra", "gpt-5.6-terra", "Europe/Warsaw"],
            },
        )

    report = audit_schema_artifacts(tmp_path)
    payload = report.to_payload()
    review_summary = payload["review_summary"]
    assert isinstance(review_summary, list)
    grouped = {(str(item["category"]), str(item["value"])): item for item in review_summary if isinstance(item, dict)}

    model = grouped[("approved_readable_value", "gpt-5.6-terra")]
    assert model["occurrence_count"] == 2
    assert model["artifact_count"] == 2
    sample_locations = model["sample_locations"]
    assert isinstance(sample_locations, list)
    assert len(sample_locations) == 2
    findings = payload["findings"]
    assert isinstance(findings, list)
    assert len(findings) == 4


def _package(*, version: str, sample_count: int, kinds: list[str]) -> dict[str, object]:
    return {
        "provider": "provider",
        "version": version,
        "sample_count": sample_count,
        "first_seen": "2026-01-01T00:00:00+00:00",
        "last_seen": "2026-02-01T00:00:00+00:00",
        "elements": [{"element_kind": kind, "schema_file": f"{kind}.schema.json.gz"} for kind in kinds],
    }


def _stage_provider(root: Path, *, catalog_package: dict[str, object], disk_package: dict[str, object]) -> None:
    version = str(disk_package["version"])
    (root / "provider" / "versions" / version).mkdir(parents=True, exist_ok=True)
    (root / "provider" / "versions" / version / "package.json").write_text(json.dumps(disk_package), encoding="utf-8")
    (root / "provider" / "catalog.json").write_text(
        json.dumps({"provider": "provider", "default_version": version, "packages": [catalog_package]}),
        encoding="utf-8",
    )


def test_promotion_audit_accepts_a_catalog_that_matches_its_packages(tmp_path: Path) -> None:
    package = _package(version="v1", sample_count=10, kinds=["session_document"])
    _stage_provider(tmp_path, catalog_package=package, disk_package=package)

    report = audit_schema_artifacts(tmp_path)

    assert [finding for finding in report.findings if finding.category == "catalog_incoherent"] == []


def test_promotion_audit_blocks_a_catalog_that_lags_its_packages(tmp_path: Path) -> None:
    """A promotion that rewrites package.json but not catalog.json is inert:
    catalog.json is what runtime_registry resolves against, so the new element
    kinds are never reachable.  This is the real 2026-07-29 regression.
    """
    _stage_provider(
        tmp_path,
        catalog_package=_package(version="v1", sample_count=10, kinds=["session_document"]),
        disk_package=_package(version="v1", sample_count=999, kinds=["session_document", "subagent_session_stream"]),
    )

    report = audit_schema_artifacts(tmp_path)

    incoherent = {finding.json_path for finding in report.blockers if finding.category == "catalog_incoherent"}
    assert "$.packages[version=v1].elements" in incoherent
    assert "$.packages[version=v1].sample_count" in incoherent


def test_promotion_audit_blocks_a_package_missing_from_the_catalog(tmp_path: Path) -> None:
    _stage_provider(
        tmp_path,
        catalog_package=_package(version="v1", sample_count=10, kinds=["session_document"]),
        disk_package=_package(version="v1", sample_count=10, kinds=["session_document"]),
    )
    orphan = tmp_path / "provider" / "versions" / "v2"
    orphan.mkdir(parents=True)
    (orphan / "package.json").write_text(
        json.dumps(_package(version="v2", sample_count=3, kinds=["session_document"])), encoding="utf-8"
    )

    report = audit_schema_artifacts(tmp_path)

    values = {finding.value for finding in report.blockers if finding.category == "catalog_incoherent"}
    assert "version=v2;reason=package_on_disk_absent_from_catalog" in values
