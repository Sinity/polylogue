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
        ("unsafe_property_name", "What should happen after approval?"),
        ("unsafe_structural_identifier", "child:mapping:2f5a7f5d-a809-469a-a79a-8f032618fa92"),
    }
    review = {(item.category, item.value) for item in report.review_items}
    assert ("filesystem_path", "/home/operator/.claude/session.jsonl") in review
    assert ("identifier", "session-123") in review
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
