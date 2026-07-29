"""The publication gate must run the enum-value privacy guard.

`check_privacy_guards` catches UUIDs, hex ids and high-entropy tokens recorded
verbatim in `x-polylogue-values` annotations. It existed, it was the sharper
check for that class, and its only caller was `devtools schema-audit` -- a
command wired into no verify gate. So nothing enforced it on the path that
actually publishes schemas to a public repository.

That was not theoretical: `promote_cluster`'s samples path calls
`generate_schema_from_samples()`, which unlike the full `generate` pipeline has
no `privacy_config` plumbed through it. A low-cardinality "safe enum" heuristic
with no UUID exemption recorded seven literal conversation UUIDs into
claude-ai v2, while the committed promotion audit reported zero blockers
throughout -- because it did not duplicate the check.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from polylogue.schemas.promotion_audit import audit_schema_artifacts


def _write_element(root: Path, provider: str, document: dict[str, object]) -> None:
    element_dir = root / "providers" / provider / "versions" / "v1" / "elements"
    element_dir.mkdir(parents=True, exist_ok=True)
    target = element_dir / "session_document.schema.json.gz"
    target.write_bytes(gzip.compress(json.dumps(document).encode("utf-8")))


def test_uuid_recorded_as_an_observed_value_blocks_promotion(tmp_path: Path) -> None:
    """The exact claude-ai v2 shape: real UUIDs captured as a "safe enum"."""
    _write_element(
        tmp_path,
        "example",
        {
            "type": "object",
            "properties": {
                "current_leaf_message_uuid": {
                    "type": "string",
                    "x-polylogue-values": [
                        "0f1e2d3c-4b5a-4968-8776-655443332211",
                        "1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d",
                    ],
                }
            },
        },
    )

    report = audit_schema_artifacts(tmp_path)

    assert report.blockers, "a UUID in x-polylogue-values must block publication"
    assert any(item.category == "unsafe_enum_value" for item in report.blockers)
    assert report.to_payload()["verdict"] == "blocked"


def test_genuine_enum_values_do_not_block(tmp_path: Path) -> None:
    """The guard must not fire on the low-cardinality vocabularies it exists to allow."""
    _write_element(
        tmp_path,
        "example",
        {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "x-polylogue-values": ["ok", "success", "failed", "in_progress"],
                }
            },
        },
    )

    report = audit_schema_artifacts(tmp_path)

    assert not [item for item in report.blockers if item.category == "unsafe_enum_value"]
