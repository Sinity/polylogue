"""End-to-end proof of the continuity replay artifact.

Runs the full wiring: the real continuity scenario catalog over real MCP stdio
JSON-RPC against a freshly seeded synthetic archive, cross-checked against the
real query-discovery catalog and combined into one JSON artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.continuity_evidence import main, run_continuity_evidence
from tests.infra.continuity import continuity_catalog_path, load_continuity_catalog, seed_continuity_archive


@pytest.mark.asyncio
async def test_continuity_evidence_end_to_end_synthetic_lane() -> None:
    report = await run_continuity_evidence(redact=False)

    assert report["schema_version"] == 3
    assert report["live_archive"] is False

    continuity = report["continuity"]
    assert isinstance(continuity, dict)
    assert continuity["scenario_count"] == 8
    assert continuity["status"] == "pass"

    discovery = report["discovery_coverage"]
    assert isinstance(discovery, dict)
    assert discovery["status"] == "pass"
    assert discovery["gaps"] == []

    assert report["status"] == "pass"

    # Full report round-trips through JSON (it must be a valid standalone artifact).
    json.dumps(report)


@pytest.mark.asyncio
async def test_supplied_archive_uses_matching_catalog_without_runtime_writes(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    seed_continuity_archive(archive_root, catalog=load_continuity_catalog())

    report = await run_continuity_evidence(
        archive_root=archive_root,
        catalog_path=continuity_catalog_path(),
        scenario_names=("resume",),
        redact=False,
    )

    assert report["status"] == "pass"
    assert report["live_archive"] is True
    assert isinstance(report["catalog_sha256"], str)
    assert not (archive_root / ".continuity-runtime").exists()


def test_main_cli_writes_json_output_and_returns_pass_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "continuity-evidence.json"

    async def fake_replay(**_kwargs: object) -> dict[str, object]:
        return {"schema_version": 3, "status": "pass"}

    monkeypatch.setattr("devtools.continuity_evidence.run_continuity_evidence", fake_replay)

    exit_code = main(
        [
            "--no-redact",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["status"] == "pass"
