from __future__ import annotations

import ast
from pathlib import Path

from devtools import repo_root
from devtools.visual_artifacts import (
    READER_VISUAL_ARTIFACTS,
    READER_VISUAL_SMOKE_DEVTOOLS_COMMAND,
    READER_VISUAL_SMOKE_PYTEST_COMMAND,
    READER_VISUAL_SMOKE_REPORT,
    reader_visual_artifact_payloads,
)


def _manifest_artifact_ids_from_visual_tests() -> set[str]:
    ids: set[str] = set()
    visual_root = repo_root() / "tests" / "visual"
    for path in sorted(visual_root.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "write_evidence_manifest":
                continue
            artifact_id = next((keyword.value for keyword in node.keywords if keyword.arg == "artifact_id"), None)
            assert isinstance(artifact_id, ast.Constant), (
                f"{path}: write_evidence_manifest requires literal artifact_id"
            )
            assert isinstance(artifact_id.value, str), f"{path}: artifact_id must be a string literal"
            ids.add(artifact_id.value)
    return ids


def test_reader_visual_artifact_inventory_matches_visual_tests() -> None:
    inventory_ids = {artifact.artifact_id for artifact in READER_VISUAL_ARTIFACTS}

    assert inventory_ids == _manifest_artifact_ids_from_visual_tests()


def test_reader_visual_artifact_inventory_is_machine_readable() -> None:
    payloads = reader_visual_artifact_payloads()

    assert [payload["artifact_id"] for payload in payloads] == [
        artifact.artifact_id for artifact in READER_VISUAL_ARTIFACTS
    ]
    assert len(payloads) == len({payload["artifact_id"] for payload in payloads})
    assert all(Path(str(payload["owner"])).parts[:2] == ("tests", "visual") for payload in payloads)
    assert all(payload["routes"] for payload in payloads)
    assert READER_VISUAL_SMOKE_PYTEST_COMMAND == ("python", "-m", "pytest", "-q", "tests/visual")
    assert READER_VISUAL_SMOKE_DEVTOOLS_COMMAND == ("uv", "run", "devtools", "test", "tests/visual")
    assert READER_VISUAL_SMOKE_REPORT.endswith("reader-visual-smoke.json")
