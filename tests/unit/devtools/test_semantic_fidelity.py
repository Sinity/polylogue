from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from devtools.semantic_fidelity import build_report, main


def test_semantic_fidelity_census_is_complete_and_catches_mutations() -> None:
    report = build_report()

    assert report["contradiction_count"] == 0
    denominator = report["denominator"]
    assert isinstance(denominator, dict)
    assert denominator["executable_origin_specs"] == 11
    controls = report["mutation_controls"]
    assert isinstance(controls, list)
    assert all(receipt["caught"] for receipt in controls)
    flow = report["construct_flow"]
    assert isinstance(flow, list)
    assert len(flow) == 12
    assert report["blind_spots"]


def test_semantic_fidelity_json_report_is_privacy_safe(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    output = StringIO()
    assert main(["--json", "--report", str(destination)], stdout=output) == 0

    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["resource_measurements"]["network"] == "none"
    assert "payload_text" not in output.getvalue()
