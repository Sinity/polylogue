from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from devtools import public_claims


def _write_minimal_repo(tmp_path: Path) -> Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / ".beads").mkdir()
    (tmp_path / "README.md").write_text("Evidence-first product.\n", encoding="utf-8")
    (tmp_path / "docs" / "proof.md").write_text("proof\n", encoding="utf-8")
    (tmp_path / ".beads" / "issues.jsonl").write_text(
        '{"_type":"issue","id":"polylogue-demo","status":"open"}\n',
        encoding="utf-8",
    )
    return tmp_path


def _ledger() -> dict[str, object]:
    return {
        "schema": public_claims.SUPPORTED_SCHEMA,
        "statuses": ["proven", "capability", "aspirational", "retired"],
        "evidence_classes": [
            "deterministic_product_proof",
            "field_observation",
            "implementation_contract",
            "architecture_decision",
            "hypothesis",
        ],
        "public_surfaces": ["README.md"],
        "retired_phrases": ["old slogan"],
        "claims": [
            {
                "id": "demo.receipt",
                "status": "proven",
                "evidence_class": "deterministic_product_proof",
                "publication": "The receipt is structurally verified.",
                "scope": "One deterministic provider-shaped fixture.",
                "evidence": ["docs/proof.md"],
                "caveat": "This does not establish private-archive prevalence.",
                "owner_beads": ["polylogue-demo"],
                "proof_commands": ["polylogue demo receipts"],
            }
        ],
    }


def test_build_report_accepts_grounded_ledger(tmp_path: Path) -> None:
    root = _write_minimal_repo(tmp_path)
    ledger_path = root / "docs" / "public-claims.yaml"
    ledger_path.write_text(yaml.safe_dump(_ledger(), sort_keys=False), encoding="utf-8")

    report = public_claims.build_report(ledger_path=ledger_path, root=root)

    assert report["ok"] is True
    assert report["claim_count"] == 1
    assert report["evidence_path_count"] == 1
    assert report["proof_command_count"] == 1


def test_build_report_rejects_missing_evidence_unknown_bead_and_retired_copy(tmp_path: Path) -> None:
    root = _write_minimal_repo(tmp_path)
    (root / "README.md").write_text("The old slogan returned.\n", encoding="utf-8")
    ledger = _ledger()
    claims = cast(list[dict[str, Any]], ledger["claims"])
    claim = claims[0]
    claim["evidence"] = ["docs/missing.md"]
    claim["owner_beads"] = ["polylogue-missing"]
    ledger_path = root / "docs" / "public-claims.yaml"
    ledger_path.write_text(yaml.safe_dump(ledger, sort_keys=False), encoding="utf-8")

    report = public_claims.build_report(ledger_path=ledger_path, root=root)
    messages = [problem["message"] for problem in report["problems"]]

    assert report["ok"] is False
    assert "evidence path does not exist: docs/missing.md" in messages
    assert "owner Bead does not exist: polylogue-missing" in messages
    assert any("retired phrase" in message for message in messages)


def test_public_claims_gate_is_declared_in_ci_and_release_readiness() -> None:
    root = Path(__file__).resolve().parents[3]
    ci = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    readiness = (root / "devtools" / "release_readiness.py").read_text(encoding="utf-8")

    assert "uv run devtools verify public-claims --json" in ci
    assert 'GateCommand(("devtools", "verify", "public-claims")' in readiness
