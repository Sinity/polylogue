from __future__ import annotations

import json
from pathlib import Path

from devtools import delivery_gate_status


def _write_jsonl(path: Path, issues: list[dict[str, object]]) -> None:
    lines = [json.dumps({"_type": "issue", **issue}) for issue in issues]
    path.write_text("\n".join(lines) + "\n")


def test_load_parses_issues_and_dependencies(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "polylogue-a",
                "status": "open",
                "labels": ["delivery:A-trust-floor"],
                "dependencies": [{"depends_on_id": "polylogue-b", "type": "blocks"}],
            },
            {"id": "polylogue-b", "status": "closed", "labels": []},
        ],
    )
    issues, deps = delivery_gate_status.load(path)
    assert set(issues) == {"polylogue-a", "polylogue-b"}
    assert ("polylogue-a", "polylogue-b", "blocks") in deps


def test_gate_counts_ready_blocked_closed(tmp_path: Path, capsys: object) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [
            {"id": "polylogue-closed", "status": "closed", "labels": ["delivery:A-trust-floor"]},
            {
                "id": "polylogue-blocked",
                "status": "open",
                "labels": ["delivery:A-trust-floor"],
                "dependencies": [{"depends_on_id": "polylogue-open-blocker", "type": "blocks"}],
            },
            {"id": "polylogue-open-blocker", "status": "open", "labels": ["delivery:A-trust-floor"]},
            {"id": "polylogue-unlabeled", "status": "open", "labels": []},
        ],
    )

    rc = delivery_gate_status.main([str(path), "--json", "--gate", "A-trust-floor"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["unlabeled_open"] == 1
    gate = next(g for g in payload["gates"] if g["gate"] == "A-trust-floor")
    assert gate["total"] == 3
    assert gate["closed"] == 1
    assert gate["blocked"] == 1
    assert "polylogue-open-blocker" in gate["ready_ids"]
    assert "polylogue-blocked" not in gate["ready_ids"]


def test_delivery_ac_patched_label_is_not_a_gate_assignment(tmp_path: Path, capsys: object) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(path, [{"id": "polylogue-a", "status": "open", "labels": ["delivery:ac-patched"]}])

    rc = delivery_gate_status.main([str(path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["unlabeled_open"] == 1


def test_unknown_gate_label_is_reported(tmp_path: Path, capsys: object) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(path, [{"id": "polylogue-a", "status": "open", "labels": ["delivery:not-a-real-gate"]}])

    rc = delivery_gate_status.main([str(path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["unknown_gates"] == ["not-a-real-gate"]
