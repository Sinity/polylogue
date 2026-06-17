from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import release_readiness


def _write_gate(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _valid_gate_text() -> str:
    headings = "\n\n".join(release_readiness.REQUIRED_DOC_HEADINGS)
    commands = "\n".join(" ".join(command.argv) for command in release_readiness.REQUIRED_COMMANDS)
    fields = "\n".join(release_readiness.REQUIRED_RELEASE_BODY_FIELDS)
    return f"""# Release Readiness Gate

{headings}

```bash
{commands}
```

Satisfied:

- #1 landed.

Still blocking external release claims:

- #2 blocks a release claim.

```text
{fields}
```
"""


def test_release_readiness_report_validates_committed_gate() -> None:
    report = release_readiness.build_report()

    assert report["ok"] is True
    assert report["gate_doc"] == "docs/plans/release-readiness-gate.md"
    assert ["devtools", "verify", "--quick"] in [command["argv"] for command in report["required_commands"]]
    assert ["devtools", "build-package"] in [command["argv"] for command in report["required_commands"]]
    assert "- Known caveats scoped out:" in report["required_release_body_fields"]
    assert report["release_status"]["satisfied"]
    assert report["release_status"]["blocking_external_claims"]
    assert report["errors"] == []


def test_release_readiness_json_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert release_readiness.main(["--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["required_commands"][0]["argv"] == ["devtools", "release-readiness"]
    assert "release_status" in payload


def test_release_readiness_rejects_missing_required_command(tmp_path: Path) -> None:
    gate_doc = tmp_path / "release-readiness-gate.md"
    _write_gate(gate_doc, _valid_gate_text().replace("devtools verify --lab\n", ""))

    report = release_readiness.build_report(gate_doc=gate_doc)

    assert report["ok"] is False
    assert any("devtools verify --lab" in error for error in report["errors"])


def test_release_readiness_rejects_missing_status_lists(tmp_path: Path) -> None:
    gate_doc = tmp_path / "release-readiness-gate.md"
    _write_gate(gate_doc, _valid_gate_text().replace("Still blocking external release claims:\n", ""))

    report = release_readiness.build_report(gate_doc=gate_doc)

    assert report["ok"] is False
    assert any("blocking release-status list" in error for error in report["errors"])


def test_release_readiness_rejects_retired_issue_refs(tmp_path: Path) -> None:
    gate_doc = tmp_path / "release-readiness-gate.md"
    _write_gate(gate_doc, _valid_gate_text().replace("#1 landed.", "#1839 was folded elsewhere."))

    report = release_readiness.build_report(gate_doc=gate_doc)

    assert report["ok"] is False
    assert "release gate references retired issue: #1839" in report["errors"]


def test_release_readiness_rejects_satisfied_issue_as_blocker_without_caveat(tmp_path: Path) -> None:
    gate_doc = tmp_path / "release-readiness-gate.md"
    _write_gate(
        gate_doc,
        _valid_gate_text().replace("#2 blocks a release claim.", "#1 still blocks the release."),
    )

    report = release_readiness.build_report(gate_doc=gate_doc)

    assert report["ok"] is False
    assert any("also cites satisfied issue(s) #1" in error for error in report["errors"])


def test_release_readiness_allows_satisfied_issue_in_scoped_caveat(tmp_path: Path) -> None:
    gate_doc = tmp_path / "release-readiness-gate.md"
    _write_gate(
        gate_doc,
        _valid_gate_text().replace(
            "#2 blocks a release claim.",
            "#1 is scoped out: do not advertise the unshipped extension.",
        ),
    )

    report = release_readiness.build_report(gate_doc=gate_doc)

    assert report["ok"] is True
