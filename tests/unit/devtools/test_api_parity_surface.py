"""Gate and renderer contract for the API parity surface.

Anti-vacuity: each test names the mutation that turns it red -- a library doc
that awaits a synchronous callable, documents a callable or keyword that does
not exist, documents no live call at all, drops a semantic operation, or a
committed ``docs/api-parity.md`` that no longer matches the declarations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from devtools import render_api_parity, verify_api_parity
from devtools.command_catalog import COMMANDS
from devtools.gate import GATES_BY_NAME
from devtools.generated_surfaces import GENERATED_SURFACE_BY_NAME

ROOT = Path(__file__).resolve().parents[3]


def _doc(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "library-api.md"
    path.write_text(body, encoding="utf-8")
    return path


def test_committed_matrix_matches_the_declarations() -> None:
    """The committed artifact is regenerated, never hand-edited.

    Anti-vacuity: changing a CLI binding or adding an MCP tool without running
    ``devtools render api-parity`` makes this red.
    """

    committed = (ROOT / render_api_parity.OUTPUT).read_text(encoding="utf-8")
    assert committed == render_api_parity.build_document()


def test_rendering_is_deterministic() -> None:
    assert render_api_parity.build_document() == render_api_parity.build_document()


def test_gate_and_command_are_registered() -> None:
    """The gate runs in the quick set and the command is in the catalog."""

    gate = GATES_BY_NAME["api-parity"]
    assert gate.in_quick and gate.blocking
    assert "verify api-parity" in COMMANDS
    assert "api-parity" in GENERATED_SURFACE_BY_NAME


def test_live_repository_passes_the_gate() -> None:
    findings = verify_api_parity.run(doc_path=ROOT / verify_api_parity.LIBRARY_DOC)
    assert findings == (), [f"{item.code}: {item.subject}: {item.message}" for item in findings]


def test_awaiting_a_synchronous_callable_fails(tmp_path: Path) -> None:
    path = _doc(tmp_path, "## Ops\n\n```python\nvalue = await archive.embedding_status()\n```\n")
    codes = {finding.code for finding in verify_api_parity.validate_library_doc(path)}
    assert "async_mismatch" in codes


def test_documenting_a_missing_callable_fails(tmp_path: Path) -> None:
    path = _doc(tmp_path, "## Ops\n\n```python\nvalue = await archive.method_that_vanished()\n```\n")
    findings = verify_api_parity.validate_library_doc(path)
    assert "unknown_documented_callable" in {finding.code for finding in findings}


def test_documenting_an_unknown_keyword_fails(tmp_path: Path) -> None:
    path = _doc(tmp_path, '## Ops\n\n```python\nvalue = await archive.resolve_ref(nope="x")\n```\n')
    codes = {finding.code for finding in verify_api_parity.validate_library_doc(path)}
    assert "unknown_documented_keyword" in codes


def test_nested_call_keywords_are_not_attributed_to_the_facade(tmp_path: Path) -> None:
    """A keyword belonging to a nested constructor is not a facade keyword."""

    path = _doc(
        tmp_path,
        "## Ops\n\n```python\nvalue = await archive.judge_assertion_candidates(items=Query(limit=5))\n```\n",
    )
    codes = {finding.code for finding in verify_api_parity.validate_library_doc(path)}
    assert "unknown_documented_keyword" not in codes


def test_chained_builder_call_is_not_an_async_mismatch(tmp_path: Path) -> None:
    path = _doc(tmp_path, "## Ops\n\n```python\nrows = await archive.filter().limit(10).list()\n```\n")
    codes = {finding.code for finding in verify_api_parity.validate_library_doc(path)}
    assert "async_mismatch" not in codes


def test_doc_without_any_live_call_fails_instead_of_passing(tmp_path: Path) -> None:
    """The verifier refuses to pass vacuously on an example-free document."""

    path = _doc(tmp_path, "## Ops\n\nProse only, no code.\n")
    codes = {finding.code for finding in verify_api_parity.validate_library_doc(path)}
    assert "no_documented_operations" in codes


def test_dropping_a_documented_operation_fails(tmp_path: Path) -> None:
    path = _doc(tmp_path, "## Ops\n\n```python\ntotals = await archive.stats()\n```\n")
    findings = verify_api_parity.validate_library_doc(path)
    subjects = {finding.subject for finding in findings if finding.code == "undocumented_operation"}
    assert "import_annotation_batch" in subjects


def test_main_reports_findings_as_a_non_zero_exit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _doc(tmp_path, "## Ops\n\nProse only, no code.\n")
    assert verify_api_parity.main(["--doc", str(path), "--json"]) == 1
    assert "no_documented_operations" in capsys.readouterr().out
