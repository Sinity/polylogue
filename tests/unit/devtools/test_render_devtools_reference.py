from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from devtools import render_devtools_reference
from devtools.render_support import write_if_changed


def test_build_command_catalog_includes_discovery_and_commands() -> None:
    rendered = render_devtools_reference.build_command_catalog()

    assert "## Core Loop" in rendered
    assert "## Executable Lab Checks" in rendered
    assert "devtools --list-commands --json" in rendered
    assert "devtools status --json" in rendered
    assert "not a proof ledger or end-user archive workflow" in rendered
    assert "| `devtools lab graph` | Render the runtime artifact, operation, and scenario-coverage map. |" in rendered
    assert "| `devtools lab projections` | Render the authored scenario-bearing verification projections. |" in rendered
    assert (
        "| `devtools lab probe capture-regression` | Capture pipeline-probe summaries as durable local regression cases. |"
        in rendered
    )
    assert (
        "| `devtools lab probe cost-reconciliation` | Reconcile Polylogue token accounting against private provider stores. |"
        in rendered
    )
    assert "### Lab Checks" in rendered
    assert "| `devtools render all` |" in rendered
    assert "| `devtools render demo-corpus-datasheet` |" in rendered
    assert "Common forms: `devtools status`" in rendered


def test_replace_marked_section_updates_catalog_block() -> None:
    source = "\n".join(
        [
            "before",
            "<!-- BEGIN GENERATED: devtools-command-catalog -->",
            "old",
            "<!-- END GENERATED: devtools-command-catalog -->",
            "after",
        ]
    )

    updated = render_devtools_reference.replace_marked_section(source, "new")

    assert "before" in updated
    assert "\nnew\n" in updated
    assert "after" in updated


def test_write_if_changed_reuses_existing_output(tmp_path: Path) -> None:
    output_path = tmp_path / "devtools.md"
    content = "hello\n"
    write_if_changed(output_path, content)
    original_mtime = output_path.stat().st_mtime_ns

    write_if_changed(output_path, content)

    assert output_path.read_text(encoding="utf-8") == content
    assert output_path.stat().st_mtime_ns == original_mtime


def test_write_if_changed_uses_unique_temp_files_for_concurrent_writers(tmp_path: Path) -> None:
    output_path = tmp_path / "cli-reference.md"

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(write_if_changed, output_path, f"content {index}\n") for index in range(32)]
        for future in futures:
            future.result()

    assert output_path.read_text(encoding="utf-8").startswith("content ")
    assert list(tmp_path.glob("*.tmp")) == []
