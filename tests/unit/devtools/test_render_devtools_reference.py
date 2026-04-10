from __future__ import annotations

from pathlib import Path

from devtools import render_devtools_reference


def test_build_command_catalog_includes_discovery_and_commands() -> None:
    rendered = render_devtools_reference.build_command_catalog()

    assert "devtools --list-commands --json" in rendered
    assert "devtools status --json" in rendered
    assert "| `devtools render-all` |" in rendered


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
    render_devtools_reference.write_if_changed(output_path, content)
    original_mtime = output_path.stat().st_mtime_ns

    render_devtools_reference.write_if_changed(output_path, content)

    assert output_path.read_text(encoding="utf-8") == content
    assert output_path.stat().st_mtime_ns == original_mtime
