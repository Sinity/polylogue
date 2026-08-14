from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from devtools import render_devtools_reference
from devtools.command_catalog import (
    control_plane_command,
    featured_command_specs,
    grouped_command_specs,
    verification_lab_command_specs,
)
from devtools.render_support import write_if_changed


def test_build_command_catalog_includes_discovery_and_commands() -> None:
    rendered = render_devtools_reference.build_command_catalog()

    assert rendered.startswith("<!-- BEGIN GENERATED: devtools-command-catalog -->")
    assert rendered.endswith("<!-- END GENERATED: devtools-command-catalog -->")
    for command in (
        control_plane_command("--help"),
        control_plane_command("--list-commands"),
        control_plane_command("--list-commands", "--json"),
        control_plane_command("status"),
        control_plane_command("status", "--json"),
    ):
        assert command in rendered
    for spec in verification_lab_command_specs():
        assert f"| `{spec.invocation}` | {spec.use_when or spec.description} |" in rendered
    for spec in featured_command_specs():
        assert f"- `{spec.invocation}`: {spec.use_when or spec.description}" in rendered
    for specs in grouped_command_specs().values():
        for spec in specs:
            assert f"| `{spec.invocation}` | {spec.description} |" in rendered


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
