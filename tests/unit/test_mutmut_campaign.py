from __future__ import annotations

import json
from pathlib import Path

from devtools.mutmut_campaign import (
    copy_workspace,
    git_status_summary,
    patch_mutmut_section,
    summarize_mutmut_results,
)


def test_patch_mutmut_section_replaces_scope_and_test_selection() -> None:
    original = """
[tool.other]
value = 1

[tool.mutmut]
paths_to_mutate = [
    "polylogue",
]
do_not_mutate = [
    "polylogue/**/__init__.py",
]
pytest_add_cli_args = ["-n", "0"]

[tool.pytest.ini_options]
addopts = "-q"
""".lstrip()

    patched = patch_mutmut_section(
        original,
        paths_to_mutate=("polylogue/archive/filter/filters.py",),
        tests=("tests/unit/core/test_filters.py", "tests/unit/core/test_filters_props.py"),
    )

    assert 'paths_to_mutate = ["polylogue/archive/filter/filters.py"]' in patched
    assert (
        'pytest_add_cli_args_test_selection = ["tests/unit/core/test_filters.py", '
        '"tests/unit/core/test_filters_props.py"]'
    ) in patched
    assert "tests_dir = []" in patched
    assert 'pytest_add_cli_args = ["-n", "0"]' in patched
    assert "[tool.pytest.ini_options]" in patched


def test_summarize_mutmut_results_filters_by_prefix_and_groups_statuses(tmp_path: Path) -> None:
    mutants_dir = tmp_path / "mutants"
    meta_dir = mutants_dir / "polylogue" / "lib"
    meta_dir.mkdir(parents=True)
    (meta_dir / "filters.py.meta").write_text(
        json.dumps(
            {
                "exit_code_by_key": {
                    "polylogue.archive.filter.filters.xǁSessionFilterǁpick__mutmut_1": 0,
                    "polylogue.archive.filter.filters.xǁSessionFilterǁpick__mutmut_2": 1,
                    "polylogue.archive.filter.filters.xǁSessionFilterǁdelete__mutmut_1": -24,
                    "polylogue.archive.models.xǁMessageǁextract_thinking__mutmut_1": 0,
                }
            }
        )
    )

    counts, survivors, timeouts, not_checked, survivor_keys, timeout_keys, not_checked_keys = summarize_mutmut_results(
        mutants_dir,
        prefixes=("polylogue.archive.filter.filters*",),
    )

    assert counts["survived"] == 1
    assert counts["killed"] == 1
    assert counts["timeout"] == 1
    assert counts["not_checked"] == 0
    assert survivors["pick"] == 1
    assert timeouts["delete"] == 1
    assert not not_checked
    assert survivor_keys == ["polylogue.archive.filter.filters.xǁSessionFilterǁpick__mutmut_1"]
    assert timeout_keys == ["polylogue.archive.filter.filters.xǁSessionFilterǁdelete__mutmut_1"]
    assert not not_checked_keys


def test_git_status_summary_ignores_campaign_artifacts(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".local" / "mutation-campaigns").mkdir(parents=True)
    (repo / "tracked.txt").write_text("base\n")

    import subprocess

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    (repo / ".local" / "mutation-campaigns" / "run.json").write_text("{}\n")
    (repo / "tracked.txt").write_text("changed\n")

    summary = git_status_summary(repo)
    assert " M tracked.txt" in summary
    assert all(".local/mutation-campaigns/" not in line for line in summary)


def test_copy_workspace_preserves_symlinked_files(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "AGENTS.md").write_text("agents\n")
    (src / "CLAUDE.md").symlink_to("AGENTS.md")

    copy_workspace(src, dst)

    copied = dst / "CLAUDE.md"
    assert copied.is_symlink()
    assert copied.read_text() == "agents\n"
