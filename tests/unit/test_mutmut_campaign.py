from __future__ import annotations

import json
from pathlib import Path

from devtools.mutmut_campaign import (
    CAMPAIGNS,
    CampaignResult,
    copy_workspace,
    format_index,
    format_markdown,
    git_status_summary,
    latest_results_by_campaign,
    load_results,
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
                    "polylogue.archive.filter.filters.xǁConversationFilterǁpick__mutmut_1": 0,
                    "polylogue.archive.filter.filters.xǁConversationFilterǁpick__mutmut_2": 1,
                    "polylogue.archive.filter.filters.xǁConversationFilterǁdelete__mutmut_1": -24,
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
    assert survivor_keys == ["polylogue.archive.filter.filters.xǁConversationFilterǁpick__mutmut_1"]
    assert timeout_keys == ["polylogue.archive.filter.filters.xǁConversationFilterǁdelete__mutmut_1"]
    assert not not_checked_keys


def test_campaign_registry_has_selection_tests_for_each_scope() -> None:
    for campaign in CAMPAIGNS.values():
        assert campaign.paths_to_mutate
        assert campaign.tests


def test_format_markdown_records_dirty_worktree_status() -> None:
    result = CampaignResult(
        campaign="json",
        description="JSON serialization and parser laws",
        commit="deadbeef",
        worktree_dirty=True,
        status_summary=[" M tests/unit/core/test_json.py", "?? .local/mutation-campaigns/foo.json"],
        created_at="2026-03-11T00:00:00+00:00",
        workspace="/tmp/example/repo",
        command=["mutmut", "run"],
        paths_to_mutate=["polylogue/core/json.py"],
        tests=["tests/unit/core/test_json.py"],
        counts={"killed": 1, "survived": 2, "timeout": 0, "not_checked": 0, "suspicious": 0, "skipped": 0},
        dominant_survivors=[("dumps", 2)],
        dominant_timeouts=[],
        dominant_not_checked=[],
        survivor_keys=["polylogue.core.json.x_dumps__mutmut_1"],
        timeout_keys=[],
        not_checked_keys=[],
        runtime_seconds=1.5,
        exit_code=0,
        notes=[],
        origin="authored.mutation-campaign",
        path_targets=["json-law-loop"],
        artifact_targets=["raw_payload", "validation_state"],
        operation_targets=["plan-validation-backlog"],
        tags=["mutation", "json"],
    )

    rendered = format_markdown(result)
    assert "- Worktree dirty: `yes`" in rendered
    assert "## Source Worktree Status" in rendered
    assert "` M tests/unit/core/test_json.py`" in rendered
    assert "## Scenario Metadata" in rendered
    assert "- Origin: `authored.mutation-campaign`" in rendered
    assert "- Path targets: `json-law-loop`" in rendered


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


def test_load_results_and_index_use_latest_campaign_entry(tmp_path: Path) -> None:
    campaign_dir = tmp_path / ".local" / "mutation-campaigns"
    campaign_dir.mkdir(parents=True)
    old = {
        "campaign": "json",
        "description": "JSON",
        "commit": "aaaaaaaaaaaa",
        "created_at": "2026-03-11T00:00:00+00:00",
        "workspace": "/tmp/old",
        "command": ["mutmut", "run"],
        "paths_to_mutate": ["polylogue/core/json.py"],
        "tests": ["tests/unit/core/test_json.py"],
        "counts": {"killed": 11, "survived": 15, "timeout": 0, "not_checked": 0, "suspicious": 0, "skipped": 0},
        "dominant_survivors": [["dumps", 14]],
        "dominant_timeouts": [],
        "dominant_not_checked": [],
        "runtime_seconds": 7.0,
        "exit_code": 0,
        "notes": [],
        "origin": "authored.mutation-campaign",
        "path_targets": ["json-law-loop"],
        "artifact_targets": ["raw_payload", "validation_state"],
        "operation_targets": ["plan-validation-backlog"],
        "tags": ["mutation", "json"],
    }
    new = {
        **old,
        "commit": "bbbbbbbbbbbb",
        "created_at": "2026-03-11T01:00:00+00:00",
        "counts": {"killed": 24, "survived": 2, "timeout": 0, "not_checked": 0, "suspicious": 0, "skipped": 0},
        "worktree_dirty": True,
        "status_summary": [" M tests/unit/core/test_json.py"],
        "survivor_keys": ["polylogue.core.json.xǁdumps__mutmut_1"],
        "timeout_keys": [],
        "not_checked_keys": [],
    }
    other = {
        **old,
        "campaign": "filters",
        "description": "Filters",
        "commit": "cccccccccccc",
        "created_at": "2026-03-11T02:00:00+00:00",
        "counts": {"killed": 486, "survived": 11, "timeout": 100, "not_checked": 0, "suspicious": 0, "skipped": 0},
    }
    (campaign_dir / "2026-03-11-json-old.json").write_text(json.dumps(old))
    (campaign_dir / "2026-03-11-json-new.json").write_text(json.dumps(new))
    (campaign_dir / "2026-03-11-filters.json").write_text(json.dumps(other))

    results = load_results(campaign_dir)
    latest = latest_results_by_campaign(results)

    assert [result.campaign for result in latest] == ["filters", "json"]
    json_result = next(result for result in latest if result.campaign == "json")
    assert json_result.commit == "bbbbbbbbbbbb"
    assert json_result.origin == "authored.mutation-campaign"
    assert json_result.path_targets == ["json-law-loop"]
    assert json_result.artifact_targets == ["raw_payload", "validation_state"]
    assert json_result.operation_targets == ["plan-validation-backlog"]
    assert json_result.tags == ["mutation", "json"]
    rendered = format_index(results)
    assert "`json` | `2026-03-11T01:00:00+00:00` | `bbbbbbbbbbbb` | 24 | 2 | 0 | 0 | yes | 7.00s |" in rendered
    assert "`filters` | `2026-03-11T02:00:00+00:00` | `cccccccccccc` | 486 | 11 | 100 | 0 | no | 7.00s |" in rendered


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
