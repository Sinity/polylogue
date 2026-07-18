from __future__ import annotations

from pathlib import Path

from devtools.codex_exec_child_census import build_report

CATALOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "codex_event_stream"


def test_census_reports_outer_only_counterfactual_and_lowered_coverage() -> None:
    """Exercise the production parser through the census entrypoint.

    Removing child lowering must drive typed/path/outcome counts to zero and
    make this test fail, while the explicit outer-only baseline remains zero.
    """
    report = build_report(
        [
            CATALOG_DIR / "functions_exec_single.jsonl",
            CATALOG_DIR / "functions_exec_multiple.jsonl",
        ]
    )

    assert report["parse_errors"] == []
    assert report["files_discovered"] == 2
    assert report["sessions_parsed"] == 2
    assert report["outer_transport_only_baseline"] == {
        "method": "counterfactual projection retaining only transport blocks (the pre-polylogue-j2zz behavior)",
        "transport_actions": 2,
        "typed_child_actions": 0,
        "child_actions_with_path": 0,
        "child_results_with_structured_outcome": 0,
    }
    assert report["lowered"] == {
        "transport_actions": 2,
        "transport_results": 2,
        "transport_result_texts_with_exit_code_token": 2,
        "typed_child_actions": 11,
        "child_results": 11,
        "paired_child_results": 11,
        "unpaired_child_actions": 0,
        "orphan_child_results": 0,
        "child_actions_with_command": 3,
        "child_actions_with_path": 3,
        "child_actions_with_byte_count": 4,
        "child_results_with_structured_outcome": 3,
        "child_results_with_exit_code": 3,
        "child_results_with_path": 3,
        "child_results_with_byte_count": 4,
        "child_results_with_unknown_outcome": 8,
        "child_result_texts_with_exit_code_token": 5,
        "children_by_registry_type": {
            "apply_patch": 1,
            "exec_command": 3,
            "image": 1,
            "mcp": 1,
            "unknown": 1,
            "update_plan": 1,
            "wait": 1,
            "web": 1,
            "write_stdin": 1,
        },
        "children_by_parse_state": {"malformed": 1, "parsed": 10},
    }
    assert report["coverage"] == {
        "typed_children_per_transport": 5.5,
        "paired_result_coverage": 1.0,
        "structured_outcome_coverage": 0.272727,
        "path_coverage": 0.272727,
    }
