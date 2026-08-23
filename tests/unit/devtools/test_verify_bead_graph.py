from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from devtools import verify_bead_graph


def _issue(bead_id: str, *, dependencies: object = None) -> dict[str, object]:
    return {"id": bead_id, "dependencies": [] if dependencies is None else dependencies}


def test_clean_structured_graph_has_no_findings() -> None:
    issues = [
        _issue("parent"),
        _issue("child", dependencies=[{"type": "parent-child", "depends_on_id": "parent"}]),
        _issue("dependent", dependencies=[{"type": "blocks", "depends_on_id": "child"}]),
    ]
    assert verify_bead_graph.collect_findings(issues) == []
    assert verify_bead_graph.canonical_parent_map(issues)["child"] == "parent"


def test_top_level_schema_findings_reject_malformed_metadata_labels_and_dependencies() -> None:
    findings = verify_bead_graph.collect_findings(
        [
            {"id": "metadata", "metadata": "not-an-object", "dependencies": []},
            {"id": "labels", "labels": {"not": "a-list"}, "dependencies": []},
            {"id": "dependencies", "dependencies": {"not": "a-list"}},
        ]
    )
    assert {finding.kind for finding in findings} >= {
        "malformed-metadata",
        "malformed-labels",
        "malformed-dependencies",
    }


def test_dependency_integrity_rejects_malformed_duplicate_missing_and_self_edges() -> None:
    issues = [
        _issue("a", dependencies="invalid"),
        _issue("b", dependencies=[None, {"type": "blocks", "depends_on_id": ""}]),
        _issue(
            "c",
            dependencies=[
                {"type": "relates-to", "depends_on_id": "missing"},
                {"type": "blocks", "depends_on_id": "c"},
                {"type": "blocks", "depends_on_id": "c"},
            ],
        ),
    ]
    kinds = {finding.kind for finding in verify_bead_graph.collect_findings(issues)}
    assert kinds >= {
        "malformed-dependencies",
        "malformed-dependency",
        "missing-dependency-target",
        "self-dependency",
        "duplicate-dependency",
    }


def test_parent_cardinality_and_parent_cycle_are_rejected() -> None:
    issues = [
        _issue("a", dependencies=[{"type": "parent-child", "depends_on_id": "b"}]),
        _issue("b", dependencies=[{"type": "parent-child", "depends_on_id": "a"}]),
        _issue(
            "c",
            dependencies=[
                {"type": "parent-child", "depends_on_id": "a"},
                {"type": "parent-child", "depends_on_id": "b"},
            ],
        ),
    ]
    findings = verify_bead_graph.collect_findings(issues)
    assert {finding.kind for finding in findings} >= {"multiple-parents", "parent-cycle"}
    assert verify_bead_graph.canonical_parent_map(issues)["c"] is None


def test_blocks_cycle_is_rejected_from_export_without_invoking_bd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    export = tmp_path / "issues.jsonl"
    export.write_text(
        "\n".join(
            json.dumps(issue)
            for issue in [
                _issue("a", dependencies=[{"type": "blocks", "depends_on_id": "b"}]),
                _issue("b", dependencies=[{"type": "blocks", "depends_on_id": "a"}]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("bd must not run"))
    assert verify_bead_graph.main(["--export", str(export), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["cycles"]["ok"] is False
    assert report["counts"]["blocks-cycle"] > 0


@pytest.mark.parametrize("payload", [["not-an-issue"], [{"id": ""}], [{"id": 42}], [{"id": "a"}, {"id": "a"}]])
def test_issue_population_validation_is_fail_closed(payload: list[object]) -> None:
    with pytest.raises(RuntimeError):
        verify_bead_graph._validated_issues(payload, source="test")


def test_export_json_error_names_line(tmp_path: Path) -> None:
    export = tmp_path / "issues.jsonl"
    export.write_text('{"id":"a"}\n{broken\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"issues\.jsonl:2"):
        verify_bead_graph._load_export(export)


def test_main_loads_live_population_and_reports_clean_graph(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(verify_bead_graph, "_run_bd_dep_cycles", lambda: (True, "no cycles"))
    monkeypatch.setattr(verify_bead_graph, "_run_bd_list_all", lambda: [_issue("a")])
    monkeypatch.setattr(verify_bead_graph, "_registry_findings", lambda _issues: [])
    assert verify_bead_graph.main(["--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["issues_scanned"] == 1
    assert report["findings"] == []


def test_main_fails_closed_when_live_cycle_probe_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(verify_bead_graph, "_run_bd_dep_cycles", lambda: (False, "a -> b -> a"))
    assert verify_bead_graph.main(["--json"]) == 1
    assert "dependency cycle check failed" in json.loads(capsys.readouterr().out)["error"]


def test_export_report_derives_transitive_forcing_closure_and_rejects_unknown_edge_kind() -> None:
    issues = [
        _issue("root", dependencies=[{"type": "blocks", "depends_on_id": "middle"}]),
        _issue("middle", dependencies=[{"type": "blocks", "depends_on_id": "leaf"}]),
        _issue("leaf"),
    ]

    report = verify_bead_graph.build_report(
        issues,
        cycles_ok=True,
        cycles_output="",
        forcing_roots=["root"],
    )

    assert report["report_version"] == 3
    assert report["forcing"][0]["blocker_ids"] == ["leaf", "middle"]

    malformed = [_issue("root", dependencies=[{"type": "blockz", "depends_on_id": "leaf"}]), _issue("leaf")]
    findings = verify_bead_graph.collect_findings(malformed)
    assert any(finding.kind == "unknown-dependency-kind" for finding in findings)


def test_malformed_graph_still_emits_structured_report_with_forcing_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Derived forcing analysis must not hide already-collected graph violations."""
    issues = [_issue("root", dependencies=[{"type": "blocks", "depends_on_id": "missing"}])]
    export = tmp_path / "issues.jsonl"
    export.write_text("\n".join(json.dumps(issue) for issue in issues) + "\n", encoding="utf-8")

    assert verify_bead_graph.main(["--export", str(export), "--forcing-root", "root", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["forcing"][0]["blocker_ids"] == []
    assert any(item["kind"] == "missing-dependency-target" for item in report["findings"])


@pytest.mark.parametrize(
    ("mutation", "expected_kind"),
    [
        ("poured-stage", "malformed-metadata-field"),
        ("adapter-membership", "malformed-metadata-field"),
        ("status", "malformed-status"),
    ],
)
def test_campaign_export_json_rejects_malformed_nested_fields(
    mutation: str,
    expected_kind: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    issues = _campaign_fixture()
    if mutation == "poured-stage":
        issues.extend(_batch_group("a"))
        stage = next(issue for issue in issues if issue.get("metadata", {}).get("stage") == "prepare")
        stage["metadata"]["stage"] = ["prepare"]
    elif mutation == "adapter-membership":
        issues.append(
            {
                "id": verify_bead_graph.CAMPAIGN_ADAPTER_ID,
                "labels": ["campaign:reindex-2026", "workstream:e", "campaign-role:implementation", "timing:prep"],
                "metadata": {"campaign_membership_kind": {"kind": "staged-adapter"}},
                "dependencies": [],
            }
        )
    else:
        issues[0]["status"] = ["open"]

    export = tmp_path / f"{mutation}.jsonl"
    export.write_text("\n".join(json.dumps(issue) for issue in issues) + "\n", encoding="utf-8")
    assert verify_bead_graph.main(["--export", str(export), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert isinstance(report, dict)
    assert any(item["kind"] == expected_kind for item in report["findings"])


def _campaign_fixture() -> list[dict[str, Any]]:
    root: dict[str, Any] = {
        "id": "polylogue-reindex-2026",
        "status": "open",
        "labels": ["campaign:reindex-2026", "campaign-role:milestone"],
        "metadata": {
            "campaign_id": "reindex-2026",
            "campaign_schema": "polylogue.campaign-control.v1",
            "implementation_lane_wip": 6,
            "merge_train_wip": 1,
            "workstream_active_batch_wip": 1,
            "source_control_plane_sha256": "df6ce27a046a6d95252282124f3f889fed41f0f6fce7b10bed462f5b66dc0d7f",
        },
        "dependencies": [],
    }
    issues: list[dict[str, Any]] = [root]
    member_ids: list[str] = []
    for workstream in "abcdefgh":
        epic_id = f"polylogue-reindex-ws-{workstream}"
        issues.append(
            {
                "id": epic_id,
                "status": "open",
                "labels": [
                    "campaign:reindex-2026",
                    f"workstream:{workstream}",
                    "campaign-role:closure-gate",
                ],
                "metadata": {
                    "campaign_id": "reindex-2026",
                    "workstream": workstream,
                    "epic_semantics": "closure-gate-not-executable",
                },
                "dependencies": [],
            }
        )
        root["dependencies"].append({"type": "blocks", "depends_on_id": epic_id})
        member_id = f"member-{workstream}"
        member_ids.append(member_id)
        issues.append(
            {
                "id": member_id,
                "status": "open",
                "labels": [
                    "campaign:reindex-2026",
                    f"workstream:{workstream}",
                    "campaign-role:plane",
                    "timing:prep",
                ],
                "metadata": {
                    "campaign_id": "reindex-2026",
                    "campaign_role": "plane",
                    "campaign_timing": "prep",
                    "campaign_workstreams": workstream.upper(),
                    "campaign_membership_source": "fixture",
                },
                "dependencies": [],
            }
        )
        issues[-2]["dependencies"].append({"type": "blocks", "depends_on_id": member_id})
    for suffix in ("a-2", "a-3"):
        member_id = f"member-{suffix}"
        issues.append(
            {
                "id": member_id,
                "status": "open",
                "labels": ["campaign:reindex-2026", "workstream:a", "campaign-role:plane", "timing:prep"],
                "metadata": {
                    "campaign_id": "reindex-2026",
                    "campaign_role": "plane",
                    "campaign_timing": "prep",
                    "campaign_workstreams": "A",
                    "campaign_membership_source": "fixture",
                },
                "dependencies": [],
            }
        )
        next(issue for issue in issues if issue["id"] == "polylogue-reindex-ws-a")["dependencies"].append(
            {"type": "blocks", "depends_on_id": member_id}
        )
    return issues


def _campaign_kinds(issues: list[dict[str, object]]) -> set[str]:
    return {finding.kind for finding in verify_bead_graph.collect_campaign_findings(issues)}


def test_campaign_projection_accepts_native_bindings_and_edges() -> None:
    findings = verify_bead_graph.collect_campaign_findings(_campaign_fixture())
    assert {finding.kind for finding in findings} <= {
        "campaign-source-anchor",
        "campaign-source-census",
        "campaign-agentctl-provenance",
        "campaign-agentctl-edge",
    }


def test_campaign_projection_rejects_missing_binding_extra_row_and_bad_edge() -> None:
    missing = deepcopy(_campaign_fixture())
    next(issue for issue in missing if issue["id"] == "member-a")["labels"] = ["workstream:a"]
    assert "campaign-missing-binding" in _campaign_kinds(missing)

    extra = deepcopy(_campaign_fixture())
    extra.append(
        {
            "id": "unowned",
            "status": "open",
            "labels": ["campaign:reindex-2026", "workstream:a"],
            "metadata": {"campaign_id": "reindex-2026"},
            "dependencies": [],
        }
    )
    assert "campaign-extra-unowned" in _campaign_kinds(extra)

    staged_without_binding = deepcopy(_campaign_fixture())
    staged_without_binding.append(
        {
            "id": "polylogue-agentctl-adapter",
            "status": "open",
            "labels": ["batch:agentctl-adapter", "workstream:e"],
            "metadata": {"campaign_id": "reindex-2026", "authoritative_beads": "member-e", "stage": "prepare"},
            "dependencies": [],
        }
    )
    assert "campaign-missing-binding" in _campaign_kinds(staged_without_binding)

    agentctl = deepcopy(_campaign_fixture())
    agentctl.append(
        {
            "id": "polylogue-agentctl-adapter",
            "status": "open",
            "labels": [
                "campaign:reindex-2026",
                "workstream:e",
                "campaign-role:implementation",
                "timing:prep",
            ],
            "metadata": {
                "native_control_ids": [
                    "polylogue-reindex-native-control-plane",
                    "polylogue-reindex-ws-e",
                    "polylogue-reindex-2026",
                ],
                "offline_follow_up": True,
            },
            "dependencies": [],
        }
    )
    next(issue for issue in agentctl if issue["id"] == "polylogue-reindex-ws-e")["dependencies"].append(
        {"type": "blocks", "depends_on_id": "polylogue-agentctl-adapter"}
    )
    assert "campaign-missing-binding" in _campaign_kinds(agentctl)

    corrected_agentctl = deepcopy(_campaign_fixture())
    root = next(issue for issue in corrected_agentctl if issue["id"] == "polylogue-reindex-2026")
    root["metadata"]["source_control_plane_sha256"] = "df6ce27a046a6d95252282124f3f889fed41f0f6fce7b10bed462f5b66dc0d7f"
    corrected_agentctl.append(
        {
            "id": "polylogue-reindex-native-control-plane",
            "status": "open",
            "labels": ["campaign:reindex-2026", "workstream:e", "campaign-role:implementation", "timing:prep"],
            "metadata": {
                "campaign_id": "reindex-2026",
                "workstream": "E",
                "campaign_membership_source": "native-control-plane",
                "source_attachment_sha256": "df6ce27a046a6d95252282124f3f889fed41f0f6fce7b10bed462f5b66dc0d7f",
            },
            "dependencies": [],
        }
    )
    corrected_agentctl.append(
        {
            "id": "polylogue-agentctl-adapter",
            "status": "open",
            "labels": [
                "campaign:reindex-2026",
                "workstream:e",
                "campaign-role:implementation",
                "timing:prep",
            ],
            "metadata": {
                "campaign_id": "reindex-2026",
                "campaign_role": "implementation",
                "campaign_timing": "prep",
                "campaign_workstreams": "E",
                "campaign_membership_source": "agentctl:staged-native-adapter",
                "campaign_membership_kind": "staged-adapter",
                "native_control_ids": [
                    "polylogue-reindex-native-control-plane",
                    "polylogue-reindex-ws-e",
                    "polylogue-reindex-2026",
                ],
            },
            "dependencies": [
                {"type": "blocks", "depends_on_id": "polylogue-reindex-native-control-plane"},
            ],
        }
    )
    ws_e = next(issue for issue in corrected_agentctl if issue["id"] == "polylogue-reindex-ws-e")
    ws_e["dependencies"].extend(
        [
            {"type": "blocks", "depends_on_id": "polylogue-agentctl-adapter"},
            {"type": "blocks", "depends_on_id": "polylogue-reindex-native-control-plane"},
        ]
    )
    root["dependencies"].append({"type": "blocks", "depends_on_id": "polylogue-reindex-native-control-plane"})
    assert not any(
        finding.bead_id == "polylogue-agentctl-adapter"
        for finding in verify_bead_graph.collect_campaign_findings(corrected_agentctl)
    )

    production_agentctl = deepcopy(corrected_agentctl)
    adapter = next(issue for issue in production_agentctl if issue["id"] == "polylogue-agentctl-adapter")
    adapter["metadata"]["campaign_membership_source"] = "agentctl:production-native-adapter"
    adapter["metadata"]["campaign_membership_kind"] = "production-adapter"
    assert not any(
        finding.bead_id == "polylogue-agentctl-adapter"
        for finding in verify_bead_graph.collect_campaign_findings(production_agentctl)
    )

    forged_provenance = deepcopy(corrected_agentctl)
    adapter = next(issue for issue in forged_provenance if issue["id"] == "polylogue-agentctl-adapter")
    adapter["metadata"]["native_control_ids"] = ["polylogue-reindex-ws-e", "forged"]
    assert "campaign-agentctl-provenance" in _campaign_kinds(forged_provenance)

    missing_edge = deepcopy(corrected_agentctl)
    adapter = next(issue for issue in missing_edge if issue["id"] == "polylogue-agentctl-adapter")
    adapter["dependencies"] = []
    assert "campaign-agentctl-edge" in _campaign_kinds(missing_edge)

    demoted = deepcopy(corrected_agentctl)
    adapter = next(issue for issue in demoted if issue["id"] == "polylogue-agentctl-adapter")
    adapter["metadata"].pop("campaign_membership_kind")
    assert "campaign-agentctl-provenance" in _campaign_kinds(demoted)

    bad_edge = deepcopy(_campaign_fixture())
    member = next(issue for issue in bad_edge if issue["id"] == "member-a")
    member["labels"] = ["campaign:reindex-2026", "workstream:b"]
    assert "campaign-workstream-edge" in _campaign_kinds(bad_edge)

    ownership_hardening = deepcopy(_campaign_fixture())
    member = next(issue for issue in ownership_hardening if issue["id"] == "member-a")
    member["labels"] = []
    assert "campaign-missing-binding" in _campaign_kinds(ownership_hardening)


def test_campaign_anchor_and_census_reject_identity_substitution() -> None:
    revision, path = verify_bead_graph.CAMPAIGN_SOURCE_REF.split(":", 1)
    mutated = verify_bead_graph._parse_jsonl_bytes(
        verify_bead_graph._git_blob(revision, path), source=verify_bead_graph.CAMPAIGN_SOURCE_REF
    )
    member = next(
        issue
        for issue in mutated
        if "campaign:reindex-2026" in issue.get("labels", [])
        and issue["id"] not in {"polylogue-reindex-2026", "polylogue-reindex-native-control-plane"}
        and not str(issue["id"]).startswith("polylogue-reindex-ws-")
    )
    member["id"] = "identity-substitution"
    findings = verify_bead_graph.collect_campaign_findings(mutated)
    assert any(f.kind == "campaign-source-census" for f in findings)

    missing_anchor = deepcopy(_campaign_fixture())
    root = next(issue for issue in missing_anchor if issue["id"] == "polylogue-reindex-2026")
    root["metadata"]["source_control_plane_sha256"] = "wrong-anchor"
    assert any(f.kind == "campaign-source-anchor" for f in verify_bead_graph.collect_campaign_findings(missing_anchor))

    simultaneous_adapter_deletion = deepcopy(_campaign_fixture())
    root = next(issue for issue in simultaneous_adapter_deletion if issue["id"] == "polylogue-reindex-2026")
    root["metadata"].pop("source_control_plane_sha256")
    assert any(
        f.kind == "campaign-source-anchor"
        for f in verify_bead_graph.collect_campaign_findings(simultaneous_adapter_deletion)
    )


def test_cli_export_reports_combined_anchor_removal_and_adapter_demotion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    issues = deepcopy(_campaign_fixture())
    root = next(issue for issue in issues if issue["id"] == verify_bead_graph.CAMPAIGN_ROOT)
    root["metadata"]["source_control_plane_sha256"] = "df6ce27a046a6d95252282124f3f889fed41f0f6fce7b10bed462f5b66dc0d7f"
    issues.extend(
        [
            {
                "id": verify_bead_graph.CAMPAIGN_NATIVE_CONTROL_ID,
                "status": "open",
                "labels": ["campaign:reindex-2026", "workstream:e", "campaign-role:implementation", "timing:prep"],
                "metadata": {
                    "campaign_id": "reindex-2026",
                    "workstream": "E",
                    "campaign_membership_source": "native-control-plane",
                    "source_attachment_sha256": root["metadata"]["source_control_plane_sha256"],
                },
                "dependencies": [],
            },
            {
                "id": verify_bead_graph.CAMPAIGN_ADAPTER_ID,
                "status": "open",
                "labels": ["campaign:reindex-2026", "workstream:e", "campaign-role:implementation", "timing:prep"],
                "metadata": {
                    "campaign_id": "reindex-2026",
                    "campaign_role": "implementation",
                    "campaign_timing": "prep",
                    "campaign_workstreams": "E",
                    "campaign_membership_source": "agentctl:staged-native-adapter",
                    "campaign_membership_kind": "staged-adapter",
                    "native_control_ids": [
                        verify_bead_graph.CAMPAIGN_NATIVE_CONTROL_ID,
                        "polylogue-reindex-ws-e",
                        verify_bead_graph.CAMPAIGN_ROOT,
                    ],
                },
                "dependencies": [
                    {"type": "blocks", "depends_on_id": verify_bead_graph.CAMPAIGN_NATIVE_CONTROL_ID},
                ],
            },
        ]
    )
    root["dependencies"].append({"type": "blocks", "depends_on_id": verify_bead_graph.CAMPAIGN_NATIVE_CONTROL_ID})
    workstream = next(issue for issue in issues if issue["id"] == "polylogue-reindex-ws-e")
    workstream["dependencies"].extend(
        [
            {"type": "blocks", "depends_on_id": verify_bead_graph.CAMPAIGN_NATIVE_CONTROL_ID},
            {"type": "blocks", "depends_on_id": verify_bead_graph.CAMPAIGN_ADAPTER_ID},
        ]
    )

    # Apply both mutations before entering the real CLI export route.
    root["metadata"].pop("source_control_plane_sha256")
    adapter = next(issue for issue in issues if issue["id"] == verify_bead_graph.CAMPAIGN_ADAPTER_ID)
    adapter["metadata"].pop("campaign_membership_kind")
    export = tmp_path / "combined-mutation.jsonl"
    export.write_text("\n".join(json.dumps(issue) for issue in issues) + "\n", encoding="utf-8")

    assert verify_bead_graph.main(["--export", str(export), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    findings = report["findings"]
    assert any(item["kind"] == "campaign-source-anchor" for item in findings)
    assert any(
        item["kind"] == "campaign-agentctl-provenance" and item["id"] == verify_bead_graph.CAMPAIGN_ADAPTER_ID
        for item in findings
    )


def test_campaign_projection_handles_malformed_nested_fields_without_traceback() -> None:
    malformed = deepcopy(_campaign_fixture())
    root = next(issue for issue in malformed if issue["id"] == "polylogue-reindex-2026")
    root["metadata"] = {"source_control_plane_sha256": ["not", "a", "digest"]}
    epic = next(issue for issue in malformed if issue["id"] == "polylogue-reindex-ws-a")
    epic["metadata"]["workstream"] = {"not": "a string"}
    malformed.append(
        {
            "id": "malformed-nested",
            "labels": [None, {"label": "bad"}],
            "metadata": ["bad"],
            "dependencies": [{"type": ["blocks"], "depends_on_id": {"id": "bad"}}],
        }
    )
    findings = verify_bead_graph.collect_campaign_findings(malformed)
    assert findings
    assert any(f.kind in {"campaign-source-anchor", "malformed-metadata", "malformed-label"} for f in findings)


def test_campaign_projection_requires_root_to_block_every_workstream_gate() -> None:
    missing_root_edge = deepcopy(_campaign_fixture())
    root = next(issue for issue in missing_root_edge if issue["id"] == "polylogue-reindex-2026")
    root["dependencies"] = [
        dependency for dependency in root["dependencies"] if dependency["depends_on_id"] != "polylogue-reindex-ws-h"
    ]
    assert "campaign-workstream-edge" in _campaign_kinds(missing_root_edge)

    extra_root_blocker = deepcopy(_campaign_fixture())
    extra_root_blocker.append({"id": "unowned-root-blocker", "status": "open", "dependencies": []})
    root = next(issue for issue in extra_root_blocker if issue["id"] == "polylogue-reindex-2026")
    root["dependencies"].append({"type": "blocks", "depends_on_id": "unowned-root-blocker"})
    assert "campaign-root-edge" in _campaign_kinds(extra_root_blocker)


def test_campaign_projection_rejects_malformed_root_wip_metadata() -> None:
    malformed = deepcopy(_campaign_fixture())
    root = next(issue for issue in malformed if issue["id"] == "polylogue-reindex-2026")
    root["metadata"]["merge_train_wip"] = "one"
    findings = verify_bead_graph.collect_campaign_findings(malformed)
    assert any(finding.kind == "campaign-wip-metadata" for finding in findings)


def test_campaign_projection_does_not_skip_when_campaign_labels_disappear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels_removed = deepcopy(_campaign_fixture())
    root = next(issue for issue in labels_removed if issue["id"] == "polylogue-reindex-2026")
    root["labels"] = []
    assert "campaign-missing-binding" in _campaign_kinds(labels_removed)
    monkeypatch.setattr(verify_bead_graph, "_registry_findings", lambda _issues: [])
    report = verify_bead_graph.build_report(labels_removed, cycles_ok=True, cycles_output="")
    assert any(item["kind"] == "campaign-missing-binding" for item in report["findings"])


def _batch_row(batch: str, *, stage: str, status: str = "in_progress") -> dict[str, Any]:
    return {
        "id": f"batch-{batch}-{stage}",
        "issue_type": "task",
        "status": status,
        "labels": ["campaign:reindex-2026", "campaign-role:batch", "timing:execution"],
        "metadata": {
            "campaign_id": "reindex-2026",
            "formula": "mol-polylogue-thematic-batch",
            "formula_version": 1,
            "molecule_type": "workflow",
            "pour_origin": "native-formula",
            "campaign_role": "batch",
            "campaign_timing": "execution",
            "stage": stage,
        },
        "description": (
            f"NATIVE_BATCH_BINDING_V1 batch={batch}\n"
            "NATIVE_BATCH_BINDING_V1 workstream=a\n"
            "NATIVE_BATCH_BINDING_V1 authoritative_beads=member-a,member-a-2,member-a-3"
        ),
        "dependencies": [{"type": "parent-child", "depends_on_id": f"molecule-{batch}"}],
    }


def _batch_group(batch: str, *, status: str = "in_progress") -> list[dict[str, Any]]:
    root_id = f"molecule-{batch}"
    gate_id = f"gate-{batch}"
    rows = [
        {
            "id": root_id,
            "issue_type": "molecule",
            "title": verify_bead_graph.BATCH_FORMULA,
            "status": status,
            "metadata": {
                "formula": verify_bead_graph.BATCH_FORMULA,
                "formula_version": 1,
                "molecule_type": "workflow",
                "pour_origin": "native-formula",
                "campaign_id": "reindex-2026",
            },
            "dependencies": [],
        },
        *(_batch_row(batch, stage=stage, status=status) for stage in verify_bead_graph.BATCH_STAGES),
        {
            "id": gate_id,
            "issue_type": "gate",
            "title": "Gate: human operator-merge-authorization",
            "status": status,
            "await_type": "human",
            "await_id": "operator-merge-authorization",
            "dependencies": [{"type": "parent-child", "depends_on_id": root_id}],
        },
    ]
    by_stage = {row["metadata"]["stage"]: row for row in rows if row.get("metadata", {}).get("stage")}
    for stage, predecessor in {
        "implement": "prepare",
        "verify": "implement",
        "merge": "verify",
        "dispositions": "merge",
    }.items():
        by_stage[stage]["dependencies"].append({"type": "blocks", "depends_on_id": by_stage[predecessor]["id"]})
    by_stage["merge"]["dependencies"].append({"type": "blocks", "depends_on_id": gate_id})
    return rows


def test_campaign_projection_rejects_batch_authority_shape_and_workstream() -> None:
    fake_batch = _campaign_fixture()
    row = _batch_row("fake", stage="prepare")
    for key in ("formula", "formula_version", "molecule_type", "pour_origin"):
        row["metadata"].pop(key)
    fake_batch.append(row)
    assert "campaign-batch-binding" in _campaign_kinds(fake_batch)

    malformed = _campaign_fixture()
    row = _batch_row("malformed", stage="prepare")
    row["metadata"]["authoritative_beads"] = "member-a"
    malformed.append(row)
    assert "campaign-batch-binding" in _campaign_kinds(malformed)

    verify_short = _campaign_fixture()
    row = _batch_row("verify-short", stage="verify")
    row["metadata"]["authoritative_beads"] = "member-a"
    verify_short.append(row)
    assert "campaign-batch-binding" in _campaign_kinds(verify_short)

    cross_workstream = _campaign_fixture()
    row = _batch_row("cross-workstream", stage="prepare")
    row["metadata"]["authoritative_beads"] = "member-a,member-b,member-c"
    cross_workstream.append(row)
    assert "campaign-batch-binding" in _campaign_kinds(cross_workstream)

    copied_fields = _campaign_fixture()
    copied_fields.append(
        {"id": "copied-molecule", "issue_type": "task", "title": verify_bead_graph.BATCH_FORMULA, "dependencies": []}
    )
    copied_fields.append(_batch_row("copied", stage="prepare"))
    copied_fields[-1]["dependencies"] = [{"type": "parent-child", "depends_on_id": "copied-molecule"}]
    assert "campaign-batch-binding" in _campaign_kinds(copied_fields)


def test_campaign_projection_rejects_invalid_native_batch_name() -> None:
    malformed = _campaign_fixture() + _batch_group("valid")
    for row in malformed:
        if row.get("metadata", {}).get("stage") in verify_bead_graph.BATCH_STAGES:
            row["description"] = row["description"].replace("batch=valid", "batch=Invalid Name")
    assert "campaign-batch-binding" in _campaign_kinds(malformed)


def test_campaign_projection_rejects_wip_limits() -> None:
    batches = _campaign_fixture()
    for name in "abcdefg":
        group = _batch_group(name)
        batches.extend(group)
        for row in group:
            if row.get("metadata", {}).get("stage") != "implement":
                row["status"] = "closed"
    batch_findings = verify_bead_graph.collect_campaign_findings(batches)
    kinds = {finding.kind for finding in batch_findings}
    assert {"campaign-active-batch-wip", "campaign-implementation-lane-wip"} <= kinds, batch_findings

    merge_ready = _campaign_fixture()
    for name in ("one", "two"):
        group = _batch_group(name)
        merge_ready.extend(group)
        for row in group:
            if row.get("metadata", {}).get("stage") != "merge":
                row["status"] = "closed"
    assert "campaign-merge-train-wip" in _campaign_kinds(merge_ready)


def test_thematic_batch_formula_is_native_pourable_shape() -> None:
    formula = json.loads(
        (Path(__file__).parents[3] / ".beads" / "formulas" / "mol-polylogue-thematic-batch.formula.json").read_text(
            encoding="utf-8"
        )
    )
    assert formula["type"] == "workflow"
    assert formula["pour"] is True
    assert [step["id"] for step in formula["steps"]] == ["prepare", "implement", "verify", "merge", "dispositions"]
    assert formula["steps"][1]["needs"] == ["prepare"]
    assert formula["steps"][4]["needs"] == ["merge"]
    for step in formula["steps"]:
        assert set(step["labels"]) == verify_bead_graph.BATCH_LABELS
        assert not any("{{" in label for label in step["labels"])
        metadata = step["metadata"]
        assert metadata["campaign_id"] == "reindex-2026"
        assert metadata["formula"] == "mol-polylogue-thematic-batch"
        assert metadata["formula_version"] == 1
        assert metadata["molecule_type"] == "workflow"
        assert metadata["pour_origin"] == "native-formula"
        assert metadata["campaign_role"] == "batch"
        assert metadata["campaign_timing"] == "execution"
        assert "NATIVE_BATCH_BINDING_V1 authoritative_beads={{beads}}" in step["description"]
        assert "NATIVE_BATCH_BINDING_V1 workstream={{workstream}}" in step["description"]


def test_native_formula_pour_materializes_supported_bindings_and_edges(tmp_path: Path) -> None:
    """Exercise the installed bd formula/pour path in an isolated repository."""
    bd = shutil.which("bd")
    if bd is None:
        pytest.skip("native bd executable is unavailable")
    repo = tmp_path / "native-beads"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        [bd, "init", "--non-interactive", "--skip-agents", "--skip-hooks", "--quiet"],
        cwd=repo,
        check=True,
        text=True,
    )
    formula_dir = repo / ".beads" / "formulas"
    formula_dir.mkdir()
    source_formula = Path(__file__).parents[3] / ".beads" / "formulas" / "mol-polylogue-thematic-batch.formula.json"
    shutil.copy2(source_formula, formula_dir / source_formula.name)
    vars_ = [
        "batch=demo",
        "workstream=a",
        "beads=member-a,member-a-2,member-a-3",
        "branch=feature/demo",
        "verification=devtools test tests/unit/devtools/test_verify_bead_graph.py",
    ]
    poured = subprocess.run(
        [bd, "mol", "pour", "mol-polylogue-thematic-batch", *sum((["--var", value] for value in vars_), []), "--json"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    receipt = json.loads(poured.stdout)
    assert receipt["created"] == 7
    records = json.loads(
        subprocess.run(
            [bd, "list", "--all", "--include-gates", "-n", "0", "--json"],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    )
    root = next(record for record in records if record["id"] == receipt["new_epic_id"])
    assert root["issue_type"] == "molecule"
    assert root["title"] == verify_bead_graph.BATCH_FORMULA
    children = [
        record
        for record in records
        if any(
            dependency.get("type") == "parent-child" and dependency.get("depends_on_id") == root["id"]
            for dependency in record.get("dependencies", [])
        )
    ]
    stage_children = [record for record in children if record.get("issue_type") == "task"]
    gate_children = [record for record in children if record.get("issue_type") == "gate"]
    assert {record["metadata"]["stage"] for record in stage_children} == verify_bead_graph.BATCH_STAGES
    assert len(gate_children) == 1
    assert gate_children[0]["await_type"] == "human"
    assert gate_children[0]["await_id"] == "operator-merge-authorization"
    assert all(set(record["labels"]) == verify_bead_graph.BATCH_LABELS for record in stage_children)
    assert all("{{" not in json.dumps(record) for record in children)
    assert all(
        verify_bead_graph._binding_fields(record)
        == {
            "batch": "demo",
            "workstream": "a",
            "authoritative_beads": "member-a,member-a-2,member-a-3",
            **({"branch": "feature/demo"} if record["metadata"]["stage"] in {"prepare", "implement"} else {}),
            **(
                {"verification": "devtools test tests/unit/devtools/test_verify_bead_graph.py"}
                if record["metadata"]["stage"] == "verify"
                else {}
            ),
        }
        for record in stage_children
    )
    by_stage = {record["metadata"]["stage"]: record for record in stage_children}
    assert any(
        dependency.get("type") == "blocks" and dependency.get("depends_on_id") == by_stage["prepare"]["id"]
        for dependency in by_stage["implement"].get("dependencies", [])
    )
    assert any(
        dependency.get("type") == "blocks" and dependency.get("depends_on_id") == by_stage["implement"]["id"]
        for dependency in by_stage["verify"].get("dependencies", [])
    )
    assert any(
        dependency.get("type") == "blocks" and dependency.get("depends_on_id") == by_stage["verify"]["id"]
        for dependency in by_stage["merge"].get("dependencies", [])
    )
    assert any(
        dependency.get("type") == "blocks" and dependency.get("depends_on_id") == by_stage["merge"]["id"]
        for dependency in by_stage["dispositions"].get("dependencies", [])
    )
    combined_findings = verify_bead_graph.collect_campaign_findings(_campaign_fixture() + records)
    poured_ids = {str(record["id"]) for record in records}
    assert not [finding for finding in combined_findings if finding.bead_id in poured_ids], combined_findings


def _commit(repo: Path, message: str) -> str:
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Tests", "-c", "user.email=tests@example.invalid", "commit", "-qm", message],
        cwd=repo,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()


def _write_snapshot(repo: Path, issues: list[dict[str, Any]]) -> None:
    beads = repo / ".beads"
    beads.mkdir(exist_ok=True)
    (beads / "issues.jsonl").write_text(
        "".join(json.dumps(issue, sort_keys=True) + "\n" for issue in issues), encoding="utf-8"
    )


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _revision_pinned_genesis(repo: Path, input_revision: str, migration_revision: str, formula_revision: str) -> None:
    input_blob = subprocess.run(
        ["git", "show", f"{input_revision}:.beads/issues.jsonl"], cwd=repo, check=True, capture_output=True
    ).stdout
    migration_blob = subprocess.run(
        ["git", "show", f"{migration_revision}:.beads/issues.jsonl"], cwd=repo, check=True, capture_output=True
    ).stdout
    formula_blob = subprocess.run(
        ["git", "show", f"{formula_revision}:.beads/formulas/mol-polylogue-thematic-batch.formula.json"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    genesis = {
        "schema": verify_bead_graph.CAMPAIGN_GENESIS_SCHEMA,
        "campaign_id": verify_bead_graph.CAMPAIGN_ID,
        "input_snapshot": {
            "revision": input_revision,
            "path": ".beads/issues.jsonl",
            "sha256": _sha256_bytes(input_blob),
        },
        "migration_snapshot": {
            "revision": migration_revision,
            "path": ".beads/issues.jsonl",
            "sha256": _sha256_bytes(migration_blob),
        },
        "formula_snapshot": {
            "revision": formula_revision,
            "path": ".beads/formulas/mol-polylogue-thematic-batch.formula.json",
            "sha256": _sha256_bytes(formula_blob),
        },
    }
    path = repo / verify_bead_graph.CAMPAIGN_GENESIS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(genesis, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _acceptance_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "acceptance"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _write_snapshot(repo, [_issue("before")])
    input_revision = _commit(repo, "input")
    _write_snapshot(repo, [_issue("after")])
    formula_path = repo / ".beads" / "formulas" / "mol-polylogue-thematic-batch.formula.json"
    formula_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).parents[3] / formula_path.relative_to(repo), formula_path)
    migration_revision = _commit(repo, "migration")
    _revision_pinned_genesis(repo, input_revision, migration_revision, migration_revision)
    _commit(repo, "acceptance genesis")
    return repo


def test_revision_pinned_campaign_acceptance_derives_and_verifies_named_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _acceptance_repo(tmp_path)
    monkeypatch.chdir(repo)
    monkeypatch.setattr(verify_bead_graph, "_registry_findings", lambda _issues: [])
    output_dir = tmp_path / "witnesses"
    provisional = output_dir / "placeholder.json"
    assert verify_bead_graph.main(["--revision", "HEAD", "--acceptance-output", str(provisional), "--json"]) == 1
    error = json.loads(capsys.readouterr().out)["error"]
    assert "acceptance output must be named" in error

    revision, snapshot, issues = verify_bead_graph._load_revision_export("HEAD")
    report = verify_bead_graph.build_report(issues, cycles_ok=True, cycles_output="")
    acceptance = verify_bead_graph._build_campaign_acceptance(revision, snapshot, issues, report)
    output = output_dir / f"reindex-2026-acceptance-{acceptance['acceptance_sha256']}.json"
    assert verify_bead_graph.main(["--revision", "HEAD", "--acceptance-output", str(output), "--json"]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["campaign_acceptance"]["derivation"]["changed_record_count"] == 2
    assert verify_bead_graph.main(["--verify-acceptance-output", str(output), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True

    tampered = json.loads(output.read_text(encoding="utf-8"))
    tampered["snapshot"]["revision"] = "forged"
    output.write_text(json.dumps(tampered), encoding="utf-8")
    assert verify_bead_graph.main(["--verify-acceptance-output", str(output), "--json"]) == 1
    assert "acceptance output digest is invalid" in json.loads(capsys.readouterr().out)["error"]


def test_revision_pinned_campaign_acceptance_rejects_hand_forged_molecule_without_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _acceptance_repo(tmp_path)
    _write_snapshot(repo, [_issue("after"), *_batch_group("forged")])
    _commit(repo, "forge molecule")
    monkeypatch.chdir(repo)
    monkeypatch.setattr(verify_bead_graph, "_registry_findings", lambda _issues: [])
    revision, snapshot, issues = verify_bead_graph._load_revision_export("HEAD")
    report = verify_bead_graph.build_report(issues, cycles_ok=True, cycles_output="")
    acceptance = verify_bead_graph._build_campaign_acceptance(revision, snapshot, issues, report)
    output = tmp_path / f"reindex-2026-acceptance-{acceptance['acceptance_sha256']}.json"
    assert verify_bead_graph.main(["--revision", "HEAD", "--acceptance-output", str(output), "--json"]) == 1
    emitted = json.loads(capsys.readouterr().out)
    assert any(item["kind"] == "campaign-pour-receipt" for item in emitted["campaign_acceptance"]["findings"])
