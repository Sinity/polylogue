"""A declared source root that narrows or moves fails the frontier check.

Anti-vacuity: every assertion here goes red if ``check_frontier`` ever reports
a deleted member, an emptied root or a vanished root as an ordinary smaller
sample set -- the silent narrowing that let a stale provider package survive
unnoticed. Making the recorded baseline advisory, dropping the refusal in
``observe_root``, or letting the ``append`` mutability downgrade a *missing*
member to a notice each turn one of these tests red.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.core.schema_subjects import INFERENCE_EXCLUDED_SUBJECTS
from polylogue.schemas.source_frontier import (
    FRONTIER_SCHEMA,
    FrontierExclusion,
    FrontierRoot,
    FrontierSubject,
    SchemaFrontier,
    SchemaFrontierError,
    check_frontier,
    frontier_from_payload,
    frontier_source_inputs,
    load_frontier,
    record_frontier,
    write_frontier,
)


def write_member(root: Path, name: str, *, marker: str = "a") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "session_meta", "id": marker}) + "\n", encoding="utf-8")
    return path


def declare(
    root: Path,
    *,
    exclusions: tuple[FrontierExclusion, ...] = (),
    admit: tuple[str, ...] = (),
    mutability: str = "frozen",
    zero_material_reason: str | None = None,
) -> SchemaFrontier:
    declared = FrontierRoot(
        path=root,
        scope="synthetic root",
        exclusions=exclusions,
        admit=admit,
        mutability=mutability,
        zero_material_reason=zero_material_reason,
    )
    return SchemaFrontier(subjects=(FrontierSubject("codex", (declared,)),))


@pytest.fixture
def recorded(tmp_path: Path) -> tuple[SchemaFrontier, Path]:
    root = tmp_path / "codex"
    write_member(root, "2026/09/15/rollout-a.jsonl", marker="a")
    write_member(root, "2026/09/15/rollout-b.jsonl", marker="b")
    return record_frontier(declare(root)), root


def test_recorded_baseline_is_conserved_across_reads(recorded: tuple[SchemaFrontier, Path], tmp_path: Path) -> None:
    frontier, _root = recorded
    document = tmp_path / "frontier.json"
    write_frontier(frontier, document)

    first = load_frontier(document)
    second = load_frontier(document)

    assert first.baseline_digest == frontier.baseline_digest
    assert second.baseline_digest == first.baseline_digest
    assert check_frontier(first).ok


def test_deleting_a_declared_member_is_red(recorded: tuple[SchemaFrontier, Path]) -> None:
    frontier, root = recorded
    (root / "2026/09/15/rollout-b.jsonl").unlink()

    check = check_frontier(frontier)

    assert not check.ok
    assert [finding.kind for finding in check.errors] == ["member_missing"]
    assert check.baseline_digest == frontier.baseline_digest


def test_moving_the_declared_root_is_red(recorded: tuple[SchemaFrontier, Path]) -> None:
    frontier, root = recorded
    root.rename(root.parent / "codex-moved")

    check = check_frontier(frontier)

    assert not check.ok
    assert [finding.kind for finding in check.errors] == ["root_unresolvable"]


def test_a_live_root_tolerates_growth_but_not_loss(tmp_path: Path) -> None:
    root = tmp_path / "codex"
    write_member(root, "rollout-a.jsonl", marker="a")
    frontier = record_frontier(declare(root, mutability="append"))

    write_member(root, "rollout-b.jsonl", marker="b")
    (root / "rollout-a.jsonl").write_text('{"type":"session_meta","id":"a"}\n{"type":"user"}\n', encoding="utf-8")
    grown = check_frontier(frontier)
    assert grown.ok
    assert {finding.kind for finding in grown.notices} == {"member_added", "member_grew"}

    (root / "rollout-a.jsonl").unlink()
    assert [finding.kind for finding in check_frontier(frontier).errors] == ["member_missing"]


def test_an_emptied_root_without_a_declared_reason_is_red(tmp_path: Path) -> None:
    root = tmp_path / "codex"
    write_member(root, "rollout-a.jsonl")
    frontier = record_frontier(declare(root))
    (root / "rollout-a.jsonl").unlink()

    kinds = [finding.kind for finding in check_frontier(frontier).errors]

    assert "root_empty" in kinds
    assert "member_missing" in kinds


def test_a_declared_zero_material_root_is_green_while_it_stays_empty(tmp_path: Path) -> None:
    root = tmp_path / "inbox"
    root.mkdir()
    frontier = record_frontier(declare(root, zero_material_reason="no export has ever been acquired"))

    assert check_frontier(frontier).ok
    assert frontier.baselines[0].member_count == 0


def test_exclusions_keep_deferred_material_out_of_the_denominator(tmp_path: Path) -> None:
    root = tmp_path / "chatgpt"
    root.mkdir()
    (root / "chatgpt-data-2026.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    write_member(root, "capture-0001.json")
    frontier = record_frontier(
        declare(
            root,
            exclusions=(FrontierExclusion("*.json", "capture-origin material", owner="reindex-2026.1"),),
        )
    )

    assert [member.relative for member in frontier.baselines[0].members] == ["chatgpt-data-2026.zip"]
    assert check_frontier(frontier).ok


def test_an_edited_document_fails_its_own_recorded_digest(recorded: tuple[SchemaFrontier, Path]) -> None:
    frontier, _root = recorded
    payload = frontier.to_payload()
    baselines = payload["baselines"]
    assert isinstance(baselines, list)
    baselines[0]["members"] = baselines[0]["members"][:1]

    with pytest.raises(SchemaFrontierError, match="recorded baseline digest"):
        frontier_from_payload(payload)


def test_an_unrecorded_frontier_refuses_to_pass_as_a_baseline(tmp_path: Path) -> None:
    root = tmp_path / "codex"
    write_member(root, "rollout-a.jsonl")

    check = check_frontier(declare(root))

    assert not check.ok
    assert {finding.kind for finding in check.errors} == {"baseline_not_recorded", "root_not_recorded"}


def test_declared_inputs_drive_generation_for_the_named_subject(recorded: tuple[SchemaFrontier, Path]) -> None:
    frontier, root = recorded

    inputs = frontier_source_inputs(frontier, "codex")

    assert [(item.provider, item.root) for item in inputs] == [("codex", root)]
    with pytest.raises(SchemaFrontierError, match="no roots for subject"):
        frontier_source_inputs(frontier, "chatgpt")


def test_a_restricted_root_hands_generation_its_recorded_members_not_the_directory(tmp_path: Path) -> None:
    """A shared drop root would otherwise feed every neighbouring file to this subject."""
    root = tmp_path / "inbox"
    write_member(root, "rollout-a.jsonl")
    write_member(root, "someone-elses-export.jsonl")
    frontier = record_frontier(declare(root, admit=("rollout-*.jsonl",)))

    inputs = frontier_source_inputs(frontier, "codex")

    assert [item.root for item in inputs] == [root / "rollout-a.jsonl"]


def test_declaration_refuses_a_subject_outside_the_inference_denominator(tmp_path: Path) -> None:
    """A subject whose wire format this repository authors has a zero
    denominator by declaration. A declared root or a recorded baseline for it
    would be exactly the "eligible material we then refused" the exclusion
    denies exists, so the document is refused rather than quietly carrying a
    member list nothing may infer from.

    Anti-vacuity: drop the guard in ``frontier_from_payload`` and a frontier
    can once again record 1,243 browser-capture members as a denominator.
    """
    excluded = next(iter(INFERENCE_EXCLUDED_SUBJECTS))
    source_root = tmp_path / "spool"
    write_member(source_root, "capture.json")

    with pytest.raises(SchemaFrontierError) as declared:
        frontier_from_payload(
            {
                "schema": FRONTIER_SCHEMA,
                "subjects": [{"subject": excluded, "roots": [{"path": str(source_root), "scope": "spool"}]}],
            }
        )
    assert "outside the schema-inference denominator" in str(declared.value)

    with pytest.raises(SchemaFrontierError) as baseline:
        frontier_from_payload(
            {
                "schema": FRONTIER_SCHEMA,
                "subjects": [{"subject": "codex", "roots": [{"path": str(source_root), "scope": "spool"}]}],
                "baselines": [
                    {
                        "subject": excluded,
                        "root": str(source_root),
                        "members": [{"relative": "capture.json", "byte_count": 1, "sha256": ""}],
                    }
                ],
            }
        )
    assert "must not carry a recorded baseline" in str(baseline.value)
