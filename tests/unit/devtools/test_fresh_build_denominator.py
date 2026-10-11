"""Native staged coverage and weighted product-receipt denominator controls."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

import pytest

from devtools.fresh_build_bench import denominator
from devtools.fresh_build_bench.denominator import inventory, project


@pytest.fixture
def private_path() -> Iterator[Path]:
    # Managed pytest basetemp and TMPDIR sit inside the checkout. These neutral
    # private-output controls need the external operator-managed scratch root.
    managed = Path("/realm/tmp/work")
    with tempfile.TemporaryDirectory(
        prefix="synthetic-denominator-", dir=managed if managed.is_dir() else "/tmp"
    ) as folder:
        yield Path(folder)


def _json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _population(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "home" / ".claude" / "projects"
    project_root = root / "-synthetic"
    project_root.mkdir(parents=True)
    for name, content in (("a", b'{"x":1}\n\n{"x":2}\n'), ("b", b"{}\n"), ("c", b"{}\n{}\n{}\n")):
        (project_root / f"{name}.jsonl").write_bytes(content)
    children = project_root / "a" / "subagents"
    children.mkdir(parents=True)
    (children / "agent-child.jsonl").write_bytes(b"{}\n")
    (project_root / "sessions-index.json").write_text("{}", encoding="utf-8")
    (root.parent / "history.jsonl").write_text("{}\n", encoding="utf-8")
    hooks = tmp_path / "hooks"
    (hooks / "2026-01-01").mkdir(parents=True)
    (hooks / "2026-01-01" / "carrier-1.ndjson").write_bytes(b"{}\n")
    plan = _json(
        tmp_path / "plan.json",
        {
            "format": "polylogue.denominator-plan.v1",
            "frontier": {"id": "synthetic-v1", "basis": "fixed synthetic tree"},
            "roots": [
                {
                    "id": "claude",
                    "origin": "claude-code",
                    "path": str(root),
                    "stratum": "ordinary",
                    "shape": "session-with-children",
                    "population_kind": "normal",
                },
                {
                    "id": "hooks",
                    "origin": "claude-code-hooks",
                    "path": str(hooks),
                    "stratum": "hook-backlog",
                    "shape": "carrier",
                    "population_kind": "exceptional-hooks",
                },
            ],
        },
    )
    return plan, root


def _stages(tmp_path: Path, plan: Path) -> Path:
    previous = None
    for ordinal in range(10):
        stage = tmp_path / f"stage-{ordinal}.jsonl"
        result = inventory(plan, stage, members=2, previous=previous)
        previous = stage
        if result["complete"]:
            return stage
    raise AssertionError("synthetic population did not finish")


def _sample(
    tmp_path: Path, root: Path, name: str, messages: int, *, weight: float, held_out: bool = False
) -> dict[str, Any]:
    unit = f"claude:-synthetic/{name}"
    selection = _json(
        tmp_path / f"selection-{name}.json",
        {
            "frontier_id": "synthetic-v1",
            "unit_id": unit,
            "stratum": "ordinary",
            "shape": "session-with-children",
            "complete_unit": True,
            "closure": _ref(_json(tmp_path / f"closure-{name}.json", {"unit_id": unit, "members": [name]})),
            "normalization_inputs": [
                {"member_id": name, "input_sha256": f"input-{name}", "recipe_sha256": "neutral-recipe"}
            ],
        },
    )
    receipt = _json(
        tmp_path / f"receipt-{name}.json",
        {
            "format": "polylogue.component-receipt.v1",
            "component": "retained-preparation",
            "candidate": {"git_sha": "neutral-candidate", "dirty": False},
            "product_tree": "neutral-tree",
            "runtime": {"path": "neutral-runtime", "sha256": "neutral-hash"},
            "imported_product_modules": {"polylogue.neutral": {"sha256": "neutral-module"}},
            "input_sha256": f"input-{name}",
            "recipe_sha256": "neutral-recipe",
            "output": {
                "normalized": {
                    "count_basis": "production-preparation-before-lineage",
                    "sessions": 1,
                    "messages": messages,
                    "blocks": 2 * messages,
                    "events": 0,
                },
                "status": "passed",
                "preparation": {
                    "state": "complete",
                    "complete_success": True,
                    "input_sha256": f"input-{name}",
                    "accepted_prefix_size": None,
                    "parsed_prefix_size": None,
                    "consumed_session_count": 1,
                },
                "session_content_hashes": ["neutral-content-hash"],
                "verdict": {
                    "sample_count": 1,
                    "error_count": 0,
                    "invalid_count": 0,
                    "drift_count": 0,
                    "strict_refusal": False,
                },
            },
        },
    )
    return {
        "receipts": [_ref(receipt)],
        "selection": _ref(selection),
        "unit_id": unit,
        "frontier_id": "synthetic-v1",
        "stratum": "ordinary",
        "shape": "session-with-children",
        "input_sha256": f"input-{name}",
        "recipe_sha256": "neutral-recipe",
        "complete_unit": True,
        "inclusion_weight": weight,
        "inclusion_probability": 1 / weight,
        "held_out": held_out,
    }


def test_native_stages_preserve_units_shared_metadata_and_stat_only_default(
    private_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = private_path
    plan, root = _population(tmp_path)
    original_open = Path.open

    def no_transcript_read(path: Path, *args: Any, **kwargs: Any) -> IO[Any]:
        if root in path.parents or path.name == "history.jsonl":
            raise AssertionError("stat-only inventory read a transcript or metadata body")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_transcript_read)
    final = _stages(tmp_path, plan)
    calibration = _json(
        tmp_path / "calibration.json",
        {
            "format": "polylogue.denominator-calibration.v1",
            "sampling_design": {"kind": "targeted-model", "mechanism": "no calibration yet"},
            "samples": [],
        },
    )
    result = project(final, calibration, tmp_path / "projection.json")
    assert result["population_complete"] is True
    assert result["input_member_count"] == 6
    assert result["shared_dependency_count"] == 4
    assert result["dependency_states"] == {"absent": 2, "present": 2}
    assert result["additional_shared_bytes"] == 3
    assert result["strata"]["ordinary"]["observed_source_units"] == 4
    assert result["strata"]["hook-backlog"]["observed_source_units"] == 1
    assert result["strata"]["hook-backlog"]["population_kind"] == "exceptional-hooks"
    assert result["strata"]["ordinary"]["estimates"]["messages"] is None
    assert "bytes_read" not in result["structural_coverage"]
    with pytest.raises(ValueError):
        inventory(plan, tmp_path / "beyond.jsonl", members=2, previous=final)
    public_output = Path(denominator.__file__).resolve().parents[2] / ".cache" / f"{tmp_path.name}.jsonl"
    with pytest.raises(ValueError):
        inventory(plan, public_output, members=2)
    assert not public_output.exists()
    original_rows = final.read_text().splitlines()
    truncated = tmp_path / "truncated.jsonl"
    truncated.write_text("\n".join(original_rows[:-1]) + "\n")
    with pytest.raises(ValueError):
        project(truncated, calibration, tmp_path / "truncated-projection.json")
    missing_authority = tmp_path / "missing-authority.jsonl"
    header = json.loads(original_rows[0])
    header["source_authority"].pop(next(iter(header["source_authority"])))
    missing_authority.write_text(json.dumps(header) + "\n" + "\n".join(original_rows[1:]) + "\n")
    with pytest.raises(ValueError):
        project(missing_authority, calibration, tmp_path / "missing-authority-projection.json")


def test_weighted_normalization_and_held_out_error_keep_unknown_denominators(private_path: Path) -> None:
    tmp_path = private_path
    plan, root = _population(tmp_path)
    stage = _stages(tmp_path, plan)
    rows = [
        _sample(tmp_path, root, "a", 10, weight=1),
        _sample(tmp_path, root, "b", 30, weight=3),
        _sample(tmp_path, root, "c", 20, weight=2, held_out=True),
    ]
    calibration = _json(
        tmp_path / "calibration.json",
        {
            "format": "polylogue.denominator-calibration.v1",
            "sampling_design": {"kind": "probability", "mechanism": "explicit synthetic unequal inclusion"},
            "samples": rows,
        },
    )
    result = project(stage, calibration, tmp_path / "projection.json")
    ordinary = result["strata"]["ordinary"]
    assert ordinary["estimates"]["messages"] == 100
    assert ordinary["held_out_error"]["messages"] == {
        "held_out_units": 1,
        "predicted": 25,
        "actual": 20,
        "relative_error": 0.25,
    }
    assert ordinary["estimates"]["revisions"] is None
    assert ordinary["estimates"]["jobs"] is None
    assert ordinary["uncertainty"]["messages"]["confidence_interval"] is None
    assert ordinary["uncertainty"]["messages"]["observed_shape_scenario_range"] == [40, 120]
    assert result["strata"]["hook-backlog"]["held_out_checked_metrics"] == []
    assert not list(tmp_path.glob("denominator-*"))
    assert result["count_basis"] == "production-preparation-before-lineage"
    equal_receipt = _json(
        tmp_path / "equal-byte-receipt.json", json.loads(Path(rows[0]["receipts"][0]["path"]).read_text())
    )
    equal_row = json.loads(json.dumps(rows[0]))
    equal_row["receipts"].append(_ref(equal_receipt))
    selection_path = Path(rows[0]["selection"]["path"])
    equal_selection = json.loads(selection_path.read_text())
    equal_selection["normalization_inputs"].append(
        {"member_id": "child-with-equal-bytes", "input_sha256": "input-a", "recipe_sha256": "neutral-recipe"}
    )
    equal_selection_path = _json(tmp_path / "equal-byte-selection.json", equal_selection)
    equal_row["selection"] = _ref(equal_selection_path)
    equal_calibration = _json(
        tmp_path / "equal-byte-calibration.json",
        {
            "format": "polylogue.denominator-calibration.v1",
            "sampling_design": {"kind": "targeted-model", "mechanism": "two distinct equal-byte unit members"},
            "samples": [equal_row],
        },
    )
    equal_projection = project(stage, equal_calibration, tmp_path / "equal-byte-projection.json")
    assert equal_projection["strata"]["ordinary"]["estimates"]["messages"] == 80
    equal_row["receipts"][1] = equal_row["receipts"][0]
    _json(
        equal_calibration,
        {
            "format": "polylogue.denominator-calibration.v1",
            "sampling_design": {"kind": "targeted-model", "mechanism": "duplicate receipt is not separate work"},
            "samples": [equal_row],
        },
    )
    with pytest.raises(ValueError):
        project(stage, equal_calibration, tmp_path / "duplicate-receipt-projection.json")
    receipt_path = Path(rows[0]["receipts"][0]["path"])
    original_receipt = json.loads(receipt_path.read_text())
    for defect in ("verdict", "status", "prefix", "complete", "basis"):
        receipt = json.loads(json.dumps(original_receipt))
        if defect == "verdict":
            del receipt["output"]["verdict"]
        elif defect == "status":
            receipt["output"]["status"] = "failed"
        elif defect == "prefix":
            receipt["output"]["preparation"]["accepted_prefix_size"] = 1
            receipt["output"]["preparation"]["parsed_prefix_size"] = 1
        elif defect == "complete":
            receipt["output"]["preparation"]["complete_success"] = False
        else:
            receipt["output"]["normalized"]["count_basis"] = "final-index-rows"
        _json(receipt_path, receipt)
        rows[0]["receipts"][0] = _ref(receipt_path)
        _json(
            calibration,
            {
                "format": "polylogue.denominator-calibration.v1",
                "sampling_design": {"kind": "probability", "mechanism": "synthetic"},
                "samples": rows,
            },
        )
        with pytest.raises(ValueError):
            project(stage, calibration, tmp_path / f"refused-{defect}.json")


def test_streaming_physical_lines_have_partial_read_and_hash_coverage(
    private_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = private_path
    plan, root = _population(tmp_path)
    stage = tmp_path / "partial.jsonl"
    result = inventory(plan, stage, members=1, structural_bytes=9)
    rows = [json.loads(line) for line in stage.read_text().splitlines()]
    observed = next(row for row in rows if row["kind"] == "member")["structural"]
    assert result["complete"] is False
    assert result["structural_bytes_read"] <= 9
    assert observed["sha256"] is None
    assert observed["complete"] is False
    assert observed["nonblank_physical_lines"] == 1
    assert observed["not_normalized_messages"] is True
    full = tmp_path / "full.jsonl"
    inventory(plan, full, members=1, structural_bytes=100)
    record = next(json.loads(line) for line in full.read_text().splitlines() if json.loads(line)["kind"] == "member")
    assert record["structural"]["nonblank_physical_lines"] == 2
    assert record["structural"]["sha256"] == hashlib.sha256((root / "-synthetic/a.jsonl").read_bytes()).hexdigest()
    original_observation = denominator._observation
    target = root / "-synthetic/a.jsonl"
    injected = False

    def changing_observation(path: Path) -> dict[str, Any]:
        nonlocal injected
        observed = original_observation(path)
        if path == target and not injected:
            target.write_bytes(b"{}\n" * 10)
            injected = True
        return observed

    monkeypatch.setattr(denominator, "_observation", changing_observation)
    changed = tmp_path / "changed.jsonl"
    summary = inventory(plan, changed, members=1, structural_bytes=1000)
    changed_member = next(
        json.loads(line) for line in changed.read_text().splitlines() if json.loads(line)["kind"] == "member"
    )
    assert injected is True
    assert summary["changed_observations"] == 1
    assert changed_member["structural"]["state"] == "changed"
    assert changed_member["structural"]["sha256"] is None


@pytest.mark.parametrize("defect", ["record-count", "selection", "weight", "candidate"])
def test_projection_refuses_unbound_or_non_normalized_calibration(private_path: Path, defect: str) -> None:
    tmp_path = private_path
    plan, root = _population(tmp_path)
    stage = _stages(tmp_path, plan)
    sample = _sample(tmp_path, root, "a", 10, weight=2)
    if defect == "record-count":
        receipt_path = Path(sample["receipts"][0]["path"])
        receipt = json.loads(receipt_path.read_text())
        receipt["output"] = {"verdict": {"sample_count": 999}}
        _json(receipt_path, receipt)
        sample["receipts"][0] = _ref(receipt_path)
    elif defect == "selection":
        Path(sample["selection"]["path"]).write_text("{}")
    elif defect == "weight":
        sample["inclusion_probability"] = 1
    elif defect == "candidate":
        receipt_path = Path(sample["receipts"][0]["path"])
        receipt = json.loads(receipt_path.read_text())
        receipt["candidate"]["dirty"] = True
        _json(receipt_path, receipt)
        sample["receipts"][0] = _ref(receipt_path)
    calibration = _json(
        tmp_path / "calibration.json",
        {
            "format": "polylogue.denominator-calibration.v1",
            "sampling_design": {"kind": "probability", "mechanism": "synthetic"},
            "samples": [sample],
        },
    )
    with pytest.raises(ValueError):
        project(stage, calibration, tmp_path / "projection.json")
    assert not (tmp_path / "projection.json").exists()


def test_refused_root_and_changed_shared_metadata_remain_visible(private_path: Path) -> None:
    tmp_path = private_path
    plan, root = _population(tmp_path)
    (root / "-synthetic" / "escape.jsonl").symlink_to(tmp_path / "unavailable")
    first = tmp_path / "first.jsonl"
    inventory(plan, first, members=1)
    (root.parent / "history.jsonl").write_text("changed\n")
    second = tmp_path / "second.jsonl"
    summary = inventory(plan, second, members=2, previous=first)
    assert summary["changed_observations"] > 0
    rows = [json.loads(line) for line in first.read_text().splitlines()]
    assert rows[-1]["dispositions"]["fault:broken_symlink"] == 1
    payload = json.loads(plan.read_text())
    payload["roots"][1]["path"] = str(tmp_path / "missing-root")
    _json(plan, payload)
    with pytest.raises(ValueError):
        inventory(plan, tmp_path / "wrong-frontier.jsonl", members=1, previous=second)
    refused = inventory(plan, tmp_path / "refused.jsonl", members=100)
    assert refused["complete"] is True  # traversal ended; fault coverage is separately explicit
    assert refused["dispositions"]["fault:walk_refused"] == 1
